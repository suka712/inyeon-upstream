from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from chromadb.telemetry.product import ProductTelemetryClient, ProductTelemetryEvent
from overrides import override  # required by Chroma's Component metaclass

from .embeddings import RAGError


class VectorStoreError(RAGError):
    """Raised when vector store operations fail."""

    pass


class NoTelemetry(ProductTelemetryClient):
    """Product telemetry that does nothing.

    anonymized_telemetry=False only stops Chroma's Posthog client from
    sending: it still reads its user_id on every event, which creates
    ~/.cache/chroma/telemetry_user_id on the user's machine. Replacing the
    client means nothing is written at all.
    """

    @override
    def capture(self, event: ProductTelemetryEvent) -> None:
        pass


_NO_TELEMETRY = f"{NoTelemetry.__module__}.{NoTelemetry.__qualname__}"


class VectorStore:
    """ChromaDB wrapper for code embeddings storage."""

    def __init__(self, persist_dir: str | None = None, collection_name: str = "code"):
        settings = Settings(
            anonymized_telemetry=False,
            chroma_product_telemetry_impl=_NO_TELEMETRY,
            chroma_telemetry_impl=_NO_TELEMETRY,
        )

        if persist_dir:
            Path(persist_dir).mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=persist_dir, settings=settings)
        else:
            self.client = chromadb.Client(settings)

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        try:
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas or [{} for _ in ids],
            )
        except Exception as e:
            raise VectorStoreError(f"Failed to add documents: {e}")

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 5,
    ) -> list[dict[str, Any]]:
        """Search for similar documents."""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )

            items = []
            for i in range(len(results["ids"][0])):
                items.append(
                    {
                        "id": results["ids"][0][i],
                        "document": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i],
                    }
                )
            return items
        except Exception as e:
            raise VectorStoreError(f"Search failed: {e}")

    def delete(self, ids: list[str]) -> None:
        """Delete documents by ID."""
        try:
            self.collection.delete(ids=ids)
        except Exception as e:
            raise VectorStoreError(f"Failed to delete documents: {e}")

    def count(self) -> int:
        """Return total document count."""
        return self.collection.count()

    def clear(self) -> None:
        """Remove all documents from collection."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"},
        )
