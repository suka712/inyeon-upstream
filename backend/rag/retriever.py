from typing import Any

from .embeddings import EmbeddingService, RAGError
from .vectorstore import VectorStore


class RetrieverError(RAGError):
    """Raised when retrieval operations fail."""

    pass


class CodeRetriever:
    """High-level interface for indexing and searching code."""

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        vector_store: VectorStore | None = None,
        persist_dir: str | None = None,
    ):
        self.embeddings = embedding_service or EmbeddingService()
        self.store = vector_store or VectorStore(persist_dir=persist_dir)

    async def index_file(self, file_path: str, content: str) -> str:
        """Index a single file."""
        doc_id = file_path.replace("/", "_").replace("\\", "_")
        embedding = await self.embeddings.embed_text(content)

        self.store.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[{"path": file_path}],
        )
        return doc_id

    async def index_files(self, files: dict[str, str]) -> list[str]:
        """Index multiple files. Keys are paths, values are content."""
        if not files:
            return []

        ids = []
        contents = []
        metadatas = []

        for path, content in files.items():
            doc_id = path.replace("/", "_").replace("\\", "_")
            ids.append(doc_id)
            contents.append(content)
            metadatas.append({"path": path})

        embeddings = await self.embeddings.embed_texts(contents)

        self.store.add(
            ids=ids,
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas,
        )
        return ids

    async def search(self, query: str, n_results: int = 5) -> list[dict[str, Any]]:
        """Search for relevant code given a query."""
        query_embedding = await self.embeddings.embed_text(query)
        results = self.store.search(query_embedding, n_results=n_results)

        return [
            {
                "path": item["metadata"]["path"],
                "content": item["document"],
                "score": 1 - item["distance"],
            }
            for item in results
        ]

    async def search_for_diff(
        self, diff: str, n_results: int = 3
    ) -> list[dict[str, Any]]:
        """Search for code relevant to a diff."""
        return await self.search(f"Code related to:\n{diff}", n_results=n_results)

    def count(self) -> int:
        """Return number of indexed documents."""
        return self.store.count()

    def clear(self) -> None:
        """Clear all indexed documents."""
        self.store.clear()
