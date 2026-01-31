from google import genai

from backend.core.config import settings


class RAGError(Exception):
    """Base exception for all RAG-related errors."""

    pass


class EmbeddingError(RAGError):
    """Raised when embedding generation fails."""

    pass


class EmbeddingService:
    """Generate embeddings using Gemini API."""

    def __init__(self, api_key: str | None = None):
        self.client = genai.Client(api_key=api_key or settings.gemini_api_key)
        self.model = "text-embedding-004"

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        try:
            response = await self.client.aio.models.embed_content(
                model=self.model,
                contents=text,
            )
            return response.embeddings[0].values
        except Exception as e:
            raise EmbeddingError(f"Failed to generate embedding: {e}")

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        try:
            response = await self.client.aio.models.embed_content(
                model=self.model,
                contents=texts,
            )
            return [emb.values for emb in response.embeddings]
        except Exception as e:
            raise EmbeddingError(f"Failed to generate embeddings: {e}")
