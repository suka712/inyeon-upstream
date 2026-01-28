from functools import lru_cache

from backend.services.ollama_client import OllamaClient


@lru_cache
def get_ollama_client() -> OllamaClient:
    """
    Get shared OllamaClient instance.

    Uses lru_cache to ensure single instance across requests.
    """
    return OllamaClient()
