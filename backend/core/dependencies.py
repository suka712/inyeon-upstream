from functools import lru_cache

from backend.services.llm import LLMProvider, create_llm_provider


@lru_cache
def get_llm_provider() -> LLMProvider:
    """
    Get shared LLM Provider instance based on configuration.
    """
    return create_llm_provider()
