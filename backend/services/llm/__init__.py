"""
LLM Provider Package
====================

This package provides a unified interface for different LLM providers.
It allows the application to work with Ollama, Gemini, OpenAI, etc.
without changing the core logic.

Design Pattern: Strategy Pattern
- LLMProvider is the abstract strategy
- OllamaProvider, GeminiProvider are concrete strategies
- The application depends on the abstraction, not implementations

This is the "Dependency Inversion Principle" from SOLID:
- High-level modules (agents) should not depend on low-level modules (Ollama API)
- Both should depend on abstractions (LLMProvider interface)
"""

from .base import LLMProvider
from .ollama import OllamaProvider, OllamaError
from .gemini import GeminiProvider, GeminiError
from .factory import create_llm_provider, ProviderType, ProviderConfigError

__all__ = [
    "LLMProvider",
    "OllamaProvider",
    "OllamaError",
    "GeminiProvider",
    "GeminiError",
    "create_llm_provider",
    "ProviderType",
    "ProviderConfigError",
]
