from abc import ABC, abstractmethod
from typing import Any


class LLMError(Exception):
    """Base exception for all LLM provider errors."""

    pass


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        json_mode: bool = False,
        temperature: float = 0.3,
    ) -> dict[str, Any]:
        """Generate a completion from the LLM."""
        pass

    @abstractmethod
    async def generate_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> dict[str, Any]:
        """Generate a completion with tool-calling capability."""
        pass

    @abstractmethod
    async def is_healthy(self) -> bool:
        """Check if the LLM provider is available."""
        pass
