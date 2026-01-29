from typing import Any

from google import genai
from google.genai import types

from .base import LLMProvider


class GeminiError(Exception):
    """Raised when Gemini request fails."""

    pass


class GeminiProvider(LLMProvider):
    """
    Google Gemini implementation of LLMProvider.

    Free tier: 15 RPM, 1M tokens/day.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        timeout: int = 120,
    ):
        self.model_name = model
        self.timeout = timeout
        self.client = genai.Client(api_key=api_key)

    async def generate(
        self,
        prompt: str,
        json_mode: bool = False,
        temperature: float = 0.3,
    ) -> dict[str, Any]:
        """Generate completion from Gemini."""
        try:
            config = types.GenerateContentConfig(
                temperature=temperature,
                response_mime_type="application/json" if json_mode else "text/plain",
            )

            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )

            if json_mode:
                import json

                return json.loads(response.text)
            return {"text": response.text}

        except Exception as e:
            raise GeminiError(f"Gemini request failed: {e}")

    async def generate_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> dict[str, Any]:
        """Generate completion with tool-calling capability."""
        try:
            gemini_tools = self._convert_tools(tools)
            contents = self._convert_messages(messages)

            config = types.GenerateContentConfig(
                tools=gemini_tools if gemini_tools else None,
            )

            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config,
            )

            result = {"content": response.text or "", "tool_calls": []}

            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if part.function_call:
                        result["tool_calls"].append(
                            {
                                "function": {
                                    "name": part.function_call.name,
                                    "arguments": dict(part.function_call.args),
                                }
                            }
                        )

            return result

        except Exception as e:
            raise GeminiError(f"Gemini request failed: {e}")

    async def is_healthy(self) -> bool:
        """Check if Gemini is available."""
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents="ping",
            )
            return response is not None
        except Exception:
            return False

    def _convert_messages(self, messages: list[dict]) -> list[types.Content]:
        """Convert standard messages to Gemini format."""
        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=msg["content"])],
                )
            )
        return contents

    def _convert_tools(self, tools: list[dict]) -> list[types.Tool]:
        """Convert standard tools to Gemini format."""
        if not tools:
            return []

        function_declarations = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                function_declarations.append(
                    types.FunctionDeclaration(
                        name=func["name"],
                        description=func["description"],
                        parameters=func.get("parameters", {}),
                    )
                )

        return (
            [types.Tool(function_declarations=function_declarations)]
            if function_declarations
            else []
        )
