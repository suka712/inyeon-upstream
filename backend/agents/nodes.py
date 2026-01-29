from typing import Any

from .state import AgentState
from .tools import read_file
from backend.services.llm.base import LLMProvider


async def analyze_diff(state: AgentState, llm: LLMProvider) -> dict[str, Any]:
    """Analyze the diff and decide if more context is needed."""
    prompt = f"""Analyze this git diff and determine:
1. What files are being changed
2. What kind of changes are being made
3. Whether you need to read any related files for context

Respond in JSON:
{{
    "summary": "Brief summary of changes",
    "change_type": "feat|fix|refactor|docs|test|chore",
    "needs_context": true or false,
    "files_to_read": ["path/to/file.py"] or [],
    "reasoning": "Why you need/don't need more context"
}}

DIFF:
{state["diff"]}
"""

    response = await llm.generate(prompt, json_mode=True)

    return {
        "analysis": response,
        "needs_context": response.get("needs_context", False),
        "files_to_read": response.get("files_to_read", []),
        "reasoning": state["reasoning"] + [response.get("reasoning", "")],
    }


async def gather_context(state: AgentState, llm: LLMProvider) -> dict[str, Any]:
    """Read files the agent requested for additional context."""
    file_contents = {}

    for file_path in state["files_to_read"]:
        content = await read_file(file_path, state["repo_path"])
        file_contents[file_path] = content

    return {
        "file_contents": file_contents,
        "reasoning": state["reasoning"]
        + [f"Read {len(file_contents)} files for context"],
    }


async def generate_commit(state: AgentState, llm: LLMProvider) -> dict[str, Any]:
    """Generate the commit message with full context."""
    context = ""
    if state.get("file_contents"):
        context = "\n\nRELATED FILES:\n"
        for path, content in state["file_contents"].items():
            context += f"\n--- {path} ---\n{content}\n"

    prompt = f"""Generate a git commit message following Conventional Commits format.

ANALYSIS:
{state.get("analysis", {})}

DIFF:
{state["diff"]}
{context}

Respond in JSON:
{{
    "type": "feat|fix|refactor|docs|test|chore",
    "scope": "optional scope",
    "subject": "imperative description under 50 chars",
    "body": "optional detailed explanation",
    "message": "full formatted commit message"
}}
"""

    response = await llm.generate(prompt, json_mode=True)

    return {
        "commit_message": response.get("message", ""),
        "reasoning": state["reasoning"] + ["Generated commit message"],
    }


def should_gather_context(state: AgentState) -> str:
    """Conditional edge: decide next node based on state."""
    if state.get("needs_context") and state.get("files_to_read"):
        return "gather_context"
    return "generate_commit"
