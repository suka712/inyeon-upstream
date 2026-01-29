from typing import Any, TypedDict


class AgentState(TypedDict):
    # --- Input ---
    diff: str
    """The git diff to analyze."""

    repo_path: str
    """Path to the repository root (for file reading)."""

    # --- Analysis Phase ---
    analysis: dict[str, Any] | None
    """Output from the analyze node: summary, change_type, etc."""

    needs_context: bool
    """Whether the agent needs to read additional files."""

    files_to_read: list[str]
    """List of file paths the agent wants to read for context."""

    # --- Context Phase ---
    file_contents: dict[str, str]
    """Contents of files read by the agent. Format: {path: content}."""

    # --- Output ---
    commit_message: str | None
    """The final generated commit message."""

    reasoning: list[str]
    """Agent's reasoning steps (for verbose/debug mode)."""
