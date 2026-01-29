"""
Agents Package
==============

This package contains agentic workflows built with LangGraph.

Agents are autonomous systems that can:
- Analyze input and decide what actions to take
- Use tools to interact with the environment
- Iterate based on results until the task is complete

Main components:
- GitAgent: Analyzes diffs and generates commit messages
- AgentState: State that flows through the agent graph
"""

from .state import AgentState
from .git_agent import GitAgent

__all__ = ["AgentState", "GitAgent"]
