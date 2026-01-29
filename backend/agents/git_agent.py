from typing import Any

from langgraph.graph import StateGraph, END

from .state import AgentState
from .nodes import analyze_diff, gather_context, generate_commit, should_gather_context
from backend.services.llm.base import LLMProvider


class GitAgent:
    """
    Git workflow agent that analyzes diffs and generates commit messages.

    Uses LangGraph to orchestrate a multi-step workflow:
    START -> analyze -> [gather_context?] -> generate_commit -> END
    """

    def __init__(self, llm: LLMProvider):
        self.llm = llm
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Construct the LangGraph state machine."""
        graph = StateGraph(AgentState)

        graph.add_node("analyze", lambda s: analyze_diff(s, self.llm))
        graph.add_node("gather_context", lambda s: gather_context(s, self.llm))
        graph.add_node("generate_commit", lambda s: generate_commit(s, self.llm))

        graph.set_entry_point("analyze")

        graph.add_conditional_edges(
            "analyze",
            should_gather_context,
            {
                "gather_context": "gather_context",
                "generate_commit": "generate_commit",
            },
        )

        graph.add_edge("gather_context", "generate_commit")
        graph.add_edge("generate_commit", END)

        return graph.compile()

    async def run(self, diff: str, repo_path: str = ".") -> dict[str, Any]:
        """Run the agent on a diff."""
        initial_state: AgentState = {
            "diff": diff,
            "repo_path": repo_path,
            "analysis": None,
            "needs_context": False,
            "files_to_read": [],
            "file_contents": {},
            "commit_message": None,
            "reasoning": [],
        }

        final_state = await self.graph.ainvoke(initial_state)

        return {
            "commit_message": final_state.get("commit_message"),
            "reasoning": final_state.get("reasoning", []),
            "analysis": final_state.get("analysis"),
        }
