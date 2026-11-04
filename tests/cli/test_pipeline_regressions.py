"""Regressions found by evals/ (2026-11-04) that the existing suite missed.

The pipeline tests mocked the commit response as {"message": ...}, a shape no
backend actually returns, so they agreed with the bug instead of the contract.
These use the real shapes.
"""

import asyncio
from unittest.mock import MagicMock

from cli.engine import SyncLocalBackend
from cli.pipeline import Pipeline

SMALL_DIFF = """diff --git a/main.py b/main.py
--- a/main.py
+++ b/main.py
@@ -1,2 +1,3 @@
 def main():
+    print("Hello")
     pass
"""


def _client(**returns) -> MagicMock:
    client = MagicMock()
    client.generate_commit.return_value = returns.get(
        "commit", {"commit_message": "feat: add print", "reasoning": []}
    )
    client.review.return_value = returns.get("review", {"review": None})
    client.generate_pr.return_value = {"pr_description": None, "error": None}
    return client


def test_commit_message_read_from_the_key_the_agent_returns():
    result = Pipeline(_client()).run(SMALL_DIFF, skip_review=True, skip_pr=True)
    assert result.commit_message == "feat: add print"
    assert "commit" in result.steps_completed


def test_commit_error_is_reported_not_swallowed():
    client = _client(commit={"error": "provider down"})
    result = Pipeline(client).run(SMALL_DIFF, skip_review=True, skip_pr=True)
    assert result.commit_message is None
    assert result.error and "provider down" in result.error
    assert "commit" not in result.steps_completed


def test_failed_review_is_skipped_not_completed():
    client = _client(review={"error": "Connection error"})
    result = Pipeline(client).run("x" * 600, skip_pr=True)
    assert "review" in result.steps_skipped
    assert "review" not in result.steps_completed


def test_empty_review_is_skipped_not_completed():
    result = Pipeline(_client(review={"review": {}})).run("x" * 600, skip_pr=True)
    assert "review" in result.steps_skipped


class _LoopBoundEngine:
    """Fails the way a cached async client did: bound to its first loop."""

    def __init__(self):
        self.loop = None

    async def _call(self):
        loop = asyncio.get_running_loop()
        if self.loop is None:
            self.loop = loop
        if loop is not self.loop:
            raise RuntimeError("Connection error: client bound to another loop")
        return MagicMock(data={"commit_message": "ok", "review": {"summary": "ok"}}, error=None)

    async def generate_commit(self, diff, issue_ref=None):
        return await self._call()

    async def review(self, diff):
        return await self._call()


def test_sync_backend_reuses_one_event_loop_across_steps():
    backend = SyncLocalBackend(_LoopBoundEngine())
    try:
        assert backend.generate_commit("d")["commit_message"] == "ok"
        assert backend.review("d")["review"] == {"summary": "ok"}
    finally:
        backend.close()


def test_local_engine_builds_a_retriever_with_its_gemini_key(monkeypatch):
    import backend.rag as rag
    from backend.engine.local import LocalEngine

    built = {}

    class FakeEmbeddings:
        def __init__(self, api_key=None):
            built["api_key"] = api_key

    class FakeRetriever:
        def __init__(self, embedding_service=None, vector_store=None, persist_dir=None):
            built["service"] = embedding_service

    monkeypatch.setattr(rag, "EmbeddingService", FakeEmbeddings)
    monkeypatch.setattr(rag, "CodeRetriever", FakeRetriever)

    retriever = LocalEngine(gemini_api_key="k")._get_retriever()
    assert isinstance(retriever, FakeRetriever)
    assert built["api_key"] == "k"
    assert isinstance(built["service"], FakeEmbeddings)


def test_embeddings_use_a_model_that_still_exists():
    from backend.rag.embeddings import EmbeddingService

    assert EmbeddingService(api_key="k").model != "text-embedding-004"
