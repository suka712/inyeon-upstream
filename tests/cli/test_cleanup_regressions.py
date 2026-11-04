"""Cleanup regressions: clients closed on their own loop, no Chroma telemetry file.

Both were seen in the evals/ runs (2026-11-04): a process that built several
SyncLocalBackends printed "RuntimeError: Event loop is closed", and every
local run wrote ~/.cache/chroma/telemetry_user_id with telemetry turned off.
"""

import asyncio

from backend.engine.local import LocalEngine
from backend.services.llm.gemini import GeminiProvider
from backend.services.llm.openai import OpenAIProvider
from cli.engine import SyncLocalBackend

DUMMY_KEY = "dummy-key-for-tests"


class _RecordingEngine:
    def __init__(self):
        self.closed_on = None
        self.loop_was_running = False

    async def aclose(self):
        self.closed_on = asyncio.get_running_loop()
        self.loop_was_running = not self.closed_on.is_closed()


def test_close_closes_the_engine_on_its_own_loop_before_the_loop():
    engine = _RecordingEngine()
    backend = SyncLocalBackend(engine)
    loop = backend._loop

    backend.close()

    assert engine.closed_on is loop
    assert engine.loop_was_running
    assert loop.is_closed()


def test_close_is_idempotent_and_tolerates_engines_without_aclose():
    backend = SyncLocalBackend(object())
    backend.close()
    backend.close()
    assert backend._loop.is_closed()


def test_openai_client_is_closed_when_the_backend_closes():
    engine = LocalEngine(llm_provider="openai", openai_api_key=DUMMY_KEY)
    provider = engine._get_llm()
    assert isinstance(provider, OpenAIProvider)

    SyncLocalBackend(engine).close()

    # Already closed, so its __del__ has nothing to schedule on a later loop.
    assert provider.client.is_closed()
    assert engine._llm is None


def test_gemini_provider_closes_cleanly():
    provider = GeminiProvider(api_key=DUMMY_KEY)
    asyncio.run(provider.aclose())


def test_vectorstore_writes_no_telemetry_file(tmp_path, monkeypatch):
    from chromadb.telemetry.product import ProductTelemetryClient

    from backend.rag.vectorstore import VectorStore

    user_id_file = tmp_path / "telemetry_user_id"
    monkeypatch.setattr(ProductTelemetryClient, "USER_ID_PATH", str(user_id_file))
    monkeypatch.setattr(ProductTelemetryClient, "_curr_user_id", None)

    store = VectorStore(collection_name="telemetry_regression")
    store.add(
        ids=["a"],
        embeddings=[[0.1, 0.2, 0.3]],
        documents=["x"],
        metadatas=[{"file": "x.py"}],
    )
    store.collection.query(query_embeddings=[[0.1, 0.2, 0.3]], n_results=1)

    assert not user_id_file.exists()
