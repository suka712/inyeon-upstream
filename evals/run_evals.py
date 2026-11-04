"""Run Inyeon agents over the dataset in evals/dataset_manifest.json.

Every config writes its own raw file: evals/results/raw/<config>.json (resumable;
finished cases are skipped on re-run). No repo source file is modified: token
usage and call counts are captured by wrapping methods on the provider
*instances* at runtime (see Meter / instrument_*).

Usage:
  python evals/run_evals.py --list
  python evals/run_evals.py split-ollama-directory commit-openai ...
  python evals/run_evals.py --budget-check
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
EVALS = ROOT / "evals"
RAW = EVALS / "results" / "raw"
os.chdir(ROOT)  # cli.config reads .env relative to cwd
sys.path.insert(0, str(ROOT))

from cli.config import settings  # noqa: E402  (reads .env; values never printed)
from backend.engine.local import LocalEngine  # noqa: E402
from backend.agents.split_agent import SplitAgent  # noqa: E402
from backend.rag.embeddings import EmbeddingService  # noqa: E402

# Prices used ONLY for the spend guard / spend report (USD per 1M tokens).
PRICES = {
    "gpt-4.1-mini": {"input": 0.40, "cached_input": 0.10, "output": 1.60},
    "gemini-embedding-001": {"input": 0.15},
}
HARD_BUDGET_USD = 1.00
RETRY_PATTERN: str | None = None  # set by --retry-errors
PACE_S = 0.0
ABORT_AT_USD = 0.90  # stop paid configs before the hard budget

_QWEN_TOK = None


def qwen_tokens(text: str) -> int | None:
    """Exact Qwen2.5 token count of the raw prompt (no chat template)."""
    global _QWEN_TOK
    path = EVALS / ".cache" / "qwen2.5-coder-7b-instruct-tokenizer.json"
    if not path.exists():
        return None
    if _QWEN_TOK is None:
        from tokenizers import Tokenizer

        _QWEN_TOK = Tokenizer.from_file(str(path))
    return len(_QWEN_TOK.encode(text).ids)


# --------------------------------------------------------------------------- meter
class Meter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.llm_calls = 0  # logical LLMProvider.generate()/generate_with_tools() calls
        self.requests: list[dict] = []  # physical HTTP requests with provider usage
        self.embed_calls: list[dict] = []
        self.call_errors: list[str] = []  # exceptions raised by LLMProvider.generate()

    def snapshot(self) -> dict:
        return {
            "llm_calls": self.llm_calls,
            "requests": list(self.requests),
            "embed_calls": list(self.embed_calls),
            "call_errors": list(self.call_errors),
        }


METER = Meter()


def instrument_llm(llm, provider: str) -> None:
    orig_generate = llm.generate
    orig_tools = llm.generate_with_tools

    async def generate(prompt, json_mode=False, temperature=0.3):
        METER.llm_calls += 1
        try:
            return await orig_generate(prompt, json_mode=json_mode, temperature=temperature)
        except Exception as e:
            METER.call_errors.append(f"{type(e).__name__}: {str(e)[:300]}")
            raise

    async def generate_with_tools(messages, tools):
        METER.llm_calls += 1
        return await orig_tools(messages, tools)

    llm.generate = generate
    llm.generate_with_tools = generate_with_tools

    if provider == "ollama":
        orig_post = llm._client.post

        async def post(url, *args, **kwargs):
            payload = kwargs.get("json") or {}
            t0 = time.perf_counter()
            resp = await orig_post(url, *args, **kwargs)
            dt = time.perf_counter() - t0
            rec = {"provider": "ollama", "url": url, "latency_s": dt, "status": resp.status_code}
            try:
                body = resp.json()
                rec["input_tokens"] = body.get("prompt_eval_count")
                rec["output_tokens"] = body.get("eval_count")
                rec["done_reason"] = body.get("done_reason")
            except Exception:
                pass
            prompt = payload.get("prompt")
            if isinstance(prompt, str):
                rec["prompt_chars"] = len(prompt)
                raw = qwen_tokens(prompt)
                rec["prompt_raw_tokens_qwen"] = raw
                if raw is not None and rec.get("input_tokens") is not None:
                    # Ollama evaluated fewer tokens than the prompt contains => it
                    # silently truncated the prompt to fit its context window.
                    rec["context_truncated"] = rec["input_tokens"] < raw
            METER.requests.append(rec)
            return resp

        llm._client.post = post

    elif provider == "openai":
        completions = llm.client.chat.completions
        orig_create = completions.create

        async def create(*args, **kwargs):
            t0 = time.perf_counter()
            resp = await orig_create(*args, **kwargs)
            dt = time.perf_counter() - t0
            rec = {"provider": "openai", "model": kwargs.get("model"), "latency_s": dt}
            u = getattr(resp, "usage", None)
            if u is not None:
                rec["input_tokens"] = u.prompt_tokens
                rec["output_tokens"] = u.completion_tokens
                det = getattr(u, "prompt_tokens_details", None)
                rec["cached_input_tokens"] = getattr(det, "cached_tokens", 0) if det else 0
            METER.requests.append(rec)
            return resp

        completions.create = create


def make_embedding_service(model: str) -> EmbeddingService:
    svc = EmbeddingService(api_key=settings.gemini_api_key)
    svc.model = model  # "text-embedding-004" is the shipped value
    orig = svc.embed_texts

    async def embed_texts(texts):
        rec = {"model": model, "n_texts": len(texts), "chars": sum(len(t) for t in texts)}
        t0 = time.perf_counter()
        try:
            out = await orig(texts)
            rec["ok"] = True
            return out
        except Exception as e:
            rec["ok"] = False
            rec["error"] = str(e)[:300]
            raise
        finally:
            rec["latency_s"] = time.perf_counter() - t0
            METER.embed_calls.append(rec)

    svc.embed_texts = embed_texts
    return svc


def make_engine(provider: str) -> LocalEngine:
    # Mirrors cli/engine.py create_engine(local=True, provider=...)
    return LocalEngine(
        llm_provider=provider,
        ollama_url=settings.ollama_url,
        ollama_model=settings.ollama_model,
        gemini_api_key=settings.gemini_api_key,
        gemini_model=settings.gemini_model,
        openai_api_key=settings.openai_api_key,
        openai_model=settings.openai_model,
        timeout=settings.ollama_timeout,
    )


# --------------------------------------------------------------------------- configs
CONFIGS: dict[str, dict] = {}
for prov in ("ollama", "openai"):
    for strat in ("directory", "semantic", "conventional", "hybrid"):
        CONFIGS[f"split-{prov}-{strat}"] = {"eval": "split", "provider": prov, "strategy": strat, "embedding": None}
    for emb, tag in (("text-embedding-004", "te004"), ("gemini-embedding-001", "gemb001")):
        for strat in ("semantic", "hybrid"):
            CONFIGS[f"split-{prov}-{strat}-inject-{tag}"] = {
                "eval": "split", "provider": prov, "strategy": strat, "embedding": emb,
            }
    CONFIGS[f"commit-{prov}"] = {"eval": "commit", "provider": prov}
    CONFIGS[f"auto-{prov}"] = {"eval": "auto", "provider": prov, "skip_pr": False}
    CONFIGS[f"auto-{prov}-nopr"] = {"eval": "auto", "provider": prov, "skip_pr": True}


# --------------------------------------------------------------------------- helpers
def git_bytes(repo: str, *args: str) -> bytes:
    return subprocess.run(["git", "-C", repo, "-c", "core.quotepath=false", *args], capture_output=True, check=True).stdout


def regen_diff(repo: str, a: str, b: str, expect_sha: str) -> str:
    diff = git_bytes(repo, "diff", "--no-color", "--no-ext-diff", a, b).decode("utf-8", errors="replace")
    got = hashlib.sha256(diff.encode("utf-8")).hexdigest()
    if got != expect_sha:
        raise RuntimeError(f"diff hash mismatch for {a}..{b}")
    return diff


def materialize(repo: str, sha: str) -> str:
    """Extract the tree at `sha` so the commit agent's read_file tool sees the
    files as of that commit (read-only; the source repo is untouched)."""
    d = tempfile.mkdtemp(prefix="inyeon-eval-")
    data = git_bytes(repo, "archive", "--format=tar", sha)
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        tf.extractall(d, filter="data")
    return d


def load_raw(name: str) -> dict:
    p = RAW / f"{name}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"config": name, "settings": CONFIGS[name], "cases": {}}


def save_raw(name: str, data: dict) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    tmp = RAW / f"{name}.json.tmp"
    tmp.write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
    tmp.replace(RAW / f"{name}.json")


def spend_of(records: list[dict]) -> float:
    usd = 0.0
    for r in records:
        if r.get("provider") == "openai" and r.get("input_tokens") is not None:
            p = PRICES["gpt-4.1-mini"]
            cached = r.get("cached_input_tokens") or 0
            usd += (r["input_tokens"] - cached) * p["input"] / 1e6 + cached * p["cached_input"] / 1e6
            usd += r["output_tokens"] * p["output"] / 1e6
    return usd


def embed_spend_upper_bound(embed_calls: list[dict]) -> float:
    # Gemini embedContent returns no token usage. Upper bound: tokens <= chars.
    return sum(e["chars"] for e in embed_calls if e.get("ok") and e["model"] == "gemini-embedding-001") * PRICES["gemini-embedding-001"]["input"] / 1e6


def total_spend_so_far() -> float:
    usd = 0.0
    if RAW.exists():
        for p in RAW.rglob("*.json"):
            d = json.loads(p.read_text(encoding="utf-8"))
            for c0 in d["cases"].values():
                for c in [c0] + c0.get("previous_attempts", []):
                    usd += spend_of(c.get("meter", {}).get("requests", []))
                    usd += embed_spend_upper_bound(c.get("meter", {}).get("embed_calls", []))
    return usd


# --------------------------------------------------------------------------- runners
async def run_split(name: str, cfg: dict, cases: list[dict]) -> None:
    data = load_raw(name)
    engine = make_engine(cfg["provider"])
    llm = engine._get_llm()
    instrument_llm(llm, cfg["provider"])
    if cfg["embedding"]:
        retriever = SimpleNamespace(embeddings=make_embedding_service(cfg["embedding"]))
        data["retriever"] = f"injected EmbeddingService(model={cfg['embedding']})"
    else:
        retriever = engine._get_retriever()  # what LocalEngine.split_diff uses
        data["retriever"] = repr(retriever)
    for case in cases:
        cid = case["case_id"]
        prev_attempts = []
        if cid in data["cases"]:
            prev = data["cases"][cid]
            if not (RETRY_PATTERN and RETRY_PATTERN in (prev.get("error") or "")):
                continue
            # Re-run a case that failed on provider quota (HTTP 429), keeping the
            # failed attempt (and its metered usage) in the record.
            prev_attempts = prev.pop("previous_attempts", []) + [prev]
            await asyncio.sleep(PACE_S)
        if cfg["provider"] == "openai" and total_spend_so_far() > ABORT_AT_USD:
            print(f"[{name}] spend guard hit; stopping", flush=True)
            break
        diff = regen_diff(case["repo_path"], case["base"], case["commits"][-1], case["diff_sha256"])
        METER.reset()
        agent = SplitAgent(llm, retriever)
        t0 = time.perf_counter()
        rec: dict = {}
        try:
            # Same as LocalEngine.split_diff, but keeps the hunk-level commit_groups
            # (the public result only exposes file lists per split).
            state = await agent.graph.ainvoke(
                agent._initial_state(engine._prepare_diff(diff), case["repo_path"], cfg["strategy"])
            )
            rec["groups"] = [
                {
                    "id": g.id,
                    "hunks": [h.hunk_id for h in g.hunks],
                    "files": g.files,
                    "suggested_type": g.suggested_type,
                }
                for g in state.get("commit_groups", [])
            ]
            rec["splits"] = state.get("splits", [])
            rec["error"] = state.get("error")
            rec["reasoning"] = state.get("reasoning", [])
        except Exception as e:
            rec["error"] = f"exception: {type(e).__name__}: {e}"
            rec["traceback"] = traceback.format_exc()[-2000:]
        rec["wall_s"] = time.perf_counter() - t0
        rec["meter"] = METER.snapshot()
        if prev_attempts:
            rec["previous_attempts"] = prev_attempts
        data["cases"][cid] = rec
        save_raw(name, data)
        print(f"[{name}] {cid} groups={len(rec.get('groups', []))} err={bool(rec.get('error'))} "
              f"calls={rec['meter']['llm_calls']} {rec['wall_s']:.1f}s", flush=True)


async def run_commit(name: str, cfg: dict, cases: list[dict]) -> None:
    data = load_raw(name)
    engine = make_engine(cfg["provider"])
    llm = engine._get_llm()
    instrument_llm(llm, cfg["provider"])
    data["retriever"] = repr(engine._get_retriever())
    for case in cases:
        cid = case["case_id"]
        if cid in data["cases"]:
            continue
        if cfg["provider"] == "openai" and total_spend_so_far() > ABORT_AT_USD:
            print(f"[{name}] spend guard hit; stopping", flush=True)
            break
        diff = regen_diff(case["repo_path"], case["parent"], case["sha"], case["diff_sha256"])
        tree = materialize(case["repo_path"], case["sha"])
        METER.reset()
        t0 = time.perf_counter()
        try:
            res = await engine.generate_commit(diff, repo_path=tree)
            rec = {
                "commit_message": res.data.get("commit_message"),
                "analysis": res.data.get("analysis"),
                "reasoning": res.reasoning,
                "error": res.error,
            }
        except Exception as e:
            rec = {"error": f"exception: {type(e).__name__}: {e}"}
        rec["wall_s"] = time.perf_counter() - t0
        rec["meter"] = METER.snapshot()
        shutil.rmtree(tree, ignore_errors=True)
        data["cases"][cid] = rec
        save_raw(name, data)
        print(f"[{name}] {cid} calls={rec['meter']['llm_calls']} {rec['wall_s']:.1f}s err={bool(rec.get('error'))} "
              f"msg={str(rec.get('commit_message'))[:70]!r}", flush=True)


def run_auto(name: str, cfg: dict, cases: list[dict]) -> None:
    """Exercise cli.pipeline.Pipeline through SyncLocalBackend exactly as
    `inyeon auto --local --dry-run` does (one asyncio.run per step)."""
    from cli.engine import SyncLocalBackend
    from cli.pipeline import Pipeline

    data = load_raw(name)
    for case in cases:
        cid = case["case_id"]
        if cid in data["cases"]:
            continue
        diff = regen_diff(case["repo_path"], case["parent"], case["sha"], case["diff_sha256"])
        engine = make_engine(cfg["provider"])
        instrument_llm(engine._get_llm(), cfg["provider"])
        METER.reset()
        t0 = time.perf_counter()
        try:
            r = Pipeline(SyncLocalBackend(engine)).run(
                diff=diff, commits=[], branch_name="eval", base_branch="main",
                skip_review=False, skip_pr=cfg["skip_pr"],
            )
            rec = {
                "steps_completed": r.steps_completed,
                "steps_skipped": r.steps_skipped,
                "commit_message": r.commit_message,
                "has_review": r.review is not None,
                "has_pr": r.pr_description is not None,
                "error": r.error,
            }
        except Exception as e:
            rec = {"error": f"exception: {type(e).__name__}: {e}", "traceback": traceback.format_exc()[-2000:]}
        rec["wall_s"] = time.perf_counter() - t0
        rec["diff_chars"] = len(diff)
        rec["meter"] = METER.snapshot()
        data["cases"][cid] = rec
        save_raw(name, data)
        print(f"[{name}] {cid} chars={len(diff)} calls={rec['meter']['llm_calls']} "
              f"done={rec.get('steps_completed')} skipped={rec.get('steps_skipped')} err={rec.get('error')}", flush=True)


def budget_check(manifest: dict) -> None:
    """Pre-run spend estimate for the paid configs (chars/3 ~ tokens, generous)."""
    p = PRICES["gpt-4.1-mini"]
    commit_chars = sum(c["diff_chars"] for c in manifest["commit_cases"])
    # commit agent: analyze prompt + generate prompt each contain the diff (+~1.5k chars overhead),
    # plus up to 10k chars per file read via gather_context (assume 2 files on every case).
    commit_in = (2 * commit_chars + 30 * (3000 + 20000)) / 3
    commit_out = 30 * 2 * 400
    n_split_cfg = 8  # all split configs run with gpt-4.1-mini (Ollama unavailable)
    split_in = n_split_cfg * 30 * (8 * 1500) / 3  # ~8 prompts/case, <=1.5k chars each
    split_out = n_split_cfg * 30 * 8 * 150
    auto = 2 * 5 * 4 * (2 * 2000 / 3)  # 2 auto configs x 5 cases x <=4 calls
    split_in += auto
    oa = (commit_in + split_in) * p["input"] / 1e6 + (commit_out + split_out) * p["output"] / 1e6
    emb_chars = 2 * sum(c["diff_chars"] for c in manifest["split_cases"])
    emb = emb_chars * PRICES["gemini-embedding-001"]["input"] / 1e6
    print(json.dumps({"openai_estimate_usd": round(oa, 4), "gemini_embedding_upper_bound_usd": round(emb, 4),
                      "total_estimate_usd": round(oa + emb, 4), "hard_budget_usd": HARD_BUDGET_USD}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="*")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--budget-check", action="store_true")
    ap.add_argument("--retry-errors", default=None, help="re-run split cases whose error contains this text")
    ap.add_argument("--pace", type=float, default=0.0, help="seconds to wait before each retried case")
    args = ap.parse_args()
    global RETRY_PATTERN, PACE_S
    RETRY_PATTERN, PACE_S = args.retry_errors, args.pace
    manifest = json.loads((EVALS / "dataset_manifest.json").read_text(encoding="utf-8"))
    if args.list:
        print("\n".join(CONFIGS))
        return
    if args.budget_check:
        budget_check(manifest)
        print("spend so far (from raw results):", round(total_spend_so_far(), 4))
        return
    for name in args.configs:
        cfg = CONFIGS[name]
        if cfg["eval"] == "split":
            asyncio.run(run_split(name, cfg, manifest["split_cases"]))
        elif cfg["eval"] == "commit":
            asyncio.run(run_commit(name, cfg, manifest["commit_cases"]))
        else:
            run_auto(name, cfg, manifest["auto_cases"])


if __name__ == "__main__":
    main()
