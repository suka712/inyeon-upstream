"""After-fixes eval: re-run `auto` through the shipped local path and calibrate
the semantic clustering threshold on a by-repo calibration/held-out split.

Reuses evals/run_evals.py (dataset regeneration, LLM metering, spend accounting)
and evals/metrics.py (ARI, bootstrap). Raw outputs go to
evals/results/raw/after_fixes/. Embedding vectors are cached in evals/.cache/
(delete it when done; it is not a result).

No harness workaround is applied: the retriever comes from
LocalEngine._get_retriever() (the shipped construction), the embedding model is
whatever backend/rag/embeddings.py ships, and the threshold comes from the
SemanticStrategy default unless a sweep passes one explicitly.

Usage:
  python evals/after_fixes.py estimate
  python evals/after_fixes.py auto               # 5 cases x {default, --no-pr}
  python evals/after_fixes.py embed              # cache hunk vectors, all 30 split cases
  python evals/after_fixes.py sweep              # offline, calibration repos only -> chosen threshold
  python evals/after_fixes.py heldout            # offline scoring on held-out repos
  python evals/after_fixes.py live-heldout       # shipped SplitAgent path on held-out repos (paid)
  python evals/after_fixes.py spend
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import statistics
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_evals as RE  # noqa: E402  (chdir to repo root, reads .env, keys never printed)
from metrics import ari, bootstrap_ci  # noqa: E402

from backend.diff import DiffParser  # noqa: E402
from backend.clustering import DirectoryStrategy, HybridStrategy, SemanticStrategy  # noqa: E402

OUT = RE.EVALS / "results" / "raw" / "after_fixes"
CACHE = RE.EVALS / ".cache" / "embeddings_gemini-embedding-001_768.json"
RE.RAW = OUT  # run_evals helpers (load_raw/save_raw/total_spend_so_far) now use after_fixes/
HARD_CAP_USD = 0.50
STOP_AT_USD = 0.45
SPLIT_SEED = 20261104
# Pre-registered grid (before looking at any after-fix result).
GRID = [round(0.50 + 0.025 * i, 3) for i in range(19)]  # 0.500 .. 0.950
MANIFEST = json.loads((RE.EVALS / "dataset_manifest.json").read_text(encoding="utf-8"))


def write_json(name: str, data) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")


# ------------------------------------------------------------------ repo split
def repo_split() -> tuple[list[str], list[str]]:
    """Deterministic by-repo split: shuffle repo names with a fixed seed, first 3
    repos (18 cases) calibrate, the remaining 2 (12 cases) are held out."""
    repos = sorted({c["repo"] for c in MANIFEST["split_cases"]})
    random.Random(SPLIT_SEED).shuffle(repos)
    return sorted(repos[:3]), sorted(repos[3:])


def cases_for(repos: list[str]) -> list[dict]:
    return [c for c in MANIFEST["split_cases"] if c["repo"] in repos]


# ------------------------------------------------------------------ spend
def spend_now() -> dict:
    oa, emb = 0.0, 0.0
    if OUT.exists():
        for p in OUT.glob("*.json"):
            d = json.loads(p.read_text(encoding="utf-8"))
            for c0 in (d.get("cases") or {}).values():
                for c in [c0] + c0.get("previous_attempts", []):
                    oa += RE.spend_of(c.get("meter", {}).get("requests", []))
                    emb += RE.embed_spend_upper_bound(c.get("meter", {}).get("embed_calls", []))
    if (OUT / "embed_log.json").exists():
        for e in json.loads((OUT / "embed_log.json").read_text(encoding="utf-8"))["calls"]:
            if e.get("ok"):
                emb += e["chars"] * RE.PRICES["gemini-embedding-001"]["input"] / 1e6
    return {"openai_usd": oa, "gemini_upper_bound_usd": emb, "total_upper_bound_usd": oa + emb}


def guard() -> None:
    s = spend_now()["total_upper_bound_usd"]
    if s > STOP_AT_USD:
        raise SystemExit(f"spend guard: {s:.4f} USD > {STOP_AT_USD}")


# ------------------------------------------------------------------ parsing / texts
def parsed_case(case: dict, engine) -> tuple[str, object]:
    diff = RE.regen_diff(case["repo_path"], case["base"], case["commits"][-1], case["diff_sha256"])
    prepared = engine._prepare_diff(diff)  # exactly what LocalEngine.split_diff feeds the agent
    return prepared, DiffParser().parse(prepared)


def hunk_text(file, hunk) -> str:
    # Must stay identical to SemanticStrategy.cluster's hunk_texts.
    return f"File: {file.path}\n{hunk.section_header}\n{hunk.content}"


def key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class CachedEmbeddings:
    """Offline stand-in for EmbeddingService: returns vectors cached from the
    shipped EmbeddingService. Used only for the offline sweep/scoring."""

    def __init__(self, cache: dict):
        self.cache = cache

    async def embed_texts(self, texts):
        return [self.cache[key(t)] for t in texts]


class NoLLM:
    """Hybrid only uses the LLM for a type label (ConventionalStrategy falls back
    to 'chore' on an exception); grouping does not depend on it."""

    async def generate(self, *a, **k):
        raise RuntimeError("offline")


# ------------------------------------------------------------------ estimate
def estimate() -> None:
    cal, held = repo_split()
    chars_all = sum(c["diff_chars"] for c in MANIFEST["split_cases"])
    chars_held = sum(c["diff_chars"] for c in cases_for(held))
    p = RE.PRICES
    emb = (chars_all + 2 * chars_held) * p["gemini-embedding-001"]["input"] / 1e6  # tokens <= chars
    # auto: 10 runs x <=5 calls x <=3k input tokens, <=600 output tokens
    auto = 10 * 5 * (3000 * p["gpt-4.1-mini"]["input"] + 600 * p["gpt-4.1-mini"]["output"]) / 1e6
    # live held-out split: 2 configs x 12 cases x <=12 calls x <=1.5k in / 200 out
    split = 2 * len(cases_for(held)) * 12 * (1500 * p["gpt-4.1-mini"]["input"] + 200 * p["gpt-4.1-mini"]["output"]) / 1e6
    est = {
        "calibration_repos": cal, "heldout_repos": held,
        "gemini_embedding_upper_bound_usd": round(emb, 4),
        "openai_auto_upper_bound_usd": round(auto, 4),
        "openai_live_heldout_split_upper_bound_usd": round(split, 4),
        "total_upper_bound_usd": round(emb + auto + split, 4),
        # Same workloads measured in the earlier run (results/raw/*.json): auto 2x5 runs
        # $0.0064; held-out split hybrid+semantic LLM $0.0108, embeddings <= $0.0438;
        # all-30 embedding pass <= ~$0.056 (chars bound). Expected total ~ $0.12.
        "expected_from_earlier_runs_usd": round(0.0064 + 0.0108 * 2 + 0.0438 + 0.056, 4),
        "hard_cap_usd": HARD_CAP_USD, "stop_at_usd": STOP_AT_USD,
    }
    write_json("budget_estimate.json", est)
    print(json.dumps(est, indent=1))


# ------------------------------------------------------------------ A: auto
def instrument_embeddings(svc) -> None:
    orig_many, orig_one = svc.embed_texts, svc.embed_text

    async def embed_texts(texts):
        rec = {"model": svc.model, "n_texts": len(texts), "chars": sum(len(t) for t in texts)}
        t0 = time.perf_counter()
        try:
            out = await orig_many(texts)
            rec["ok"] = True
            return out
        except Exception as e:
            rec["ok"], rec["error"] = False, str(e)[:300]
            raise
        finally:
            rec["latency_s"] = time.perf_counter() - t0
            RE.METER.embed_calls.append(rec)

    async def embed_text(text):
        rec = {"model": svc.model, "n_texts": 1, "chars": len(text)}
        try:
            out = await orig_one(text)
            rec["ok"] = True
            return out
        except Exception as e:
            rec["ok"], rec["error"] = False, str(e)[:300]
            raise
        finally:
            RE.METER.embed_calls.append(rec)

    svc.embed_texts, svc.embed_text = embed_texts, embed_text


def run_auto() -> None:
    from cli.engine import SyncLocalBackend
    from cli.pipeline import Pipeline

    for name, skip_pr in (("auto-openai", False), ("auto-openai-nopr", True)):
        data = {"config": name, "settings": {"skip_pr": skip_pr, "skip_review": False}, "cases": {}}
        p = OUT / f"{name}.json"
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
        for case in MANIFEST["auto_cases"]:
            cid = case["case_id"]
            if cid in data["cases"]:
                continue
            guard()
            diff = RE.regen_diff(case["repo_path"], case["parent"], case["sha"], case["diff_sha256"])
            engine = RE.make_engine("openai")
            RE.instrument_llm(engine._get_llm(), "openai")
            retr = engine._get_retriever()  # shipped construction (bug 3 fix); metered only
            if retr is not None:
                instrument_embeddings(retr.embeddings)
            RE.METER.reset()
            backend = SyncLocalBackend(engine)
            t0 = time.perf_counter()
            try:
                r = Pipeline(backend).run(
                    diff=diff, commits=[], branch_name="eval", base_branch="main",
                    skip_review=False, skip_pr=skip_pr,
                )
                rec = {"result": asdict(r)}
            except Exception as e:
                rec = {"error": f"exception: {type(e).__name__}: {e}", "traceback": traceback.format_exc()[-2000:]}
            finally:
                backend.close()  # what cli/commands/auto.py now does
            rec["wall_s"] = time.perf_counter() - t0
            rec["diff_chars"] = len(diff)
            rec["retriever"] = type(retr).__name__ if retr is not None else None
            rec["meter"] = RE.METER.snapshot()
            blob = json.dumps(rec, default=str)
            rec["connection_errors"] = blob.count("Connection error")
            data["cases"][cid] = rec
            OUT.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
            res = rec.get("result", {})
            print(f"[{name}] {cid} chars={len(diff)} calls={rec['meter']['llm_calls']} "
                  f"msg={bool(res.get('commit_message'))} done={res.get('steps_completed')} "
                  f"skipped={res.get('steps_skipped')} conn_err={rec['connection_errors']} "
                  f"call_errors={len(rec['meter']['call_errors'])} err={res.get('error') or rec.get('error')}",
                  flush=True)


# ------------------------------------------------------------------ B: embeddings cache
async def embed_all() -> None:
    from backend.rag.embeddings import EmbeddingService

    svc = EmbeddingService(api_key=RE.settings.gemini_api_key)  # shipped model + dims
    engine = RE.make_engine("openai")
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    logp = OUT / "embed_log.json"
    log = json.loads(logp.read_text(encoding="utf-8")) if logp.exists() else {"model": svc.model, "calls": []}
    for case in MANIFEST["split_cases"]:
        _, parsed = parsed_case(case, engine)
        texts = [hunk_text(f, h) for f, h in parsed.get_all_hunks()]
        todo = [t for t in texts if key(t) not in cache]
        if not todo:
            continue
        guard()
        for attempt in range(8):
            rec = {"case_id": case["case_id"], "n_texts": len(todo), "chars": sum(map(len, todo)), "attempt": attempt}
            try:
                vecs = await svc.embed_texts(todo)  # one batch per case, as SemanticStrategy does
                rec["ok"] = True
                log["calls"].append(rec)
                for t, v in zip(todo, vecs):
                    cache[key(t)] = list(v)
                break
            except Exception as e:
                rec["ok"], rec["error"] = False, str(e)[:300]
                log["calls"].append(rec)
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait = min(90, 10 * 2 ** attempt)
                    print(f"{case['case_id']} 429, sleeping {wait}s", flush=True)
                    await asyncio.sleep(wait)
                    continue
                raise
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache), encoding="utf-8")
        write_json("embed_log.json", log)
        print(f"{case['case_id']} embedded {len(todo)} hunks (dims={len(cache[key(todo[0])])})", flush=True)
        await asyncio.sleep(3)  # pacing
    print("cached vectors:", len(cache))


# ------------------------------------------------------------------ B: offline scoring
async def offline_labels(cases: list[dict], strategy: str, threshold: float | None) -> dict[str, dict]:
    cache = json.loads(CACHE.read_text(encoding="utf-8"))
    emb = CachedEmbeddings(cache)
    engine = RE.make_engine("openai")
    out = {}
    orig_default = SemanticStrategy.__init__.__defaults__
    for case in cases:
        _, parsed = parsed_case(case, engine)
        if strategy == "semantic":
            kw = {} if threshold is None else {"similarity_threshold": threshold}
            groups = await SemanticStrategy(emb, **kw).cluster(parsed)
        elif strategy == "hybrid":
            # Hybrid constructs SemanticStrategy(embedding_service) with the class
            # default; to score hybrid at threshold t offline, set the default to t.
            if threshold is not None:
                SemanticStrategy.__init__.__defaults__ = (threshold,) + orig_default[1:]
            try:
                groups = await HybridStrategy(NoLLM(), emb).cluster(parsed)
            finally:
                SemanticStrategy.__init__.__defaults__ = orig_default
        elif strategy == "directory":
            groups = await DirectoryStrategy().cluster(parsed)
        else:
            raise ValueError(strategy)
        pred = {}
        for gi, g in enumerate(groups):
            for h in g.hunks:
                pred.setdefault(h.hunk_id, f"g{gi}")
        for hid in case["hunk_labels"]:
            pred.setdefault(hid, f"unassigned-{hid}")
        out[case["case_id"]] = {"pred": pred, "n_groups": len(groups)}
    return out


def score(cases: list[dict], preds: dict[str, dict]) -> dict:
    vals, match = [], 0
    for c in cases:
        p = preds[c["case_id"]]
        vals.append(ari(c["hunk_labels"], p["pred"]))
        match += int(p["n_groups"] == c["n_commits"])
    return {"per_case": vals, "mean": statistics.fmean(vals), "ci95": bootstrap_ci(vals),
            "count_match_pct": 100 * match / len(cases), "n": len(cases),
            "mean_groups": statistics.fmean(preds[c["case_id"]]["n_groups"] for c in cases)}


def per_file(cases):
    return {c["case_id"]: {"pred": {h: c["hunk_files"][h] for h in c["hunk_labels"]},
                           "n_groups": len(set(c["hunk_files"][h] for h in c["hunk_labels"]))} for c in cases}


def one_group(cases):
    return {c["case_id"]: {"pred": {h: "all" for h in c["hunk_labels"]}, "n_groups": 1} for c in cases}


def paired_delta(a: list[float], b: list[float]) -> dict:
    d = [x - y for x, y in zip(a, b)]
    return {"mean": statistics.fmean(d), "ci95": bootstrap_ci(d)}


def similarity_distribution(cases: list[dict]) -> dict:
    import numpy as np

    cache = json.loads(CACHE.read_text(encoding="utf-8"))
    engine = RE.make_engine("openai")
    same, diff = [], []
    for case in cases:
        _, parsed = parsed_case(case, engine)
        hunks = parsed.get_all_hunks()
        ids = [h.id for _, h in hunks]
        X = np.array([cache[key(hunk_text(f, h))] for f, h in hunks])
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        S = X @ X.T
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                (same if case["hunk_labels"][ids[i]] == case["hunk_labels"][ids[j]] else diff).append(float(S[i, j]))
    q = lambda v: {str(k): round(float(np.quantile(v, k)), 4) for k in (0.05, 0.25, 0.5, 0.75, 0.95)} if v else None
    return {"same_commit_pairs": len(same), "cross_commit_pairs": len(diff),
            "same_commit_cosine_quantiles": q(same), "cross_commit_cosine_quantiles": q(diff)}


def strip(s: dict) -> dict:
    return {k: (round(v, 4) if isinstance(v, float) else [round(x, 4) for x in v] if k == "ci95" else v)
            for k, v in s.items() if k != "per_case"}


async def sweep() -> None:
    cal, held = repo_split()
    cases = cases_for(cal)
    dist = similarity_distribution(cases)
    print(json.dumps(dist, indent=1))
    # Extend the pre-registered grid if the cosine distribution sits above it.
    grid = list(GRID)
    hi = dist["cross_commit_cosine_quantiles"]["0.95"]
    if hi > grid[-1]:
        t = grid[-1]
        while t < min(0.995, hi + 0.025):
            t = round(t + 0.0125, 4)
            grid.append(t)
    rows = []
    for t in grid:
        s = score(cases, await offline_labels(cases, "semantic", t))
        h = score(cases, await offline_labels(cases, "hybrid", t))
        rows.append({"threshold": t, "semantic": strip(s), "hybrid": strip(h)})
        print(f"t={t:.4f} semantic ARI={s['mean']:.3f} groups={s['mean_groups']:.1f} | "
              f"hybrid ARI={h['mean']:.3f} groups={h['mean_groups']:.1f}", flush=True)
    best = max(r["semantic"]["mean"] for r in rows)
    # Pre-registered tie rule: lowest threshold among those within 1e-9 of the best.
    chosen = min(r["threshold"] for r in rows if r["semantic"]["mean"] >= best - 1e-9)
    base = {k: strip(score(cases, v)) for k, v in (
        ("per_file", per_file(cases)), ("one_group", one_group(cases)),
        ("directory", await offline_labels(cases, "directory", None)))}
    write_json("threshold_sweep_calibration.json", {
        "calibration_repos": cal, "heldout_repos": held, "n_cases": len(cases),
        "selection_rule": "max mean semantic ARI on calibration; ties -> lowest threshold",
        "similarity_distribution": dist, "grid": grid, "rows": rows, "baselines": base,
        "chosen_threshold": chosen,
    })
    print("baselines:", json.dumps(base))
    print("CHOSEN", chosen)


async def heldout() -> None:
    cal, held = repo_split()
    chosen = json.loads((OUT / "threshold_sweep_calibration.json").read_text(encoding="utf-8"))["chosen_threshold"]
    cases = cases_for(held)
    configs = {
        f"semantic@{chosen}": await offline_labels(cases, "semantic", chosen),
        f"hybrid@{chosen}": await offline_labels(cases, "hybrid", chosen),
        "semantic@shipped_default": await offline_labels(cases, "semantic", None),
        "semantic@0.5": await offline_labels(cases, "semantic", 0.5),
        "hybrid@0.5": await offline_labels(cases, "hybrid", 0.5),
        "directory": await offline_labels(cases, "directory", None),
        "baseline_per_file": per_file(cases),
        "baseline_one_group": one_group(cases),
    }
    scores = {k: score(cases, v) for k, v in configs.items()}
    pf, dr = scores["baseline_per_file"]["per_case"], scores["directory"]["per_case"]
    out = {"heldout_repos": held, "n_cases": len(cases), "n_hunks": sum(c["n_hunks"] for c in cases),
           "chosen_threshold": chosen,
           "shipped_default_now": SemanticStrategy.__init__.__defaults__[0],
           "configs": {k: {**strip(s), "per_case": [round(x, 4) for x in s["per_case"]],
                           "delta_vs_per_file": {kk: (round(vv, 4) if isinstance(vv, float) else [round(x, 4) for x in vv])
                                                 for kk, vv in paired_delta(s["per_case"], pf).items()},
                           "delta_vs_directory": {kk: (round(vv, 4) if isinstance(vv, float) else [round(x, 4) for x in vv])
                                                  for kk, vv in paired_delta(s["per_case"], dr).items()}}
                       for k, s in scores.items()},
           "case_ids": [c["case_id"] for c in cases]}
    write_json("threshold_heldout_offline.json", out)
    for k, s in out["configs"].items():
        print(f"{k:28s} ARI={s['mean']:.3f} {s['ci95']} match={s['count_match_pct']:.1f}% groups={s['mean_groups']:.2f} "
              f"dPF={s['delta_vs_per_file']['mean']:+.3f} {s['delta_vs_per_file']['ci95']} "
              f"dDir={s['delta_vs_directory']['mean']:+.3f} {s['delta_vs_directory']['ci95']}")


# ------------------------------------------------------------------ B: live shipped path on held-out
async def live_heldout() -> None:
    from backend.agents.split_agent import SplitAgent

    _, held = repo_split()
    cases = cases_for(held)
    for strategy in ("semantic", "hybrid"):
        name = f"split-openai-{strategy}-shipped-heldout"
        p = OUT / f"{name}.json"
        data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"config": name, "cases": {}}
        engine = RE.make_engine("openai")
        llm = engine._get_llm()
        RE.instrument_llm(llm, "openai")
        retr = engine._get_retriever()  # shipped construction, no injection
        data["retriever"] = f"{type(retr).__name__}(embeddings={type(retr.embeddings).__name__}(model={retr.embeddings.model}))"
        data["semantic_default_threshold"] = SemanticStrategy.__init__.__defaults__[0]
        instrument_embeddings(retr.embeddings)
        for case in cases:
            cid = case["case_id"]
            if cid in data["cases"] and not data["cases"][cid].get("error"):
                continue
            prev = [data["cases"][cid]] if cid in data["cases"] else []
            guard()
            diff = RE.regen_diff(case["repo_path"], case["base"], case["commits"][-1], case["diff_sha256"])
            for attempt in range(7):
                RE.METER.reset()
                agent = SplitAgent(llm, retr)  # what LocalEngine.split_diff constructs
                t0 = time.perf_counter()
                rec = {}
                try:
                    state = await agent.graph.ainvoke(agent._initial_state(engine._prepare_diff(diff), case["repo_path"], strategy))
                    rec["groups"] = [{"id": g.id, "hunks": [h.hunk_id for h in g.hunks], "files": g.files,
                                      "suggested_type": g.suggested_type} for g in state.get("commit_groups", [])]
                    rec["error"] = state.get("error")
                    rec["reasoning"] = state.get("reasoning", [])
                except Exception as e:
                    rec["error"] = f"exception: {type(e).__name__}: {e}"
                rec["wall_s"] = time.perf_counter() - t0
                rec["meter"] = RE.METER.snapshot()
                if rec["error"] and ("429" in rec["error"] or "RESOURCE_EXHAUSTED" in rec["error"]):
                    prev.append(rec)
                    await asyncio.sleep(min(90, 15 * 2 ** attempt))
                    continue
                break
            else:
                prev.pop()  # every attempt hit 429: keep the last one as the record
            if prev:
                rec["previous_attempts"] = prev
            data["cases"][cid] = rec
            p.write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
            print(f"[{name}] {cid} groups={len(rec.get('groups', []))} err={rec.get('error')} "
                  f"calls={rec['meter']['llm_calls']} embeds={len(rec['meter']['embed_calls'])}", flush=True)
            await asyncio.sleep(3)
    live_summary()


def live_summary() -> None:
    """Score the live shipped-path runs (whatever finished) and compare each case
    with the offline cached-vector grouping at the same threshold."""
    _, held = repo_split()
    cases = cases_for(held)
    off = json.loads((OUT / "threshold_heldout_offline.json").read_text(encoding="utf-8"))
    chosen = off["chosen_threshold"]
    summary = {}
    for strategy in ("semantic", "hybrid"):
        d = json.loads((OUT / f"split-openai-{strategy}-shipped-heldout.json").read_text(encoding="utf-8"))
        vals, same, match, used, not_run, paired_off, errs = [], 0, 0, set(), [], [], []
        offline_vals = off["configs"][f"{strategy}@{chosen}"]["per_case"]
        for i, c in enumerate(cases):
            rec = d["cases"].get(c["case_id"])
            if rec is None:
                not_run.append(c["case_id"])
                continue
            if rec.get("error"):
                vals.append(None)
                errs.append(f"{c['case_id']}: {rec['error'][:70]}")
                continue
            paired_off.append(offline_vals[i])
            pred = {}
            for gi, g in enumerate(rec["groups"]):
                for h in g["hunks"]:
                    pred.setdefault(h, f"g{gi}")
            for hid in c["hunk_labels"]:
                pred.setdefault(hid, f"unassigned-{hid}")
            v = ari(c["hunk_labels"], pred)
            vals.append(v)
            same += int(abs(v - offline_vals[i]) < 5e-5)  # offline per-case values are rounded to 4 dp
            match += int(len(rec["groups"]) == c["n_commits"])
            used.update(r for r in rec.get("reasoning", []) if "strategy" in r)
        ok = [v for v in vals if v is not None]
        summary[strategy] = {"n_scored": len(ok), "errors": errs, "not_run": not_run,
                             "offline_mean_on_same_cases": round(statistics.fmean(paired_off), 4) if paired_off else None,
                             "mean": round(statistics.fmean(ok), 4) if ok else None,
                             "ci95": [round(x, 4) for x in bootstrap_ci(ok)] if ok else None,
                             "count_match_pct": round(100 * match / max(1, len(ok)), 1),
                             "per_case_equal_to_offline": same,
                             "strategy_reasoning_seen": sorted({u.split("using ")[-1] for u in used}),
                             "llm_calls_mean": round(statistics.fmean(d["cases"][c["case_id"]]["meter"]["llm_calls"]
                                                                      for c in cases if c["case_id"] in d["cases"]), 2)}
    write_json("threshold_heldout_live_summary.json", summary)
    print(json.dumps(summary, indent=1))


def main() -> None:
    cmd = sys.argv[1]
    if cmd == "estimate":
        estimate()
    elif cmd == "auto":
        run_auto()
    elif cmd == "embed":
        asyncio.run(embed_all())
    elif cmd == "sweep":
        asyncio.run(sweep())
    elif cmd == "heldout":
        asyncio.run(heldout())
    elif cmd == "live-heldout":
        asyncio.run(live_heldout())
    elif cmd == "live-summary":
        live_summary()
    elif cmd == "spend":
        s = spend_now()
        write_json("spend.json", {k: round(v, 4) for k, v in s.items()})
        print(json.dumps(s, indent=1))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
