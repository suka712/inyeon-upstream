"""Compute every reported number from evals/results/raw/*.json + the manifest.

Writes evals/results/summary.json. RESULTS.md quotes numbers from summary.json.
Usage: python evals/report.py
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

from metrics import CC_TYPES, ari, bootstrap_ci, cc_type, percentile, strict_cc_check

EVALS = Path(__file__).resolve().parent
RAW = EVALS / "results" / "raw"
PRICES = {"gpt-4.1-mini": {"input": 0.40, "cached_input": 0.10, "output": 1.60}, "gemini-embedding-001": 0.15}


def r4(x):
    return None if x is None else round(x, 4)


def load(name):
    p = RAW / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def openai_usd(reqs):
    p = PRICES["gpt-4.1-mini"]
    usd = 0.0
    for r in reqs:
        if r.get("provider") == "openai" and r.get("input_tokens") is not None:
            c = r.get("cached_input_tokens") or 0
            usd += (r["input_tokens"] - c) * p["input"] / 1e6 + c * p["cached_input"] / 1e6 + r["output_tokens"] * p["output"] / 1e6
    return usd


def summarize_ci(vals, seed=0):
    return {"mean": r4(statistics.fmean(vals)) if vals else None, "ci95": [r4(x) for x in bootstrap_ci(vals, seed=seed)] if vals else None, "n": len(vals)}


# ------------------------------------------------------------------ split
def split_predictions(case, rec):
    """Return (hunk-level labels, as-executed file-level labels) or None on error."""
    if rec.get("error") or not rec.get("groups"):
        return None
    hunk_pred, file_pred = {}, {}
    for gi, g in enumerate(rec["groups"]):
        for h in g["hunks"]:
            hunk_pred.setdefault(h, f"g{gi}")
    # `inyeon auto` / `inyeon split` commit with `git add <files>` per split, in order,
    # so a hunk actually lands in the first group that lists its file.
    for hid, path in case["hunk_files"].items():
        for gi, g in enumerate(rec["groups"]):
            if path in g["files"]:
                file_pred[hid] = f"g{gi}"
                break
    for hid in case["hunk_labels"]:
        hunk_pred.setdefault(hid, f"unassigned-{hid}")
        file_pred.setdefault(hid, f"unassigned-{hid}")
    return hunk_pred, file_pred


def split_section(manifest):
    cases = manifest["split_cases"]
    out = {"n_cases": len(cases)}
    out["dataset"] = {
        "cases_per_repo": dict(Counter(c["repo"] for c in cases)),
        "commits_per_case": dict(sorted(Counter(c["n_commits"] for c in cases).items())),
        "total_hunks": sum(c["n_hunks"] for c in cases),
        "hunks_per_case_median": statistics.median(c["n_hunks"] for c in cases),
        "hunks_per_case_range": [min(c["n_hunks"] for c in cases), max(c["n_hunks"] for c in cases)],
        "files_per_case_median": statistics.median(c["n_files"] for c in cases),
        "files_per_case_range": [min(c["n_files"] for c in cases), max(c["n_files"] for c in cases)],
        "diff_chars_median": statistics.median(c["diff_chars"] for c in cases),
        # Cases where one file carries hunks from >1 original commit (file-level
        # grouping cannot be perfect there).
        "cases_with_file_shared_across_commits": sum(
            1 for c in cases
            if any(len({c["hunk_labels"][h] for h, f in c["hunk_files"].items() if f == path}) > 1
                   for path in set(c["hunk_files"].values()))
        ),
    }
    # Baselines
    base = {"one_group": [], "per_file": [], "per_hunk": []}
    base_cnt = {"one_group": 0, "per_file": 0, "per_hunk": 0}
    per_file_by_case = {}
    for c in cases:
        t = c["hunk_labels"]
        preds = {
            "one_group": {h: "all" for h in t},
            "per_file": {h: c["hunk_files"][h] for h in t},
            "per_hunk": {h: h for h in t},
        }
        for k, p in preds.items():
            v = ari(t, p)
            base[k].append(v)
            base_cnt[k] += int(len(set(p.values())) == c["n_commits"])
        per_file_by_case[c["case_id"]] = base["per_file"][-1]
    out["baselines"] = {
        k: {**summarize_ci(v), "group_count_match_pct": r4(100 * base_cnt[k] / len(cases))} for k, v in base.items()
    }

    configs = {}
    for p in sorted(RAW.glob("split-*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        name = d["config"]
        aris, aris_exec, match, deltas, calls, wall, errors, strat_used = [], [], 0, [], [], [], [], Counter()
        reqs_all, embed_all = [], []
        scored_ids = []
        for c in cases:
            rec = d["cases"].get(c["case_id"])
            if rec is None:
                continue
            calls.append(rec["meter"]["llm_calls"])
            wall.append(rec["wall_s"])
            reqs_all += rec["meter"]["requests"]
            embed_all += rec["meter"]["embed_calls"]
            for line in rec.get("reasoning", []):
                if "commit groups using" in line:
                    strat_used[line.split("'")[1]] += 1
            pr = split_predictions(c, rec)
            if pr is None:
                errors.append({"case_id": c["case_id"], "error": (rec.get("error") or "no groups")[:240]})
                continue
            hp, fp = pr
            a = ari(c["hunk_labels"], hp)
            aris.append(a)
            aris_exec.append(ari(c["hunk_labels"], fp))
            deltas.append(a - per_file_by_case[c["case_id"]])
            match += int(len(rec["groups"]) == c["n_commits"])
            scored_ids.append(c["case_id"])
        n_run = len(calls)
        configs[name] = {
            "settings": d["settings"],
            "retriever": d.get("retriever"),
            "n_run": n_run,
            "n_errors": len(errors),
            "n_scored": len(aris),
            "errors": errors,
            "error_classes": dict(Counter(
                "404_model_not_found" if "404" in e["error"] else "429_quota" if "429" in e["error"] else "other"
                for e in errors)),
            "n_cases_retried_after_429": sum(1 for c in cases if d["cases"].get(c["case_id"], {}).get("previous_attempts")),
            "strategy_reported_by_agent": dict(strat_used),
            "ari": summarize_ci(aris),
            "ari_as_executed_file_level": summarize_ci(aris_exec),
            "delta_ari_vs_per_file_baseline": summarize_ci(deltas, seed=1),
            "per_file_baseline_on_scored_cases": summarize_ci([per_file_by_case[i] for i in scored_ids]),
            "group_count_match_pct": r4(100 * match / len(aris)) if aris else None,
            "llm_calls_per_run_mean": r4(statistics.fmean(calls)) if calls else None,
            "llm_calls_per_run_range": [min(calls), max(calls)] if calls else None,
            "wall_s_p50": r4(percentile(wall, 0.5)),
            "wall_s_p95": r4(percentile(wall, 0.95)),
            "embed_requests": len(embed_all),
            "embed_requests_failed": sum(1 for e in embed_all if not e.get("ok")),
            "embed_chars_ok": sum(e["chars"] for e in embed_all if e.get("ok")),
            "openai_usd": r4(openai_usd(reqs_all)),
            "per_case_ari": dict(zip(scored_ids, [r4(a) for a in aris])),
        }
    out["configs"] = configs
    return out


# ------------------------------------------------------------------ commit
def commit_section(manifest):
    cases = manifest["commit_cases"]
    author_types = {c["case_id"]: cc_type(c["author_message"]) for c in cases}
    author_strict = {c["case_id"]: strict_cc_check(c["author_message"])[0] for c in cases}
    out = {
        "n_cases": len(cases),
        "cases_per_repo": dict(Counter(c["repo"] for c in cases)),
        "author_messages_conventional_n": sum(1 for t in author_types.values() if t),
        "author_messages_strict_valid_n": sum(author_strict.values()),
        "author_type_distribution": dict(Counter(t for t in author_types.values() if t)),
        "providers": {},
    }
    for name in sorted(x.stem for x in RAW.glob("commit-*.json")):
        d = load(name)
        if not d:
            continue
        valid, fail_reasons, type_match, type_n, calls, wall = 0, Counter(), 0, 0, [], []
        in_tok, out_tok, trunc_calls, trunc_runs, n_req, errors, empty = [], [], 0, 0, 0, [], 0
        gen_types = Counter()
        usage_missing = 0
        lenient = 0
        valid_list, match_list = [], []
        reqs_all = []
        n = 0
        for c in cases:
            rec = d["cases"].get(c["case_id"])
            if rec is None:
                continue
            n += 1
            msg = rec.get("commit_message")
            if rec.get("error"):
                errors.append({"case_id": c["case_id"], "error": rec["error"][:240]})
            if not msg:
                empty += 1
            ok, why = strict_cc_check(msg)
            valid += ok
            valid_list.append(100.0 * ok)
            if not ok:
                fail_reasons[why] += 1
            lenient += int(cc_type(msg) is not None)
            gt = cc_type(msg)
            gen_types[gt or "none"] += 1
            if author_types[c["case_id"]]:
                type_n += 1
                type_match += int(gt == author_types[c["case_id"]])
                match_list.append(100.0 * (gt == author_types[c["case_id"]]))
            m = rec["meter"]
            calls.append(m["llm_calls"])
            wall.append(rec["wall_s"])
            reqs = m["requests"]
            reqs_all += reqs
            n_req += len(reqs)
            if any(r.get("input_tokens") is None for r in reqs):
                usage_missing += 1
            in_tok.append(sum(r.get("input_tokens") or 0 for r in reqs))
            out_tok.append(sum(r.get("output_tokens") or 0 for r in reqs))
            t = sum(1 for r in reqs if r.get("context_truncated"))
            trunc_calls += t
            trunc_runs += int(t > 0)
        out["providers"][name] = {
            "n": n,
            "n_errors": len(errors),
            "errors": errors,
            "n_empty_message": empty,
            "strict_cc_valid_pct": r4(100 * valid / n) if n else None,
            "strict_cc_valid_n": valid,
            "strict_cc_valid_pct_ci95": [r4(x) for x in bootstrap_ci(valid_list)] if valid_list else None,
            "type_match_pct_ci95": [r4(x) for x in bootstrap_ci(match_list)] if match_list else None,
            "strict_cc_fail_reasons": dict(fail_reasons),
            "strict_cc_failing_headers": [
                (d["cases"][c["case_id"]].get("commit_message") or "").split("\n", 1)[0]
                for c in cases if c["case_id"] in d["cases"]
                and not strict_cc_check(d["cases"][c["case_id"]].get("commit_message"))[0]],
            "lenient_cc_type_parse_pct": r4(100 * lenient / n) if n else None,
            "type_match_pct": r4(100 * type_match / type_n) if type_n else None,
            "type_match_n": type_match,
            "type_eval_n": type_n,
            "generated_type_distribution": dict(gen_types),
            "llm_calls_per_run_mean": r4(statistics.fmean(calls)) if calls else None,
            "llm_calls_per_run_distribution": dict(sorted(Counter(calls).items())),
            "http_requests_total": n_req,
            "runs_missing_usage": usage_missing,
            "input_tokens_per_run_mean": r4(statistics.fmean(in_tok)) if in_tok else None,
            "input_tokens_per_run_median": percentile(in_tok, 0.5),
            "output_tokens_per_run_mean": r4(statistics.fmean(out_tok)) if out_tok else None,
            "output_tokens_per_run_median": percentile(out_tok, 0.5),
            "context_truncated_calls": trunc_calls,
            "context_truncated_runs": trunc_runs,
            "latency_s_p50": r4(percentile(wall, 0.5)),
            "latency_s_p95": r4(percentile(wall, 0.95)),
            "latency_s_max": r4(max(wall)) if wall else None,
            "openai_usd": r4(openai_usd(reqs_all)),
        }
    # Baseline for type match: always predict the most common author type.
    if out["author_type_distribution"]:
        top, cnt = Counter(out["author_type_distribution"]).most_common(1)[0]
        out["type_majority_baseline"] = {"type": top, "pct": r4(100 * cnt / out["author_messages_conventional_n"])}
    return out


# ------------------------------------------------------------------ auto
def auto_section(manifest):
    out = {}
    for name in sorted(x.stem for x in RAW.glob("auto-*.json")):
        d = load(name)
        if not d:
            continue
        rows = []
        for c in manifest["auto_cases"]:
            rec = d["cases"].get(c["case_id"])
            if rec is None:
                continue
            rows.append({
                "case_id": c["case_id"], "repo": c["repo"], "sha": c["sha"][:10], "diff_chars": rec["diff_chars"],
                "llm_calls": rec["meter"]["llm_calls"],
                "http_requests_succeeded": len(rec["meter"]["requests"]),
                "llm_call_errors": rec["meter"].get("call_errors", []),
                "review_returned": rec.get("has_review"), "pr_returned": rec.get("has_pr"),
                "steps_completed": rec.get("steps_completed"),
                "steps_skipped": rec.get("steps_skipped"), "commit_message_returned": bool(rec.get("commit_message")),
                "error": rec.get("error"), "wall_s": r4(rec["wall_s"]),
            })
        out[name] = rows
    return out


def spend_section():
    oa_usd, emb_chars = 0.0, 0
    for p in RAW.rglob("*.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        for rec0 in d["cases"].values():
          for rec in [rec0] + rec0.get("previous_attempts", []):
            oa_usd += openai_usd(rec.get("meter", {}).get("requests", []))
            emb_chars += sum(e["chars"] for e in rec.get("meter", {}).get("embed_calls", []) if e.get("ok") and e["model"] == "gemini-embedding-001")
    emb_ub = emb_chars * PRICES["gemini-embedding-001"] / 1e6
    return {
        "openai_usd_from_usage": r4(oa_usd),
        "gemini_embedding_chars": emb_chars,
        "gemini_embedding_usd_upper_bound": r4(emb_ub),
        "total_usd_upper_bound": r4(oa_usd + emb_ub),
        "prices_usd_per_1m_tokens": PRICES,
    }


def main():
    manifest = json.loads((EVALS / "dataset_manifest.json").read_text(encoding="utf-8"))
    pytest_txt = (EVALS / "results" / "pytest_output.txt")
    summary = {
        "split": split_section(manifest),
        "commit": commit_section(manifest),
        "auto": auto_section(manifest),
        "spend": spend_section(),
    }
    if pytest_txt.exists():
        last = [l for l in pytest_txt.read_text(encoding="utf-8").splitlines() if " passed" in l or " failed" in l]
        summary["pytest"] = last[-1].strip() if last else None
    (EVALS / "results" / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "split"}, indent=1)[:6000])
    s = summary["split"]
    print("baselines", json.dumps(s["baselines"]))
    for k, v in s["configs"].items():
        print(k, "n", v["n_scored"], "/", v["n_run"], "err", v["n_errors"], "ARI", v["ari"], "exec", v["ari_as_executed_file_level"]["mean"],
              "match%", v["group_count_match_pct"], "calls", v["llm_calls_per_run_mean"], "strat", v["strategy_reported_by_agent"])


if __name__ == "__main__":
    main()
