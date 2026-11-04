# Inyeon v4.0.0 — measured eval results

- **Date:** 2026-11-04
- **Hardware:** Windows 11 laptop, AMD Ryzen 7 7735HS. Python 3.12.8. All agents ran in-process via the `LocalEngine` path (`backend/engine/local.py`), not the Railway backend.
- **LLM used for every number below:** OpenAI `gpt-4.1-mini` (Chat Completions, the repo's `OpenAIProvider`, temperature 0.3 as shipped).
- **Embeddings (semantic split only):** Gemini `text-embedding-004` (the shipped model id) and `gemini-embedding-001` (substituted by the harness, see below).
- **Sources:** every number is computed by `evals/report.py` from `evals/results/raw/*.json` and written to `evals/results/summary.json`. Each run happened once, with no repeated sampling.

> **The local model was removed partway through the session.** The plan was to run the split and commit evals on Ollama `qwen2.5-coder:7b`. Ollama and its model store were uninstalled from the machine before any Ollama eval case finished. The single Ollama attempt (`split-ollama-directory`) failed with "All connection attempts failed" and was thrown away. **No number in this file comes from Ollama.** Every LLM-backed number comes from `gpt-4.1-mini`. A few Ollama probes from before the removal are stored in `results/ollama_manual_probes.json`, kept for context only and not used in any table. The Ollama-vs-OpenAI latency comparison was dropped.

## Dataset (`evals/dataset_manifest.json`)

Real commits by "Anh D Tran" from 5 local repos: kiln, ferrox, commma-coding-progress-tracker, inyeon-upstream and draft-thinker. The manifest records the HEAD, every SHA and a sha256 of every diff. The runner checks that each regenerated diff is byte-identical before it uses it. `build_dataset.py` scans the repos deterministically and samples round-robin across them with a fixed seed (20261104).

**Split cases (n = 30, 6 per repo).** Each case is 2–4 adjacent first-parent commits squashed into one diff with `git diff <base> <tip>`. Mix: 18 cases of 2 commits, 11 of 3 and 1 of 4. Each case has 2–15 files (median 5) and 2–36 hunks (median 8), 322 hunks in total. The median diff is 12,710 chars.

Inclusion rules for split cases:
- No merges, no lockfile/generated-only commits, no binary files, no renames.
- Diff ≤ 30,000 chars and ≥ 2 files. Two files is the `auto` split threshold.
- Each hunk's origin commit comes from `git blame` (added lines) and `git blame --reverse` (removed lines) over `base..tip`. Every hunk must map to exactly one commit.
- Each commit's added/removed line counts must survive the squash unchanged. This rejects cases where commit 2 edits lines that commit 1 added.

Only 1 of the 30 cases has a file that carries hunks from more than one commit. In practice, the ground truth almost always groups whole files.

**Commit-message cases (n = 30, 6 per repo).** Single non-merge commits that pass the same filters and are not used in any split case. 20 of the 30 author messages are Conventional Commits (fix 3, docs 5, feat 9, test 3). All 20 also pass the strict check defined below.

**`auto` cases (n = 5).** Single-file commits: 3 with diffs under 500 chars (151, 183, 170) and 2 with diffs of 500–3,000 chars (688, 845).

## Eval A: Split (hunk grouping vs. the author's real commits)

Metric: Adjusted Rand Index (ARI) over hunks, comparing predicted groups against the original commits. The 95% CI comes from a 10,000-resample percentile bootstrap over cases. "Count match" is the % of cases where the number of groups equals the number of original commits.

The harness invokes `SplitAgent.graph` exactly as `LocalEngine.split_diff` does. It keeps the full final state so the hunk-level `commit_groups` can be scored, because the public result only lists files per split.

### Shipped behaviour (LocalEngine, gpt-4.1-mini, n = 30, 0 errors)

| Config | Mean ARI [95% CI] | Δ vs per-file baseline [95% CI] | Count match | LLM calls/run (mean, range) | Wall s p50 / p95 |
|---|---|---|---|---|---|
| `directory` | 0.474 [0.328, 0.618] | +0.090 [−0.056, +0.235] | 40.0% | 2.77 (1–7) | 2.21 / 5.62 |
| `conventional` | 0.400 [0.261, 0.539] | +0.015 [−0.141, +0.155] | 33.3% | 3.60 (2–5) | 4.23 / 7.16 |
| `hybrid` (default) | 0.474 [0.328, 0.618] | +0.090 [−0.056, +0.235] | 40.0% | 4.03 (2–9) | 3.79 / 6.98 |
| `semantic` (requested) | 0.474 [0.328, 0.618] | +0.090 [−0.056, +0.235] | 40.0% | 4.03 (2–9) | 3.67 / 7.11 |
| **Baseline:** one group | 0.000 [0.000, 0.000] | — | 0.0% | 0 | — |
| **Baseline:** one group per file | 0.385 [0.259, 0.516] | — | 23.3% | 0 | — |
| **Baseline:** one group per hunk | 0.067 [0.000, 0.167] | — | 6.7% | 0 | — |

Findings from the shipped path:

1. **`semantic` never runs in the shipped code.** `LocalEngine._get_retriever()` calls `CodeRetriever(api_key=...)`, but `CodeRetriever.__init__` accepts no `api_key` argument. The resulting `TypeError` is swallowed, so the retriever is always `None`. The HTTP routers also pass `retriever=None` to `SplitAgent`. With no embedding service, `_get_strategy` leaves `"semantic"` unregistered and falls back to `hybrid`. The agent's own reasoning reports `'hybrid'` on all 30 `semantic` runs.
2. **`hybrid` produces exactly `directory`'s grouping** (per-case ARI identical on 30/30). Without embeddings, hybrid groups by directory and then asks the LLM only for a type label. The extra LLM calls (4.03 vs 2.77 per run) do not change the grouping.
3. **No strategy beats one-group-per-file with confidence.** Every paired Δ CI includes 0. `conventional` scores lower than `directory` and is essentially equal to the per-file baseline.

### As designed: embeddings injected by the harness (`SplitAgent(llm, retriever)` with an `EmbeddingService`, gpt-4.1-mini)

| Config | Scored / run | Errors | Mean ARI [95% CI] | Same-case per-file baseline | Count match |
|---|---|---|---|---|---|
| `semantic` + `text-embedding-004` (shipped id) | 0 / 30 | 30 × HTTP 404 `models/text-embedding-004 is not found` | — | — | — |
| `hybrid` + `text-embedding-004` | 12 / 30 | 18 × HTTP 404 | 0.508 [0.251, 0.757] | 0.485 | 50.0% |
| `semantic` + `gemini-embedding-001` (substituted) | 30 / 30 | 0 | 0.000 [0.000, 0.000] | 0.385 | 0.0% |
| `hybrid` + `gemini-embedding-001` (substituted) | 30 / 30 | 0 | 0.474 [0.328, 0.618] | 0.385 | 40.0% |

Notes on these runs:
- **The shipped embedding model id no longer exists on the Gemini API.** Under `hybrid`, any directory group with more than 3 hunks fails the whole split. The 12 successful `hybrid` + `text-embedding-004` cases are exactly the ones that never called the embedder, so their numbers describe a different and easier subset.
- **With a live model (`gemini-embedding-001`), the 0.5 cosine threshold merges everything.** `semantic` returned a single group on every case, which scores ARI 0, the same as the one-group baseline. `hybrid`'s semantic refinement therefore changes nothing and matches `directory` on 30/30 cases.
- **Some embedding calls hit HTTP 429.** On first pass, 31 cases failed with Gemini HTTP 429 `RESOURCE_EXHAUSTED`: 23 for `semantic` and 8 for `hybrid`. They were re-run with pacing until all succeeded. Every failed attempt is kept under `previous_attempts` in the raw JSON. These are quota failures, not agent failures.

### Why the strategies tie on directory
In these repos, adjacent commits mostly touch different directories, so directory grouping does well. The per-file baseline over-splits: it matches the commit count in only 23.3% of cases.

**Caveat on the metric.** ARI against the author's real commits rewards reproducing *this author's* boundaries. A grouping that is valid but different from the author's is penalized, for example putting docs and code for one feature in a single commit. These scores measure agreement with the author, not correctness.

## Eval B: Commit messages (gpt-4.1-mini, n = 30, 0 errors, 0 empty)

Strict Conventional Commit check (`evals/metrics.py`):
- Header regex: `^(build|chore|ci|docs|feat|fix|perf|refactor|revert|style|test)(\([A-Za-z0-9_./-]+\))?!?: \S(.*\S)?$`
- Header ≤ 100 chars and no trailing period.
- A blank line before any body.

| Metric | gpt-4.1-mini | Baseline / reference |
|---|---|---|
| Strict-valid Conventional Commit | 93.3% (28/30), 95% CI [83.3, 100] | Author's own messages on the same 30 commits: 66.7% (20/30) |
| Type matches the author's type (author message is conventional, **n = 20**) | 65.0% (13/20), 95% CI [45.0, 85.0] | Always predicting the most common type (`feat`): 45.0% |
| LLM calls per run | 2.0 (all 30 runs: analyze + generate) | — |
| Input tokens per run (from `usage`) | mean 3,552, median 2,014 | — |
| Output tokens per run (from `usage`) | mean 306, median 305 | — |
| Wall-clock latency per run | p50 3.14 s, p95 5.21 s (max 9.55 s) | — |

Notes:
- **The 2 strict failures** both use comma-separated multi-scopes (`refactor(llm, config, docker): …`, `feat(cmd,config): …`). My regex treats these as invalid. Every generated message has a parseable conventional type.
- **Where types differ, the model leans toward `refactor`.** The generated type distribution was feat 14, refactor 6, docs 5, test 3, fix 1, chore 1.

## `auto`: LLM calls per run (README: "As few as 2 LLM calls for simple changes")

The harness ran `cli.pipeline.Pipeline` through `SyncLocalBackend(LocalEngine("openai"))`, which is the `inyeon auto --local` path, on 5 single-file commits.

| Case (diff chars) | Default: LLM calls attempted / succeeded | `--no-pr`: attempted / succeeded |
|---|---|---|
| auto-00 (151) | 3 / 2 | 2 / 2 |
| auto-01 (183) | 3 / 2 | 2 / 2 |
| auto-02 (170) | 3 / 2 | 2 / 2 |
| auto-03 (688) | 5 / 4 | 3 / 2 |
| auto-04 (845) | 5 / 4 | 3 / 2 |

- **The 2-call minimum holds only with `--no-pr` on diffs under 500 chars.** It holds in 3/3 such runs: split is skipped for 1 file and review is skipped under 500 chars, leaving commit's analyze + generate. With default flags, the PR step adds calls, and the minimum observed was 3 attempted.
- **Local-mode bugs found during these runs:**
  - **The commit message is always lost.** `Pipeline._step_commit` reads `resp.get("message")`, but `LocalEngine.generate_commit` returns the message under the key `commit_message`. `commit_message` came back `None` in 10/10 runs, so `inyeon auto --local` has nothing to commit.
  - **The first LLM call of every later step fails.** `SyncLocalBackend` calls `asyncio.run()` once per step, but `LocalEngine` caches one `AsyncOpenAI` client. In every run where a second step executed, its first LLM call failed with `OpenAIError: Connection error`: 7/7 runs. The pipeline swallows this. The review came back empty in all 4 runs that reached the review step, yet "review" is still listed under `steps_completed`. The PR came back in only 2/5 default runs.

## Existing test suite

`pytest` (repo `tests/`): **245 passed, 1 skipped** in 86.6 s (`results/pytest_output.txt`). The skip is one of the two live-Ollama tests (`test_analyze.py` / `test_commit.py`), whose skip message reads "Ollama not available or returned invalid format". Ollama was still installed when the suite ran.

## Spend

| Item | USD |
|---|---|
| OpenAI `gpt-4.1-mini`, computed from per-request `usage` (all configs, including superseded and retried attempts) | $0.1819 |
| Gemini `gemini-embedding-001`, upper bound (the API returns no token usage; tokens ≤ chars × $0.15/1M, 565,625 chars) | ≤ $0.0848 |
| **Total (upper bound)** | **≤ $0.267** (hard budget $1.00) |

Prices assumed: gpt-4.1-mini $0.40 / 1M input ($0.10 cached), $1.60 / 1M output; gemini-embedding-001 $0.15 / 1M. `text-embedding-004` calls returned 404 and were not billed. Pre-run estimates are in `results/budget_estimate_*.txt`. The harness also has a spend guard that stops paid configs at $0.90.

## Limitations

- **Single model, single sample.** No Ollama/qwen numbers exist because the model was uninstalled mid-session. Each config ran once at temperature 0.3, with no seeds and no repeated sampling, so run-to-run variance in the LLM-dependent configs is unmeasured. This affects `conventional` grouping, messages and types.
- **Small n.** The CIs are wide: n = 30 split cases, 30 commits, and only 20 for type agreement. The `auto` call counts come from 5 cases and are descriptive only.
- **Agreement, not correctness.** Split ARI and type match measure agreement with one author's choices, which are imperfect labels. Some of the author's own commit messages may themselves be AI-assisted. Valid-but-different groupings are penalized.
- **Selection bias toward easy splits.** The filters (clean non-overlapping hunks, no renames/binaries, ≤ 30k chars, ≥ 2 files) favour file-disjoint commits; only 1/30 cases splits a single file across commits. Interleaved edits within one file were not tested.
- **Semantic results are not the shipped configuration.** They required injecting an `EmbeddingService` that the shipped code never constructs. The `gemini-embedding-001` runs also substitute a model id. They show what the code *would* do, not what users get today.
- **The harness reads past the public API.** It reads the agent's internal `commit_groups` to score at hunk level. The CLI actually commits whole files per split (`git add <files>`); for every config here, hunk-level and file-level ("as executed") ARI were identical (see `summary.json`).
- **Token usage scope.** Counts are exact OpenAI `usage` fields; Gemini embedding usage is not reported by the API, hence the char-based upper bound. Latency is wall-clock per agent run from this laptop over the public internet and includes network time.
- **Local path only.** Only the in-process `LocalEngine` path was evaluated. The hosted Railway backend and the `HttpEngine` path were not.

## Reproduce

```bash
python evals/build_dataset.py                 # -> evals/dataset_manifest.json (needs the 5 repos at the recorded HEADs)
python evals/run_evals.py --budget-check
python evals/run_evals.py split-openai-directory split-openai-conventional split-openai-hybrid split-openai-semantic \
  split-openai-semantic-inject-te004 split-openai-hybrid-inject-te004 \
  split-openai-semantic-inject-gemb001 split-openai-hybrid-inject-gemb001 \
  commit-openai auto-openai auto-openai-nopr
python evals/run_evals.py split-openai-semantic-inject-gemb001 split-openai-hybrid-inject-gemb001 --retry-errors RESOURCE_EXHAUSTED --pace 8
python evals/report.py                        # -> evals/results/summary.json
```

Keys are read from `.env` by `cli.config.settings` and are never printed or written.

## After fixes (2026-11-04)

Re-measured after four local-path fixes (uncommitted in the working tree at the time): `cli/pipeline.py` reads `commit_message`, reports commit errors, and skips a failed or empty review; `cli/engine.py` `SyncLocalBackend` runs every step on one persistent event loop and `cli/commands/auto.py` closes it; `backend/engine/local.py` builds `CodeRetriever(embedding_service=EmbeddingService(...))`; `backend/rag/embeddings.py` uses `gemini-embedding-001` with `output_dimensionality=768`. The harness is `evals/after_fixes.py`, which reuses `run_evals.py` (diff regeneration with sha256 check, LLM metering) and `metrics.py` (ARI, 10,000-resample bootstrap, seed 0). **No embedding service or retriever is injected.** The retriever comes from `LocalEngine._get_retriever()`, and the model id and threshold are the shipped ones. Raw outputs are in `results/raw/after_fixes/`. Every LLM call used `gpt-4.1-mini`, and Gemini was used for embeddings only.

### A. `inyeon auto --local` path (5 cases × 2 flag sets, n = 10 runs)

Same 5 single-file commits and same flags as above. `Pipeline(SyncLocalBackend(LocalEngine("openai")))` was run, then `close()` was called. "Connection error" counts every occurrence in the run record, including LLM call exceptions and step errors.

| Case (diff chars) | Flags | `commit_message` | "Connection error" | steps_completed | steps_skipped | LLM calls (all succeeded) | Before fixes: attempted / succeeded |
|---|---|---|---|---|---|---|---|
| auto-00 (151) | default | present | 0 | commit, pr | split, review | 4 | 3 / 2 |
| auto-01 (183) | default | present | 0 | commit, pr | split, review | 4 | 3 / 2 |
| auto-02 (170) | default | present | 0 | commit, pr | split, review | 4 | 3 / 2 |
| auto-03 (688) | default | present | 0 | commit, review, pr | split | 5 | 5 / 4 |
| auto-04 (845) | default | present | 0 | commit, review, pr | split | 5 | 5 / 4 |
| auto-00 (151) | `--no-pr` | present | 0 | commit | split, review, pr | 2 | 2 / 2 |
| auto-01 (183) | `--no-pr` | present | 0 | commit | split, review, pr | 2 | 2 / 2 |
| auto-02 (170) | `--no-pr` | present | 0 | commit | split, review, pr | 2 | 2 / 2 |
| auto-03 (688) | `--no-pr` | present | 0 | commit, review | split, pr | 3 | 3 / 2 |
| auto-04 (845) | `--no-pr` | present | 0 | commit, review | split, pr | 3 | 3 / 2 |

- **`commit_message` is present in 10/10 runs (before: 0/10). Connection errors: 0 (before: 7/7 runs that reached a second step). LLM call failures: 0/34.**
- **Review results.** Review completed with content in all 4 runs over 500 chars (keys `summary`, `quality_score`, `issues`, `positives`). Before the fixes, all 4 came back empty but were listed as completed. Under 500 chars, review is skipped by design.
- **PR results.** A PR description was produced in 5/5 default runs (before: 2/5).
- **The 2-call minimum still holds only with `--no-pr` on diffs under 500 chars** (3/3). The default flags now cost 4 calls there, because PR generation actually runs (2 calls) instead of failing on its first call.
- **The retriever is now a real `CodeRetriever` in every run.** It made 0 embedding calls in `auto`, because the in-memory vector store is empty and RAG search is skipped.

### B. Semantic threshold recalibration

**Setup.** The 30 split cases were divided by repo, with no repo on both sides. The repo names were shuffled with seed 20261104, and the first 3 became the calibration set.
- **Calibration:** commma, inyeon, kiln. 18 cases, 164 hunks.
- **Held-out:** draft-thinker, ferrox. 12 cases, 158 hunks.

**Embeddings.** Each case's hunks were embedded once with the shipped `EmbeddingService` (one batch per case, the same hunk text `SemanticStrategy` builds) and cached locally. The cache was deleted afterwards. 30 batch calls succeeded. 7 attempts hit Gemini 429s and were retried with backoff.

**Sweep.** The shipped `SemanticStrategy` / `HybridStrategy` / `DirectoryStrategy` classes were run offline on the cached vectors. Hybrid's LLM is used only for the type label, not the grouping, so it was stubbed for the sweep.

**Selection rule (fixed before the sweep).** Pick the threshold with the highest mean semantic ARI on the calibration set; on a tie, take the lowest threshold.

**Cosine similarity of hunk pairs (calibration):**

| Pair type | n | p5 | p50 | p95 |
|---|---|---|---|---|
| same-commit | 614 | 0.592 | 0.720 | 0.872 |
| cross-commit | 503 | 0.562 | 0.623 | 0.747 |

Almost every pair is above 0.5. That is why `distance_threshold = 0.5` (similarity 0.5) merged everything. The p95 sits below 0.95, so the pre-registered 0.50–0.95 grid (step 0.025) covered the range and was not extended.

**Calibration (n = 18).** Baselines: per-file 0.407 [0.255, 0.567]; directory 0.496 [0.320, 0.664]; one group 0.000.

| similarity_threshold | 0.50–0.55 | 0.60 | 0.65 | 0.675 | 0.70 | 0.725 | **0.75** | 0.775 | 0.80 | 0.85 | 0.90 | 0.95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| semantic mean ARI | 0.000 | 0.089 | 0.357 | 0.501 | 0.584 | 0.592 | **0.661** | 0.506 | 0.429 | 0.298 | 0.140 | 0.056 |
| semantic mean groups | 1.0 | 1.2 | 1.8 | 2.1 | 2.5 | 3.3 | 3.8 | 4.6 | 5.3 | 6.9 | 8.5 | 9.1 |
| hybrid mean ARI | 0.496 | 0.496 | 0.541 | 0.541 | 0.491 | 0.483 | 0.459 | 0.430 | 0.409 | 0.321 | 0.271 | 0.255 |

**Chosen: 0.75.** Calibration semantic ARI was 0.661 [0.514, 0.802]. The full grid, including the rows not shown, is in `threshold_sweep_calibration.json`. The peak is narrow: 0.725 gives 0.592 and 0.775 gives 0.506.

**Held-out (n = 12, never used for selection).** Bootstrap 95% CIs; Δ values are paired per case.

| Config | Mean ARI [95% CI] | Δ vs per-file [95% CI] | Δ vs directory [95% CI] | Count match | Mean groups |
|---|---|---|---|---|---|
| `semantic` @ 0.75 (new default) | 0.513 [0.277, 0.735] | +0.161 [−0.116, +0.410] | +0.071 [−0.159, +0.245] | 16.7% | 2.92 |
| `hybrid` @ 0.75 (shipped default strategy) | **0.525 [0.287, 0.746]** | **+0.174 [+0.015, +0.351]** | **+0.084 [+0.002, +0.192]** | 25.0% | 3.33 |
| `semantic` @ 0.5 (old default) | 0.000 [0.000, 0.000] | −0.352 [−0.590, −0.137] | −0.441 [−0.696, −0.182] | 0.0% | 1.00 |
| `hybrid` @ 0.5 (old default) | 0.441 [0.182, 0.696] | +0.090 [−0.056, +0.247] | 0.000 (identical to directory on 12/12) | 33.3% | 2.58 |
| `directory` | 0.441 [0.182, 0.696] | +0.090 [−0.056, +0.247] | — | 33.3% | 2.58 |
| **Baseline:** one group per file | 0.352 [0.137, 0.590] | — | −0.090 [−0.247, +0.056] | 33.3% | 5.83 |
| **Baseline:** one group | 0.000 | −0.352 | −0.441 | 0.0% | 1.00 |

Reading the held-out results:
- **`semantic` @ 0.75 no longer collapses everything.** It moves from ARI 0 to 0.513. That is above directory and per-file on the mean, but both paired CIs include 0, so this is not a demonstrated win over the simple baselines.
- **`hybrid` @ 0.75 beats both baselines, with paired CIs that just exclude 0.** The lower bounds are +0.015 and +0.002, on n = 12. Treat this as weak evidence. On calibration, hybrid was *worse* at 0.75 than at 0.5 (0.459 vs 0.496). So the hybrid gain did not replicate across the split, and its direction is unsettled.
- **Hybrid inherits the threshold.** It constructs `SemanticStrategy(embedding_service)` without passing a threshold, so the new default applies to hybrid's sub-clustering of directory groups with more than 3 hunks. Hybrid was never tuned on its own.

**Live check of the shipped path (held-out, paid).** `SplitAgent(llm, LocalEngine._get_retriever())` was run exactly as `LocalEngine.split_diff` builds it, with no injection.
- **`semantic`:** 10/12 cases scored, ARI 0.443 [0.185, 0.690]. The agent's reasoning reports `'semantic' strategy`, so semantic now really runs; before the fixes it fell back to hybrid on 30/30. Groupings were **identical to the offline cached-vector result on 10/10 cases**.
- **Failures.** The other 2 semantic cases, both hybrid cases attempted (split-06, split-07) and the remaining 10 hybrid cases (not run) were blocked by the Gemini free-tier daily quota. The quota is `EmbedContentRequestsPerDayPerUserPerProjectPerModel-FreeTier`, limit 1000, where each text in a batch counts as one request; the API asked for a retry in about 11.8 h.
- **What the held-out table rests on.** Its hybrid numbers come from the offline run of the shipped classes on cached shipped-model vectors. The semantic live check shows that this offline path reproduces the live grouping exactly.

**Code change.** In `backend/clustering/semantic.py`, the `SemanticStrategy.__init__` default `similarity_threshold` changed from 0.5 to 0.75, with a one-line comment citing this file. `HybridStrategy` passes no threshold, so nothing else changed.

### C. Test suite

`python -m pytest -p no:cacheprovider` after the threshold change gave **251 passed, 2 skipped** (`results/raw/after_fixes/pytest_output.txt`).
- 251 = 244 pre-existing tests + 7 in `tests/cli/test_pipeline_regressions.py`, all passing.
- The 2 skips are the live-Ollama tests; Ollama is not installed.
- No existing test asserts the threshold, so no test was edited.

### Spend (this section)

| Item | USD |
|---|---|
| OpenAI `gpt-4.1-mini`, exact from per-request `usage`: `auto` 10 runs plus live held-out semantic split | $0.0155 |
| Gemini `gemini-embedding-001`: the key is on the **free tier** (per the 429 response), so the billed amount is $0. The bound at the paid rate ($0.15/1M, tokens ≤ chars) is shown for reference. | $0 billed (≤ $0.0746 at paid rate) |
| **Total** | **$0.0155 billed; ≤ $0.090 even at paid embedding rates** (cap $0.50; estimate in `budget_estimate.json`) |

### Limitations (this section)
- **One run per config.** LLM outputs (messages, reviews, PRs) were not resampled. The clustering itself is deterministic given the vectors.
- **Small held-out set.** n = 12 cases from 2 repos. The CIs are wide, and hybrid's CI clears 0 by only +0.002 against directory. A different repo split could pick a different threshold, since the calibration peak is narrow (±0.025 drops ARI by ≥ 0.07).
- **Partial live check.** Most of the live hybrid check was blocked by the Gemini daily quota. Hybrid held-out numbers are offline only, using the shipped classes and shipped-model vectors.
- **Labels and selection bias.** The caveats above still apply: ARI measures agreement with one author's commit boundaries, and the dataset is biased toward file-disjoint splits.
- **Harmless shutdown noise in multi-run processes.** In a process that builds several `SyncLocalBackend`s, a closed backend's pooled HTTPS connections can log `RuntimeError: Event loop is closed` when garbage-collected, because `close()` closes the loop without first closing the async LLM client. It did not affect any result, and it was not reproduced in a single-run process.
- **Chroma side effect.** Building the retriever now creates an in-memory Chroma client on every local run. That client writes `~/.cache/chroma/telemetry_user_id` even with `anonymized_telemetry=False`; it was deleted after this run.
