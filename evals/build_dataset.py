"""Build the eval dataset manifest from real commits in local git repos.

Reproducible: given the same repos at the same HEADs, this produces the same
manifest (deterministic scan + seeded sampling). The manifest stores repo paths,
commit SHAs, hunk ids and ground-truth labels, plus a sha256 of every diff so
`run_evals.py` can verify it regenerates byte-identical diffs.

Usage:  python evals/build_dataset.py
Output: evals/dataset_manifest.json
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.diff import DiffParser, LineType  # noqa: E402  (Inyeon's own parser -> same hunk ids)

AUTHOR_NAME = "Anh D Tran"
MAX_DIFF_CHARS = 30_000  # backend/utils/cost.py DEFAULT_MAX_DIFF_CHARS
SEED = 20261104
SPLIT_TARGET = 30
COMMIT_TARGET = 30
AUTO_TARGET = 5

REPOS = {
    "kiln": "D:/Projects/K8s-Platform/kiln",
    "ferrox": "D:/Projects/Trading-Matching-Engine/ferrox",
    "commma": "D:/Projects/Commma-Tracker/commma-coding-progress-tracker",
    "inyeon": "D:/Projects/Inyeon-TerminalCLI/inyeon-upstream",
    "draft-thinker": "D:/Projects/Cost-Aware_LLM_Gateway/draft-thinker",
}

LOCK_BASENAMES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "cargo.lock", "poetry.lock",
    "uv.lock", "pipfile.lock", "go.sum", "composer.lock", "gemfile.lock", "bun.lockb",
    "npm-shrinkwrap.json", ".terraform.lock.hcl", "go.work.sum",
}
GENERATED_RE = re.compile(
    r"(^|/)(dist|build|out|node_modules|\.next|target|__pycache__|[^/]+\.egg-info)/"
    r"|\.min\.(js|css)$|\.map$|\.snap$|_pb2\.py$|\.pb\.go$|\.generated\.|\.lock$"
)


def git(repo: str, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", repo, "-c", "core.quotepath=false", *args],
        capture_output=True,
        check=True,
    )
    return out.stdout.decode("utf-8", errors="replace")


def diff_text(repo: str, a: str, b: str) -> str:
    # Same defaults the CLI gets from `git diff --cached` (rename detection on).
    return git(repo, "diff", "--no-color", "--no-ext-diff", a, b)


def is_generated(path: str) -> bool:
    p = path.replace("\\", "/")
    return p.rsplit("/", 1)[-1].lower() in LOCK_BASENAMES or bool(GENERATED_RE.search(p))


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


class RepoInfo:
    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path
        self.head = git(path, "rev-parse", "HEAD").strip()
        self.branch = git(path, "rev-parse", "--abbrev-ref", "HEAD").strip()
        raw = git(path, "log", "--first-parent", "--reverse", "--format=%H%x1f%P%x1f%an%x1f%B%x1e", "HEAD")
        self.commits: list[dict] = []
        for rec in raw.split("\x1e"):
            rec = rec.strip("\n")
            if not rec:
                continue
            sha, parents, author, body = rec.split("\x1f", 3)
            self.commits.append(
                {"sha": sha, "parents": parents.split(), "author": author, "message": body.strip()}
            )
        self._numstat: dict[str, list] = {}
        self._namestatus: dict[str, list] = {}

    def numstat(self, sha: str, parent: str) -> list[tuple[str, str, str]]:
        if sha not in self._numstat:
            out = git(self.path, "diff", "--numstat", "--no-renames", parent, sha)
            rows = []
            for line in out.splitlines():
                a, r, f = line.split("\t", 2)
                rows.append((a, r, f))
            self._numstat[sha] = rows
        return self._numstat[sha]

    def name_status(self, a: str, b: str) -> list[str]:
        key = f"{a}..{b}"
        if key not in self._namestatus:
            out = git(self.path, "diff", "--name-status", "-M", a, b)
            self._namestatus[key] = [l.split("\t", 1)[0] for l in out.splitlines() if l]
        return self._namestatus[key]


def commit_ok(repo: RepoInfo, c: dict) -> tuple[bool, str]:
    """Single-commit filters shared by both evals."""
    if len(c["parents"]) != 1:
        return False, "merge_or_root"
    if c["author"] != AUTHOR_NAME:
        return False, "other_author"
    rows = repo.numstat(c["sha"], c["parents"][0])
    if not rows:
        return False, "empty"
    if all(is_generated(f) for _, _, f in rows):
        return False, "lockfile_or_generated_only"
    if any(a == "-" for a, _, _ in rows):
        return False, "binary"
    return True, ""


def blame_commits(repo: str, rng: str, path: str, reverse: bool) -> list[str]:
    args = ["blame", "-l", "-s"]
    if reverse:
        args.append("--reverse")
    out = git(repo, *args, rng, "--", path)
    shas = []
    for line in out.splitlines():
        tok = line.split(" ", 1)[0].lstrip("^")
        shas.append(tok)
    return shas


def attribute_window(repo: RepoInfo, window: list[dict]) -> tuple[dict | None, str]:
    """Return a split case (or None, reason). Ground truth via git blame."""
    base = window[0]["parents"][0]
    tip = window[-1]["sha"]
    chain = [base] + [c["sha"] for c in window]

    statuses = repo.name_status(base, tip)
    if any(s.startswith(("R", "C")) for s in statuses):
        return None, "rename_or_copy"
    for c in window:
        if any(s.startswith(("R", "C")) for s in repo.name_status(c["parents"][0], c["sha"])):
            return None, "rename_or_copy"

    diff = diff_text(repo.path, base, tip)
    if not diff.strip():
        return None, "empty"
    if len(diff) > MAX_DIFF_CHARS:
        return None, "too_large"
    parsed = DiffParser().parse(diff)
    if len(parsed.files) < 2:
        return None, "fewer_than_2_files"  # auto pipeline only splits >= 2 files
    if any(f.is_binary or not f.hunks for f in parsed.files):
        return None, "binary_or_no_hunk_file"

    rng = f"{base}..{tip}"
    labels: dict[str, int] = {}
    attributed_added = Counter()
    attributed_removed = Counter()
    for f in parsed.files:
        fwd = rev = None
        for h in f.hunks:
            owners = set()
            for ln in h.lines:
                if ln.line_type == LineType.ADDED:
                    if fwd is None:
                        fwd = blame_commits(repo.path, rng, f.path, reverse=False)
                    sha = fwd[ln.target_line_no - 1]
                    owner = next((i for i, s in enumerate(chain) if s.startswith(sha) or sha.startswith(s)), None)
                    if owner is None or owner == 0:
                        return None, "blame_unattributed"
                    owners.add(owner)
                    attributed_added[owner] += 1
                elif ln.line_type == LineType.REMOVED:
                    if rev is None:
                        rev = blame_commits(repo.path, rng, f.path, reverse=True)
                    sha = rev[ln.source_line_no - 1]
                    last = next((i for i, s in enumerate(chain) if s.startswith(sha) or sha.startswith(s)), None)
                    if last is None or last + 1 >= len(chain):
                        return None, "blame_unattributed"
                    owners.add(last + 1)
                    attributed_removed[last + 1] += 1
            if len(owners) != 1:
                return None, "overlapping_hunk"
            labels[h.id] = owners.pop() - 1  # 0-based index into window

    # Conservation check: every commit's own line counts must survive the squash
    # unchanged (catches "c1 adds a line, c2 edits it", which blame alone hides).
    for i, c in enumerate(window, start=1):
        rows = repo.numstat(c["sha"], c["parents"][0])
        add = sum(int(a) for a, _, _ in rows)
        rem = sum(int(r) for _, r, _ in rows)
        if add != attributed_added[i] or rem != attributed_removed[i]:
            return None, "interacting_commits"

    if len(set(labels.values())) != len(window):
        return None, "commit_without_hunk"

    return {
        "repo": repo.name,
        "repo_path": repo.path,
        "base": base,
        "commits": [c["sha"] for c in window],
        "commit_subjects": [c["message"].splitlines()[0] for c in window],
        "n_commits": len(window),
        "n_files": len(parsed.files),
        "n_hunks": len(labels),
        "diff_chars": len(diff),
        "diff_sha256": sha256(diff),
        "hunk_labels": labels,
        "hunk_files": {h.id: f.path for f in parsed.files for h in f.hunks},
    }, ""


def scan_split_windows(repo: RepoInfo, rejects: Counter) -> list[dict]:
    """Greedy, deterministic, non-overlapping scan over the first-parent chain."""
    cases = []
    commits = repo.commits
    i = 0
    while i < len(commits):
        order = [[2, 3, 4], [3, 4, 2], [4, 2, 3]][i % 3]  # vary window size
        taken = None
        for k in order:
            window = commits[i : i + k]
            if len(window) < k:
                continue
            bad = None
            for c in window:
                ok, why = commit_ok(repo, c)
                if not ok:
                    bad = why
                    break
            if bad:
                rejects[bad] += 1
                continue
            try:
                case, why = attribute_window(repo, window)
            except (subprocess.CalledProcessError, IndexError) as e:
                case, why = None, f"git_error:{type(e).__name__}"
            if case is None:
                rejects[why] += 1
                continue
            taken = case
            break
        if taken:
            cases.append(taken)
            i += taken["n_commits"]
        else:
            i += 1
    return cases


def single_commit_entry(repo: RepoInfo, c: dict) -> tuple[dict | None, str]:
    ok, why = commit_ok(repo, c)
    if not ok:
        return None, why
    diff = diff_text(repo.path, c["parents"][0], c["sha"])
    if len(diff) > MAX_DIFF_CHARS:
        return None, "too_large"
    parsed = DiffParser().parse(diff)
    if not any(f.hunks for f in parsed.files):
        return None, "no_text_hunks"
    return {
        "repo": repo.name,
        "repo_path": repo.path,
        "sha": c["sha"],
        "parent": c["parents"][0],
        "author_message": c["message"],
        "n_files": len(parsed.files),
        "diff_chars": len(diff),
        "diff_sha256": sha256(diff),
    }, ""


def balanced_sample(by_repo: dict[str, list], target: int, rnd: random.Random) -> list:
    pools = {k: list(v) for k, v in sorted(by_repo.items())}
    for v in pools.values():
        rnd.shuffle(v)
    picked = []
    while len(picked) < target and any(pools.values()):
        for k in sorted(pools):
            if pools[k] and len(picked) < target:
                picked.append(pools[k].pop())
    return picked


def main() -> None:
    rnd = random.Random(SEED)
    repos = {name: RepoInfo(name, path) for name, path in REPOS.items()}

    split_by_repo: dict[str, list] = {}
    split_rejects: dict[str, Counter] = {}
    for name, repo in repos.items():
        rej = Counter()
        split_by_repo[name] = scan_split_windows(repo, rej)
        split_rejects[name] = rej
        print(f"[split] {name}: {len(split_by_repo[name])} valid windows; rejects={dict(rej)}", flush=True)
    split_cases = balanced_sample(split_by_repo, SPLIT_TARGET, rnd)
    split_cases.sort(key=lambda c: (c["repo"], c["commits"][0]))
    for i, c in enumerate(split_cases):
        c["case_id"] = f"split-{i:02d}"

    used = {s for c in split_cases for s in c["commits"]}
    commit_by_repo: dict[str, list] = defaultdict(list)
    commit_rejects: dict[str, Counter] = {}
    auto_pool = []
    for name, repo in repos.items():
        rej = Counter()
        for c in repo.commits:
            if c["sha"] in used:
                rej["used_in_split_eval"] += 1
                continue
            e, why = single_commit_entry(repo, c)
            if e is None:
                rej[why] += 1
                continue
            commit_by_repo[name].append(e)
            if e["n_files"] == 1:
                auto_pool.append(e)
        commit_rejects[name] = rej
        print(f"[commit] {name}: {len(commit_by_repo[name])} eligible; rejects={dict(rej)}", flush=True)
    commit_cases = balanced_sample(commit_by_repo, COMMIT_TARGET, rnd)
    commit_cases.sort(key=lambda c: (c["repo"], c["sha"]))
    for i, c in enumerate(commit_cases):
        c["case_id"] = f"commit-{i:02d}"

    # `auto` call-count cases: small single-file commits not used above.
    taken = {c["sha"] for c in commit_cases}
    auto_pool = [e for e in auto_pool if e["sha"] not in taken]
    small = sorted([e for e in auto_pool if e["diff_chars"] < 500], key=lambda e: (e["repo"], e["sha"]))
    medium = sorted([e for e in auto_pool if 500 <= e["diff_chars"] <= 3000], key=lambda e: (e["repo"], e["sha"]))
    rnd.shuffle(small)
    rnd.shuffle(medium)
    auto_cases = small[:3] + medium[:2]
    for i, c in enumerate(auto_cases):
        c["case_id"] = f"auto-{i:02d}"

    manifest = {
        "created": "2026-11-04",
        "seed": SEED,
        "author_filter": AUTHOR_NAME,
        "max_diff_chars": MAX_DIFF_CHARS,
        "repos": {n: {"path": r.path, "branch": r.branch, "head": r.head, "first_parent_commits": len(r.commits)} for n, r in repos.items()},
        "selection_rules": {
            "split": "Windows of 2-4 ADJACENT first-parent commits, all non-merge and authored by "
            "'Anh D Tran'; no lockfile/generated-only commit; no binary files; no renames/copies; squashed "
            "diff (git diff <base> <tip>) <= 30,000 chars and touching >= 2 files; every squashed hunk "
            "attributed (git blame / blame --reverse over base..tip) to exactly one commit; per-commit "
            "added/removed line counts conserved in the squash. Greedy non-overlapping scan, then "
            "seeded round-robin sampling across repos.",
            "commit": "Single non-merge commits by 'Anh D Tran', not lockfile/generated-only, no binary "
            "files, diff <= 30,000 chars, not used in any split case. Seeded round-robin sampling across repos.",
            "auto": "Single-file commits not used above: 3 with diff < 500 chars (review skipped) and 2 "
            "with 500-3000 chars (review runs).",
        },
        "split_valid_windows_per_repo": {k: len(v) for k, v in split_by_repo.items()},
        "split_reject_reasons": {k: dict(v) for k, v in split_rejects.items()},
        "commit_eligible_per_repo": {k: len(v) for k, v in commit_by_repo.items()},
        "commit_reject_reasons": {k: dict(v) for k, v in commit_rejects.items()},
        "split_cases": split_cases,
        "commit_cases": commit_cases,
        "auto_cases": auto_cases,
    }
    out = Path(__file__).resolve().parent / "dataset_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {out}: {len(split_cases)} split, {len(commit_cases)} commit, {len(auto_cases)} auto cases")


if __name__ == "__main__":
    main()
