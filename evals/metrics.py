"""Scoring helpers shared by report.py (no LLM calls, no network)."""

from __future__ import annotations

import random
import re
import statistics

from sklearn.metrics import adjusted_rand_score

CC_TYPES = ("build", "chore", "ci", "docs", "feat", "fix", "perf", "refactor", "revert", "style", "test")

# Strict Conventional Commits header (commitlint config-conventional flavour):
#   <type>[(<scope>)][!]: <description>
#   - type: lowercase, one of CC_TYPES
#   - scope: optional, non-empty, no spaces or parentheses
#   - exactly ": " after type/scope/!, description non-empty, no leading/trailing space
#   - description must not end with "."
#   - header <= 100 chars; if there is a body, line 2 must be blank
CC_HEADER_RE = re.compile(
    r"^(?P<type>" + "|".join(CC_TYPES) + r")"
    r"(?:\((?P<scope>[A-Za-z0-9_./-]+)\))?"
    r"(?P<bang>!)?"
    r": (?P<desc>\S(?:.*\S)?)$"
)
LENIENT_TYPE_RE = re.compile(r"^\s*(?P<type>[A-Za-z]+)(?:\([^)\n]*\))?!?\s*:\s*\S")


def strict_cc_check(message: str | None) -> tuple[bool, str]:
    if not message or not message.strip():
        return False, "empty"
    lines = message.split("\n")
    header = lines[0].rstrip("\r")
    m = CC_HEADER_RE.match(header)
    if not m:
        return False, "header_regex"
    if len(header) > 100:
        return False, "header_too_long"
    if m.group("desc").endswith("."):
        return False, "trailing_period"
    if len(lines) > 1 and lines[1].strip() != "":
        return False, "no_blank_line_before_body"
    return True, "ok"


def cc_type(message: str | None) -> str | None:
    """Lenient type extraction; returns a CC type or None if not conventional."""
    if not message:
        return None
    m = LENIENT_TYPE_RE.match(message.split("\n", 1)[0])
    if not m:
        return None
    t = m.group("type").lower()
    return t if t in CC_TYPES else None


def ari(true: dict[str, int], pred: dict[str, str]) -> float:
    keys = sorted(true)
    return float(adjusted_rand_score([true[k] for k in keys], [pred[k] for k in keys]))


def bootstrap_ci(values: list[float], n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05):
    if not values:
        return None
    rnd = random.Random(seed)
    n = len(values)
    means = sorted(statistics.fmean(rnd.choices(values, k=n)) for _ in range(n_boot))
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot) - 1]
    return [lo, hi]


def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolated percentile (numpy 'linear' method)."""
    if not values:
        return None
    v = sorted(values)
    pos = (len(v) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)
