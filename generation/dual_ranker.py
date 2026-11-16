"""Deterministic, evidence-aware selection of research candidates.

This is deliberately a ranking *scaffold*: it consumes human/model supplied
fields but does not claim to score novelty or repair malformed scores.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping

EVIDENCE_STATUSES = {"covered", "incremental", "unresolved", "plausibly_distinct"}


@dataclass(frozen=True)
class Selection:
    id: str
    row: Mapping[str, Any]
    reason: str
    promoted_confidence: bool
    exploration: bool


@dataclass(frozen=True)
class SelectionResult:
    selected: tuple[Selection, ...]
    rejected: Mapping[str, str] = field(default_factory=dict)


def _finite(value: Any, name: str, *, nonnegative: bool = False, unit_interval: bool = False) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number or None")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    if unit_interval and not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")


def _validate(row: Mapping[str, Any]) -> None:
    if not row.get("id") or not row.get("problem_family"):
        raise ValueError("each row requires id and problem_family")
    d, e = row.get("discovery", {}), row.get("execution", {})
    if not isinstance(d, Mapping) or not isinstance(e, Mapping):
        raise ValueError("discovery and execution must be objects")
    status = d.get("evidence_status")
    if status is not None and status not in EVIDENCE_STATUSES:
        raise ValueError(f"unknown evidence_status: {status}")
    _finite(d.get("novelty_estimate"), "novelty_estimate", unit_interval=True)
    _finite(d.get("uncertainty"), "uncertainty", unit_interval=True)
    _finite(e.get("feasibility"), "feasibility", unit_interval=True)
    _finite(e.get("expected_gain"), "expected_gain")
    _finite(e.get("utility_uncertainty"), "utility_uncertainty", nonnegative=True)


def _context(row: Mapping[str, Any]) -> tuple[Any, ...]:
    e = row["execution"]
    budget = e.get("budget")
    try:
        budget_key = json.dumps(budget, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        budget_key = repr(budget)
    return (row["problem_family"], e.get("metric"), e.get("baseline_id"), budget_key)


def _utility_comparable(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    return _context(a) == _context(b)


def dominates(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    """Return whether ``a`` Pareto-dominates ``b`` on known, comparable axes."""
    da, db = a["discovery"], b["discovery"]
    ea, eb = a["execution"], b["execution"]
    axes = [(da.get("novelty_estimate"), db.get("novelty_estimate")),
            (ea.get("feasibility"), eb.get("feasibility"))]
    # A three-axis claim is invalid when usefulness is not on the same unit.
    # Do not let novelty/feasibility accidentally dominate across metrics.
    if not _utility_comparable(a, b):
        return False
    axes.append((ea.get("expected_gain"), eb.get("expected_gain")))
    if any(x is None or y is None for x, y in axes):
        return False
    return all(x >= y for x, y in axes) and any(x > y for x, y in axes)


def d_is_known(row: Mapping[str, Any]) -> bool:
    """Whether all three ranking axes and their comparison context are known."""
    d, e = row["discovery"], row["execution"]
    return (d.get("source_consistent") is True
            and d.get("novelty_estimate") is not None
            and e.get("feasibility") is not None
            and e.get("expected_gain") is not None
            and e.get("metric") not in (None, "")
            and e.get("baseline_id") not in (None, "")
            and e.get("budget") is not None)


def rank_and_select(rows: list[Mapping[str, Any]], count: int, *, seed: int = 0,
                    feasibility_threshold: float | None = None,
                    exploration_quota: int = 0) -> SelectionResult:
    """Select at most ``count`` rows with deterministic exploitation/exploration.

    Unknown values are never converted to zero.  They can enter only through
    the explicit exploration quota (or when the quota exceeds known rows).
    """
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("count must be a nonnegative integer")
    if isinstance(exploration_quota, bool) or not isinstance(exploration_quota, int) or exploration_quota < 0:
        raise ValueError("exploration_quota must be a nonnegative integer")
    if feasibility_threshold is not None:
        _finite(feasibility_threshold, "feasibility_threshold", unit_interval=True)
    for r in rows:
        _validate(r)
    # Duplicate ids make deterministic rationales ambiguous.
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("candidate ids must be unique")
    rejected: dict[str, str] = {}
    eligible: list[Mapping[str, Any]] = []
    for r in rows:
        d, e = r["discovery"], r["execution"]
        if r.get("known_invalid") or d.get("known_invalid") or e.get("known_invalid"):
            rejected[r["id"]] = "known_invalid"
        elif d.get("source_consistent") is False:
            rejected[r["id"]] = "source_inconsistent"
        elif feasibility_threshold is not None and e.get("feasibility") is not None and e["feasibility"] < feasibility_threshold:
            rejected[r["id"]] = "below_feasibility_threshold"
        else:
            eligible.append(r)
    def tie(r: Mapping[str, Any]) -> str:
        return hashlib.sha256(f"{seed}:{r['id']}".encode()).hexdigest()
    known = [r for r in eligible
             if d_is_known(r)]
    frontier = [r for r in known if not any(dominates(o, r) for o in known if o["id"] != r["id"])]
    # First meet diversity quota: one candidate per family where possible.
    chosen: list[Mapping[str, Any]] = []
    families = list({r["problem_family"] for r in eligible})
    families.sort(key=lambda fam: hashlib.sha256(f"{seed}:family:{fam}".encode()).hexdigest())
    for fam in families:
        if len(chosen) >= min(count, exploration_quota):
            break
        pool = [r for r in eligible if r["problem_family"] == fam and r not in chosen]
        if pool:
            # Exploration slots deliberately admit missing axes.  This is the
            # only path by which an unknown candidate is promoted into output.
            pool.sort(key=lambda r: (d_is_known(r),
                                      r not in frontier, tie(r)))
            chosen.append(pool[0])
    # Unknown values are eligible only in explicit exploration slots.
    remaining = [r for r in known if r not in chosen]
    remaining.sort(key=lambda r: (r not in frontier, r["discovery"].get("novelty_estimate") is None,
                                 r["execution"].get("feasibility") is None, tie(r)))
    chosen.extend(remaining)
    chosen = chosen[:count]
    out = []
    for r in chosen:
        explored = len(out) < min(count, exploration_quota)
        out.append(Selection(r["id"], r, "exploration_quota" if explored else ("pareto_frontier" if r in frontier else "best_known"), not explored, explored))
    return SelectionResult(tuple(out), rejected)


select_candidates = rank_and_select

