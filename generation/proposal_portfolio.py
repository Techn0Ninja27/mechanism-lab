"""Deterministic bookkeeping for comparing a small proposal portfolio.

This module does not score novelty or usefulness.  It only fingerprints the
causal interface a proposal declares and checks that a final review covers
every candidate and pair exactly once.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any


class PortfolioError(ValueError):
    pass


_REQUIRED_KEYS = ("baseline", "operator_delta", "data_paths")
_RELATIONSHIPS = {"same_mechanism", "different_mechanism", "uncertain"}


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PortfolioError(f"{field} must be a non-empty string")
    return " ".join(value.split())


def _canonical(value: Any) -> Any:
    """Canonicalize text and mapping keys while preserving list causal order."""
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, Mapping):
        return {str(k): _canonical(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, list):
        return [_canonical(x) for x in value]
    if isinstance(value, tuple):
        return [_canonical(x) for x in value]
    return value


def _fingerprint(proposal: Mapping[str, Any]) -> str:
    payload = {key: _canonical(proposal[key]) for key in _REQUIRED_KEYS}
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _tokens(proposal: Mapping[str, Any]) -> set[str]:
    fields = [proposal[key] for key in ("baseline", "operator_delta")]
    fields.append(proposal["data_paths"])
    return set(re.findall(r"[a-z0-9_]+", json.dumps(_canonical(fields), ensure_ascii=False).casefold()))


def annotate_candidates(proposals: Sequence[Mapping[str, Any]], size: int = 3) -> list[dict[str, Any]]:
    """Assign local IDs and deterministic duplicate/near-duplicate metadata."""
    if not isinstance(proposals, Sequence) or isinstance(proposals, (str, bytes)):
        raise PortfolioError("proposals must be a sequence")
    if len(proposals) != size or size < 1 or size > 8:
        raise PortfolioError(f"exactly {size} candidates are required")
    rows: list[dict[str, Any]] = []
    first_by_fp: dict[str, str] = {}
    token_sets: dict[str, set[str]] = {}
    for index, proposal in enumerate(proposals):
        if not isinstance(proposal, Mapping):
            raise PortfolioError(f"candidate {index} is not an object")
        missing = [key for key in _REQUIRED_KEYS if key not in proposal]
        if missing:
            raise PortfolioError(f"candidate {index} missing {', '.join(missing)}")
        _text(proposal["baseline"], f"candidate {index}.baseline")
        _text(proposal["operator_delta"], f"candidate {index}.operator_delta")
        if not isinstance(proposal["data_paths"], list) or not proposal["data_paths"]:
            raise PortfolioError(f"candidate {index}.data_paths must be a non-empty list")
        for path_index, path in enumerate(proposal["data_paths"]):
            if not isinstance(path, Mapping) or not all(isinstance(path.get(k), str) and path[k].strip() for k in ("input", "operation", "output")):
                raise PortfolioError(f"candidate {index}.data_paths[{path_index}] is incomplete")
        candidate_id = f"cand-{index:02d}"
        fingerprint = _fingerprint(proposal)
        exact = first_by_fp.get(fingerprint)
        if exact is None:
            first_by_fp[fingerprint] = candidate_id
        tokens = _tokens(proposal)
        token_sets[candidate_id] = tokens
        # No model assertion automatically makes a candidate experiment-eligible.
        rows.append({"id": candidate_id, "proposal": dict(proposal), "fingerprint": fingerprint,
                     "exact_duplicate_of": exact, "auto_eligible": False,
                     "near_duplicate_of": []})
    for row in rows:
        mine = token_sets[row["id"]]
        for other in rows:
            if other["id"] == row["id"] or other["exact_duplicate_of"] == row["id"]:
                continue
            other_tokens = token_sets[other["id"]]
            union = mine | other_tokens
            similarity = len(mine & other_tokens) / len(union) if union else 1.0
            if similarity >= 0.8 and row["exact_duplicate_of"] is None and other["id"] not in row["near_duplicate_of"]:
                row["near_duplicate_of"].append(other["id"])
        row["near_duplicate_of"].sort()
    return rows


def compare_structures(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not isinstance(candidates, Sequence):
        raise PortfolioError("candidates must be a sequence")
    ids = [row.get("id") for row in candidates]
    if len(ids) != len(set(ids)) or any(not isinstance(x, str) for x in ids):
        raise PortfolioError("candidate IDs must be unique strings")
    groups: dict[str, list[str]] = defaultdict(list)
    for row in candidates:
        groups[row["fingerprint"]].append(row["id"])
    return {"exact_duplicate_groups": [sorted(v) for v in groups.values() if len(v) > 1],
            "near_duplicate_pairs": sorted({tuple(sorted((row["id"], other))) for row in candidates for other in row.get("near_duplicate_of", [])}),
            "scope": "Deterministic structural diagnostics only; no novelty or usefulness score."}


def validate_portfolio_review(value: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]], review_validator: Callable[[Mapping[str, Any], Mapping[str, Any]], Any] | None = None) -> dict[str, Any]:
    """Validate portfolio coverage and delegate each review to parent validator."""
    if not isinstance(value, Mapping) or set(value) != {"reviews", "comparisons", "selected_candidate_id", "selection_reason"}:
        raise PortfolioError("portfolio review schema mismatch")
    ids = [row.get("id") for row in candidates]
    if len(ids) != len(set(ids)) or any(not isinstance(x, str) for x in ids):
        raise PortfolioError("candidate IDs must be unique")
    if not isinstance(value["selection_reason"], str) or not value["selection_reason"].strip() or len(value["selection_reason"])>2400:
        raise PortfolioError("selection_reason is required")
    selected = value["selected_candidate_id"]
    if selected is not None and selected not in ids:
        raise PortfolioError("selected_candidate_id is unknown")
    exact_ids = {row["id"] for row in candidates if row.get("exact_duplicate_of") is not None}
    if selected in exact_ids:
        raise PortfolioError("exact duplicate cannot be selected")
    reviews = value["reviews"]
    if not isinstance(reviews, list) or len(reviews) != len(ids):
        raise PortfolioError("one review is required for every candidate")
    seen = set()
    for row in reviews:
        if not isinstance(row, Mapping) or set(row) != {"candidate_id", "review"} or row["candidate_id"] not in ids or row["candidate_id"] in seen:
            raise PortfolioError("reviews must cover each candidate exactly once")
        seen.add(row["candidate_id"])
        if review_validator is not None:
            review_validator(row["review"], next(c for c in candidates if c["id"] == row["candidate_id"]))
    comparisons = value["comparisons"]
    expected = {tuple(sorted((a, b))) for i, a in enumerate(ids) for b in ids[i + 1:]}
    actual = set()
    if not isinstance(comparisons, list) or len(comparisons) != len(expected):
        raise PortfolioError("one comparison is required for every candidate pair")
    for row in comparisons:
        pair = row.get("candidate_ids") if isinstance(row, Mapping) else None
        if not isinstance(pair, list) or len(pair) != 2 or pair[0] == pair[1] or any(x not in ids for x in pair):
            raise PortfolioError("invalid candidate pair")
        key = tuple(sorted(pair))
        if key in actual or set(row) != {"candidate_ids", "relationship", "reason"}:
            raise PortfolioError("candidate pairs must be unique and complete")
        if row["relationship"] not in _RELATIONSHIPS or not isinstance(row["reason"], str) or not row["reason"].strip() or len(row["reason"])>2400:
            raise PortfolioError("invalid comparison")
        actual.add(key)
    if actual != expected or seen != set(ids):
        raise PortfolioError("portfolio review is incomplete")
    return {"reviews": reviews, "comparisons": comparisons, "selected_candidate_id": selected,
            "selection_reason": value["selection_reason"], "novelty_established": False,
            "scope": "Review bookkeeping only; no automatic novelty, usefulness, or promotion decision."}
