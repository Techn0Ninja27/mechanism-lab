"""Offline schemas and blind exports for the world-search pilot.

This module intentionally does not score novelty.  It makes arm labels and
generation metadata unavailable to a downstream judge, checks that an idea has
an executable mechanism schema, and reports only descriptive coverage and
exact-string duplicate rates.  Semantic originality requires independent
prior-art retrieval and human adjudication.
"""

from __future__ import annotations

import copy
import random
import re
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any, Iterable, Mapping


MECHANISM_FIELDS = (
    "title", "intervention", "mechanism", "assumptions", "baseline",
    "metric", "falsification", "implementation", "queries",
)
TEXT_FIELDS = ("title", "intervention", "mechanism", "baseline", "metric",
               "falsification", "implementation")


class WorldEvaluationError(ValueError):
    """Raised when a pilot record cannot be evaluated safely."""


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorldEvaluationError(f"{field} must be a non-empty string")
    return re.sub(r"\s+", " ", value).strip()


def _list(value: Any, field: str) -> list[str]:
    if isinstance(value, str) and value.strip():
        value = [value]
    if not isinstance(value, list) or not value or not all(isinstance(x, str) and x.strip() for x in value):
        raise WorldEvaluationError(f"{field} must be a non-empty list of strings")
    return [re.sub(r"\s+", " ", x).strip() for x in value]


def validate_mechanism(idea: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize the required mechanism record."""
    if not isinstance(idea, Mapping):
        raise WorldEvaluationError("idea must be an object")
    out: dict[str, Any] = {}
    for field in TEXT_FIELDS:
        out[field] = _text(idea.get(field), field)
    out["assumptions"] = _list(idea.get("assumptions"), "assumptions")
    out["queries"] = _list(idea.get("queries"), "queries")
    return out


def _idea_from_record(record: Mapping[str, Any]) -> Mapping[str, Any]:
    idea = record.get("idea", record.get("proposal"))
    if not isinstance(idea, Mapping):
        raise WorldEvaluationError("record needs an idea or proposal object")
    return idea


def validate_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a generated case while retaining arm metadata for analysis."""
    if not isinstance(record, Mapping):
        raise WorldEvaluationError("record must be an object")
    case_id = _text(record.get("id", record.get("case_id")), "id")
    arm = _text(record.get("arm"), "arm")
    problem = _text(record.get("problem"), "problem")
    idea = validate_mechanism(_idea_from_record(record))
    return {"id": case_id, "arm": arm, "problem": problem, "idea": idea}


def normalized_mechanism(idea: Mapping[str, Any]) -> str:
    """Return only the operative mechanism for exact duplicate diagnostics."""
    checked = validate_mechanism(idea)
    # Titles and implementation prose may differ while the mechanism is the same.
    return checked["mechanism"].casefold()


PRIMITIVE_GAPS = {"none", "possible", "clear", "unknown"}
LITERATURE_STATUS = {"covered", "incremental", "plausibly_distinct", "insufficient_evidence"}
UTILITY_STATUS = {"untested", "supported", "failed", "unknown"}
ATOM_ACTIONS = {"retain_existing", "propose_for_review", "reject", "unknown"}


def validate_primitive_triage(value: Mapping[str, Any], evidence_ids: Iterable[str] = ()) -> dict[str, Any]:
    """Validate separate representation, literature, and utility claims.

    A missing atom is only a representation observation. It is never evidence
    of literature novelty or usefulness, and no action admits a new atom.
    """
    if not isinstance(value, Mapping):
        raise WorldEvaluationError("primitive triage must be an object")
    gap, literature, utility = (value.get(k) for k in ("representation_gap", "literature_novelty", "empirical_utility"))
    action = value.get("atom_action", "unknown")
    if gap not in PRIMITIVE_GAPS:
        raise WorldEvaluationError(f"representation_gap must be one of {sorted(PRIMITIVE_GAPS)}")
    if literature not in LITERATURE_STATUS:
        raise WorldEvaluationError(f"literature_novelty must be one of {sorted(LITERATURE_STATUS)}")
    if utility not in UTILITY_STATUS:
        raise WorldEvaluationError(f"empirical_utility must be one of {sorted(UTILITY_STATUS)}")
    if action not in ATOM_ACTIONS:
        raise WorldEvaluationError(f"atom_action must be one of {sorted(ATOM_ACTIONS)}")
    cited = value.get("evidence_source_ids", [])
    if not isinstance(cited, list) or not all(isinstance(x, str) and x.strip() for x in cited):
        raise WorldEvaluationError("evidence_source_ids must be a list of strings")
    unknown = set(cited) - {str(x) for x in evidence_ids}
    if unknown:
        raise WorldEvaluationError(f"unknown evidence source IDs: {sorted(unknown)}")
    if value.get("human_adjudicated", False) is not False:
        raise WorldEvaluationError("automated triage cannot mark human_adjudicated true")
    return {"representation_gap": gap, "literature_novelty": literature,
            "empirical_utility": utility, "atom_action": action,
            "evidence_source_ids": list(cited), "human_adjudicated": False,
            "train_eligible": False}


def blind_cases(records: Iterable[Mapping[str, Any]], seed: int = 0) -> list[dict[str, Any]]:
    """Shuffle cases and remove arm/source/generation metadata for blind review.

    The returned ``blind_id`` is a stable opaque index within this export.  The
    mapping is not included in the export; keep it separately for analysis.
    """
    result, _ = blind_cases_with_mapping(records, seed=seed)
    return result


def blind_cases_with_mapping(records: Iterable[Mapping[str, Any]], seed: int = 0) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Return a blind export and a separate blind-id to source-id mapping.

    The mapping is deliberately returned separately so it can be retained for
    paired analysis without being exposed to a reviewer.
    """
    checked = [validate_record(record) for record in records]
    order = list(range(len(checked)))
    random.Random(seed).shuffle(order)
    result = []
    mapping: dict[str, str] = {}
    for blind_index, source_index in enumerate(order):
        case = checked[source_index]
        blind_id = f"case-{blind_index:04d}"
        mapping[blind_id] = case["id"]
        result.append({
            "blind_id": blind_id,
            "problem": case["problem"],
            "idea": copy.deepcopy(case["idea"]),
        })
    return result, mapping


def summarize(records: Iterable[Mapping[str, Any]], reviews: Iterable[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Report schema coverage and exact duplicates, without novelty claims."""
    raw = list(records)
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing: list[dict[str, str]] = []
    for index, record in enumerate(raw):
        try:
            checked = validate_record(record)
        except WorldEvaluationError as exc:
            arm = str(record.get("arm", "unknown")) if isinstance(record, Mapping) else "unknown"
            missing.append({"index": str(index), "arm": arm, "error": str(exc)})
            continue
        by_arm[checked["arm"]].append(checked)
    arms = {}
    arm_signatures: dict[str, set[str]] = {}
    pooled_signatures: list[str] = []
    for arm, cases in sorted(by_arm.items()):
        signatures = [normalized_mechanism(case["idea"]) for case in cases]
        counts = Counter(signatures)
        arm_signatures[arm] = set(signatures)
        pooled_signatures.extend(signatures)
        duplicate_cases = sum(n - 1 for n in counts.values() if n > 1)
        arms[arm] = {
            "records": len(cases),
            "unique_exact_mechanisms": len(counts),
            "exact_duplicate_cases": duplicate_cases,
            "exact_duplicate_rate": duplicate_cases / len(cases) if cases else 0.0,
        }
    pooled_counts = Counter(pooled_signatures)
    pooled_duplicate_cases = sum(n - 1 for n in pooled_counts.values() if n > 1)
    cross_arm: dict[str, dict[str, int]] = {}
    for left, right in combinations(sorted(arm_signatures), 2):
        shared = arm_signatures[left] & arm_signatures[right]
        cross_arm[f"{left}__{right}"] = {
            "shared_unique_mechanisms": len(shared),
            "left_cases": sum(1 for sig in [normalized_mechanism(c["idea"]) for c in by_arm[left]] if sig in shared),
            "right_cases": sum(1 for sig in [normalized_mechanism(c["idea"]) for c in by_arm[right]] if sig in shared),
        }
    review_status = Counter()
    for review in reviews or []:
        if isinstance(review, Mapping):
            review_status[str(review.get("decision", "unread"))] += 1
    return {
        "records": len(raw),
        "valid_records": sum(len(x) for x in by_arm.values()),
        "missing_or_invalid": missing,
        "arms": arms,
        "pooled": {
            "records": len(pooled_signatures),
            "unique_exact_mechanisms": len(pooled_counts),
            "exact_duplicate_cases": pooled_duplicate_cases,
            "exact_duplicate_rate": pooled_duplicate_cases / len(pooled_signatures) if pooled_signatures else 0.0,
        },
        "cross_arm_exact_overlap": cross_arm,
        "review_decisions": dict(sorted(review_status.items())),
        "interpretation": "Exact duplicate rates are descriptive string diagnostics, not semantic novelty or external prior-art evidence.",
    }

