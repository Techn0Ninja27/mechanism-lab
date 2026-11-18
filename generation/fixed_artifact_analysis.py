"""Pure analysis helpers for the fixed-artifact comparison.

This module deliberately does not execute candidates, read the data set, or
write a budget ledger.  It adapts the 192-slot prospective inventory to the
old campaign inventory interface and annotates already validated final
reports before passing them to :mod:`fixed_artifact_decisions`.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Mapping, Sequence

from generation.fixed_artifact_decisions import EXPECTED_SLOTS, classify_fixed_artifact
from generation.campaign_selection import canonical_hash


MODEL_ARMS = ("task-only", "raw", "atoms", "distill")
DECODERS = ("opus-5.5", "fable-5.2")
MODEL_SLOTS = 192
SHARED_CONTROL_SLOTS = 28


def _rows(value: Any, key: str = "candidates") -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        value = value.get(key)
    if not isinstance(value, list) or any(not isinstance(x, Mapping) for x in value):
        raise ValueError(f"{key} must be a list of objects")
    return [dict(x) for x in value]


def _validate_ids(rows: Iterable[Mapping[str, Any]], label: str) -> None:
    ids: set[str] = set()
    for row in rows:
        cid = row.get("id")
        if not isinstance(cid, str) or not cid:
            raise ValueError(f"{label} row has malformed id")
        if cid in ids:
            raise ValueError(f"duplicate {label} id: {cid}")
        ids.add(cid)


def _control_rows(sharedcontrols: Any, model_ids: set[str] = set()) -> list[dict[str, Any]]:
    rows = _rows(sharedcontrols, "candidates") if isinstance(sharedcontrols, Mapping) else _rows(sharedcontrols)
    _validate_ids(rows, "shared control")
    for row in rows:
        if row.get("id") in model_ids:
            raise ValueError(f"control/model ID collision: {row['id']}")
        if row.get("decoder") not in (None, ""):
            raise ValueError("shared controls must not carry a decoder")
        if row.get("arm") == "linear_control":
            row["arm"] = "linear-control"
        if row.get("arm") not in {"random", "linear-control", "control"}:
            # Accept the historical control labels while normalizing only the
            # label consumed by the classifier.
            if row.get("control_kind") == "random":
                row["arm"] = "random"
            elif row.get("control_kind") == "linear":
                row["arm"] = "linear-control"
            else:
                raise ValueError(f"shared control has unsupported arm: {row.get('arm')}")
        if not isinstance(row.get("call_id"), str) or not row["call_id"] or row.get("status","valid") != "valid":
            raise ValueError("control call/status malformed")
        if not isinstance(row.get("program"), Mapping) or row.get("program_sha256") != canonical_hash(row["program"]):
            raise ValueError(f"shared control program hash mismatch: {row['id']}")
    random_calls = {r.get("call_id") for r in rows if r.get("arm") == "random"}
    if len(random_calls) != 6 or any(sum(r.get("call_id") == c for r in rows if r.get("arm") == "random") != 4 for c in random_calls):
        raise ValueError("random controls must contain six groups of four")
    if sum(r.get("arm") == "linear-control" for r in rows) != 4 or len({r["call_id"] for r in rows if r["arm"] == "linear-control"}) != 1:
        raise ValueError("linear controls must contain four rows")
    return rows


def split_inventory(overall: Any, sharedcontrols: Any, *, expected_cases: Sequence[Mapping[str, Any]], decoders: Sequence[str] = DECODERS) -> dict[str, dict[str, Any]]:
    """Split the prospective inventory into one old-compatible inventory per decoder.

    ``overall`` contains exactly 192 model candidates (48 calls x four slots),
    while ``sharedcontrols`` contains the independent 24 random and four fixed
    controls.  Controls are copied into both outputs and retain their original
    program hashes.  The complete model schedule is required: missing slots
    are represented by invalid/unattempted rows, never removed from a
    denominator.
    """
    models = _rows(overall)
    cases = [dict(x) for x in expected_cases]
    if len(cases) != 48 or any(not isinstance(x, Mapping) for x in expected_cases):
        raise ValueError("expected_cases must contain exactly 48 cases")
    if len({x.get("id") for x in cases}) != 48 or any(not isinstance(x.get("id"), str) for x in cases):
        raise ValueError("expected cases must have unique IDs")
    if any(sum(x.get("decoder") == d and x.get("arm") == a for x in cases) != 6
           for d in decoders for a in MODEL_ARMS):
        raise ValueError("expected cases must contain six calls per decoder/arm")
    if len(models) != MODEL_SLOTS:
        raise ValueError("overall inventory must contain exactly 192 model slots")
    if not decoders or len(set(decoders)) != len(decoders):
        raise ValueError("decoders must be nonempty and unique")
    _validate_ids(models, "model")
    model_ids = {x["id"] for x in models}
    controls = _control_rows(sharedcontrols, model_ids)
    if len(controls) != SHARED_CONTROL_SLOTS:
        raise ValueError("shared controls must contain exactly 28 rows")
    for row in models:
        if row.get("decoder") not in decoders:
            raise ValueError("model row has unknown decoder")
        if row.get("arm") not in MODEL_ARMS:
            raise ValueError(f"model row has unsupported arm: {row.get('arm')}")
        case = next((x for x in cases if x["id"] == row.get("call_id")), None)
        if case is None or case.get("decoder") != row.get("decoder") or case.get("arm") != row.get("arm"):
            raise ValueError(f"model row does not match expected case: {row.get('id')}")
        if row.get("id") not in {f"{case['id']}-cand-{i:02d}" for i in range(4)}:
            raise ValueError(f"model row has unexpected candidate slot ID: {row.get('id')}")
        if row.get("status", "valid") == "valid" and (not isinstance(row.get("program"), Mapping) or row.get("program_sha256") != canonical_hash(row["program"])):
            raise ValueError("model program/hash invalid")
        if row.get("status", "valid") not in {"valid", "invalid"}:
            raise ValueError(f"model row has malformed status: {row.get('id')}")
    for case in cases:
        if sum(row.get("call_id") == case["id"] for row in models) != 4:
            raise ValueError(f"expected four slots for case {case['id']}")
    counts = {d: sum(r.get("decoder") == d for r in models) for d in decoders}
    if any(n != 96 for n in counts.values()):
        raise ValueError("each decoder must have 96 model candidates")
    result: dict[str, dict[str, Any]] = {}
    for decoder in decoders:
        selected = [deepcopy(r) for r in models if r.get("decoder") == decoder]
        # The old runner does not know about decoders; do not leak this field
        # into its candidate identity or alter the shared control hashes.
        for row in selected:
            row.pop("decoder", None)
        candidates = selected + [deepcopy(r) for r in controls]
        result[decoder] = {"schema": "causal-candidate-inventory.v1",
                           "campaign": "fixed-artifact-comparison-v1",
                           "decoder": decoder, "candidates": candidates,
                           "model_candidate_count": 96,
                           "shared_control_count": len(controls),
                           "planned_model_slots": 96,
                           "train_eligible": False, "novelty_established": False}
    return result


def _fidelity_annotation(row: Mapping[str, Any], fidelity: Mapping[str, Any] | None) -> dict[str, Any]:
    arm = row.get("arm")
    if arm not in {"atoms", "distill", "prose"}:
        return {}
    if not isinstance(fidelity, Mapping):
        raise ValueError("positive prepared-arm pick requires fidelity measurement")
    key = "distill" if arm == "prose" else arm
    value = fidelity.get(key, fidelity.get("prose"))
    if not isinstance(value, Mapping):
        raise ValueError(f"missing fidelity table for positive {arm} row")
    fields = value.get("fields", [])
    if not isinstance(fields, list):
        raise ValueError("fidelity fields must be a list")
    flagged = [deepcopy(f) for f in fields if isinstance(f, Mapping) and f.get("flagged_by_either")]
    summary = value.get("summary") if isinstance(value.get("summary"), Mapping) else {}
    coverage = {"valid_reviewer_slots": summary.get("valid_reviewer_slots"),
                "total_reviewer_slots": summary.get("total_reviewer_slots")}
    return {"fidelity_flags": flagged,
            "fidelity_flagged_by_both_count": summary.get("flagged_by_both"),
            "fidelity_measurement_coverage": coverage,
            "fidelity_representation_sha256": value.get("representation_sha256"),
            "fidelity": deepcopy(dict(value))}


def summarize_fixed_artifact(per_decoder_reports: Any, *, selections: Any = None,
                            inventories: Any = None, sharedcontrols: Any = None,
                            fidelity: Mapping[str, Any] | None = None,
                            expected_decoders: Sequence[str] = DECODERS,
                            expected_slots: Mapping[str, int] = EXPECTED_SLOTS) -> dict[str, Any]:
    """Annotate validated final rows and classify each decoder independently.

    ``per_decoder_reports`` may be a decoder->rows mapping or a flat list.
    Optional selections/inventories are used to add explicit no-eligible rows
    via the same semantics as ``_campaign2_report_rows``.  A missing decoder
    remains an empty expected group, so it is reported as incomplete rather
    than silently disappearing.
    """
    expected = tuple(expected_decoders)
    if isinstance(per_decoder_reports, Mapping):
        grouped = {}
        for k, value in per_decoder_reports.items():
            if isinstance(value, Mapping) and isinstance(value.get("reports"), list):
                value = value["reports"]
            if not isinstance(value, list):
                raise ValueError("decoder report must be a list or report object")
            grouped[str(k)] = [dict(r) for r in value]
    elif isinstance(per_decoder_reports, list):
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in per_decoder_reports:
            if not isinstance(row, Mapping):
                raise ValueError("report row must be an object")
            decoder = row.get("decoder")
            if not isinstance(decoder, str) or not decoder:
                raise ValueError("flat report rows require decoder")
            grouped.setdefault(decoder, []).append(dict(row))
    else:
        raise ValueError("per_decoder_reports must be a mapping or list")
    if set(grouped) - set(expected):
        raise ValueError("unexpected decoder in reports")

    # Add no-eligible calls from each decoder's selection/inventory without
    # changing any existing validated final-report row.
    for decoder in expected:
        rows = grouped.setdefault(decoder, [])
        selection = selections.get(decoder) if isinstance(selections, Mapping) else None
        inventory = inventories.get(decoder) if isinstance(inventories, Mapping) else None
        if isinstance(selection, Mapping) and isinstance(inventory, Mapping):
            inv_by_call = {x.get("call_id"): x for x in inventory.get("candidates", []) if isinstance(x, Mapping)}
            present = {x.get("call_id") for x in rows}
            manifest = selection.get("manifest", selection).get("selected", []) if isinstance(selection.get("manifest", selection), Mapping) else []
            for item in manifest:
                if isinstance(item, Mapping) and item.get("status") == "no_eligible_candidate" and item.get("call_id") not in present:
                    call = item.get("call_id"); candidate = inv_by_call.get(call, {})
                    rows.append({"call_id": call, "arm": candidate.get("arm"), "decoder": decoder,
                                 "status": "no_eligible_candidate", "program_sha256": None,
                                 "failed_final_seeds": None, "low_precision_control": True,
                                 "better_than_control": None})
        for row in rows:
            if row.get("decoder") not in (None, decoder): raise ValueError("report decoder disagrees with group")
            row["decoder"] = decoder
            positive = row.get("better_than_control") is True
            if positive:
                row.update(_fidelity_annotation(row, fidelity))
    decision = classify_fixed_artifact(sum((grouped[d] for d in expected), []),
                                       expected_slots=expected_slots, expected_decoders=expected)
    decision["reports"] = {d: grouped[d] for d in expected}
    decision["missing_decoders"] = [d for d in expected if not grouped[d]]
    return decision


# Descriptive aliases make the helper easy to discover without multiplying
# implementations.
analyze_final_reports = summarize_fixed_artifact
final_summary = summarize_fixed_artifact
