"""Predeclared descriptive decisions for a fixed-artifact comparison.

This classifier is intentionally separate from the stopped Campaign 2 gate.
It describes one recorded artifact per arm and never claims a causal
representation effect, novelty, or source truth.  Rows are evaluated per
decoder; positives from different decoders are never pooled.
"""
from __future__ import annotations

from typing import Any, Mapping


EXPECTED_SLOTS = {
    "task-only": 6, "raw": 6, "atoms": 6, "distill": 6,
    "random": 6, "linear-control": 1,
}
SOURCE_ARMS = ("raw", "atoms", "distill")
CONTROL_ARMS = frozenset({"random", "linear-control", "control"})


def _one(rows: list[Mapping[str, Any]], slots: Mapping[str, int]) -> dict[str, Any]:
    by_call: dict[tuple[str, str], Mapping[str, Any]] = {}
    errors: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("arm"), str) or not isinstance(row.get("call_id"), str):
            errors.append("malformed report row")
            continue
        arm, call = row["arm"], row["call_id"]
        if arm not in slots:
            errors.append(f"unexpected arm {arm}")
            continue
        if call in seen or (arm, call) in by_call:
            errors.append(f"duplicate call {arm}/{call}")
            continue
        seen.add(call); by_call[(arm, call)] = row
    observed = {arm: sum(a == arm for a, _ in by_call) for arm in slots}
    missing = {arm: slots[arm] - observed[arm] for arm in slots if observed[arm] < slots[arm]}
    errors += [f"extra calls for {arm}" for arm in slots if observed[arm] > slots[arm]]
    incomplete: list[dict[str, Any]] = []
    positives: list[Mapping[str, Any]] = []
    for row in by_call.values():
        failed = row.get("failed_final_seeds")
        low = row.get("low_precision_control", row.get("low_precision"))
        reason = None
        if row.get("status") == "no_eligible_candidate": reason = "no_eligible_candidate"
        elif not isinstance(failed, list) or failed: reason = "failed_final_seed"
        elif low is not False: reason = "low_precision"
        elif not isinstance(row.get("program_sha256"), str) or not row["program_sha256"].strip(): reason = "missing_program_identity"
        elif type(row.get("better_than_control")) is not bool: reason = "missing_strong_control_comparison"
        if reason:
            incomplete.append({"arm": row.get("arm"), "call_id": row.get("call_id"), "reason": reason})
        elif row.get("better_than_control") is True:
            positives.append(row)
    distinct = {
        arm: len({r.get("program_sha256") for r in positives if r.get("arm") == arm and isinstance(r.get("program_sha256"), str)})
        for arm in slots
    }
    control_positive = any(r.get("arm") in CONTROL_ARMS for r in positives)
    task_n = distinct.get("task-only", 0)
    source = {arm: distinct.get(arm, 0) for arm in SOURCE_ARMS}
    complete = not errors and not missing and not incomplete
    decision, reason = "incomplete", "missing_or_invalid_fixed_artifact_call"
    if complete and control_positive:
        decision, reason = "no_representation_attribution", "control_call_beats_frozen_strong_control"
    elif complete and not positives:
        decision, reason = "close_fixed_artifact_case", "no_call_beats_frozen_strong_control"
    elif complete and task_n >= 3:
        decision, reason = "task_or_dsl_lead", "three_distinct_task_only_calls_positive"
    elif complete and task_n == 2:
        decision, reason = "attribution_blocked", "two_task_only_calls_positive"
    elif complete and source["atoms"] >= 3 and source["raw"] <= 1 and source["distill"] <= 1 and task_n <= 1:
        decision, reason = "atom_artifact_lead", "atoms_three_or_more_and_other_source_arms_and_task_at_most_one"
    elif complete and all(source[a] >= 3 for a in SOURCE_ARMS) and task_n <= 1:
        decision, reason = "source_artifact_lead", "all_source_arms_three_or_more_no_typed_structure_claim"
    elif complete and source["atoms"] <= 1 and (source["raw"] >= 3 or source["distill"] >= 3) and task_n <= 1:
        decision, reason = "against_this_atom_artifact", "raw_or_prose_leads_while_atoms_do_not"
    elif complete:
        decision, reason = "mixed_source_result", "mixed_source_positive_counts_no_predeclared_lead"
    return {
        "schema": "fixed-artifact-decision.v1", "status": "complete" if complete else "incomplete",
        "decision": decision, "reason": reason, "expected_slots": dict(slots),
        "observed_slots": observed, "missing_slots": missing, "malformed_rows": errors,
        "incomplete_rows": incomplete, "positive_call_count": len(positives),
        "distinct_positive_programs_by_arm": distinct, "source_positive_counts": source,
        "task_only_positive_count": task_n, "control_positive": control_positive,
        "interpretation": "descriptive fixed-artifact result; no novelty, source-truth, or causal representation claim",
        "auto_execute": False, "train_eligible": False, "novelty_established": False,
    }


def classify_fixed_artifact(reports: Any, *, expected_slots: Mapping[str, int] = EXPECTED_SLOTS, expected_decoders: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Classify rows independently for each decoder, never pooling decoders."""
    if not isinstance(reports, list):
        raise ValueError("reports must be a list")
    slots = {str(k): int(v) for k, v in expected_slots.items()}
    if any(v <= 0 for v in slots.values()):
        raise ValueError("expected slot counts must be positive")
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in reports:
        decoder = row.get("decoder", "default") if isinstance(row, Mapping) else "default"
        if not isinstance(decoder, str) or not decoder.strip():
            raise ValueError("decoder must be a nonempty string")
        groups.setdefault(decoder, []).append(row)
    if expected_decoders is not None:
        if not expected_decoders or len(set(expected_decoders))!=len(expected_decoders): raise ValueError("invalid expected decoders")
        if set(groups)-set(expected_decoders): raise ValueError("unexpected decoder")
        for decoder in expected_decoders: groups.setdefault(decoder,[])
    if not groups: groups['default']=[]
    per_decoder = {decoder: _one(rows, slots) for decoder, rows in sorted(groups.items())}
    if len(per_decoder) == 1:
        result = dict(next(iter(per_decoder.values())))
        result["decoder"] = next(iter(per_decoder))
        result["per_decoder"] = per_decoder
        return result
    return {"schema": "fixed-artifact-decision.v1", "status": "complete" if all(x["status"]=="complete" for x in per_decoder.values()) else "incomplete", "decision": "per_decoder",
            "cross_decoder_interpretation": ("incomplete_no_cross_decoder_comparison" if any(x["status"] != "complete" for x in per_decoder.values()) else "same_label_only_not_equal_results" if len({x["decision"] for x in per_decoder.values()})==1 else "decoder_dependent_no_representation_reading"),
            "reason": "decoder results are reported separately and never pooled", "per_decoder": per_decoder,
            "auto_execute": False, "train_eligible": False, "novelty_established": False}
