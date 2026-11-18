"""Descriptive stopping rules for the second matched-representation campaign.

This module consumes final-report rows only.  It does not compare model arms
statistically and cannot establish a representation effect or novelty.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

EXPECTED_SLOTS = {"task-only": 6, "raw": 6, "atoms": 6, "distill": 6, "random": 6, "linear-control": 1}
SOURCE_ARMS = frozenset({"raw", "atoms", "distill"})
CONTROL_ARMS = frozenset({"random", "linear-control", "control"})


def classify_campaign(reports: Any, *, expected_slots: Mapping[str, int] = EXPECTED_SLOTS) -> dict[str, Any]:
    if not isinstance(reports, list):
        raise ValueError("reports must be a list")
    slots = {str(k): int(v) for k, v in expected_slots.items()}
    if any(v <= 0 for v in slots.values()):
        raise ValueError("expected slot counts must be positive")
    by_call: dict[tuple[str, str], Mapping[str, Any]] = {}
    errors: list[str] = []
    seen_call_ids: set[str] = set()
    for row in reports:
        if not isinstance(row, Mapping) or not isinstance(row.get("call_id"), str) or not isinstance(row.get("arm"), str):
            errors.append("malformed report row"); continue
        if row["arm"] not in slots:
            errors.append(f"unexpected arm {row['arm']}"); continue
        if row["call_id"] in seen_call_ids:
            errors.append(f"duplicate final report call_id {row['call_id']}"); continue
        seen_call_ids.add(row["call_id"])
        key = (row["arm"], row["call_id"])
        if key in by_call:
            errors.append(f"duplicate final report call {key[0]}/{key[1]}"); continue
        by_call[key] = row
    counts = {arm: len({call for a, call in by_call if a == arm}) for arm in slots}
    missing = {arm: slots[arm] - counts.get(arm, 0) for arm in slots if counts.get(arm, 0) < slots[arm]}
    incomplete_rows = []
    positive: list[Mapping[str, Any]] = []
    for row in by_call.values():
        failed = row.get("failed_final_seeds")
        low = row.get("low_precision_control", row.get("low_precision"))
        if row.get("status") == "no_eligible_candidate" or not isinstance(failed, list) or failed or low is not False:
            incomplete_rows.append({"arm": row.get("arm"), "call_id": row.get("call_id"), "reason": "failure_or_low_precision"})
            continue
        if row.get("better_than_control") is True:
            positive.append(row)
        elif row.get("better_than_control") is not False:
            incomplete_rows.append({"arm": row.get("arm"), "call_id": row.get("call_id"), "reason": "missing_strong_control_comparison"})
    distinct_by_arm = {}
    for arm in slots:
        hashes = {r.get("program_sha256") for r in positive if r.get("arm") == arm and isinstance(r.get("program_sha256"), str)}
        distinct_by_arm[arm] = len(hashes)
    source_max = max((distinct_by_arm.get(a, 0) for a in SOURCE_ARMS), default=0)
    task_n = distinct_by_arm.get("task-only", 0)
    control_positive = any(r.get("arm") in CONTROL_ARMS for r in positive)
    complete = not errors and not missing and not incomplete_rows
    reason = "incomplete_campaign"
    decision = "incomplete"
    if complete and not positive:
        decision, reason = "close_benchmark", "all_calls_complete_precise_and_no_call_beats_strong_control"
    elif complete and control_positive:
        decision, reason = "no_representation_attribution", "random_or_control_call_beats_strong_control"
    elif complete and source_max >= 3 and task_n <= 1:
        decision, reason = "replication_candidate", "one_source_arm_has_three_distinct_positive_programs"
    elif complete and task_n >= 3:
        decision, reason = "task_or_dsl_lead", "three_distinct_task_only_programs_beat_strong_control"
    elif complete and task_n == 2:
        decision, reason = "attribution_blocked", "two_task_only_programs_are_insufficient_for_dsl_lead"
    elif complete:
        decision, reason = "no_escalation", "fewer_than_three_distinct_source_programs_and_no_task_lead"
    return {"schema": "campaign2-decision.v1", "status": "complete" if complete else "incomplete",
            "decision": decision, "reason": reason, "expected_slots": slots, "observed_slots": counts,
            "missing_slots": missing, "malformed_rows": errors, "incomplete_rows": incomplete_rows,
            "positive_call_count": len(positive), "distinct_positive_programs_by_arm": distinct_by_arm,
            "source_positive_max": source_max, "task_only_positive_count": task_n,
            "auto_execute": False, "train_eligible": False, "novelty_established": False}
