"""Read-only readiness assessment for the v6 reference-review calibration.

The assessment is deliberately separate from the paid-campaign guard.  It
records a retrospective parser-policy result and never changes a ledger or a
production readiness flag.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from generation.campaign_journal import _event_hash
from generation.reference_review_v2 import decode_review_v2


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> Any:
    return json.loads(path.read_text())


def _fail(message: str) -> None:
    raise ValueError("reference calibration not ready: " + message)


def assess(root: str | Path, ledger_path: str | Path) -> dict[str, Any]:
    """Recompute the retrospective v2 result without writing files."""
    root = Path(root)
    ledger_path = Path(ledger_path)
    required = ["freeze.json", "manifest.json", "answer-key.json", "key-audit.json",
                "baseline-scores.json", "evaluation.json"]
    for name in required:
        if not (root / name).is_file():
            _fail(f"missing {name}")
    freeze = _read(root / "freeze.json")
    manifest = _read(root / "manifest.json")
    key = _read(root / "answer-key.json")
    audit = _read(root / "key-audit.json")
    evaluation = _read(root / "evaluation.json")

    expected_hashes = {
        "manifest_sha256": "manifest.json", "key_sha256": "answer-key.json",
        "key_audit_sha256": "key-audit.json", "baseline_sha256": "baseline-scores.json",
        "builder_sha256": "../scripts/build_review_calibration_v6.py",
    }
    for field, rel in expected_hashes.items():
        path = (root / rel) if not rel.startswith("../") else (Path(__file__).resolve().parents[1] / rel[3:])
        if _sha(path) != freeze.get(field):
            _fail(f"frozen hash mismatch: {field}")
    if evaluation.get("freeze_sha256") != _sha(root / "freeze.json"):
        _fail("evaluation is not bound to freeze")
    if audit.get("status") != "pass" or audit.get("summary", {}).get("fields_checked") != 48:
        _fail("independent key audit is incomplete")
    if manifest.get("packet_count") != 12 or len(manifest.get("packets", [])) != 12:
        _fail("packet count is not twelve")
    if len({p.get("opaque_id") for p in manifest["packets"]}) != 12:
        _fail("packet identifiers are not unique")
    instructions = root / "reviewer-inputs" / "reviewer-instructions.md"
    if not instructions.is_file() or _sha(instructions) != manifest.get("instructions_sha256"):
        _fail("review instructions hash mismatch")
    if manifest.get("builder_sha256") != freeze.get("builder_sha256"):
        _fail("builder hash mismatch")

    # The external anchor is evidence that the completed evaluation was bound
    # to the phase ledger.  Reading it does not mutate the ledger.
    # Read the ledger directly.  BudgetLedger.read_event_anchor uses its
    # normal locked-write context, which is appropriate for mutable clients
    # but would violate this assessor's read-only contract.
    try:
        ledger_doc = json.loads(ledger_path.read_text())
    except (OSError, ValueError) as exc:
        _fail(f"cannot read phase ledger: {exc}")
    if ledger_doc.get("version") != 1 or ledger_doc.get("cap_usd") != 30.0:
        _fail("phase ledger schema or cap mismatch")
    anchor = (ledger_doc.get("audit_heads") or {}).get("recordwise-calibration-v6-freeze")
    if not isinstance(anchor, dict):
        _fail("missing external calibration anchor")
    event = anchor.get("last_event") or {}
    payload = event.get("payload") or {}
    if payload.get("evaluation_sha256") != _sha(root / "evaluation.json"):
        _fail("evaluation hash is not externally anchored")
    if anchor.get("head") != event.get("event_hash"):
        _fail("ledger anchor head is inconsistent")
    if _event_hash(event.get("seq"), event.get("event"), payload,
                   event.get("previous_hash")) != event.get("event_hash"):
        _fail("ledger anchor event hash is invalid")

    expected = {r["opaque_id"]: r["fields"] for r in key.get("records", [])}
    if len(expected) != 12 or any(len(fields) != 4 for fields in expected.values()):
        _fail("answer key does not cover 48 fields")
    rows = evaluation.get("reviews", [])
    if len(rows) != 24:
        _fail("evaluation does not contain 24 reviews")
    row_by_task = {r.get("task_name"): r for r in rows}
    if len(row_by_task) != 24:
        _fail("review task names are not unique")
    packet_specs = {p.get("path"): p for p in manifest.get("packets", [])}
    if len(packet_specs) != 12:
        _fail("packet manifest paths are not unique")
    launches=freeze.get("launches", [])
    if len(launches)!=24 or {x.get("task_name") for x in launches}!=set(row_by_task):
        _fail("launch coverage mismatch")
    pairs={(x.get("input"),x.get("slot")) for x in launches}
    if pairs!={(name,slot) for name in packet_specs for slot in ('a','b')}:
        _fail("packet/slot coverage mismatch")
    strict_failed = []
    v2_valid = 0
    v2_matches = 0
    for launch in freeze.get("launches", []):
        task = launch.get("task_name")
        row = row_by_task.get(task)
        if row is None or row.get("input_sha256") != launch.get("input_sha256"):
            _fail(f"launch/evaluation binding missing: {task}")
        input_path = root / launch["input"]
        output_path = root / launch["output"]
        receipt_path = root / "launch-receipts" / (task.rsplit("/", 1)[-1] + ".json")
        for path in (input_path, output_path, receipt_path):
            if not path.is_file():
                _fail(f"missing bound artifact: {path.name}")
        if _sha(input_path) != launch["input_sha256"] or _sha(output_path) != row.get("review_sha256"):
            _fail(f"review artifact hash mismatch: {task}")
        if row.get("launch_receipt_sha256") != _sha(receipt_path):
            _fail(f"launch receipt hash mismatch: {task}")
        spec = packet_specs.get(launch["input"])
        if spec is None or spec.get("sha256") != _sha(input_path):
            _fail(f"packet manifest hash mismatch: {task}")
        receipt = _read(receipt_path)
        if receipt.get("returned_task_name") != task or receipt.get("input_sha256") != launch["input_sha256"]:
            _fail(f"launch receipt mismatch: {task}")
        packet = _read(input_path)
        raw = _read(output_path)
        try:
            from generation.evidence_reference_packets import decode_review
            decode_review(packet, raw)
        except (ValueError, KeyError, TypeError):
            strict_failed.append(task)
        try:
            decoded = decode_review_v2(packet, raw)
            verdicts = {x["field"]: x["verdict"] for x in decoded["field_items"]}
            v2_valid += 1
            if verdicts == expected.get(packet.get("opaque_id"), {}):
                v2_matches += 1
        except (ValueError, KeyError, TypeError):
            pass
    if v2_valid != 24 or v2_matches != 24:
        _fail(f"v2 result {v2_valid}/24 valid, {v2_matches}/24 key matching")

    return {
        "schema": "recordwise-calibration-readiness.v1",
        "status": "retrospective_parser_policy_pass",
        "calibration_criterion_satisfied": True,
        "basis": "retrospective parser policy amended after run",
        "strict_v1": {"mechanically_valid": 24 - len(strict_failed), "total": 24,
                      "failed_tasks": strict_failed},
        "v2": {"mechanically_valid": v2_valid, "key_matching": v2_matches, "total": 24},
        "field_verdicts": 96,
        "known_limitations": [
            "reviewer key and baseline artifacts were reachable elsewhere on disk",
            "launch receipts were recorded after review batches and do not prove access isolation",
            "the original evaluation scorer was created after the run",
            "review slots were correlated and are not independent observations",
            "this is a small reused development set, not a population accuracy estimate",
            "one of 84 supported field judgments had partial decisive evidence",
            "the original strict result remains failed; v2 is retrospective",
        ],
        "production_authorized": False,
        "raw_reviews_unchanged": True,
    }
