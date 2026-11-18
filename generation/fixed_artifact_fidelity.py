"""Read-only fidelity table builder for fixed prepared artifacts.

This module only reads the two frozen reviewer slots.  It deliberately does
not adjudicate flags, infer truth, or make a gate/selection decision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from generation.reference_review_v2 import decode_review_v2


SLOTS = ("a", "b")


def _sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _load(path: Path) -> tuple[Any | None, str | None]:
    """Return a JSON value and a stable error label; never repair input."""
    if not path.is_file():
        return None, "missing"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return None, f"malformed:{type(exc).__name__}"


def _packet_signature(packet: Mapping[str, Any]) -> tuple[Any, ...]:
    meta = packet.get("record_metadata")
    if not isinstance(meta, Mapping):
        raise ValueError("packet record_metadata malformed")
    fields = packet.get("fields")
    if not isinstance(fields, list):
        raise ValueError("packet fields malformed")
    desc = []
    field_ids = set()
    for field in fields:
        if not isinstance(field, Mapping) or not isinstance(field.get("field_id"), str):
            raise ValueError("packet field malformed")
        if field["field_id"] in field_ids:
            raise ValueError("duplicate packet field_id")
        field_ids.add(field["field_id"])
        desc.append((field.get("field_id"), field.get("field"), field.get("value"), bool(field.get("is_unknown"))))
    evidence = packet.get("evidence")
    if not isinstance(evidence, list):
        raise ValueError("packet evidence malformed")
    return (
        packet.get("schema"),
        packet.get("opaque_id"),
        meta.get("record_id"),
        meta.get("source_id"),
        meta.get("source_pack_digest"),
        json.dumps(dict(meta), ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        tuple(desc),
        json.dumps(evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    )


def _field_type(name: Any, is_unknown: bool) -> str:
    if is_unknown:
        return "unknown"
    if isinstance(name, str) and "[" in name:
        return name.split("[", 1)[0]
    return "statement" if name == "statement" else "field"


def build_report(reference_slots: str | Path, expected_indices: list[str] | tuple[str, ...] | None = None) -> dict[str, Any]:
    """Build a descriptive fidelity report from ``reference-slots``.

    ``reference_slots`` contains ``a/input-XX.json`` and ``b/input-XX.json``.
    Missing artifacts have null hashes; malformed artifacts retain the hash of
    their raw bytes and a malformed status. They are never silently replaced.
    A disagreement between two usable packets is a
    structural error because the reviewers did not review the same artifact.
    """
    root = Path(reference_slots)
    if not root.is_dir():
        raise ValueError(f"reference-slots directory not found: {root}")
    discovered = {p.stem.split("-", 1)[1] for slot in SLOTS for p in (root / slot).glob("input-*.json") if "-" in p.stem}
    discovered |= {p.stem.split("-", 1)[1] for slot in SLOTS for p in (root / slot).glob("review-*.json") if "-" in p.stem}
    indices = list(expected_indices) if expected_indices is not None else [f"{n:02d}" for n in range(12)]
    if not indices or any(not isinstance(x, str) or not x for x in indices) or len(set(indices)) != len(indices):
        raise ValueError("expected indices must be unique nonempty strings")
    if discovered - set(indices):
        raise ValueError("unexpected record index")
    indices = sorted(indices)
    records: list[dict[str, Any]] = []
    all_fields: dict[str, dict[str, Any]] = {}
    for index in indices:
        inputs: dict[str, Any] = {}
        reviews: dict[str, Any] = {}
        packets: dict[str, Mapping[str, Any]] = {}
        signatures = []
        for slot in SLOTS:
            input_path = root / slot / f"input-{index}.json"
            review_path = root / slot / f"review-{index}.json"
            input_value, input_error = _load(input_path)
            review_value, review_error = _load(review_path)
            inputs[slot] = {"path": str(input_path), "sha256": _sha(input_path), "status": input_error or "present"}
            reviews[slot] = {"path": str(review_path), "sha256": _sha(review_path), "status": review_error or "present"}
            if input_error is None and isinstance(input_value, Mapping):
                try:
                    signature = _packet_signature(input_value)
                except ValueError as exc:
                    inputs[slot]["status"] = f"malformed:{exc}"
                else:
                    packets[slot] = input_value
                    signatures.append((slot, signature))
            elif input_error is None:
                inputs[slot]["status"] = "malformed:TypeError"
        if len(signatures) == 2 and signatures[0][1] != signatures[1][1]:
            raise ValueError(f"structural packet mismatch at input-{index}")
        packet = next(iter(packets.values()), None)
        field_defs = packet.get("fields", []) if packet is not None else []
        if not isinstance(field_defs, list):
            raise ValueError(f"malformed fields at input-{index}")
        record_id = packet.get("opaque_id") if isinstance(packet, Mapping) else index
        if any(existing["record_id"] == record_id for existing in records):
            raise ValueError(f"duplicate record id: {record_id}")
        field_rows = []
        decoded: dict[str, dict[str, Any]] = {}
        for slot in SLOTS:
            if reviews[slot]["status"] != "present":
                continue
            if slot not in packets:
                reviews[slot]["status"] = "missing_measurement:input_unavailable"
                continue
            try:
                raw = json.loads((root / slot / f"review-{index}.json").read_text(encoding="utf-8"))
                out = decode_review_v2(packets[slot], raw)
                decoded[slot] = {row["field"]: row for row in out["field_items"]}
            except Exception as exc:
                # A malformed review is one missing measurement; never salvage fields.
                reviews[slot]["status"] = f"malformed:{type(exc).__name__}: {exc}"
        for field in field_defs:
            field_id, name = field.get("field_id"), field.get("field")
            key = f"{record_id}:{field_id}"
            flags = [slot for slot in SLOTS if slot in decoded and decoded[slot].get(name, {}).get("verdict") != "supported"]
            row = {"record_id": record_id, "field_id": field_id, "field": name,
                   "type": _field_type(name, bool(field.get("is_unknown"))),
                   "is_unknown": bool(field.get("is_unknown")),
                   "valid_reviewer_slots": len(decoded), "flagged_by_slots": flags,
                   "flagged_by_either": bool(flags), "flagged_by_both": len(flags) == len(SLOTS)}
            field_rows.append(row)
            all_fields[key] = row
        records.append({"record_id": record_id, "index": index, "inputs": inputs, "reviews": reviews, "fields": field_rows,
                        "valid_reviewer_slots": len(decoded), "missing_or_malformed_reviews": [s for s in SLOTS if reviews[s]["status"] != "present"]})
    flagged = [r for r in all_fields.values() if r["flagged_by_either"]]
    unknown = [r for r in all_fields.values() if r["is_unknown"]]
    return {"schema": "fixed-artifact-fidelity-table.v1", "status": "descriptive", "records": records,
            "fields": list(all_fields.values()),
            "summary": {"record_count": len(records), "field_count": len(all_fields),
                        "valid_reviewer_slots": sum(r["valid_reviewer_slots"] for r in records),
                        "total_reviewer_slots": len(records) * len(SLOTS),
                        "missing_or_malformed_review_slots": sum(len(r["missing_or_malformed_reviews"]) for r in records),
                        "flagged_by_either": len(flagged), "flagged_by_both": sum(r["flagged_by_both"] for r in flagged),
                        "unknown_field_count": len(unknown),
                        "unknown_flagged_by_either": sum(r["flagged_by_either"] for r in unknown),
                        "unknown_flagged_by_both": sum(r["flagged_by_both"] for r in unknown)},
            "unknowns": unknown, "interpretation": "Flags are reviewer measurements; neither agreement nor adjudication is formal truth. No gate or true-error claim is made."}


build_fidelity_report = build_report
build_fidelity_table = build_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_slots", type=Path)
    args = parser.parse_args()
    print(json.dumps(build_report(args.reference_slots), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
