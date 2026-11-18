"""Deterministic scorer for the offline field-review calibration."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any

def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _fields(record: dict[str, Any], representation: str) -> dict[str, Any]:
    out = {"statement": record["statement"]} if representation == "atom" else {"summary": record["summary"]}
    keys = ("inputs", "outputs", "conditions", "unknowns") if representation == "atom" else ("unknowns",)
    for key in keys:
        for i, value in enumerate(record.get(key, [])):
            out[f"{key}[{i}]"] = value
    return out

def _items(row: dict[str, Any]) -> list[dict[str, Any]]:
    value = row.get("field_items", row.get("fields"))
    if not isinstance(value, list): raise ValueError("review row has no field list")
    return value

def score_calibration(folder: str | Path, instructions_path: str | Path) -> dict[str, Any]:
    root, instructions = Path(folder), Path(instructions_path)
    freeze = json.loads((root / "calibration-freeze.json").read_text())
    packet = json.loads((root / "reviewer-inputs.json").read_text())
    key = json.loads((root / "scoring-key.json").read_text())
    review_paths = [root / "reviewer-a.json", root / "reviewer-b.json"]
    reviews = [json.loads(p.read_text()) for p in review_paths]
    input_hash, key_hash, instruction_hash = _sha(root / "reviewer-inputs.json"), _sha(root / "scoring-key.json"), _sha(instructions)
    errors = []
    if freeze.get("input_sha256") != input_hash: errors.append("calibration input hash mismatch")
    if freeze.get("scoring_key_sha256") != key_hash: errors.append("scoring key hash mismatch")
    if freeze.get("instructions_sha256") != instruction_hash: errors.append("instructions hash mismatch")
    if freeze.get("reviewer_count") != 2 or freeze.get("single_pass") is not True: errors.append("freeze policy mismatch")
    source_hash = str(packet.get("source_pack_sha256", ""))
    if not source_hash or key.get("source_hashes", {}).get("sourcepack.json") != source_hash: errors.append("source pack hash mismatch")
    aliases = {x["record"]["id"]: x for x in packet.get("records", []) if isinstance(x, dict) and isinstance(x.get("record"), dict)}
    if len(aliases) != 28: errors.append("calibration must contain exactly 28 unique records")
    expected = {a: _fields(x["record"], x["representation"]) for a, x in aliases.items()}
    all_results, metadata, reviewer_ids = [], [], set()
    for path, review in zip(review_paths, reviews):
        rid = review.get("reviewer_id")
        if not isinstance(rid, str) or not rid or rid in reviewer_ids: errors.append("reviewer identity missing or duplicated")
        reviewer_ids.add(rid)
        declared = review.get("input_sha256")
        if not isinstance(declared, dict) or declared.get("source_pack") != source_hash: errors.append(f"{rid}: source input hash mismatch")
        rows = review.get("records")
        if not isinstance(rows, list) or len(rows) != 28: errors.append(f"{rid}: record count mismatch"); continue
        seen, field_count = set(), 0
        for row in rows:
            alias = row.get("record_id") if isinstance(row, dict) else None
            if alias not in aliases or alias in seen: errors.append(f"{rid}: foreign or duplicate record {alias}"); continue
            seen.add(alias); by_field = {}
            for item in _items(row):
                field = item.get("field") if isinstance(item, dict) else None
                if field not in expected[alias] or field in by_field or item.get("value") != expected[alias][field]:
                    errors.append(f"{rid}: field coverage/value mismatch {alias}/{field}"); continue
                by_field[field] = item
                if item.get("verdict") not in {"supported", "unsupported", "contradicted", "indeterminate"}: errors.append(f"{rid}: invalid verdict {alias}/{field}")
            if set(by_field) != set(expected[alias]): errors.append(f"{rid}: incomplete fields {alias}")
            field_count += len(by_field)
            all_results.extend({"reviewer_id": rid, "record_id": alias, "field": f, "verdict": x.get("verdict")} for f, x in by_field.items())
        if seen != set(aliases): errors.append(f"{rid}: incomplete record coverage")
        if field_count != 133: errors.append(f"{rid}: field item count {field_count} != 133")
        metadata.append({"reviewer_id": rid, "artifact_sha256": _sha(path), "records": len(rows), "field_items": field_count})
    amap = key.get("reviewer_aliases", {})
    def verdicts(alias, field=None): return [x["verdict"] for x in all_results if x["record_id"] == alias and (field is None or x["field"] == field)]
    planted = key.get("planted_bad_records", {}); positives = key.get("required_positive_checks", {}); extra = key.get("required_additional_negative", {})
    detected = {name: any(v in {"unsupported", "contradicted", "indeterminate"} for v in verdicts(amap.get(name, ""))) for name in planted}
    positive_ok = {name: bool(verdicts(amap.get(name, ""))) and all(v == "supported" for v in verdicts(amap.get(name, ""))) for name in positives}
    extra_v = verdicts(extra.get("record_id"), extra.get("field"))
    if len(planted)!=4 or len(positives)!=2 or not extra.get("record_id") or not extra.get("field"):
        errors.append("known-answer check shape is invalid")
    per_reviewer = {}
    for rid in reviewer_ids:
        subset=[x for x in all_results if x["reviewer_id"]==rid]
        def rv(alias,field=None):return [x["verdict"] for x in subset if x["record_id"]==alias and (field is None or x["field"]==field)]
        per_reviewer[rid] = (all(any(v in {"unsupported","contradicted"} for v in rv(amap.get(name,""))) for name in planted)
                            and all(bool(rv(amap.get(name,""))) and all(v=="supported" for v in rv(amap.get(name,""))) for name in positives)
                            and any(v in {"unsupported","contradicted"} for v in rv(extra.get("record_id"),extra.get("field"))))
    ready = not errors and len(per_reviewer)==2 and all(per_reviewer.values())
    return {"schema": "field-calibration-score.v2", "ready": ready, "errors": errors, "reviewers": metadata,
            "record_count": len(aliases), "field_item_count": len(all_results), "planted_detected": detected,
            "positive_checks_supported": positive_ok, "additional_negative": {"record_id": extra.get("record_id"), "field": extra.get("field"), "verdicts": extra_v},
            "per_reviewer_pass":per_reviewer, "source_pack_sha256": source_hash, "artifact_sha256": {"reviewer-inputs.json": input_hash, "scoring-key.json": key_hash, "reviewer-instructions.md": instruction_hash, "reviewer-a.json": _sha(review_paths[0]), "reviewer-b.json": _sha(review_paths[1])},
            "limitations": ["Hand-built calibration only; not a population error estimate.", "score.json flags are ignored; readiness is recomputed from reviewer records and the key."]}
