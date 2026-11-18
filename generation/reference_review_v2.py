"""Versioned parser for harmless nested subclaim metadata.

The original ID-review decoder remains strict.  Version 2 accepts one optional
``subclaim_id`` bookkeeping key on each nested claim assessment, validates its
shape, removes it on a deep copy, and delegates every semantic/evidence check
to the original decoder.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from generation.evidence_reference_packets import (
    REVIEW_INSTRUCTIONS,
    decode_review,
)

SCHEMA = "record-review-evidence-references.v2-parser"
REVIEW_INSTRUCTIONS_V2 = REVIEW_INSTRUCTIONS + """

Version 2 permits one optional bookkeeping key, subclaim_id, inside each
claim_assessments object. It must be a nonempty string of at most 100
characters and has no semantic meaning. No other additional keys are allowed.
"""


def _validate_subclaim_metadata(model_output: Mapping[str, Any]) -> None:
    if not isinstance(model_output, Mapping) or set(model_output) != {"fields"}:
        raise ValueError("wrong version-2 output schema")
    fields = model_output.get("fields")
    if not isinstance(fields, list):
        raise ValueError("fields must be a list")
    for item in fields:
        if not isinstance(item, Mapping):
            raise ValueError("model field item malformed")
        claims = item.get("claim_assessments")
        if claims is None:
            continue
        if not isinstance(claims, list):
            raise ValueError("claim assessments malformed")
        for claim in claims:
            if not isinstance(claim, Mapping):
                raise ValueError("claim assessment malformed")
            extra = set(claim) - {"claim", "verdict", "reason", "evidence_ids", "subclaim_id"}
            if extra:
                raise ValueError("unknown claim assessment key")
            if "subclaim_id" in claim:
                value = claim["subclaim_id"]
                if not isinstance(value, str) or not value.strip() or len(value) > 100:
                    raise ValueError("subclaim_id must be a nonempty string of at most 100 characters")


def decode_review_v2(packet: Mapping[str, Any], model_output: Mapping[str, Any]) -> dict[str, Any]:
    """Decode using the original strict semantics plus optional subclaim IDs."""
    _validate_subclaim_metadata(model_output)
    cleaned = copy.deepcopy(model_output)
    for item in cleaned["fields"]:
        for claim in item.get("claim_assessments") or []:
            claim.pop("subclaim_id", None)
    return decode_review(packet, cleaned)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reanalyse_v6(root: Path, output: Path) -> dict[str, Any]:
    """Reanalyse existing v6 raw files without changing them."""
    if output.exists():
        raise FileExistsError("refusing existing reanalysis output")
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    key = json.loads((root / "answer-key.json").read_text())
    expected = {row["opaque_id"]: row["fields"] for row in key["records"]}
    freeze = json.loads((root / 'freeze.json').read_text())
    for filename, keyname in [('manifest.json','manifest_sha256'),('answer-key.json','key_sha256'),('key-audit.json','key_audit_sha256')]:
        if _sha(root/filename) != freeze[keyname]: raise ValueError('frozen calibration artifact changed')
    original = json.loads((root/'evaluation.json').read_text())
    original_hashes={x['task_name']:x['review_sha256'] for x in original['reviews']}
    specs={Path(x['path']).name:x for x in manifest['packets']}
    rows = []
    strict_valid = 0
    strict_failed = []
    for i in range(manifest["packet_count"]):
        packet_path = root / "reviewer-inputs" / f"input-{i:02d}.json"
        if _sha(packet_path) != specs[packet_path.name]['sha256']: raise ValueError('frozen packet changed')
        packet = json.loads(packet_path.read_text())
        for slot in "ab":
            raw_path = root / "reviews" / f"review-{i:02d}-{slot}.json"
            if _sha(raw_path) != original_hashes[f'/root/v6_review_{i:02d}{slot}']: raise ValueError('raw review changed since strict result')
            raw = json.loads(raw_path.read_text())
            try:
                decode_review(packet,raw)
                strict_valid += 1
            except (ValueError, KeyError, TypeError):
                strict_failed.append(raw_path.name)
            row = {"packet": packet["opaque_id"], "slot": slot,
                   "raw_sha256": _sha(raw_path)}
            try:
                decoded = decode_review_v2(packet, raw)
                verdicts = {x["field"]: x["verdict"] for x in decoded["field_items"]}
                row.update(mechanically_valid=True,
                           matches_key=verdicts == expected[packet["opaque_id"]],
                           verdicts=verdicts)
            except (ValueError, KeyError, TypeError) as exc:
                row.update(mechanically_valid=False, matches_key=False,
                           error=f"{type(exc).__name__}: {exc}")
            rows.append(row)
    result = {
        "schema": "recordwise-calibration-v6-parser-v2-reanalysis.v1",
        "status": "retrospective_parser_reanalysis",
        "review_count": len(rows),
        "strict_v1_result": {"mechanically_valid": strict_valid, "total": len(rows),
                              "failed_files": strict_failed},
        "v2_result": {
            "mechanically_valid": sum(x["mechanically_valid"] for x in rows),
            "matches_key": sum(x["matches_key"] for x in rows),
            "total": len(rows),
        },
        "policy_change": "Allow only optional nonsemantic subclaim_id metadata; delegate all semantic and evidence validation to strict v1 decoder.",
        "no_new_reviews": True,
        "production_authorized": False,
        "raw_reviews_unchanged": True,
        "parser_sha256": _sha(Path(__file__)),
        "manifest_sha256": _sha(root / "manifest.json"),
        "reviews": rows,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result
