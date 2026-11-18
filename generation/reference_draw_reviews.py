"""Offline bridge between ID-only reviews and the frozen draw-review schema.

This module does not call a model and does not alter campaign readiness.  It
builds one isolated packet per representation record and materialises reviewer
selected evidence IDs into a new schema that preserves all evidence spans.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from generation.evidence_reference_packets import build_evidence_packet
from generation.reference_review_v2 import REVIEW_INSTRUCTIONS_V2 as REVIEW_INSTRUCTIONS, decode_review_v2 as decode_review
from scripts import prepare_chunk_representations as prep

SCHEMA = "reference-draw-review-set.v1"


def _canon(v: Any) -> bytes:
    return json.dumps(v, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def _sha(v: Any) -> str:
    return hashlib.sha256(_canon(v)).hexdigest()


def _file_sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _instruction_sha() -> str:
    return hashlib.sha256((REVIEW_INSTRUCTIONS + "\n").encode()).hexdigest()


def _builder_sha() -> str:
    return _file_sha(Path(__file__).with_name("evidence_reference_packets.py"))


def _packet_with_context(kind, record, pack, plan_sha256, draw, slot, index):
    packet = build_evidence_packet(kind, record, pack)
    packet["review_context"] = {"plan_sha256": plan_sha256, "kind": kind,
                                 "draw": draw, "slot": slot,
                                 "record_index": index}
    packet["packet_sha256"] = _sha({k: packet[k] for k in
                                    ("schema", "opaque_id", "record_metadata",
                                     "fields", "evidence")})
    return packet


def _kind_key(kind: str) -> str:
    if kind not in {"atoms", "distillations"}:
        raise ValueError("kind must be atoms or distillations")
    return kind


def build_review_set(out: str | Path, *, plan_sha256: str, kind: str,
                     draw: int, slot: str, pack: Mapping[str, Any],
                     representation: Mapping[str, Any]) -> dict[str, Any]:
    """Write a frozen, source-only packet set for one reviewer slot.

    ``out`` must be a new directory.  The slot is part of every packet's
    immutable metadata. Context checks detect misplaced packets; raw-output copying
    is not proof of independent review and requires procedural controls.
    """
    kind = _kind_key(kind)
    if not isinstance(plan_sha256, str) or len(plan_sha256) != 64:
        raise ValueError("plan_sha256 must be a sha256")
    if type(draw) is not int or draw < 0:
        raise ValueError("draw must be a nonnegative integer")
    if slot not in {"a", "b"}:
        raise ValueError("slot must be a or b")
    root = Path(out)
    if root.exists():
        raise FileExistsError("refusing to overwrite review set")
    key = kind
    records = representation.get(key)
    if not isinstance(records, list) or len(records) != 12:
        raise ValueError("representation must contain exactly twelve records")
    root.mkdir(parents=True)
    packets = []
    for index, record in enumerate(records):
        packet = _packet_with_context(kind, record, pack, plan_sha256,
                                      draw, slot, index)
        path = root / f"input-{index:02d}.json"
        path.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n")
        packets.append({"path": path.name, "record_id": packet["opaque_id"],
                        "sha256": _file_sha(path),
                        "packet_sha256": packet["packet_sha256"]})
    manifest = {"schema": SCHEMA, "plan_sha256": plan_sha256, "kind": kind,
                "draw": draw, "slot": slot,
                "pack_digest": pack.get("digest"),
                "representation_sha256": _sha(representation),
                "instruction_sha256": _instruction_sha(),
                "builder_sha256": _builder_sha(),
                "parser_sha256": _file_sha(Path(__file__).with_name("reference_review_v2.py")),
                "adapter_sha256": _file_sha(Path(__file__)),
                "packets": packets}
    (root / "reviewer-instructions.md").write_text(REVIEW_INSTRUCTIONS + "\n")
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def materialize_review(root: str | Path, *, plan_sha256: str, kind: str,
                       draw: int, slot: str, pack: Mapping[str, Any],
                       representation: Mapping[str, Any], reviewer_id: str,
                       output: str | Path | None = None) -> dict[str, Any]:
    """Decode all twelve ID-only outputs and emit a reference draw review.

    The resolved ``field_items`` retain every selected span and nested
    subclaim.  The result deliberately uses a new schema: the legacy validator
    accepts only one copied quote and cannot represent multiple spans or
    nested claim evidence without loss.
    """
    kind = _kind_key(kind)
    if not isinstance(reviewer_id, str) or not reviewer_id.strip():
        raise ValueError("reviewer_id is required")
    root = Path(root)
    if not root.is_dir():
        raise ValueError("review set directory missing")
    # Verify context and all packet files before decoding any reviewer output.
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("plan_sha256") != plan_sha256 or manifest.get("kind") != kind or manifest.get("draw") != draw or manifest.get("slot") != slot:
        raise ValueError("review set context mismatch")
    if manifest.get("pack_digest") != pack.get("digest"):
        raise ValueError("source pack digest mismatch")
    if (manifest.get("instruction_sha256") != _instruction_sha() or
            manifest.get("builder_sha256") != _builder_sha() or
            manifest.get("parser_sha256") != _file_sha(Path(__file__).with_name("reference_review_v2.py")) or
            manifest.get("adapter_sha256") != _file_sha(Path(__file__))):
        raise ValueError("review instruction or builder changed")
    instruction_path = root / "reviewer-instructions.md"
    if not instruction_path.is_file() or _file_sha(instruction_path) != manifest.get("instruction_sha256"):
        raise ValueError("review instructions missing or changed")
    if manifest.get("representation_sha256") != _sha(representation):
        raise ValueError("representation hash mismatch")
    records = representation[kind]
    by_id = {r["id"]: r for r in records}
    rows = []
    if len(manifest.get("packets", [])) != 12:
        raise ValueError("review set must cover twelve packets")
    for index, spec in enumerate(manifest["packets"]):
        path = root / spec["path"]
        if not path.is_file() or _file_sha(path) != spec["sha256"]:
            raise ValueError("packet changed or missing")
        packet = json.loads(path.read_text())
        expected_packet = _packet_with_context(kind, records[index], pack,
                                               plan_sha256, draw, slot, index)
        if packet != expected_packet or spec.get("packet_sha256") != packet["packet_sha256"]:
            raise ValueError("packet does not match regenerated source packet")
        context = packet.get("review_context")
        if context != {"plan_sha256": plan_sha256, "kind": kind, "draw": draw, "slot": slot, "record_index": index}:
            raise ValueError("packet context mismatch")
        raw_path = root / f"review-{index:02d}.json"
        if not raw_path.is_file():
            raise ValueError(f"missing reviewer output {raw_path.name}")
        raw_output = json.loads(raw_path.read_text())
        raw_hash = _file_sha(raw_path)
        decoded = decode_review(packet, raw_output)
        rid = decoded["opaque_id"]
        if rid not in by_id:
            raise ValueError("review record is not in representation")
        items = decoded["field_items"]
        non_supported = [x for x in items if x["verdict"] != "supported"]
        row = {"representation": "atom" if kind == "atoms" else "distillation",
               "record_id": rid, "criterion": "cited_chunk_support.v1",
               "verdict": "supported" if not non_supported else non_supported[0]["verdict"],
               "cited_chunk_ids": sorted({e["chunk_id"] for x in items for e in x.get("evidence", [])}),
               "disputed_claims": [x.get("claim_assessments", []) for x in non_supported],
               "disputed_fields": [x["field"] for x in non_supported],
               "field_items": items,
               "raw_review_sha256": raw_hash, "raw_review_content_sha256": _sha(raw_output)}
        rows.append(row)
    if {r["record_id"] for r in rows} != set(by_id) or len(rows) != 12:
        raise ValueError("review coverage is incomplete or duplicated")
    rep_hash = _sha(representation)
    review = {"schema": "reference-draw-review.v1", "kind": kind,
              "reviewer_id": reviewer_id,
              "context": {"plan_sha256": plan_sha256, "kind": kind, "draw": draw, "slot": slot},
              "input_sha256": {"source_pack": pack["digest"], kind: rep_hash},
              "records": rows,
              "provenance": {"manifest_sha256": _file_sha(root / "manifest.json"),
                             "instruction_sha256": manifest["instruction_sha256"],
                             "builder_sha256": manifest["builder_sha256"],
                             "parser_sha256": manifest["parser_sha256"],
                             "adapter_sha256": manifest["adapter_sha256"],
                             "packet_sha256": [x["sha256"] for x in manifest["packets"]]},
              "legacy_compatibility": {"status": "not_compatible",
                  "reason": "legacy draw schema cannot preserve multiple evidence spans or nested claims"}}
    if output is not None:
        path = Path(output)
        if path.exists():
            raise FileExistsError("refusing to overwrite materialized review")
        path.write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n")
    return review


def combine_reviews(review_a: Mapping[str, Any], review_b: Mapping[str, Any], *,
                    plan_sha256: str, kind: str, draw: int) -> dict[str, Any]:
    """Combine exactly one materialized ``a`` and ``b`` review.

    This is an offline structural combiner.  It makes no claim that reviewer
    identities establish independence.
    """
    kind = _kind_key(kind)
    reviews = (review_a, review_b)
    for slot, review in zip(("a", "b"), reviews):
        if review.get("schema") != "reference-draw-review.v1" or review.get("kind") != kind:
            raise ValueError("wrong materialized review schema or kind")
        if review.get("context") != {"plan_sha256": plan_sha256, "kind": kind,
                                      "draw": draw, "slot": slot}:
            raise ValueError("materialized review context mismatch")
        if not isinstance(review.get("reviewer_id"), str) or not review["reviewer_id"].strip():
            raise ValueError("reviewer identity is required")
        records = review.get("records", [])
        if len(records) != 12 or len({r.get("record_id") for r in records}) != 12:
            raise ValueError("each review must contain twelve records")
    if review_a["reviewer_id"] == review_b["reviewer_id"]:
        raise ValueError("reviewer identities must differ")
    if {r["record_id"] for r in review_a["records"]} != {r["record_id"] for r in review_b["records"]}:
        raise ValueError("review slots cover different records")
    content_a={r['record_id']:r.get('raw_review_content_sha256') for r in review_a['records']}
    content_b={r['record_id']:r.get('raw_review_content_sha256') for r in review_b['records']}
    if any(not isinstance(v,str) or len(v)!=64 for v in (*content_a.values(),*content_b.values())):
        raise ValueError("raw review content hashes missing")
    if content_a==content_b:
        raise ValueError("duplicate raw review sets are not independent artifacts")
    if review_a.get("input_sha256") != review_b.get("input_sha256"):
        raise ValueError("review slots bind different inputs")
    for review in reviews:
        for row in review["records"]:
            items = row.get("field_items")
            if not isinstance(items, list) or not items:
                raise ValueError("empty field review cannot pass")
            if len({x.get("field") for x in items}) != len(items):
                raise ValueError("duplicate field reviews")
            for item in items:
                if item.get("verdict") not in {"supported", "unsupported", "contradicted", "indeterminate"}:
                    raise ValueError("invalid field verdict")
                claims = item.get("claim_assessments", [])
                if item["verdict"] == "supported" and any(c.get("verdict") != "supported" for c in claims):
                    raise ValueError("supported field has failed subclaim")
    a_hash = _sha(review_a); b_hash = _sha(review_b)
    if a_hash == b_hash:
        raise ValueError("review artifacts must differ")
    counts = []
    for review in reviews:
        counts.append({v: sum(item.get("verdict") == v for row in review["records"]
                              for item in row.get("field_items", []))
                       for v in ("supported", "unsupported", "contradicted", "indeterminate")})
    status = "passed" if all(c["unsupported"] == c["contradicted"] == c["indeterminate"] == 0 for c in counts) else "failed"
    return {"schema": "reference-dual-draw-review.v1", "kind": kind,
            "status": status, "context": {"plan_sha256": plan_sha256,
                                             "kind": kind, "draw": draw},
            "reviews": [{"slot": "a", "reviewer_id": review_a["reviewer_id"],
                         "sha256": a_hash, "counts": counts[0]},
                        {"slot": "b", "reviewer_id": review_b["reviewer_id"],
                         "sha256": b_hash, "counts": counts[1]}],
            "independence": "declared identities and distinct artifacts; not mechanically proven"}
