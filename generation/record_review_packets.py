"""Small, offline packets for testing source-record review.

The packet deliberately separates scientific claims from bookkeeping metadata.
It is an instrument prototype; it does not produce a fidelity verdict.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from scripts import prepare_chunk_representations as prep

SCHEMA = "record-review-packet.v1"
INSTRUCTIONS = """Review only the scientific field items in this packet against the
complete text of the cited rendered chunks. Metadata (IDs, source IDs, evidence
IDs, interpretation flags and eligibility flags) is bookkeeping and must not be
judged. A claim is supported only when it is stated or follows without an
outside assumption. Contradicted means the cited text says the opposite.
Unsupported means the claim could be false while the cited text remained true.
Indeterminate is reserved for malformed evidence or an unresolved referent.
For a supported positive claim, return one or more exact evidence spans of at
most 300 characters each, with their chunk ID, and explain the entailment in
record-specific language. For an unsupported or contradicted item, name the
exact field and the missing or conflicting condition. For an unknown item, say
what information is absent from the supplied chunks; if the chunks answer it,
the unknown is unsupported. Return only JSON with field_items, one item for
each scientific field item, preserving field names and values. Do not invent
verdicts for metadata. Each field_items entry must contain field, value,
verdict, reason and evidence (a list of {chunk_id, quote} objects). For supported
unknowns use evidence=[] and an absence_reason scoped to the supplied chunks.
For non-supported claims explain a concrete missing assumption or a counterexample
consistent with the passage. If a field asserts several claims, list them in
claim_assessments with separate verdicts and evidence; the field cannot pass
unless every claim passes. Use item-specific explanations, not merely varied
wording. Do not calculate hashes: the calling program records provenance."""


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def build_record_packet(kind: str, record: Mapping[str, Any], pack: Mapping[str, Any]) -> dict[str, Any]:
    """Build one packet from a validated record and its cited rendered chunks."""
    if kind not in {"atoms", "distillations"}:
        raise ValueError("kind must be atoms or distillations")
    pack = prep.base._validated_pack(pack)
    key = "atoms" if kind == "atoms" else "distillations"
    schema = prep.ATOM_SCHEMA if kind == "atoms" else prep.DISTILL_SCHEMA
    if not isinstance(record, Mapping) or record.get("source_id") is None:
        raise ValueError("record must include source_id")
    if record.get("interpretation") not in (None, "model_proposed_not_curated"):
        raise ValueError("unexpected interpretation")
    fields = prep._draw_fields(kind, record)
    chunks = {c["id"]: c for c in prep.render_source_context(pack)["chunks"]}
    evidence = record.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("record has no evidence")
    cited_ids = []
    for item in evidence:
        cid = item.get("chunk_id") if isinstance(item, Mapping) else None
        if cid not in chunks or cid in cited_ids:
            raise ValueError("record cites an unknown or duplicate chunk")
        if chunks[cid].get("source_id") != record["source_id"]:
            raise ValueError("record cites a chunk from another source")
        cited_ids.append(cid)
    rendered = []
    for cid in cited_ids:
        chunk = chunks[cid]
        rendered.append({"chunk_id": cid, "source_id": chunk["source_id"],
                         "text": chunk["text"],
                         "text_sha256": _sha_bytes(chunk["text"].encode())})
    return {
        "schema": SCHEMA,
        "opaque_id": str(record.get("id", "record")),
        "representation": kind,
        "record_metadata": {
            "record_id": record.get("id"),
            "source_id": record["source_id"],
            "representation_schema": schema,
            "evidence_chunk_ids": cited_ids,
            "source_pack_digest": pack["digest"],
        },
        "scientific_fields": [
            {"field": name, "value": value, "is_unknown": is_unknown}
            for name, value, is_unknown in fields
        ],
        "cited_chunks": rendered,
        "labels_excluded": True,
    }


def build_probe_v2(source_dir: Path, output_dir: Path, pack_path: Path) -> dict[str, Any]:
    """Convert v1 probe inputs without reading its answer key or reviews."""
    if output_dir.exists():
        raise FileExistsError("refusing an existing probe output")
    pack = prep._pack(pack_path)
    output_dir.mkdir(parents=True)
    packets = []
    for source in sorted(source_dir.glob("input-*.json")):
        value = json.loads(source.read_text())
        kind = "atoms" if value["representation"] == "atom" else "distillations"
        packet = build_record_packet(kind, value["record"], pack)
        packet["opaque_id"] = value["opaque_id"]
        packet["record_metadata"]["record_id"] = value["opaque_id"]
        path = output_dir / source.name
        path.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n")
        packets.append({"path": path.name, "sha256": _sha_bytes(path.read_bytes()),
                        "opaque_id": packet["opaque_id"]})
    manifest = {"schema": "record-review-probe-v2-manifest.v1", "pack_sha256": _sha_bytes(pack_path.read_bytes()),
                "instruction_sha256": _sha_bytes((INSTRUCTIONS + "\n").encode()), "packets": packets,
                "labels_excluded": True}
    (output_dir / "reviewer-instructions.md").write_text(INSTRUCTIONS + "\n")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest

