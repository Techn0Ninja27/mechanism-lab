"""Provenance checks for atom IDs, scorer vocabularies, and paper evidence.

Numeric cluster IDs are only meaningful relative to the exact clustering and
token-mapping artifacts that produced a checkpoint.  This module makes that
relationship explicit and refuses silently mixed artifacts.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


class ProvenanceError(ValueError):
    """Raised when artifacts cannot safely be joined by atom ID."""


def _read(path: str | Path) -> dict:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def _fingerprint(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _mapping_ids(path: str | Path) -> set[int]:
    data = _read(path)
    return {int(x) for x in data.get("cluster_to_token_id", {})}


def audit_provenance(
    assignment_clusters: str | Path,
    canonical_clusters: str | Path | None = None,
    coherence_mapping: str | Path | None = None,
    availability_mapping: str | Path | None = None,
) -> dict:
    """Audit exact ID coverage and (optionally) cross-file label agreement.

    ``assignment_clusters`` must be the cluster file used to construct the
    scorer datasets.  ``canonical_clusters`` is a separate semantic catalogue
    and is compared by the stable ``(paper_id, text)`` key, never by position.
    A low agreement score is evidence of a relabelled/mixed release and makes
    the numeric IDs unsafe to join to canonical descriptions.
    """
    assignments = _read(assignment_clusters)
    ids = {int(x) for x in assignments.get("clusters", {})}
    out = {
        "assignment_clusters": str(Path(assignment_clusters)),
        "assignment_sha256": _fingerprint(assignment_clusters),
        "assignment_ids": sorted(ids),
        "assignment_count": len(ids),
    }
    mapping_ids: dict[str, set[int]] = {}
    for name, path in (("coherence", coherence_mapping), ("availability", availability_mapping)):
        if path is None:
            continue
        mids = _mapping_ids(path)
        mapping_ids[name] = mids
        out[name + "_mapping"] = str(Path(path))
        out[name + "_mapping_sha256"] = _fingerprint(path)
        out[name + "_ids"] = sorted(mids)
        if mids != ids:
            raise ProvenanceError(f"{name} mapping IDs do not exactly match assignment clusters")

    if canonical_clusters is not None:
        canonical = _read(canonical_clusters)
        canonical_by_unit = {
            (str(x.get("paper_id")), str(x.get("text"))): int(x.get("cluster"))
            for x in canonical.get("atom_assignments", [])
            if x.get("cluster") is not None
        }
        counts = Counter(); compared = 0
        for row in assignments.get("atom_assignments", []):
            key = (str(row.get("paper_id")), str(row.get("text")))
            if key in canonical_by_unit:
                compared += 1
                counts[(int(row.get("cluster")), canonical_by_unit[key])] += 1
        same = sum(n for (a, b), n in counts.items() if a == b)
        out["canonical_clusters"] = str(Path(canonical_clusters))
        out["canonical_sha256"] = _fingerprint(canonical_clusters)
        out["canonical_label_agreement"] = same / compared if compared else 0.0
        out["canonical_units_compared"] = compared
        out["canonical_label_mismatch_warning"] = bool(compared and same / compared < 0.95)
    return out


def validate_candidate_ids(cluster_ids: Iterable[int], audit: dict) -> tuple[int, ...]:
    """Validate and normalize candidate IDs against an audit result."""
    ids = tuple(int(x) for x in cluster_ids)
    allowed = set(audit.get("assignment_ids", []))
    missing = sorted(set(ids) - allowed)
    if missing:
        raise ProvenanceError(f"candidate IDs absent from assignment clusters: {missing}")
    if len(set(ids)) != len(ids):
        raise ProvenanceError("candidate contains duplicate atom IDs")
    return ids


def assignment_records(clusters_path: str | Path, cluster_ids: Iterable[int], limit: int = 3) -> list[dict]:
    """Return assignment evidence only from the exact scorer cluster file."""
    if limit < 1:
        raise ValueError("limit must be positive")
    wanted = set(int(x) for x in cluster_ids)
    data = _read(clusters_path)
    grouped: dict[int, list[dict]] = {cid: [] for cid in wanted}
    for row in data.get("atom_assignments", []):
        cid = int(row.get("cluster", -1))
        if cid in grouped and len(grouped[cid]) < limit:
            grouped[cid].append({"paper_id": row.get("paper_id"), "text": row.get("text"), "cluster": cid})
    return [r for cid in sorted(grouped) for r in grouped[cid]]
