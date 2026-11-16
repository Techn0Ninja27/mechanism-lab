"""Export a read-only, evidence-qualified ledger for the local workbench."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError, TypeError):
        return default


def _rel(path: Path, root: Path = ROOT) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _hash(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _metric(v: dict[str, Any]) -> dict[str, Any]:
    keys = ("primary_metric", "mean_candidate", "mean_control", "mean_strongest_control", "strongest_control")
    return {k: v[k] for k in keys if k in v}


GROUNDED_FAILURES = {
    "abstracts-0": ("unidentifiable_transfer_assumption", "reconstruction-studies/abstracts-0/README.md"),
    "abstracts-1": ("remaining_mask_information_path", "reconstruction-studies/abstracts-1/README.md"),
    "atoms-0": ("unidentifiable_transfer_assumption", "reconstruction-studies/atoms-0/README.md"),
    "atoms-1": ("remaining_mask_information_path", "reconstruction-studies/atoms-1/README.md"),
    "molecules-0": ("empirical_failure_of_added_component", "reconstruction-studies/molecules-0/README.md"),
    "molecules-1": ("unidentifiable_transfer_assumption", "reconstruction-studies/molecules-1/README.md"),
}


def build_ledger(root: Path = ROOT) -> dict[str, Any]:
    """Build the serializable ledger without writing files (safe for the server)."""
    root = Path(root).resolve()
    audit = _read(root / "candidate-studies/audit.json", {})
    cohorts = audit.get("cohort", {}) if isinstance(audit, dict) else {}
    audit_labels = {sid: label for label, ids in cohorts.items() if isinstance(ids, list) for sid in ids}
    high_risk = audit.get("high_risk_positive_labels", {}) if isinstance(audit, dict) else {}
    invalid_reason = audit.get("invalid_reason", {}) if isinstance(audit, dict) else {}
    studies: list[dict[str, Any]] = []
    for verdict_path in sorted((root / "candidate-studies").glob("*/verdict.json")):
        sid = verdict_path.parent.name
        verdict = _read(verdict_path, {})
        proposal_path = verdict_path.parent / "proposal.json"
        proposal = _read(proposal_path, {})
        title = proposal.get("proposal", {}).get("title") if isinstance(proposal.get("proposal"), dict) else None
        studies.append({
            "id": sid, "cohort": "candidate", "title": title,
            "raw_verdict": verdict.get("status", "unknown"),
            "audit_qualification": audit_labels.get(sid, "unknown"),
            "audit_note": high_risk.get(sid) or invalid_reason.get(sid),
            "primary_metrics": _metric(verdict),
            "secondary_metrics": {},
            "novelty_established": bool(verdict.get("novelty_established", False)),
            "human_adjudicated": bool(verdict.get("human_adjudicated", False)),
            "prior_art_status": verdict.get("prior_art_status"),
            "limitations": verdict.get("limitations", []),
            "artifact_refs": [{"path": _rel(p, root), "sha256": _hash(p)} for p in (verdict_path, proposal_path) if p.exists()],
            "evidence": {"source": "verdict.json", "audit": "candidate-studies/audit.json"},
        })
    harmonized = _read(root / "reconstruction-studies/harmonized-results.json", {})
    secondary = {x.get("id"): x for x in harmonized.get("comparisons", []) if isinstance(x, dict)}
    manifest = _read(root / "reconstruction-studies/manifest.json", [])
    manifest_by_id = {x.get("id"): x for x in manifest if isinstance(x, dict)}
    for verdict_path in sorted((root / "reconstruction-studies").glob("*/verdict.json")):
        sid = verdict_path.parent.name
        verdict = _read(verdict_path, {})
        proposal = _read(verdict_path.parent / "proposal.json", {})
        title = manifest_by_id.get(sid, {}).get("title")
        studies.append({
            "id": sid, "cohort": "reconstruction", "title": title,
            "raw_verdict": verdict.get("status", verdict.get("verdict", "unknown")),
            "audit_qualification": "pre_final_audit_reviewed",
            "audit_note": "Final descriptive evaluation cannot reverse failed validation; see study README.",
            "primary_metrics": _metric(verdict),
            "secondary_metrics": secondary.get(sid, {}),
            "novelty_established": bool(verdict.get("novelty_established", False)),
            "human_adjudicated": bool(verdict.get("human_adjudicated", False)),
            "prior_art_status": verdict.get("prior_art_status"),
            "limitations": verdict.get("limitations", []),
            "failure_assessment": ({"type": GROUNDED_FAILURES[sid][0], "evidence_path": GROUNDED_FAILURES[sid][1]} if sid in GROUNDED_FAILURES else None),
            "artifact_refs": [{"path": _rel(p, root), "sha256": _hash(p)} for p in (verdict_path, verdict_path.parent / "proposal.json", verdict_path.parent / "README.md") if p.exists()],
            "evidence": {"source": "study verdict/README", "secondary_alignment": "reconstruction-studies/harmonized-results.json"},
        })
    ledger = {
        "schema_version": "1.0",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "read_only": True,
        "novelty_policy": "No automatic novelty or promotion decisions; novelty_established and human_adjudicated are copied from artifacts.",
        "study_count": len(studies), "studies": studies,
        "cohort_sources": {"candidate_audit": _rel(root / "candidate-studies/audit.json", root), "reconstruction_report": _rel(root / "reconstruction-studies/README.md", root), "failure_assessments": "notes/2026-11-16-reconstruction-review/ASSESSMENT.md"},
    }
    return ledger


def export(output: Path, root: Path = ROOT) -> dict[str, Any]:
    ledger = build_ledger(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    return ledger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({"output": str(args.output), "study_count": export(args.output)["study_count"]}))


if __name__ == "__main__":
    main()
