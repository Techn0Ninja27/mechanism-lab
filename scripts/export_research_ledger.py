"""Export a read-only, evidence-qualified ledger for the local workbench."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

# The documented CLI is run as ``python scripts/export_research_ledger.py``;
# in that mode Python puts ``scripts/`` on sys.path rather than the repository
# root.  Bootstrap the package path before importing generation modules.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generation.experiment_feedback import inspect_feedback


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


def _headroom_metrics(root: Path, record: dict[str, Any], warnings: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Project the bounded numeric summary of the GNN headroom study.

    The surrounding feedback validator has already checked the artifact hash.
    This adapter is intentionally specific to this one schema: it exposes
    descriptive development measurements and never turns them into a quality
    or promotion signal.
    """
    if record.get("id") != "gnn-predictor-headroom-v1":
        return {}, {}
    path = root / record["result_path"]
    raw = _read(path, None)
    if not isinstance(raw, dict) or raw.get("schema") != "gnn-predictor-headroom-v1" or raw.get("status") != "complete":
        warnings.append("gnn-predictor-headroom-v1: result schema is missing or malformed")
        return {}, {}
    summary = raw.get("schedule_summary")
    if not isinstance(summary, dict) or set(summary) != {"constant", "lr_drop"}:
        warnings.append("gnn-predictor-headroom-v1: schedule summary is missing or malformed")
        return {}, {}

    def number(value: Any, label: str) -> float | None:
        try:
            valid = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))
        except (ValueError, OverflowError):
            valid = False
        if not valid:
            warnings.append(f"gnn-predictor-headroom-v1: invalid numeric metric {label}")
            return None
        return float(value)

    primary: dict[str, float] = {}
    secondary: dict[str, Any] = {"scope": "development-only open-loop trajectory replay; no downstream accuracy or novelty claim"}
    names = {"constant": "constant_lr", "lr_drop": "lr_drop"}
    for schedule, label in names.items():
        row = summary[schedule]
        selected = row.get("selected_candidates") if isinstance(row, dict) else None
        reductions = row.get("reduction_vs_sni") if isinstance(row, dict) else None
        if not isinstance(row, dict) or not isinstance(selected, dict) or not isinstance(reductions, dict):
            warnings.append(f"gnn-predictor-headroom-v1: malformed metrics for {schedule}")
            return {}, {}
        fields = {
            f"{label}_sni_error": row.get("sni"),
            f"{label}_damped_mix_error": selected.get("damped_mix"),
            f"{label}_ridge_error": selected.get("ridge"),
            f"{label}_damped_mix_reduction_vs_sni": reductions.get("damped_mix"),
            f"{label}_ridge_reduction_vs_sni": reductions.get("ridge"),
        }
        for metric, value in fields.items():
            parsed = number(value, metric)
            if parsed is None:
                return {}, {}
            primary[metric] = parsed
    # The card shows four primary numbers: preserve both schedules there.
    all_metrics = primary
    primary = {k: all_metrics[k] for k in (
        "constant_lr_sni_error", "constant_lr_damped_mix_error",
        "lr_drop_sni_error", "lr_drop_damped_mix_error")}
    secondary = {**{k: all_metrics[k] for k in (
        "constant_lr_damped_mix_reduction_vs_sni", "lr_drop_damped_mix_reduction_vs_sni",
        "constant_lr_ridge_reduction_vs_sni", "lr_drop_ridge_reduction_vs_sni",
        "constant_lr_ridge_error", "lr_drop_ridge_error")}, **secondary}
    secondary["interpretation"] = raw.get("interpretation", "")[:2000] if isinstance(raw.get("interpretation"), str) else ""
    return primary, secondary


def _feedback_studies(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Expose hash-checked feedback without promoting it to evidence.

    The generation pipeline additionally requires a matching source pack.  The
    read-only workbench registry has no single pack for all historical studies,
    so it supplies each record's own canonical arXiv URLs to the same validator
    and labels the result as registry provenance only.  Invalid records are
    omitted and their validator warnings are retained in the ledger.
    """
    path = root / "docs/method-grounding/experiment-feedback.json"
    registry = _read(path, {})
    if not isinstance(registry, dict) or not isinstance(registry.get("records"), list):
        return [], ["experiment feedback registry is missing or malformed"]
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    seen: set[str] = set()
    for raw in registry["records"]:
        if not isinstance(raw, dict):
            warnings.append("experiment feedback record is not an object")
            continue
        sid = raw.get("id")
        if isinstance(sid, str) and sid in seen:
            warnings.append(f"duplicate experiment feedback id {sid!r}")
            continue
        if isinstance(sid, str):
            seen.add(sid)
        urls = raw.get("source_urls")
        pack = {"sources": [{"url": x} for x in urls]} if isinstance(urls, list) else {}
        checked = inspect_feedback(pack, root, {"schema_version": "experiment-feedback.v1", "records": [raw]})
        warnings.extend(f"{sid or '<unknown>'}: {message}" for message in checked.get("warnings", []))
        valid = checked.get("records", [])
        if not valid:
            continue
        record = valid[0]
        report = record["report_path"]
        result = record["result_path"]
        primary_metrics, secondary_metrics = _headroom_metrics(root, record, warnings)
        rows.append({
            "id": record["id"], "cohort": "experiment_feedback",
            "title": record["id"], "artifact_directory": str(Path(report).parent),
            "raw_verdict": record["verdict"],
            "audit_qualification": "registry_hash_and_paths_verified",
            "audit_note": ("Registry and result hash/path verified; this is not independent scientific validation. "
                           f"Tested change: {record['tested_change']} Scope: {record['scope']}"),
            "primary_metrics": primary_metrics, "secondary_metrics": secondary_metrics,
            "novelty_established": False, "human_adjudicated": False,
            "prior_art_status": "Source-linked local experiment; no novelty established.",
            "limitations": list(record["limitations"]),
            "artifact_refs": [{"path": report, "sha256": _hash(root / report)},
                              {"path": result, "sha256": record["result_sha256"]}],
            "evidence": {"source": "docs/method-grounding/experiment-feedback.json",
                         "source_urls": list(record["source_urls"]),
                         "tested_change": record["tested_change"], "scope": record["scope"],
                         "train_eligible": False, "novelty_established": False},
        })
    return rows, warnings


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
    # Explicitly registered experimental cohorts; never infer promotion from a score.
    for sid, directory, filename, title in (
        ("integer-composition", "operator-studies/integer-composition", "results.json", "Executable compression composition"),
        ("utility-ranker", "reports/utility-ranker", "final.json", "Utility ranking on unseen problem families"),
    ):
        folder = root / directory
        result_path = folder / filename
        result = _read(result_path, None)
        if not isinstance(result, dict):
            continue
        summary = result.get("summary", {})
        if not isinstance(summary, dict):
            summary = {}
        frozen = _read(folder / "freeze.json", {})
        sources = frozen.get("source_hashes", {}) if isinstance(frozen, dict) else {}
        issues = []
        if not isinstance(sources, dict) or not sources:
            issues.append("Missing frozen source hashes")
            sources = {}
        for name, expected in sources.items():
            target = (root / name).resolve()
            if root not in target.parents or _hash(target) != expected:
                issues.append("Frozen source mismatch: " + name)
        if result.get("source_hashes") != sources:
            issues.append("Result and freeze source hashes differ")
        if not frozen.get("validation_sha256") or _hash(folder / "validation.json") != frozen.get("validation_sha256"):
            issues.append("Frozen validation hash mismatch")
        is_utility = sid == "utility-ranker"
        metrics = ({"primary_metric": "selection_regret_lower_is_better", "mean_candidate": summary.get("regret"),
                    "mean_control": summary.get("fixed_regret"), "mean_random": summary.get("random_regret")}
                   if is_utility else
                   {"primary_metric": "bytes_per_sample_lower_is_better", "mean_candidate": summary.get("candidate_bytes_per_sample"),
                    "mean_control": summary.get("control_bytes_per_sample"), "control": summary.get("control")})
        studies.append({
            "id": sid, "cohort": "utility" if is_utility else "composition", "title": title,
            "artifact_directory": directory,
            "raw_verdict": result.get("status", "unknown") if is_utility else summary.get("status", "unknown"),
            "audit_qualification": "artifact_mismatch" if issues else "frozen_sources_and_validation_match",
            "audit_note": "; ".join(issues) if issues else "Hash agreement is provenance evidence, not independent replication.",
            "primary_metrics": metrics,
            "secondary_metrics": summary.get("per_family", {}) if is_utility else {},
            "novelty_established": False, "human_adjudicated": False,
            "prior_art_status": "Known compression operators; no new algorithm established.",
            "limitations": (["Not promoted: worse than the fixed baseline in every held-out family.",
                              "Synthetic tasks only; ranking scores are not calibrated utility predictions.",
                              "BLAS warnings remain unresolved; independent inference selected identical chains."]
                             if is_utility else
                             ["Three synthetic shifted streams only; no production-codec comparison.",
                              "Known operators enumerated deterministically, not an Alien beta-ranked result."])
                            + ["No independent replication or established scientific novelty."] + issues,
            "artifact_refs": [{"path": _rel(f, root), "sha256": _hash(f)} for f in
                              (result_path, folder / "README.md", folder / "freeze.json", folder / "validation.json") if f.is_file()],
            "evidence": {"source": _rel(result_path, root)},
        })
    diagnostic_dir = root / 'operator-studies/mechanism-review'
    diagnostic = _read(diagnostic_dir / 'result.json', None)
    if isinstance(diagnostic, dict) and isinstance(diagnostic.get('arms'), dict):
        metrics = {}
        for arm, record in diagnostic['arms'].items():
            if arm in {'old', 'specialization'} and isinstance(record, dict):
                metrics[arm + '_reference_status_matches'] = record.get('label_matches')
                metrics[arm + '_cases'] = record.get('count')
        studies.append({
            'id': 'mechanism-review', 'cohort': 'review_diagnostic',
            'title': 'Mechanism review: exposed analytic cases',
            'artifact_directory': 'operator-studies/mechanism-review',
            'raw_verdict': 'development_diagnostic_only',
            'audit_qualification': 'not_independent_quality_evidence',
            'audit_note': 'Literal reference-status agreement; raw model reasons include a status contradiction and a disputed mapping.',
            'primary_metrics': metrics, 'secondary_metrics': {},
            'novelty_established': False, 'human_adjudicated': False,
            'prior_art_status': 'Synthetic analytic cases; no literature novelty assessment.',
            'limitations': ['Six exposed root-authored cases, one batch per arm.',
                            'Same-family native label review is not independent human adjudication.',
                            'Not the full production proposal/review pipeline.',
                            'No calibrated ranker improvement, usefulness or novelty claim.'],
            'artifact_refs': [{'path': _rel(f, root), 'sha256': _hash(f)} for f in
                              (diagnostic_dir / name for name in ('README.md', 'cases.json', 'labels.json', 'result.json',
                               'independent-review.json', 'output-audit.json', 'plan.json', 'receipts.json')) if f.is_file()],
            'evidence': {'source': _rel(diagnostic_dir / 'result.json', root)},
        })
    codec_dir = root / 'operator-studies/graph-walk-codecs'
    codec_result = _read(codec_dir / 'results.json', None)
    if isinstance(codec_result, dict) and isinstance(codec_result.get('summary'), dict):
        metrics = {}
        for graph, values in codec_result['summary'].items():
            if graph in {'amazon', 'dblp'} and isinstance(values, dict):
                for metric in ('charged_ratio_to_packed', 'resident_ratio_to_packed'):
                    metrics[graph + '_' + metric] = values.get(metric)
        studies.append({
            'id': 'graph-walk-codecs', 'cohort': 'codec_development',
            'title': 'Graph-walk varint coding versus packed controls',
            'artifact_directory': 'operator-studies/graph-walk-codecs',
            'raw_verdict': codec_result.get('status', 'unknown'),
            'audit_qualification': 'development_replay_not_systems_validation',
            'audit_note': 'Serialized byte counts and exact round trips; same-code reproduction is not independent replication.',
            'primary_metrics': metrics, 'secondary_metrics': {},
            'novelty_established': False, 'human_adjudicated': False,
            'prior_art_status': 'Known rank/LEB128 composition; no novel codec established.',
            'limitations': ['Two development graphs, five seeds, finite first-round profiles.',
                            'Uniform-transition walks, not FeLoG/FeeST training traces.',
                            'Resident dictionary is an optimistic counterfactual.',
                            'No GPU, link, latency, energy, graph-task or held-out result.'],
            'artifact_refs': [{'path': _rel(f, root), 'sha256': _hash(f)} for f in
                              (codec_dir / name for name in ('README.md', 'protocol.json', 'results.json', 'freeze.json',
                               'source-receipts.json', 'reproduction.json', 'codecs.py', 'run.py')) if f.is_file()],
            'evidence': {'source': _rel(codec_dir / 'results.json', root)},
        })
    utility_probe_dir = root / 'operator-studies/utility-judgment-pilot'
    utility_probe = _read(utility_probe_dir / 'summary.json', None)
    if isinstance(utility_probe, dict):
        studies.append({
            'id': 'utility-judgment-pilot', 'cohort': 'utility_judgment_diagnostic',
            'title': 'Outcome-withheld usefulness judgments on three exposed method pairs',
            'artifact_directory': 'operator-studies/utility-judgment-pilot',
            'raw_verdict': utility_probe.get('status', 'unknown'),
            'audit_qualification': 'retrospective_development_not_ranker_validation',
            'audit_note': 'Three cases selected after outcomes; two orders per case are not independent observations.',
            'primary_metrics': {key: utility_probe.get(key) for key in
                                ('cases','presentations','abstentions','answered','correct_answered')},
            'secondary_metrics': {'conditional_mean_brier': utility_probe.get('conditional_mean_brier'),
                                  'changed_pairs': utility_probe.get('preference_or_abstention_changed_pairs')},
            'novelty_established': False, 'human_adjudicated': False,
            'prior_art_status': 'Known methods and exposed local outcomes; no novelty assessment.',
            'limitations': ['No outcomes supplied to judge, but cases were selected retrospectively.',
                            'Abstentions remain unknown; only three numeric answers.',
                            'Order sensitivity also includes possible hosted inference variability.',
                            'No trained ranker, independent calibration or stronger-model comparison.'],
            'artifact_refs': [{'path': _rel(f, root), 'sha256': _hash(f)} for f in
                              (utility_probe_dir / name for name in ('README.md','cases.json','labels.json',
                               'predictions.json','evaluation.json','summary.json','plan.json','receipts.json','run.py')) if f.is_file()],
            'evidence': {'source': _rel(utility_probe_dir / 'evaluation.json', root)},
        })
    feedback_rows, feedback_warnings = _feedback_studies(root)
    existing_ids = {item.get("id") for item in studies}
    canonical_dirs = {
        str(item.get("artifact_directory"))
        for item in studies
        if isinstance(item.get("artifact_directory"), str)
    }
    for item in feedback_rows:
        if item.get("id") in existing_ids:
            feedback_warnings.append(f"duplicate experiment feedback id {item['id']!r} omitted")
        elif item.get("artifact_directory") in canonical_dirs:
            feedback_warnings.append(
                f"experiment feedback {item['id']!r} omitted: artifact directory already registered"
            )
        else:
            studies.append(item)
            existing_ids.add(item.get("id"))
            canonical_dirs.add(item.get("artifact_directory"))
    ledger = {
        "schema_version": "1.0",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "read_only": True,
        "novelty_policy": "No automatic novelty or promotion decisions; novelty_established and human_adjudicated are copied from artifacts.",
        "study_count": len(studies), "studies": studies,
        "feedback_diagnostics": feedback_warnings,
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
