"""Replay the frozen proposal-readiness gate on historical local results.

This is a structural development audit. It never writes to the original result
files, performs no model calls, and does not claim scientific validity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from generation.proposal_readiness import evaluate_result

DEFAULT_MANIFEST = ROOT / "reports/proposal-readiness-v1/manifest.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay(manifest_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("Refusing to overwrite a replay; choose a fresh output")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "proposal-readiness-manifest.v1":
        raise ValueError("Unsupported readiness manifest")
    records = manifest.get("records")
    if not isinstance(records, list) or not 1 <= len(records) <= 20:
        raise ValueError("Manifest must contain a bounded record list")
    results = []
    for item in records:
        allowed = {"id", "path", "sha256", "expected_status", "expected_candidate_statuses"}
        if not isinstance(item, dict) or not set(item) <= allowed or not {"id", "path", "sha256", "expected_status"} <= set(item):
            raise ValueError("Malformed manifest record")
        path = (ROOT / item["path"]).resolve()
        if ROOT not in path.parents or not path.is_file() or path.is_symlink():
            raise ValueError(f"Result path is missing or outside the repository: {item['id']}")
        actual = sha(path)
        if actual != item["sha256"]:
            raise ValueError(f"Frozen result hash mismatch: {item['id']}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        gate = evaluate_result(raw)
        if gate.get("status") != item["expected_status"]:
            raise AssertionError(
                f"Disposition mismatch for {item['id']}: "
                f"expected {item['expected_status']}, got {gate.get('status')}"
            )
        expected_candidates = item.get("expected_candidate_statuses")
        if expected_candidates is not None:
            actual_candidates = gate.get("candidate_gates")
            if not isinstance(expected_candidates, dict) or not isinstance(actual_candidates, dict):
                raise AssertionError(f"Candidate statuses missing for {item['id']}")
            actual_statuses = {key: value.get("status") for key, value in actual_candidates.items()}
            if actual_statuses != expected_candidates:
                raise AssertionError(
                    f"Candidate disposition mismatch for {item['id']}: "
                    f"expected {expected_candidates}, got {actual_statuses}"
                )
        results.append({
            "id": item["id"],
            "path": item["path"],
            "result_sha256": actual,
            "status": raw.get("status"),
            "readiness": gate,
        })
    report = {
        "schema": "proposal-readiness-replay.v1",
        "manifest_sha256": sha(manifest_path),
        "classifier_sha256": sha(ROOT / "generation/proposal_readiness.py"),
        "records": results,
        "scope": (
            "Structural consistency replay over frozen local outputs. "
            "The gate is a workflow allocation signal; it does not establish "
            "novelty, usefulness, feasibility or scientific validity."
        ),
        "paid_calls": 0,
        "experiment_launches": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.manifest.resolve(), args.output.resolve())
    print(json.dumps({"schema": result["schema"], "records": len(result["records"]), "output": str(args.output)}))
