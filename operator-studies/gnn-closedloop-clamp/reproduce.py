"""Reproduce the clamp study in a fresh, hash-checked workspace.

Default execution prepares the copied source tree, runs all 40 arms, and
analyses them. ``--prepare-only`` performs only preparation and freeze
validation. Existing destinations and changed frozen inputs are rejected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENGINE_ROOT = HERE.parents[1]
FREEZE = HERE / "freeze.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_path(relative: str) -> Path:
    path = ENGINE_ROOT / relative
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"frozen source is missing or symlinked: {relative}")
    return path


def prepare(output: Path, *, prepare_only: bool, workers: int) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing existing output directory: {output}")
    output = output.resolve()
    if ENGINE_ROOT == output or ENGINE_ROOT in output.parents:
        raise ValueError("output must be outside the original engine root")
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    expected = frozen.get("sha256")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("clamp freeze has no source hash map")
    actual = {relative: digest(source_path(relative)) for relative in expected}
    if actual != expected:
        raise ValueError("frozen clamp input hash mismatch")
    for relative in expected:
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path(relative), target)
    clamp_dir = output / "operator-studies/gnn-closedloop-clamp"
    shutil.copy2(HERE / "requirements.txt", clamp_dir / "requirements.txt")
    subprocess.run([sys.executable, "clamp.py", "freeze"], cwd=clamp_dir, check=True)
    fresh = json.loads((clamp_dir / "freeze.json").read_text(encoding="utf-8"))
    if fresh.get("sha256") != expected:
        raise AssertionError("fresh clamp freeze does not reproduce source hashes")
    receipt = {
        "schema": "gnn-closedloop-clamp-reproduction.v1",
        "source_freeze_sha256": digest(FREEZE), "copied_source_hashes": actual,
        "fresh_freeze_sha256": digest(clamp_dir / "freeze.json"),
        "output": str(output), "workers": workers,
        "prepare_only": prepare_only, "training_runs": 0,
    }
    if prepare_only:
        receipt["status"] = "prepared"
    else:
        subprocess.run([sys.executable, "clamp.py", "run", "--workers", str(workers)], cwd=clamp_dir, check=True)
        subprocess.run([sys.executable, "clamp.py", "analyse"], cwd=clamp_dir, check=True)
        raw = json.loads((clamp_dir / "results/raw.json").read_text(encoding="utf-8"))
        summary = clamp_dir / "results/summary.json"
        if raw.get("n_runs") != 40 or not summary.is_file():
            raise AssertionError("completed clamp reproduction did not produce 40 runs and a summary")
        receipt.update({"status": "completed", "training_runs": raw["n_runs"],
                        "raw_sha256": digest(clamp_dir / "results/raw.json"),
                        "summary_sha256": digest(summary)})
    output.mkdir(parents=True, exist_ok=True)
    (output / "reproduction-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    receipt = prepare(args.output, prepare_only=args.prepare_only, workers=args.workers)
    print(json.dumps({"status": receipt["status"], "output": receipt["output"], "training_runs": receipt["training_runs"]}))


if __name__ == "__main__":
    main()
