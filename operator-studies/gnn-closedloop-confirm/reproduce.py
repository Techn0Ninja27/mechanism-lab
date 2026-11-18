"""Reproduce the confirmation in a fresh, hash-checked workspace.

The default command copies the minimal source tree into a new engine-root
directory, regenerates the confirmation freeze, runs the 64-run confirmation,
and analyses it. ``--prepare-only`` performs only the copy and freeze check.

Example:
    python reproduce.py --output /tmp/gnn-confirm-reproduction
    python reproduce.py --output /tmp/gnn-confirm-preflight --prepare-only
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
# confirm/ -> operator-studies/ -> upstream/
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
    if not FREEZE.is_file():
        raise FileNotFoundError(f"confirmation freeze is missing: {FREEZE}")

    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    expected = frozen.get("sha256")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("confirmation freeze has no source hash map")
    actual = {relative: digest(source_path(relative)) for relative in expected}
    if actual != expected:
        mismatches = [k for k in expected if actual.get(k) != expected[k]]
        raise ValueError(f"frozen input hash mismatch: {mismatches[:5]}")

    for relative in expected:
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path(relative), target)
    shutil.copy2(HERE / "interpret.py", output / "operator-studies/gnn-closedloop-confirm/interpret.py")
    requirements = HERE / "requirements.txt"
    shutil.copy2(requirements, output / "operator-studies/gnn-closedloop-confirm/requirements.txt")

    # The old confirmation freeze is deliberately not copied.  Regenerating it
    # in the fresh tree proves the copied files bind to one another.
    confirm_dir = output / "operator-studies/gnn-closedloop-confirm"
    subprocess.run(
        [sys.executable, "confirm.py", "freeze"],
        cwd=confirm_dir,
        check=True,
    )
    new_freeze = json.loads((confirm_dir / "freeze.json").read_text(encoding="utf-8"))
    if new_freeze.get("sha256") != expected:
        raise AssertionError("fresh freeze does not reproduce the frozen source hashes")
    receipt = {
        "schema": "gnn-closedloop-confirm-reproduction.v1",
        "source_freeze_sha256": digest(FREEZE),
        "copied_source_hashes": actual,
        "fresh_freeze_sha256": digest(confirm_dir / "freeze.json"),
        "output": str(output),
        "prepare_only": prepare_only,
        "workers": workers,
        "training_runs": 0,
    }
    if prepare_only:
        receipt["status"] = "prepared"
    else:
        subprocess.run(
            [sys.executable, "confirm.py", "run", "--workers", str(workers)],
            cwd=confirm_dir,
            check=True,
        )
        subprocess.run([sys.executable, "confirm.py", "analyse"], cwd=confirm_dir, check=True)
        subprocess.run([sys.executable, "interpret.py", "--summary", "results/summary.json", "--output", "results/interpretation.json"], cwd=confirm_dir, check=True)
        raw = json.loads((confirm_dir / "results/raw.json").read_text(encoding="utf-8"))
        summary = confirm_dir / "results/summary.json"
        if raw.get("n_runs") != 64 or not summary.is_file():
            raise AssertionError("completed reproduction did not produce 64 runs and a summary")
        receipt.update({
            "status": "completed",
            "training_runs": raw["n_runs"],
            "raw_sha256": digest(confirm_dir / "results/raw.json"),
            "summary_sha256": digest(summary),
        })
    (output / "reproduction-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true",
                        help="prepare files and freeze without training")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    receipt = prepare(args.output, prepare_only=args.prepare_only, workers=args.workers)
    print(json.dumps({"status": receipt["status"], "output": receipt["output"],
                      "training_runs": receipt["training_runs"]}))


if __name__ == "__main__":
    main()
