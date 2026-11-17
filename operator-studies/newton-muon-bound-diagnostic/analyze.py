"""Compare valid PSD spectral upper bounds on frozen Newton--Muon captures."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

POLY = Path(__file__).resolve().parents[1] / "newton-muon-polynomial"
PLANS = json.loads((POLY / "plans.json").read_text())["plans"]


def choose(k: np.ndarray, upper: float) -> dict:
    n = k.shape[0]
    delta = float(0.2 * np.trace(k) / n)
    epsilon = float(delta / (upper + delta))
    eligible = [
        (i, p) for i, p in enumerate(PLANS)
        if p["epsilon"] <= epsilon and p["total_sypp"] <= 13 and p["s_out"] <= 0.01
    ]
    selected = min(eligible, key=lambda x: (x[1]["total_sypp"], x[1]["s_out"], x[0])) if eligible else None
    return {
        "upper": float(upper), "delta": delta, "epsilon": epsilon,
        "plan_index": selected[0] if selected else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite result")
    rows = []
    source_files = []
    for width in (256, 512):
        manifest_path = args.capture / f"d{width}.manifest.json"
        npz_path = args.capture / f"d{width}.npz"
        manifest = json.loads(manifest_path.read_text())
        digest = hashlib.sha256(npz_path.read_bytes()).hexdigest()
        if digest != manifest["npz_sha256"]:
            raise ValueError(f"capture hash mismatch for width {width}")
        source_files.append({"width": width, "npz_sha256": digest, "source": manifest["source"]})
        with np.load(npz_path, allow_pickle=False) as arrays:
            for name, meta in manifest["arrays"].items():
                for block, raw in enumerate(arrays[meta["K_key"]]):
                    k = np.asarray(raw, dtype=np.float64)
                    asym = float(np.linalg.norm(k - k.T, "fro") / max(np.linalg.norm(k, "fro"), 1e-30))
                    if asym > 1e-5:
                        raise ValueError(f"materially asymmetric K: {width} {name} {block}")
                    k = (k + k.T) * 0.5
                    fro = float(np.linalg.norm(k, "fro"))
                    row = float(np.max(np.sum(np.abs(k), axis=1)))
                    trace = float(np.trace(k))
                    eigenvalues = np.linalg.eigvalsh(k)
                    min_eig = float(eigenvalues[0])
                    if min_eig < -1e-6 * max(fro, 1e-30):
                        raise ValueError(f"PSD check failed: {width} {name} {block}")
                    bounds = {
                        "frobenius": 1.001 * fro,
                        "row_sum": 1.001 * row,
                        "trace": 1.001 * trace,
                        "eigvalsh_oracle": 1.001 * float(eigenvalues[-1]),
                    }
                    selections = {key: choose(k, value) for key, value in bounds.items()}
                    rows.append({
                        "width": width, "module": name, "block": block,
                        "relative_asymmetry": asym, "minimum_eigenvalue": min_eig,
                        "frobenius": fro, "max_abs_row_sum": row, "trace": trace,
                        "lambda_max_float64": float(eigenvalues[-1]),
                        "selections": selections,
                    })
    coverage = {
        key: sum(row["selections"][key]["plan_index"] is not None for row in rows)
        for key in ("frobenius", "row_sum", "trace", "eigvalsh_oracle")
    }
    out = {
        "schema": "newton-muon-bound-diagnostic.v1",
        "scope": "CPU float64 bound comparison on frozen captures; no runtime certificate, timing, training-quality, or novelty claim",
        "filters": {"epsilon_at_least_table": True, "sypp_budget": 13, "reported_residual_at_most": 0.01},
        "bounds": {
            "frobenius": "1.001 * ||K||_F (original policy)",
            "row_sum": "1.001 * max_i sum_j |K_ij| (valid induced-norm bound for symmetric K)",
            "trace": "1.001 * trace(K) (valid for PSD K)",
            "eigvalsh_oracle": "1.001 * largest float64 eigvalsh value (offline diagnostic only)",
        },
        "sources": source_files, "rows": rows,
        "summary": {"blocks": len(rows), "coverage": coverage,
                    "uncovered": {key: len(rows) - value for key, value in coverage.items()},
                    "all_psd_checked": True, "max_relative_asymmetry": max(r["relative_asymmetry"] for r in rows)},
        "hashes": {"analyze.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "plans.json": hashlib.sha256((POLY / "plans.json").read_bytes()).hexdigest()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
