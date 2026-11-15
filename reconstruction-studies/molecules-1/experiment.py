#!/usr/bin/env python3
"""Fixed validation/final reconstruction for molecules-1; final is parent-gated."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).resolve().parent
FIXTURE = ROOT / "scripts" / "missingness_shift_fixture.py"
SEEDS = (9401, 9403, 9407)
R = 3
K = 8
MODEL = dict(max_iter=80, max_leaf_nodes=15, min_samples_leaf=20,
             learning_rate=0.08, early_stopping=False, random_state=47)
METHODS = (
    "anchor_full", "uniform_augmentation", "native_nan_duplicate",
    "impute_indicators", "anchor_train_only", "anchor_test_only",
    "anchor_k1", "permuted_anchor_rates",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def model(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(**{**MODEL, "random_state": seed})


def estimate_rate_model(x: np.ndarray, *, permute: bool, seed: int) -> tuple[np.ndarray, float]:
    # The mask is read from the already-masked training matrix only.
    anchors = np.asarray(x[:, :2], dtype=float).copy()
    missing_fraction = np.isnan(x[:, 2:]).mean(axis=1)
    if permute:
        rng = np.random.default_rng(seed)
        anchors = anchors[rng.permutation(len(anchors))]
    # Rank-based quantile bins avoid unstable duplicate quantile cut points.
    score = (anchors[:, 0] - np.nanmean(anchors[:, 0])) / (np.nanstd(anchors[:, 0]) + 1e-12)
    score += (anchors[:, 1] - np.nanmean(anchors[:, 1])) / (np.nanstd(anchors[:, 1]) + 1e-12)
    order = np.argsort(np.argsort(score, kind="mergesort"), kind="mergesort")
    bins = np.minimum(4, order * 5 // len(order))
    rates = np.array([missing_fraction[bins == j].mean() for j in range(5)], dtype=float)
    return rates, float(missing_fraction.mean())


def rates_for(x: np.ndarray, rates: np.ndarray) -> np.ndarray:
    score = (x[:, 0] - np.nanmean(x[:, 0])) / (np.nanstd(x[:, 0]) + 1e-12)
    score += (x[:, 1] - np.nanmean(x[:, 1])) / (np.nanstd(x[:, 1]) + 1e-12)
    # Training-defined cut points are represented by empirical anchor-score
    # quintiles. For prediction, use the score's empirical percentile against
    # training scores (saved in closure by make_rate_function below).
    return rates[np.minimum(4, np.maximum(0, np.floor((score + 4.0) * 5 / 8).astype(int)))]


def make_rate_function(train_x: np.ndarray, rates: np.ndarray):
    def score(x):
        a0 = (x[:, 0] - train_x[:, 0].mean()) / (train_x[:, 0].std() + 1e-12)
        a1 = (x[:, 1] - train_x[:, 1].mean()) / (train_x[:, 1].std() + 1e-12)
        return a0 + a1
    train_score = score(train_x)
    cutpoints = np.quantile(train_score, [0.2, 0.4, 0.6, 0.8])
    def fn(x):
        return rates[np.digitize(score(x), cutpoints)]
    return fn


def add_mask(x: np.ndarray, row_rates: np.ndarray, rng: np.random.Generator,
             multiplier: bool = True) -> np.ndarray:
    out = x.copy()
    u = rng.uniform(0.5, 2.0, len(x)) if multiplier else np.ones(len(x))
    p = np.clip(row_rates * u, 0.0, 1.0)
    observed = np.isfinite(out[:, 2:])
    hide = (rng.random(out[:, 2:].shape) < p[:, None]) & observed
    block = out[:, 2:]
    block[hide] = np.nan
    out[:, 2:] = block
    return out


def augmented(x, y, rate_fn, global_rate, style, seed):
    copies = [x]
    labels = [y]
    rng = np.random.default_rng(seed)
    for _ in range(R):
        if style == "anchor":
            rate = rate_fn(x)
        else:
            rate = np.full(len(x), global_rate)
        copies.append(add_mask(x, rate, rng))
        labels.append(y)
    return np.vstack(copies), np.concatenate(labels)


def prediction(model_, x, rate_fn, global_rate, method, seed):
    # This API accepts masked X only. It has no condition name and no labels.
    if method in {"anchor_full", "uniform_augmentation", "anchor_test_only", "anchor_k1", "permuted_anchor_rates"}:
        k = 1 if method == "anchor_k1" else K
        rng = np.random.default_rng(seed + 5000)
        preds = []
        for _ in range(k):
            rate = np.full(len(x), global_rate) if method == "uniform_augmentation" else rate_fn(x)
            perturbed = add_mask(x, rate, rng)
            preds.append(model_.predict_proba(perturbed)[:, 1])
        return np.mean(preds, axis=0)
    return model_.predict_proba(x)[:, 1]


def run_seed(seed: int, mode: str):
    # Import only the documented public fixture API. Validation mode exposes
    # no test arrays or labels.
    import importlib.util
    spec = importlib.util.spec_from_file_location("missingness_shift_fixture", FIXTURE)
    fixture_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = fixture_module
    spec.loader.exec_module(fixture_module)
    fixture = fixture_module.load_fixture(seed, mode=mode)
    x, y = fixture.train_x, fixture.train_y
    evaluation_cases = fixture.validation_cases if mode == "validation" else fixture.final_cases
    if evaluation_cases is None:
        raise RuntimeError("final fixture did not provide final_cases")
    out = {"seed": seed, "mode": mode, "fixture_metadata": fixture.metadata,
           "evaluation_conditions": list(evaluation_cases), "methods": {}, "errors": []}
    rates, global_rate = estimate_rate_model(x, permute=False, seed=seed)
    rate_fn = make_rate_function(x, rates)
    perm_rates, _ = estimate_rate_model(x, permute=True, seed=seed + 17)
    perm_rate_fn = make_rate_function(x, perm_rates)
    for method_name in METHODS:
        started = time.perf_counter()
        try:
            if method_name in ("anchor_full", "anchor_k1", "anchor_train_only", "uniform_augmentation", "permuted_anchor_rates"):
                style = "uniform" if method_name == "uniform_augmentation" else "anchor"
                fit_x, fit_y = augmented(x, y, perm_rate_fn if method_name == "permuted_anchor_rates" else rate_fn,
                                         global_rate, style, seed + 100)
            elif method_name == "anchor_train_only":
                fit_x, fit_y = augmented(x, y, rate_fn, global_rate, "anchor", seed + 100)
            else:
                fit_x, fit_y = np.tile(x, (R + 1, 1)), np.tile(y, R + 1)
            imputer = None
            if method_name == "impute_indicators":
                imputer = SimpleImputer(strategy="mean", add_indicator=True)
                # Fit imputation statistics on the original training rows only.
                imputer.fit(x)
                fit_x = imputer.transform(fit_x)
            fitted = model(seed).fit(fit_x, fit_y)
            cases = {}
            for ci, (condition, (vx, vy)) in enumerate(evaluation_cases.items()):
                pred_input = imputer.transform(vx) if imputer is not None else vx
                p = prediction(fitted, pred_input,
                               perm_rate_fn if method_name == "permuted_anchor_rates" else rate_fn,
                               global_rate, method_name, seed + ci * 31)
                cases[condition] = {"ap": fixture_module.evaluate(vy, p),
                                    "brier": fixture_module.evaluate(vy, p, "brier")}
            out["methods"][method_name] = {
                "cases": cases,
                "mean_ap_all": float(np.mean([v["ap"] for v in cases.values()])),
                "mean_ap_shifted": float(np.mean([v["ap"] for k, v in cases.items()
                                                   if k not in {"clean", "mcar_0.2"}])),
                "mean_brier": float(np.mean([v["brier"] for v in cases.values()])),
                "runtime_seconds": time.perf_counter() - started,
                "n_fit_rows": int(len(fit_y)),
                "rate_bins": rates.tolist() if method_name != "permuted_anchor_rates" else perm_rates.tolist(),
            }
        except Exception as exc:
            out["errors"].append({"method": method_name, "error": repr(exc)})
    out["data_hashes"] = fixture.metadata["fixture_hashes"]
    if mode == "final":
        out["final_case_hashes"] = fixture.metadata["final_case_hashes"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("validation", "final"), required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.mode == "final" and args.output.exists():
        raise SystemExit(f"refusing to overwrite final output: {args.output}")
    if args.mode == "validation" and args.output.exists() and args.output.name != "validation.json":
        raise SystemExit(f"refusing to overwrite unexpected output: {args.output}")
    if args.mode == "final":
        approval_path = STUDY / "parent-approved-final.json"
        if not approval_path.exists():
            raise SystemExit("refusing final fixture access without parent-approved-final.json")
        approval = json.loads(approval_path.read_text())
        if approval.get("code_sha256_before_final") != sha256(Path(__file__)):
            raise SystemExit("refusing final fixture access: experiment.py hash differs from approval record")
        if approval.get("fixture_sha256") != sha256(FIXTURE):
            raise SystemExit("refusing final fixture access: shared fixture hash differs from approval record")
        val_path = STUDY / "validation.json"
        if approval.get("validation_artifact_sha256") != sha256(val_path):
            raise SystemExit("refusing final fixture access: validation artifact hash differs from approval record")
    results = []
    with threadpool_limits(limits=1):
        for seed in SEEDS:
            results.append(run_seed(seed, args.mode))
    payload = {
        "study_id": "molecules-1", "mode": args.mode, "methods": list(METHODS),
        "fixed_model": MODEL, "remask_copies": R, "prediction_masks": K,
        "seed_results": results,
        "source_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in
                          (FIXTURE, STUDY / "proposal.json", STUDY / "experiment.py", ROOT / "results/atomization-study-v1/expanded-papers.json", ROOT / "results/atomization-study-v1/context-atoms.json")},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "seed_count": len(results), "errors": sum(len(r["errors"]) for r in results)}, indent=2))


if __name__ == "__main__":
    main()
