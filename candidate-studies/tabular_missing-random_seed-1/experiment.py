#!/usr/bin/env python3
"""Faithful pilot for mirrored missingness augmentation on tabular data.

Validation uses the validation split only. Final mode requires a preregistration
and refuses to overwrite existing frozen output unless --rerun-final is passed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

SEEDS = (137, 271, 389)
LEVELS = (0.0, 0.10, 0.30, 0.50)
METHODS = ("median_hgb", "native_hgb", "mirror_mask", "mirror_no_mask", "mask_only")
PARAMS = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=42)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def arr_hash(a: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(a).tobytes())


def data(seed: int):
    x, y = make_classification(n_samples=5000, n_features=20, n_informative=5,
        n_redundant=5, weights=[0.8, 0.2], class_sep=1.0, random_state=seed)
    ix = np.arange(len(y))
    tr, rem = train_test_split(ix, test_size=0.4, random_state=seed, stratify=y)
    va, te = train_test_split(rem, test_size=0.5, random_state=seed + 1, stratify=y[rem])
    return x.astype(np.float64), y.astype(np.int64), {"train": np.sort(tr), "validation": np.sort(va), "test": np.sort(te)}


def model():
    return HistGradientBoostingClassifier(**PARAMS)


def mirror_augment(x, y, seed, include_masks=True):
    rng = np.random.default_rng(seed)
    p = x.shape[1]
    parts = [np.hstack((x, np.zeros((len(x), p)))) if include_masks else x.copy()]
    ys = [y]
    for _ in range(4):
        rates = rng.beta(0.5, 2.0, size=p)
        mask = rng.random(x.shape) < rates
        xc = x.copy()
        xc[mask] = np.nan
        parts.append(np.hstack((xc, mask.astype(float))) if include_masks else xc)
        ys.append(y)
    return np.vstack(parts), np.concatenate(ys)


def corrupt_eval(x, level, seed, with_mask=True):
    if level == 0:
        mask = np.zeros(x.shape, dtype=bool)
    else:
        mask = np.random.default_rng(seed).random(x.shape) < level
    z = x.copy()
    z[mask] = np.nan
    return np.hstack((z, mask.astype(float))) if with_mask else z


def fit_method(name, x, y, idx, seed):
    xt, yt = x[idx], y[idx]
    t0 = time.perf_counter()
    if name == "median_hgb":
        imputer = SimpleImputer(strategy="median")
        z = imputer.fit_transform(xt)
        est = model().fit(z, yt)
        predictor = lambda xx: est.predict_proba(imputer.transform(xx))[:, 1]
    elif name == "native_hgb":
        est = model().fit(xt, yt)
        predictor = lambda xx: est.predict_proba(xx)[:, 1]
    elif name in ("mirror_mask", "mirror_no_mask"):
        ax, ay = mirror_augment(xt, yt, seed + 707, name == "mirror_mask")
        est = model().fit(ax, ay)
        predictor = (lambda xx: est.predict_proba(xx)[:, 1])
    elif name == "mask_only":
        ax, ay = mirror_augment(xt, yt, seed + 707, True)
        # Ablation: preserve the mask path but replace all observed feature values
        # with train medians, isolating information available from masks alone.
        med = np.nanmedian(xt, axis=0)
        ax[:, :20] = np.where(np.isfinite(ax[:, :20]), med, np.nan)
        est = model().fit(ax, ay)
        def predictor(xx):
            xx = xx.copy()
            observed = np.isfinite(xx[:, :20])
            xx[:, :20] = np.where(observed, med[None, :], np.nan)
            return est.predict_proba(xx)[:, 1]
    else:
        raise ValueError(name)
    return predictor, time.perf_counter() - t0


def run(mode, output: Path, rerun_final=False):
    if mode not in ("validation", "final"):
        raise ValueError(mode)
    if mode == "final":
        prereg = output.parent / "preregistration.json"
        if not prereg.exists():
            raise RuntimeError("final mode requires preregistration.json")
        if output.exists() and not rerun_final:
            raise RuntimeError(f"refusing to overwrite frozen result: {output}")
    started = time.perf_counter()
    rows, split_hashes = [], {}
    for seed in SEEDS:
        x, y, splits = data(seed)
        split_hashes[str(seed)] = {k: arr_hash(v) for k, v in splits.items()}
        eval_idx = splits["validation" if mode == "validation" else "test"]
        fitted = {}
        for name in METHODS:
            fitted[name] = fit_method(name, x, y, splits["train"], seed)
        # Identical per-seed and per-level MCAR draws are shared across methods.
        eval_masks = {lev: (np.zeros((len(eval_idx), 20), bool) if lev == 0 else
            np.random.default_rng(seed + 10000 + int(lev * 100)).random((len(eval_idx), 20)) < lev)
            for lev in LEVELS}
        for name, (pred_fn, train_s) in fitted.items():
            infer_start = time.perf_counter()
            for lev in LEVELS:
                mask = eval_masks[lev]
                z = x[eval_idx].copy()
                z[mask] = np.nan
                if name in ("mirror_mask", "mask_only"):
                    z = np.hstack((z, mask.astype(float)))
                scores = pred_fn(z)
                row = {"seed": seed, "method": name, "missing_rate": lev,
                    "average_precision": float(average_precision_score(y[eval_idx], scores)),
                    "n_eval": int(len(eval_idx)), "train_seconds": float(train_s)}
                if name == "mirror_mask":
                    # Keep NaN positions fixed while shuffling explicit indicator
                    # values across rows, as the proposal's mask-path ablation.
                    shuffled = z.copy()
                    rng = np.random.default_rng(seed + 50000 + int(lev * 1000))
                    for j in range(20):
                        shuffled[:, 20 + j] = rng.permutation(shuffled[:, 20 + j])
                    shuffled_scores = pred_fn(shuffled)
                    row["permuted_mask_average_precision"] = float(
                        average_precision_score(y[eval_idx], shuffled_scores))
                    row["mask_permutation_ap_drop"] = row["average_precision"] - row["permuted_mask_average_precision"]
                rows.append(row)
            infer_s = time.perf_counter() - infer_start
            for row in rows:
                if row["seed"] == seed and row["method"] == name:
                    row["inference_seconds_all_conditions"] = float(infer_s)
    payload = {"config": {"mode": mode, "seeds": SEEDS, "conditions": LEVELS,
        "metric": "mean average precision over clean, 10%, 30%, 50% MCAR",
        "dataset": "make_classification(5000,20,5 informative,5 redundant, weights .8/.2)",
        "split": "stratified 60/20/20; validation/test each from remaining 40%, random_state seed then seed+1",
        "model_params": PARAMS, "methods": METHODS, "threads": 1, "sklearn": sklearn.__version__,
        "python": sys.version, "platform": platform.platform(), "test_evaluated": mode == "final"},
        "split_hashes": split_hashes, "results": rows, "elapsed_seconds": time.perf_counter() - started}
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not (mode == "final" and rerun_final):
        raise RuntimeError(f"refusing to overwrite existing output: {output}")
    output.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["validation", "final"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rerun-final", action="store_true")
    a = p.parse_args()
    with threadpool_limits(limits=1):
        r = run(a.mode, a.output, a.rerun_final)
    print(json.dumps({"mode": a.mode, "rows": len(r["results"]), "elapsed_seconds": r["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
