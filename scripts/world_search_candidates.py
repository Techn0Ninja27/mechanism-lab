"""Validation-only executable tests for three selected world-search candidates.

The candidate IDs and proposals are fixed before this script is run.  This
script never reads or scores the test split; a separate final mode should only
be enabled after the validation configuration is frozen by the parent harness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.datasets import load_digits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.world_search_benchmark import _split_indices, _digit_corrupt, SEED


IDS = ("digits_shift-direct-3", "digits_shift-random_seed-2", "digits_shift-world-3")
VAL_CONDITIONS = ((0.0, 0.0), (0.2, 0.0), (0.4, 0.0), (0.6, 0.0),
                  (0.0, 0.05), (0.0, 0.15), (0.0, 0.30), (0.4, 0.15), (0.6, 0.30))


def _load() -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    d = load_digits()
    x, y = d.data.astype(np.float32), d.target.astype(np.int64)
    return x, y, _split_indices(y, SEED)


def _augment(x: np.ndarray, y: np.ndarray, mode: str, seed: int = 42):
    copies = []
    for i, noise in enumerate((0.15, 0.30)):
        rng = np.random.default_rng(seed + i)
        z = np.clip(x + rng.normal(0, noise * 16, x.shape), 0, 16)
        use_block = mode == "block" or (mode == "mixed" and i == 0)
        for row in z:
            if use_block:
                mask = np.zeros((8,8), dtype=bool)
                while mask.sum() < 8:
                    a,b = rng.integers(0,7,size=2)
                    if not mask[a:a+2,b:b+2].any(): mask[a:a+2,b:b+2] = True
                row[mask.ravel()] = 0
            else:
                row[rng.choice(64,8,replace=False)] = 0
        copies.append(z)
    return np.concatenate([x, *copies]), np.tile(y, len(copies) + 1)


def _corrupt_eval(x: np.ndarray, noise: float, dropout: float, seed: int):
    return _digit_corrupt(x, noise, dropout, seed)


def _score(model, x, y):
    return float(accuracy_score(y, model.predict(x)))


def _mean_conditions(model, scaler, x, y, indices):
    vals = []
    for k, (noise, dropout) in enumerate(VAL_CONDITIONS):
        z = _corrupt_eval(x[indices], noise, dropout, 70_000 + k)
        vals.append(_score(model, scaler.transform(z), y[indices]))
    return float(np.mean(vals)), vals


def _fit_svm(x, y, kind="ordinary", pca=False):
    xa, ya = _augment(x, y, kind)
    scaler = StandardScaler().fit(xa)
    z = scaler.transform(xa).astype(np.float64, copy=False)
    if pca:
        model = make_pipeline(PCA(n_components=32, random_state=SEED), SVC(C=10, gamma="scale", kernel="rbf"))
    else:
        model = SVC(C=10, gamma="scale", kernel="rbf")
    model.fit(z, ya)
    return model, scaler


def _direct(x, y, train, val, sigmas=(0.0, 0.4, 0.6, 0.8, 1.0)):
    rows = []
    for sigma in sigmas:
        shape = lambda z: gaussian_filter(z.reshape(-1, 8, 8), sigma=(0, sigma, sigma)).reshape(-1, 64)
        raw_xa, ya = _augment(x[train], y[train], "ordinary")
        xa = shape(raw_xa)
        scaler = StandardScaler().fit(xa)
        model = SVC(C=10, gamma="scale", kernel="rbf").fit(scaler.transform(xa), ya)
        vals = []
        for k, (noise, dropout) in enumerate(VAL_CONDITIONS):
            z = shape(_corrupt_eval(x[val], noise, dropout, 70_000 + k))
            vals.append(_score(model, scaler.transform(z), y[val]))
        rows.append({"variant": f"gaussian_sigma_{sigma}", "mean_accuracy": float(np.mean(vals)), "conditions": vals})
    return rows


def _random_seed(x, y, train, val):
    def phases(kind):
        if kind == "sinusoidal": return [0.5 * (1 + np.sin(2 * np.pi * i / 8)) for i in range(8)]
        if kind == "uniform": return [i / 7 for i in range(8)]
        return [0.5]
    rows = []
    for kind in ("sinusoidal", "uniform", "constant_mid", "single_mid"):
        ph = phases("constant" if kind == "constant_mid" else "single" if kind == "single_mid" else kind)
        if kind == "constant_mid":
            ph = [0.5] * 8
        rng = np.random.default_rng(0)
        parts, labels = [x[train]], [y[train]]
        for s in ph:
            parts.append(_digit_corrupt(x[train], 0.5 * s, 0.15 * s, int(rng.integers(1_000_000))))
            labels.append(y[train])
        xa, ya = np.concatenate(parts), np.concatenate(labels)
        scaler = StandardScaler().fit(x[train])
        model = SVC(C=10, gamma="scale", kernel="rbf").fit(scaler.transform(xa), ya)
        vals = [_score(model, scaler.transform(_corrupt_eval(x[val], n, d, 70_000 + i)), y[val]) for i, (n, d) in enumerate(VAL_CONDITIONS)]
        rows.append({"variant": kind, "mean_accuracy": float(np.mean(vals)), "conditions": vals, "n_train": len(xa)})
    return rows


def _world(x, y, train, val, weight_indices=None):
    weight_indices = val if weight_indices is None else weight_indices
    models = []
    for name, aug, pca in (("random_dropout", "ordinary", False), ("block_dropout", "block", False), ("mixed", "mixed", False)):
        model, scaler = _fit_svm(x[train], y[train], aug, pca)
        models.append((name, model, scaler))
    individual = []
    for name, model, scaler in models:
        mean, vals = _mean_conditions(model, scaler, x, y, val)
        individual.append({"variant": name, "mean_accuracy": mean, "conditions": vals})
    # Margin weights are mean top-1 minus top-2 decision gaps on validation.
    # SVC scores are not calibrated across pipelines, so the final ensemble
    # uses these weights only for hard-label voting.
    margins = []
    for _, model, scaler in models:
        gaps = []
        for k, (noise, dropout) in enumerate(VAL_CONDITIONS):
            scores = model.decision_function(scaler.transform(_corrupt_eval(x[weight_indices], noise, dropout, 70_000 + k)))
            ordered = np.sort(scores, axis=1)
            gaps.extend((ordered[:, -1] - ordered[:, -2]).tolist())
        margins.append(float(np.mean(gaps)))
    weights = np.maximum(np.asarray(margins), 1e-6)
    weights /= weights.sum()
    weighted_vals, uniform_vals = [], []
    for k, (noise, dropout) in enumerate(VAL_CONDITIONS):
        z = _corrupt_eval(x[val], noise, dropout, 70_000 + k)
        preds = np.stack([m.predict(s.transform(z)) for _, m, s in models], axis=0)
        weighted_vals.append(float(accuracy_score(y[val], np.apply_along_axis(lambda a: np.bincount(a, weights=weights, minlength=10).argmax(), 0, preds))))
        uniform_vals.append(float(accuracy_score(y[val], np.apply_along_axis(lambda a: np.bincount(a, minlength=10).argmax(), 0, preds))))
    return {"individual": individual, "ensemble": [{"variant": "margin_weighted_label_vote", "mean_accuracy": float(np.mean(weighted_vals)), "conditions": weighted_vals, "weights": weights.tolist()}, {"variant": "uniform_label_vote", "mean_accuracy": float(np.mean(uniform_vals)), "conditions": uniform_vals}]}


def run(output: Path, mode="validation", selection=None) -> dict:
    x, y, splits = _load()
    train, val = splits["train"], splits["validation"]
    sigmas = (0.0, 0.4, 0.6, 0.8, 1.0)
    if mode == "final":
        if not selection: raise ValueError("Final evaluation requires frozen validation result")
        previous = json.loads(Path(selection).read_text())
        best = max(previous["results"]["direct"], key=lambda r:r["mean_accuracy"])
        sigma = float(best["variant"].split("_")[-1])
        sigmas = tuple(sorted({0.0, sigma}))
        val = splits["test"]
    proposals = {}
    source = Path("results/world-search-pilot/candidates")
    for ident in IDS:
        proposals[ident] = json.loads((source / f"{ident}.json").read_text())["proposal"]
    start = time.perf_counter()
    result = {"config": {"mode": mode, "selection_file": str(selection), "seed": SEED, "selected_ids": IDS, "test_evaluated": mode == "final", "conditions": VAL_CONDITIONS, "explicit_deviations": ["Final mode selects Gaussian sigma from frozen validation results; ensemble weights use validation rows only.", "Candidate 03 uses validation top-1/top-2 SVC decision gaps as margin weights and weighted hard-label votes because scores are not calibrated across pipelines.", "Candidate 02's stated four-condition metric is expanded to the shared nine-condition grid.", "Matched controls: two augmented copies, same Gaussian noise levels and exactly eight dropped pixels per image; geometry alone changes random vs two disjoint2x2 blocks. No PCA. Fixed train set; no refit on validation."], "candidate_proposals": proposals}, "split_hashes": {k: hashlib.sha256(np.asarray(v).tobytes()).hexdigest() for k, v in splits.items()}, "results": {"direct": _direct(x, y, train, val, sigmas), "random_seed": _random_seed(x, y, train, val), "world": _world(x, y, train, val, splits["validation"])}, "elapsed_seconds": time.perf_counter() - start}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--output", type=Path, required=True); p.add_argument("--mode", choices=["validation","final"],default="validation"); p.add_argument("--selection",type=Path); args = p.parse_args(); r = run(args.output,args.mode,args.selection); print(json.dumps({"test_evaluated": r["config"]["test_evaluated"], "elapsed_seconds": r["elapsed_seconds"]}, indent=2))
