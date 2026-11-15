"""Offline benchmark fixtures for world-search proposals.

The benchmark deliberately has no model or network dependency.  It provides a
small executable target for candidate mechanisms before they are tried on a
Kaggle competition.  Validation mode never touches the test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits, make_classification
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier


SEED = 42
TABULAR_MISSING_LEVELS = (0.0, 0.10, 0.30)
DIGIT_NOISE_LEVELS = (0.0, 0.15, 0.30)
DIGIT_DROPOUT_LEVELS = (0.0, 0.15, 0.30)


def _hash_array(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def _split_indices(y: np.ndarray, seed: int = SEED) -> dict[str, np.ndarray]:
    idx = np.arange(len(y))
    train, rest = train_test_split(idx, test_size=0.40, random_state=seed, stratify=y)
    val, test = train_test_split(rest, test_size=0.50, random_state=seed + 1, stratify=y[rest])
    return {"train": np.sort(train), "validation": np.sort(val), "test": np.sort(test)}


def _tabular(seed: int = SEED) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    x, y = make_classification(
        n_samples=5000, n_features=20, n_informative=5, n_redundant=5,
        weights=[0.8, 0.2], class_sep=1.0, random_state=seed,
    )
    return x.astype(np.float32), y.astype(np.int64), _split_indices(y, seed)


def _digits(seed: int = SEED) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    data = load_digits()
    return data.data.astype(np.float32), data.target.astype(np.int64), _split_indices(data.target, seed)


def _mask(x: np.ndarray, level: float, seed: int) -> np.ndarray:
    if level == 0:
        return x.copy()
    rng = np.random.default_rng(seed)
    out = x.copy()
    out[rng.random(out.shape) < level] = np.nan
    return out


def _digit_corrupt(x: np.ndarray, noise: float, dropout: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = x.copy()
    if noise:
        out += rng.normal(0, noise * 16.0, size=out.shape).astype(np.float32)
    if dropout:
        out[rng.random(out.shape) < dropout] = 0.0
    return np.clip(out, 0.0, 16.0)


def _fit_tabular(x: np.ndarray, y: np.ndarray, idx: np.ndarray, augmented: bool):
    train_x = x[idx]
    train_y = y[idx]
    if augmented:
        # Train-only missingness augmentation.  The validation/test rows are
        # never used to choose masks or fit this transform.
        rng = np.random.default_rng(SEED + 101)
        masked = train_x.copy()
        masked[rng.random(masked.shape) < 0.20] = np.nan
        train_x = np.concatenate([train_x, masked], axis=0)
        train_y = np.concatenate([train_y, train_y], axis=0)
    model = make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingClassifier(random_state=SEED, max_iter=120))
    model.fit(train_x, train_y)
    return model


def _fit_digits(x: np.ndarray, y: np.ndarray, idx: np.ndarray, augmented: bool):
    train_x = x[idx]
    train_y = y[idx]
    if augmented:
        rng = np.random.default_rng(SEED + 202)
        noise = np.clip(train_x + rng.normal(0, 0.15 * 16.0, train_x.shape), 0, 16)
        dropout = train_x.copy()
        dropout[rng.random(dropout.shape) < 0.15] = 0
        train_x = np.concatenate([train_x, noise, dropout])
        train_y = np.tile(train_y, 3)
    model = make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=False, random_state=SEED))
    model.fit(train_x, train_y)
    return model


def run(mode: str, output: Path, seed: int = SEED) -> dict:
    if mode not in {"validation", "final"}:
        raise ValueError(mode)
    started = time.perf_counter()
    tx, ty, ts = _tabular(seed)
    dx, dy, ds = _digits(seed)
    if mode == "validation":
        # The test indices are constructed and hashed for reproducibility, but
        # no model prediction or test metric is computed in this branch.
        tab_eval, dig_eval = ts["validation"], ds["validation"]
    else:
        tab_eval, dig_eval = ts["test"], ds["test"]
    tab_models = {"median_hgb": _fit_tabular(tx, ty, ts["train"], False),
                  "missing_augmentation": _fit_tabular(tx, ty, ts["train"], True)}
    dig_models = {"svm": _fit_digits(dx, dy, ds["train"], False),
                  "corruption_augmentation": _fit_digits(dx, dy, ds["train"], True)}
    rows = []
    for name, model in tab_models.items():
        for level in TABULAR_MISSING_LEVELS:
            pred = model.predict_proba(_mask(tx[tab_eval], level, seed + 10_000 + int(level * 100)))[:, 1]
            rows.append({"task": "tabular_missing", "method": name, "condition": {"missing": level}, "metric": "average_precision", "value": float(average_precision_score(ty[tab_eval], pred))})
    for name, model in dig_models.items():
        for noise in DIGIT_NOISE_LEVELS:
            for dropout in DIGIT_DROPOUT_LEVELS:
                data = _digit_corrupt(dx[dig_eval], noise, dropout, seed + 20_000 + int(noise * 1000) + int(dropout * 1000))
                rows.append({"task": "digits_shift", "method": name, "condition": {"noise": noise, "dropout": dropout}, "metric": "accuracy", "value": float(accuracy_score(dy[dig_eval], model.predict(data)))})
    payload = {"config": {"mode": mode, "seed": seed, "tabular_missing_levels": TABULAR_MISSING_LEVELS, "digit_noise_levels": DIGIT_NOISE_LEVELS, "digit_dropout_levels": DIGIT_DROPOUT_LEVELS, "train_only_transforms": True, "test_evaluated": mode == "final", "sklearn": __import__("sklearn").__version__, "python": sys.version, "platform": platform.platform()}, "splits": {"tabular": {k: _hash_array(v) for k, v in ts.items()}, "digits": {k: _hash_array(v) for k, v in ds.items()}}, "results": rows, "elapsed_seconds": time.perf_counter() - started}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["validation", "final"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.mode, args.output)
    print(json.dumps({"mode": args.mode, "n_results": len(result["results"]), "elapsed_seconds": result["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
