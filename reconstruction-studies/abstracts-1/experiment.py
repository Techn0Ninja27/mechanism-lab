#!/usr/bin/env python3
"""Phase-one validation for abstracts-1; final mode is gated and one-shot."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import time
import traceback

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
import missingness_shift_fixture as fixture_module

BASE = dict(max_iter=100, max_leaf_nodes=15, learning_rate=0.1,
            l2_regularization=1.0, early_stopping=False)
K = 8
SEEDS = (9401, 9403, 9407)
FINAL_PATH = HERE / "results.json"
VALIDATION_PATH = HERE / "validation.json"


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(path: pathlib.Path) -> str:
    return sha_bytes(path.read_bytes())


def arr_hash(a: np.ndarray) -> str:
    return sha_bytes(np.ascontiguousarray(a).tobytes())


def model_cfg(seed: int, **extra):
    return {**BASE, "random_state": int(seed), **extra}


def make_views(x: np.ndarray, seed: int, kinds: tuple[str, ...] = ("mcar", "group")) -> list[np.ndarray]:
    """Hide only finite values in the already-masked training matrix."""
    rng = np.random.default_rng(seed + 17001)
    views = []
    for k in range(K):
        kind = kinds[k % len(kinds)]
        out = x.copy()
        if kind == "mcar":
            hide = rng.random(x.shape) < 0.20
            hide[:, :2] = False
            out[hide & np.isfinite(x)] = np.nan
        elif kind == "group":
            # Two non-anchor contiguous feature groups; hidden cells remain hidden.
            start = int(rng.integers(2, x.shape[1] - 2))
            width = int(rng.integers(2, min(6, x.shape[1] - start) + 1))
            rows = rng.random(len(x)) < 0.5
            take = np.zeros(x.shape, dtype=bool)
            take[rows, start:start + width] = True
            out[take & np.isfinite(x)] = np.nan
        else:
            raise ValueError(kind)
        views.append(out)
    return views


def fit_teacher(x, y, views, seed):
    xx = np.vstack([x, *views])
    yy = np.tile(y, len(views) + 1)
    model = HistGradientBoostingClassifier(**model_cfg(seed))
    model.fit(xx, yy)
    return model, xx, yy


def predict(model, x):
    """Prediction takes features only; no labels or condition identifiers."""
    return np.asarray(model.predict_proba(x)[:, 1], dtype=float)


def candidate_predictions(cases, views, teacher, train_x, y, seed, target_views=K,
                          raw_targets=False):
    if raw_targets:
        targets = y.astype(float)
    else:
        p = np.vstack([predict(teacher, views[k]) for k in range(target_views)])
        targets = p.mean(axis=0)
    eps = 1e-5
    logits = np.log(np.clip(targets, eps, 1 - eps) / (1 - np.clip(targets, eps, 1 - eps)))
    student = HistGradientBoostingRegressor(**model_cfg(seed + 11))
    student.fit(train_x, logits)
    result = {}
    for name, (vx, _) in cases.items():
        pred_logit = np.clip(student.predict(vx), -30, 30)
        result[name] = 1.0 / (1.0 + np.exp(-pred_logit))
    return result


def eval_predictions(cases, predictions):
    ans = {}
    for name, (xx, yy) in cases.items():
        # Metrics are evaluated outside predict(); labels never enter a model.
        p = predictions[name]
        ans[name] = {"ap": fixture_module.evaluate(yy, p, "average_precision"),
                     "brier": fixture_module.evaluate(yy, p, "brier")}
    return ans


def one_seed(seed: int, mode: str):
    started = time.perf_counter()
    f = fixture_module.load_fixture(seed, mode)
    x, y = f.train_x, f.train_y
    views = make_views(x, seed)
    mcar_views = make_views(x, seed, ("mcar",))
    cases = f.validation_cases if mode == "validation" else f.final_cases
    errors = []
    outputs = {}

    def timed(name, fn):
        t = time.perf_counter()
        try:
            outputs[name] = {"predictions": fn(), "runtime_seconds": time.perf_counter() - t}
        except Exception as e:
            errors.append({"method": name, "error": repr(e), "traceback": traceback.format_exc()})

    # Native NaNs and mask-blind median imputation are the primary controls.
    def native_preds():
        model = HistGradientBoostingClassifier(**model_cfg(seed)).fit(x, y)
        return {n: predict(model, c[0]) for n, c in cases.items()}

    timed("native_hgb", native_preds)

    def median_preds():
        imp = SimpleImputer(strategy="median", keep_empty_features=True)
        tx = imp.fit_transform(x)
        model = HistGradientBoostingClassifier(**model_cfg(seed)).fit(tx, y)
        return {n: predict(model, imp.transform(c[0])) for n, c in cases.items()}
    timed("median_imputed_hgb_no_indicator", median_preds)

    teacher = None
    try:
        t0 = time.perf_counter()
        teacher, aug_x, aug_y = fit_teacher(x, y, views, seed + 31)
        teacher_time = time.perf_counter() - t0
        outputs["teacher_fit"] = {"runtime_seconds": teacher_time,
                                  "augmented_rows": int(len(aug_x)),
                                  "x_hash": arr_hash(aug_x), "y_hash": arr_hash(aug_y)}
    except Exception as e:
        errors.append({"method": "teacher_fit", "error": repr(e), "traceback": traceback.format_exc()})

    def map_model(model, extra_x=None):
        return {n: predict(model, extra_x(c[0]) if extra_x else c[0]) for n, c in cases.items()}

    if teacher is not None:
        timed("random_mask_augmentation_hgb", lambda: map_model(teacher))
        # Same synthetic rows/labels but doubled tree count gives a conservative
        # compute-matched direct-augmentation comparison to the teacher+student path.
        def aug_200_preds():
            model = HistGradientBoostingClassifier(**model_cfg(seed + 31, max_iter=200)).fit(aug_x, aug_y)
            return map_model(model)
        timed("random_mask_augmentation_hgb_200_trees", aug_200_preds)

        for name, kw in [
            ("candidate_mixed_k8", dict(target_views=8)),
            ("ablation_mixed_k4", dict(target_views=4)),
            ("ablation_mixed_k1", dict(target_views=1)),
            ("ablation_raw_y_distillation", dict(target_views=8, raw_targets=True)),
            ("ablation_indicator_shuffle_noop", dict(target_views=8)),
        ]:
            timed(name, lambda kw=kw: {
                **candidate_predictions(cases, views, teacher, x, y, seed, **kw)})

        # Ablation of grouped supermasks: separate teacher on MCAR-only views.
        try:
            mcar_teacher, mx, my = fit_teacher(x, y, mcar_views, seed + 31)
            outputs["mcar_only_teacher_fit"] = {"runtime_seconds": None,
                                                "augmented_rows": int(len(mx)),
                                                "x_hash": arr_hash(mx), "y_hash": arr_hash(my)}
            timed("ablation_mcar_only", lambda: {
                **candidate_predictions(cases, mcar_views, mcar_teacher, x, y, seed, target_views=8)})
        except Exception as e:
            errors.append({"method": "ablation_mcar_only", "error": repr(e), "traceback": traceback.format_exc()})

    metrics = {}
    for name, record in outputs.items():
        if name in {"teacher_fit", "mcar_only_teacher_fit"}:
            continue
        metrics[name] = eval_predictions(cases, record["predictions"])
    return {
        "seed": seed, "mode": mode, "metadata": f.metadata,
        "train_x_sha256": arr_hash(x), "train_y_sha256": arr_hash(y),
        "train_mask_pattern_sha256": arr_hash(np.isnan(x)),
        "view_hashes": [arr_hash(v) for v in views],
        "mcar_view_hashes": [arr_hash(v) for v in mcar_views],
        "methods": metrics,
        "runtimes_seconds": {name: rec.get("runtime_seconds") for name, rec in outputs.items()},
        "augmentation": {name: rec for name, rec in outputs.items() if name in {"teacher_fit", "mcar_only_teacher_fit"}},
        "errors": errors, "total_runtime_seconds": time.perf_counter() - started,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("validation", "final"), default="validation")
    p.add_argument("--all-seeds", action="store_true")
    p.add_argument("--seed", type=int, choices=SEEDS)
    a = p.parse_args()
    mode = a.mode
    out_path = VALIDATION_PATH if mode == "validation" else FINAL_PATH
    if mode == "final" and FINAL_PATH.exists():
        raise SystemExit(f"refusing to overwrite existing final output: {FINAL_PATH}")
    if not a.all_seeds and a.seed is None:
        p.error("provide --all-seeds or --seed")
    seeds = SEEDS if a.all_seeds else (a.seed,)
    data = {
        "candidate_id": "abstracts-1", "mode": mode,
        "configuration": {"base_hgb": BASE, "K": K, "teacher_views": ["mcar", "group"],
                           "synthetic_mcar_probability": 0.20, "anchor_columns": [0, 1],
                           "validation_seeds": list(SEEDS), "calibration": "none (avoids validation-label reuse)"},
        "source_hashes": {"experiment.py": sha_file(pathlib.Path(__file__)),
                          "proposal.json": sha_file(HERE / "proposal.json"),
                          "fixture.py": sha_file(ROOT / "scripts" / "missingness_shift_fixture.py")},
        "results": [],
    }
    for seed in seeds:
        data["results"].append(one_seed(seed, mode))
    # save atomically; final guard above prevents silently replacing held-out results
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, out_path)
    print(json.dumps({"saved": str(out_path), "mode": mode,
                      "seeds": [r["seed"] for r in data["results"]],
                      "seconds": [r["total_runtime_seconds"] for r in data["results"]]}, indent=2))


if __name__ == "__main__":
    main()
