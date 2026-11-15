"""Frozen validation/final runner for atoms-0.

Final mode is intentionally not executed during phase one.
"""
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
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE.parents[1]
sys.path.insert(0, str(UPSTREAM / "scripts"))
import missingness_shift_fixture as fixture_mod

SEEDS = (9401, 9403, 9407)
K = 5
CV_THRESHOLD = 1.0
MIN_IMPORTANCE = 1e-4
HGB_PARAMS = dict(max_iter=80, max_leaf_nodes=15, learning_rate=0.08,
                  l2_regularization=1.0, random_state=0)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def model(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(**{**HGB_PARAMS, "random_state": seed})


def mask_overlay(x: np.ndarray, y: np.ndarray, kind: str, rng: np.random.Generator,
                 base_mean: np.ndarray, base_std: np.ndarray) -> np.ndarray:
    """Further hide only observed non-anchor entries; never inspect hidden values."""
    out = x.copy()
    available = np.isfinite(out[:, 2:])
    if kind == "base":
        return out
    if kind == "mcar":
        rate = rng.uniform(0.0, 0.4, size=(len(out), 1))
        hide = (rng.random(available.shape) < rate) & available
    elif kind == "mar":
        z = (np.nan_to_num(out[:, 0], nan=base_mean[0]) - base_mean[0]) / base_std[0]
        rate = 0.05 + 0.5 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        hide = (rng.random(available.shape) < rate[:, None]) & available
    elif kind == "mnar":
        # Synthetic self-masking overlay, computed from observed values only.
        vals = out[:, 2:]
        z = (np.nan_to_num(vals, nan=0.0) - base_mean[None, 2:]) / base_std[None, 2:]
        rate = 0.05 + 0.5 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        hide = (rng.random(available.shape) < rate) & available
    else:
        raise ValueError(kind)
    target = out[:, 2:]
    target[hide] = np.nan
    out[:, 2:] = target
    return out


def augmented_views(x: np.ndarray, y: np.ndarray, seed: int,
                    mean: np.ndarray, std: np.ndarray, augment: bool = True):
    views = [(x, y)]
    if augment:
        for i, kind in enumerate(("mcar", "mar", "mnar")):
            masked = mask_overlay(x, y, kind, np.random.default_rng(seed + 700 + i), mean, std)
            views.append((masked, y))
    return views


def fit_pairwise_gaussian(x: np.ndarray):
    """Pairwise estimates using observed training pairs, projected to PSD."""
    p = x.shape[1]
    mean = np.nanmean(x, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    cov = np.zeros((p, p), dtype=float)
    for i in range(p):
        for j in range(i, p):
            ok = np.isfinite(x[:, i]) & np.isfinite(x[:, j])
            if ok.sum() < 2:
                cij = 0.0
            else:
                cij = float(np.mean((x[ok, i] - mean[i]) * (x[ok, j] - mean[j])))
            cov[i, j] = cov[j, i] = cij
    # Pairwise-deletion covariance need not be PSD. Shrink toward its diagonal;
    # this keeps the matrix positive definite for this low-dimensional fixture.
    cov = (cov + cov.T) / 2.0
    diagonal = np.diag(np.maximum(np.diag(cov), 1e-6))
    cov = 0.9 * cov + 0.1 * diagonal
    return mean, cov


def conditional_draws(x: np.ndarray, mean: np.ndarray, cov: np.ndarray, k: int,
                      seed: int, mode: str = "conditional") -> list[np.ndarray]:
    """Draw missing inputs from training-only Gaussian conditional distributions."""
    rng = np.random.default_rng(seed)
    out = [x.copy() for _ in range(k)]
    cache = {}
    for row_i, row in enumerate(x):
        obs = np.isfinite(row)
        miss = ~obs
        if not miss.any():
            continue
        if mode == "unconditional":
            mu = mean[miss]
            cm = cov[np.ix_(miss, miss)]
        else:
            key = (tuple(np.flatnonzero(obs)), tuple(np.flatnonzero(miss)))
            if key not in cache:
                oi, mi = np.asarray(key[0]), np.asarray(key[1])
                soo = cov[np.ix_(oi, oi)] + np.eye(len(oi)) * 1e-3
                smo = cov[np.ix_(mi, oi)]
                try:
                    beta = np.linalg.solve(soo, smo.T).T
                except np.linalg.LinAlgError:
                    beta = smo @ np.linalg.pinv(soo)
                with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                    cm0 = cov[np.ix_(mi, mi)] - beta @ cov[np.ix_(oi, mi)]
                cm0 = (cm0 + cm0.T) / 2.0
                e, v = np.linalg.eigh(cm0)
                with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                    cm0 = (v * np.maximum(e, 1e-8)) @ v.T
                if not np.isfinite(cm0).all():
                    cm0 = np.diag(np.maximum(np.diag(cov)[mi], 1e-8))
                cache[key] = (mi, oi, beta, cm0)
            mi, oi, beta, cm = cache[key]
            mu = mean[mi] + beta @ (row[oi] - mean[oi])
            miss = np.zeros_like(obs)
            miss[mi] = True
        if mode == "mean":
            draws = np.repeat(mu[None, :], k, axis=0)
        else:
            draws = rng.multivariate_normal(mu, cm, size=k, check_valid="ignore")
        for d in range(k):
            out[d][row_i, miss] = draws[d]
    return out


def model_input(x: np.ndarray, mask_keep: np.ndarray | None, values: np.ndarray | None = None):
    if values is None:
        values = x
    mask = (~np.isfinite(x)).astype(float)
    if mask_keep is None:
        return np.concatenate([values, mask], axis=1)
    return np.concatenate([values, mask[:, mask_keep]], axis=1)


def fit_imputed(train_x, train_y, seed, keep, mean, cov, augment=True,
                impute_mode="conditional"):
    views = augmented_views(train_x, train_y, seed, mean, np.sqrt(np.diag(cov)), augment)
    feats, labels = [], []
    for idx, (vx, vy) in enumerate(views):
        filled = conditional_draws(vx, mean, cov, 1, seed + 1000 + idx, impute_mode)[0]
        feats.append(model_input(vx, keep, filled))
        labels.append(vy)
    clf = model(seed)
    clf.fit(np.concatenate(feats), np.concatenate(labels))
    return clf


def predict_imputed(clf, x, mean, cov, seed, keep, impute_mode="conditional", k=K):
    predictions = []
    for draw in conditional_draws(x, mean, cov, k, seed, impute_mode):
        predictions.append(clf.predict_proba(model_input(x, keep, draw))[:, 1])
    return np.mean(predictions, axis=0)


def fit_native(train_x, train_y, seed, augment):
    # Native NaN model with the same additional corruption views where requested.
    finite_mean = np.nanmean(train_x, axis=0)
    finite_mean = np.where(np.isfinite(finite_mean), finite_mean, 0.0)
    finite_std = np.nanstd(train_x, axis=0)
    finite_std = np.where(finite_std > 1e-8, finite_std, 1.0)
    views = augmented_views(train_x, train_y, seed, finite_mean, finite_std, augment)
    xx = np.concatenate([v[0] for v in views])
    yy = np.concatenate([v[1] for v in views])
    clf = model(seed)
    clf.fit(xx, yy)
    return clf


def predict_native(clf, x):
    return clf.predict_proba(x)[:, 1]


def estimate_gate(train_x, train_y, seed, mean, cov):
    idx_a, idx_b = train_test_split(np.arange(len(train_y)), test_size=0.2,
                                    stratify=train_y, random_state=seed + 500)
    xa, ya, xb, yb = train_x[idx_a], train_y[idx_a], train_x[idx_b], train_y[idx_b]
    ma, ca = fit_pairwise_gaussian(xa)
    clf = fit_imputed(xa, ya, seed + 510, None, ma, ca, augment=True)
    m = train_x.shape[1]
    base_mean = mean
    base_std = np.sqrt(np.maximum(np.diag(cov), 1e-8))
    importances = []
    for vi, kind in enumerate(("base", "mcar", "mar", "mnar")):
        xv = mask_overlay(xb, yb, kind, np.random.default_rng(seed + 520 + vi), base_mean, base_std)
        filled = conditional_draws(xv, ma, ca, 1, seed + 530 + vi)[0]
        inp = model_input(xv, None, filled)
        baseline = clf.predict_proba(inp)[:, 1]
        base_loss = log_loss(yb, baseline, labels=[0, 1])
        scores = np.zeros(m)
        for j in range(2, m):
            perm = np.random.default_rng(seed + vi * 100 + j).permutation(len(yb))
            altered = inp.copy()
            altered[:, m + j] = inp[perm, m + j]
            scores[j] = log_loss(yb, clf.predict_proba(altered)[:, 1], labels=[0, 1]) - base_loss
        importances.append(scores)
    imp = np.stack(importances)
    keep = np.zeros(m, dtype=bool)
    stats = []
    for j in range(2, m):
        vals = imp[:, j]
        avg = float(np.mean(vals))
        cv = float(np.std(vals, ddof=0) / max(abs(avg), 1e-12))
        keep[j] = bool(np.all(vals > 0) and avg >= MIN_IMPORTANCE and cv <= CV_THRESHOLD)
        stats.append({"feature": j, "per_mechanism_importance": vals.tolist(),
                      "mean": avg, "cv": cv, "kept": bool(keep[j])})
    return keep, stats


def score_one(y, p):
    return {"ap": fixture_mod.evaluate(y, p, "average_precision"),
            "brier": fixture_mod.evaluate(y, p, "brier")}


def run_seed(seed):
    started = time.monotonic()
    f = fixture_mod.load_fixture(seed, "validation")
    x, y = f.train_x, f.train_y
    mean, cov = fit_pairwise_gaussian(x)
    std = np.sqrt(np.maximum(np.diag(cov), 1e-8))
    keep, gate_stats = estimate_gate(x, y, seed, mean, cov)
    n_keep = int(keep.sum())
    rng = np.random.default_rng(seed + 800)
    random_keep = np.zeros(x.shape[1], dtype=bool)
    if n_keep:
        random_keep[rng.choice(np.arange(2, x.shape[1]), size=n_keep, replace=False)] = True

    arms = {}
    # Native-NaN HGB: raw train and matched augmentation.
    for name, aug in (("native_nan", False), ("augmented_native_nan", True)):
        clf = fit_native(x, y, seed + (0 if not aug else 1), aug)
        arms[name] = (lambda v, clf=clf: predict_native(clf, v))

    # All imputed arms share the same training augmentation and tree capacity.
    configurations = {
        "all_masks": (None, True, "conditional", K),
        "stable_gate": (keep, True, "conditional", K),
        "no_masks": (np.zeros(x.shape[1], dtype=bool), True, "conditional", K),
        "random_gate": (random_keep, True, "conditional", K),
        "single_conditional_mean": (None, True, "mean", 1),
        "unconditional_mean": (None, True, "unconditional", K),
        "no_augmentation": (None, False, "conditional", K),
    }
    fitted = {}
    for ai, (name, (mask_keep, aug, imode, pred_k)) in enumerate(configurations.items()):
        clf = fit_imputed(x, y, seed + 10 + ai, mask_keep, mean, cov, aug, imode)
        fitted[name] = (clf, mask_keep, imode, pred_k)
        arms[name] = lambda v, clf=clf, mk=mask_keep, im=imode, pk=pred_k, ai=ai: predict_imputed(
            clf, v, mean, cov, seed + 900 + ai, mk, im, pk)

    results = {}
    for condition, (vx, vy) in f.validation_cases.items():
        results[condition] = {}
        for name, pred_fn in arms.items():
            results[condition][name] = score_one(vy, pred_fn(vx))
    elapsed = time.monotonic() - started
    return {"seed": seed, "runtime_seconds": elapsed, "fixture_metadata": f.metadata,
            "error_logs": [],
            "gate": {"threshold_cv": CV_THRESHOLD, "min_mean_importance": MIN_IMPORTANCE,
                     "selected_features": np.flatnonzero(keep).tolist(), "feature_stats": gate_stats},
            "results": results}


def run_final_seed(seed):
    """Frozen final scoring; labels are used only after all predictors are fitted."""
    started = time.monotonic()
    f = fixture_mod.load_fixture(seed, "final")
    x, y = f.train_x, f.train_y
    mean, cov = fit_pairwise_gaussian(x)
    keep, gate_stats = estimate_gate(x, y, seed, mean, cov)
    arms = {}
    for name, aug in (("native_nan", False), ("augmented_native_nan", True)):
        clf = fit_native(x, y, seed + (0 if not aug else 1), aug)
        arms[name] = lambda vx, clf=clf: predict_native(clf, vx)
    configurations = {
        "all_masks": (None, True, "conditional", K),
        "stable_gate": (keep, True, "conditional", K),
        "no_masks": (np.zeros(x.shape[1], dtype=bool), True, "conditional", K),
        "single_conditional_mean": (None, True, "mean", 1),
        "unconditional_mean": (None, True, "unconditional", K),
    }
    for ai, (name, (mask_keep, aug, imode, pred_k)) in enumerate(configurations.items()):
        clf = fit_imputed(x, y, seed + 10 + ai, mask_keep, mean, cov, aug, imode)
        arms[name] = lambda vx, clf=clf, mk=mask_keep, im=imode, pk=pred_k, ai=ai: predict_imputed(
            clf, vx, mean, cov, seed + 900 + ai, mk, im, pk)
    scores = {}
    for condition, (vx, vy) in f.final_cases.items():
        scores[condition] = {name: score_one(vy, fn(vx)) for name, fn in arms.items()}
    return {"seed": seed, "runtime_seconds": time.monotonic() - started,
            "error_logs": [],
            "fixture_metadata": f.metadata,
            "gate": {"threshold_cv": CV_THRESHOLD, "min_mean_importance": MIN_IMPORTANCE,
                     "selected_features": np.flatnonzero(keep).tolist(), "feature_stats": gate_stats},
            "results": scores}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("validation", "final"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, choices=SEEDS, action="append")
    args = parser.parse_args()
    output = args.output or HERE / ("validation.json" if args.mode == "validation" else "results.json")
    if args.mode == "final" and output.exists():
        raise SystemExit(f"refusing to overwrite existing final output: {output}")
    selected = tuple(args.seed) if args.seed else SEEDS
    if args.mode == "validation":
        all_results = [run_seed(s) for s in selected]
        result = {"mode": "validation", "seeds": selected,
                  "fixture_source_sha256": sha256_file(UPSTREAM / "scripts" / "missingness_shift_fixture.py"),
                  "experiment_source_sha256": sha256_file(Path(__file__)),
                  "records": all_results}
    else:
        # Final fixture is only loaded here; do not run before parent's explicit phase-two follow-up.
        records = [run_final_seed(s) for s in selected]
        result = {"mode": "final", "seeds": selected,
                  "fixture_source_sha256": sha256_file(UPSTREAM / "scripts" / "missingness_shift_fixture.py"),
                  "experiment_source_sha256": sha256_file(Path(__file__)),
                  "records": records}
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, output)


if __name__ == "__main__":
    main()
