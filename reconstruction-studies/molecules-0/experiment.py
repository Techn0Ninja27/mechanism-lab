#!/usr/bin/env python3
"""Train-only mechanism-diversified masking validation for molecules-0.

The prediction interface is intentionally just ``predict_masked(X)``. Mode,
condition names, and labels remain in the outer evaluation driver.
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
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from scripts.missingness_shift_fixture import SEEDS, evaluate, load_fixture


OUT_VALIDATION = HERE / "validation.json"
OUT_FINAL = HERE / "results.json"
MASK_START = 2
ETA_GRID = (0.5, 1.0, 2.0)
N_GROUPS = 4
N_DRO_ROUNDS = 3
N_FOLDS = 3
MODEL_PARAMS = {
    "max_iter": 80,
    "learning_rate": 0.08,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 20,
    "l2_regularization": 0.01,
    "early_stopping": False,
    "random_state": 0,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def hash_array(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(**MODEL_PARAMS)


def _ecdf_ranks(train_x: np.ndarray) -> np.ndarray:
    """Return rank CDFs for finite training entries; NaNs stay NaN."""
    q = np.full(train_x.shape, np.nan, dtype=np.float64)
    for j in range(MASK_START, train_x.shape[1]):
        observed = train_x[np.isfinite(train_x[:, j]), j]
        if not len(observed):
            continue
        ordered = np.sort(observed)
        vals = train_x[:, j]
        finite = np.isfinite(vals)
        # Mid-rank ECDF keeps ties stable and avoids using any hidden values.
        lo = np.searchsorted(ordered, vals[finite], side="left")
        hi = np.searchsorted(ordered, vals[finite], side="right")
        q[finite, j] = ((lo + hi) / 2.0) / len(ordered)
    return q


def _apply_extra_mask(x: np.ndarray, p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = x.copy()
    observed = np.isfinite(out)
    eligible = np.zeros(out.shape, dtype=bool)
    eligible[:, MASK_START:] = observed[:, MASK_START:]
    drops = (rng.random(out.shape) < p) & eligible
    out[drops] = np.nan
    return out


def make_groups(train_x: np.ndarray, seed: int, kind: str) -> np.ndarray:
    """Four paired-size groups derived only from already-masked train_x.

    kind is one of candidate, random, one_direction, shuffled. Random controls
    use 0.50 masks for groups 2/3 to match the self-mask groups' marginal rate.
    """
    rng = np.random.default_rng(seed + 4100)
    q = _ecdf_ranks(train_x)
    plus = np.zeros(train_x.shape, dtype=float)
    minus = np.zeros(train_x.shape, dtype=float)
    finite = np.isfinite(q)
    plus[finite] = 1.0 / (1.0 + np.exp(-np.clip(4.0 * (q[finite] - 0.5), -30, 30)))
    minus[finite] = 1.0 / (1.0 + np.exp(-np.clip(-4.0 * (q[finite] - 0.5), -30, 30)))
    if kind == "shuffled":
        # Preserve each feature's empirical rank distribution while severing
        # each row's own value-to-mask association.
        for j in range(MASK_START, train_x.shape[1]):
            ix = np.flatnonzero(np.isfinite(train_x[:, j]))
            permuted = q[ix, j].copy()
            rng.shuffle(permuted)
            plus[ix, j] = 1.0 / (1.0 + np.exp(-np.clip(4.0 * (permuted - 0.5), -30, 30)))
            minus[ix, j] = 1.0 / (1.0 + np.exp(-np.clip(-4.0 * (permuted - 0.5), -30, 30)))

    groups = [train_x.copy()]
    # Proposal's random comparison group (extra MCAR 0.20).
    groups.append(_apply_extra_mask(train_x, np.full(train_x.shape, 0.20), rng))
    if kind == "candidate" or kind == "shuffled":
        groups.append(_apply_extra_mask(train_x, plus, rng))
        groups.append(_apply_extra_mask(train_x, minus, rng))
    elif kind == "random":
        groups.append(_apply_extra_mask(train_x, np.full(train_x.shape, 0.50), rng))
        groups.append(_apply_extra_mask(train_x, np.full(train_x.shape, 0.50), rng))
    elif kind == "one_direction":
        groups.append(_apply_extra_mask(train_x, plus, rng))
        groups.append(_apply_extra_mask(train_x, np.full(train_x.shape, 0.50), rng))
    else:
        raise ValueError(f"unknown group kind: {kind}")
    return np.stack(groups, axis=0)


def make_mask_blind_groups(train_x: np.ndarray) -> np.ndarray:
    # Kept as a separate one-group arm; no indicator columns are added.
    return np.stack([train_x], axis=0)


def _fit_stack(groups: np.ndarray, y: np.ndarray, group_weights: np.ndarray,
               row_ids: np.ndarray | None = None) -> HistGradientBoostingClassifier:
    if row_ids is None:
        row_ids = np.arange(len(y))
    x_stack = np.concatenate([groups[g, row_ids] for g in range(groups.shape[0])], axis=0)
    y_stack = np.tile(y[row_ids], groups.shape[0])
    w_stack = np.repeat(group_weights, len(row_ids))
    w_stack = w_stack / np.mean(w_stack)
    est = model()
    est.fit(x_stack, y_stack, sample_weight=w_stack)
    return est


def _oof_group_losses(groups: np.ndarray, y: np.ndarray, seed: int,
                      group_weights: np.ndarray) -> np.ndarray:
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed + 7200)
    losses = np.zeros(groups.shape[0], dtype=float)
    counts = np.zeros(groups.shape[0], dtype=int)
    indices = np.arange(len(y))
    for train_idx, held_idx in splitter.split(indices, y):
        est = _fit_stack(groups, y, group_weights, train_idx)
        for g in range(groups.shape[0]):
            p = est.predict_proba(groups[g, held_idx])[:, 1]
            losses[g] += log_loss(y[held_idx], p, labels=[0, 1]) * len(held_idx)
            counts[g] += len(held_idx)
    return losses / counts


def fit_group_dro(groups: np.ndarray, y: np.ndarray, seed: int, eta: float,
                  rounds: int = N_DRO_ROUNDS) -> tuple[HistGradientBoostingClassifier, dict]:
    weights = np.full(groups.shape[0], 1.0 / groups.shape[0])
    history = []
    for _ in range(rounds):
        losses = _oof_group_losses(groups, y, seed, weights)
        # Stable exponentiation; centering does not alter normalized weights.
        weights *= np.exp(eta * (losses - losses.mean()))
        weights = weights / weights.sum()
        history.append({"oof_group_logloss": losses.tolist(), "group_weights": weights.tolist()})
    est = _fit_stack(groups, y, weights)
    return est, {"eta": eta, "rounds": rounds, "history": history, "final_group_weights": weights.tolist()}


def fit_uniform(groups: np.ndarray, y: np.ndarray) -> HistGradientBoostingClassifier:
    return _fit_stack(groups, y, np.full(groups.shape[0], 1.0 / groups.shape[0]))


def fit_mask_blind(train_x: np.ndarray, y: np.ndarray) -> HistGradientBoostingClassifier:
    imputer = SimpleImputer(strategy="median")
    x = imputer.fit_transform(train_x)
    est = model()
    est.fit(x, y)
    # lightweight interface adapter for the evaluation driver
    est._molecules_imputer = imputer
    est._molecules_mask_blind = True
    return est


def predict_masked(est, masked_x: np.ndarray) -> np.ndarray:
    """Prediction accepts only the masked matrix, never condition/labels."""
    if getattr(est, "_molecules_mask_blind", False):
        masked_x = est._molecules_imputer.transform(masked_x)
    return est.predict_proba(masked_x)[:, 1]


def score_model(est, cases: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    scores = {}
    for condition, (masked_x, labels) in cases.items():
        p = predict_masked(est, masked_x)
        scores[condition] = {
            "ap": evaluate(labels, p, "average_precision"),
            "brier": evaluate(labels, p, "brier"),
        }
    return scores


def run_seed(seed: int, mode: str) -> dict:
    start = time.perf_counter()
    fixture = load_fixture(seed, mode=mode)
    train_x, train_y = fixture.train_x, fixture.train_y
    if mode == "validation":
        cases = fixture.validation_cases
    else:
        if fixture.final_cases is None:
            raise RuntimeError("final fixture did not expose final_cases")
        cases = fixture.final_cases

    arm_results = {}
    errors = []
    model_starts = time.perf_counter()
    try:
        candidate_groups = make_groups(train_x, seed, "candidate")
        candidate_by_eta = {}
        if mode == "validation":
            fit_etas = ETA_GRID
        else:
            # Final mode reads the eta frozen from validation. It does not score
            # alternate etas against held-out labels.
            validation_payload = json.loads(OUT_VALIDATION.read_text())
            validation_record = next(r for r in validation_payload["records"] if r["seed"] == seed)
            fit_etas = (float(validation_record["arms"]["candidate_selfmask_dro"]["selected_eta"]),)
        for eta in fit_etas:
            est, dro = fit_group_dro(candidate_groups, train_y, seed, eta)
            candidate_by_eta[eta] = {"model": est, "dro": dro}
        if mode == "validation":
            # Model selection is restricted to the shared validation cases.
            eta_scores = {
                str(eta): score_model(candidate_by_eta[eta]["model"], cases)
                for eta in ETA_GRID
            }
            eta_means = {
                eta: float(np.mean([eta_scores[str(eta)][c]["ap"] for c in cases]))
                for eta in ETA_GRID
            }
            selected_eta = sorted(ETA_GRID, key=lambda e: (-eta_means[e], e))[0]
            selected_scores = eta_scores[str(selected_eta)]
        else:
            selected_eta = fit_etas[0]
            eta_scores = None
            eta_means = None
            selected_scores = score_model(candidate_by_eta[selected_eta]["model"], cases)
        candidate = candidate_by_eta[selected_eta]
        arm_results["candidate_selfmask_dro"] = {
            "scores": selected_scores, "selected_eta": selected_eta,
            "eta_validation_mean_ap": {str(k): v for k, v in eta_means.items()} if eta_means is not None else None,
            "dro": candidate["dro"],
        }
        # Primary matched control: same group count/capacity and self-trained
        # DRO calibration, but replace value-dependent groups by random masks.
        random_groups = make_groups(train_x, seed, "random")
        random_est, random_dro = fit_group_dro(random_groups, train_y, seed, selected_eta)
        arm_results["randommask_dro_matched"] = {
            "scores": score_model(random_est, cases), "selected_eta_from_candidate": selected_eta,
            "dro": random_dro,
        }
        # Component removal: candidate self-mask groups with uniform weights.
        uniform_est = fit_uniform(candidate_groups, train_y)
        arm_results["selfmask_uniform"] = {"scores": score_model(uniform_est, cases)}
        # Single direction; the fourth group remains a rate-matched random mask.
        one_groups = make_groups(train_x, seed, "one_direction")
        one_est = fit_uniform(one_groups, train_y)
        arm_results["one_direction_uniform"] = {"scores": score_model(one_est, cases)}
        # Shuffled-rank ablation removes the own-value relation, keeping its rank
        # marginal and group count, then uses the selected DRO strength.
        shuffled_groups = make_groups(train_x, seed, "shuffled")
        shuffled_est, shuffled_dro = fit_group_dro(shuffled_groups, train_y, seed, selected_eta)
        arm_results["shuffled_rank_dro"] = {
            "scores": score_model(shuffled_est, cases), "selected_eta_from_candidate": selected_eta,
            "dro": shuffled_dro,
        }
        # Native-NaN HGB, unaugmented baseline.
        native = model().fit(train_x, train_y)
        arm_results["native_nan_hgb"] = {"scores": score_model(native, cases)}
        # Mask-blind ablation: median fill without missingness indicators.
        blind = fit_mask_blind(train_x, train_y)
        arm_results["mask_blind_median_hgb"] = {"scores": score_model(blind, cases)}
    except Exception as e:
        errors.append({"type": type(e).__name__, "message": str(e)})
        raise
    runtime = time.perf_counter() - model_starts
    return {
        "seed": seed,
        "mode": mode,
        "fixture_metadata": fixture.metadata,
        "input_hashes": {"train_masked": hash_array(train_x), "train_y": hash_array(train_y)},
        "validation_case_hashes": {
            name: hash_array(x) for name, (x, _) in fixture.validation_cases.items()
        } if mode == "validation" else None,
        "arms": arm_results,
        "runtime_seconds": {"modeling_and_scoring": runtime, "total": time.perf_counter() - start},
        "error_log": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["validation", "final"], required=True)
    parser.add_argument("--seeds", type=int, nargs="+", choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or (OUT_VALIDATION if args.mode == "validation" else OUT_FINAL)
    if args.mode == "final" and output.exists():
        raise FileExistsError(f"refusing to overwrite held-out result: {output}")
    if args.mode == "final" and len(args.seeds) != len(SEEDS):
        raise ValueError("final mode must run all three preregistered seeds together")
    if args.mode == "final":
        if not OUT_VALIDATION.exists():
            raise FileNotFoundError("final mode requires the frozen validation artifact")
        validation_hash = sha256_file(OUT_VALIDATION)
        prereg = json.loads((HERE / "preregistration.json").read_text())
        if validation_hash != prereg.get("validation_artifact_sha256"):
            raise ValueError("validation artifact hash does not match preregistration")
        validation_payload = json.loads(OUT_VALIDATION.read_text())
        selected = {r["seed"]: r["arms"]["candidate_selfmask_dro"]["selected_eta"] for r in validation_payload["records"]}
        expected_selected = {9401: 0.5, 9403: 2.0, 9407: 2.0}
        if selected != expected_selected or set(selected) != set(args.seeds):
            raise ValueError(f"validation eta selections mismatch frozen selections: {selected}")
    # Enforce a single numerical thread even when the caller does not use the
    # compute-slot wrapper.
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    records = []
    for seed in args.seeds:
        records.append(run_seed(seed, args.mode))
    payload = {
        "study_id": "molecules-0", "mode": args.mode, "seeds": list(args.seeds),
        "model_params": MODEL_PARAMS, "eta_grid": ETA_GRID,
        "records": records,
    }
    if args.mode == "final":
        payload["validation_artifact_sha256"] = sha256_file(OUT_VALIDATION)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "final":
        # Exclusive creation provides a second guard against accidental reruns.
        with output.open("x") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
    else:
        tmp = output.with_suffix(output.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        tmp.replace(output)
    print(json.dumps({"output": str(output), "mode": args.mode, "seed_count": len(records)}, indent=2))


if __name__ == "__main__":
    main()
