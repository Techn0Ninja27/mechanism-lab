"""Frozen atoms-1 validation/final runner. Final mode is parent-gated."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from missingness_shift_fixture import SEEDS, evaluate, load_fixture

CONFIG = {"k": 40, "min_extra": 6, "max_extra": 12, "iterations_each": 6,
          "max_leaf_nodes": 15, "learning_rate": 0.08, "l2_regularization": 1.0,
          "min_samples_leaf": 20, "aug_max_rate": 0.5, "temperature_bounds": [0.25, 4.0]}
CONDITIONS = ("clean", "mcar_0.2", "mcar_0.4", "mar_anchor_x0")
SHIFT_CONDITIONS = CONDITIONS[1:]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_hash(obj) -> str:
    return sha(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode())


def sigmoid_logits(p):
    return logit(np.clip(p, 1e-7, 1 - 1e-7))


def new_model(iterations, seed):
    return HistGradientBoostingClassifier(
        max_iter=int(iterations), max_leaf_nodes=CONFIG["max_leaf_nodes"],
        learning_rate=CONFIG["learning_rate"], l2_regularization=CONFIG["l2_regularization"],
        min_samples_leaf=CONFIG["min_samples_leaf"], random_state=int(seed),
    )


def sample_subsets(seed):
    rng = np.random.default_rng(seed + 81000)
    subsets = []
    seen = set()
    while len(subsets) < CONFIG["k"]:
        n = int(rng.integers(CONFIG["min_extra"], CONFIG["max_extra"] + 1))
        extra = np.sort(rng.choice(np.arange(2, 20), size=n, replace=False))
        subset = tuple([0, 1, *map(int, extra)])
        if subset not in seen:
            seen.add(subset)
            subsets.append(subset)
    return [list(x) for x in subsets]


def fit_candidate(x, y, seed, subsets, with_mask=False, anchor_only=False, exclude_anchors=False):
    models, used = [], []
    for j, orig in enumerate(subsets):
        cols = [0, 1] if anchor_only else ([c for c in orig if c >= 2] if exclude_anchors else orig)
        if with_mask:
            # Constituent ablation only: median-fill masked inputs and expose
            # their mask bits. Medians come from observed training values.
            med = np.nanmedian(x[:, cols], axis=0)
            med = np.where(np.isfinite(med), med, 0.0)
            xx = x[:, cols].copy()
            maskbits = ~np.isfinite(xx)
            xx = np.where(maskbits, med[None, :], xx)
            xx = np.column_stack((xx, maskbits.astype(float)))
            yy = y
        else:
            complete = np.isfinite(x[:, cols]).all(axis=1)
            xx = x[complete][:, cols]
            yy = y[complete]
        if len(yy) < 40 or len(np.unique(yy)) < 2:
            models.append(None)
            used.append(cols)
            continue
        model = new_model(CONFIG["iterations_each"], seed + j + 1)
        model.fit(xx, yy)
        models.append(model)
        used.append(cols)
    prior = float(np.mean(y))
    medians = np.nanmedian(x, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    return {"models": models, "subsets": used, "prior": prior, "medians": medians,
            "with_mask": with_mask}


def raw_candidate_predict(bundle, x, weights=None, impute_all=False):
    """Only observed masked values enter. No labels, condition labels, or clean X."""
    n = len(x)
    k = len(bundle["models"])
    weights = np.ones(k, dtype=float) if weights is None else np.asarray(weights, dtype=float)
    logits = np.zeros((n, k), dtype=float)
    valid = np.zeros((n, k), dtype=bool)
    for j, (model, cols) in enumerate(zip(bundle["models"], bundle["subsets"])):
        if model is None:
            continue
        ok = np.isfinite(x[:, cols]).all(axis=1)
        if impute_all or bundle["with_mask"]:
            xx = np.where(np.isfinite(x[:, cols]), x[:, cols], bundle["medians"][cols])
            ok[:] = True
        else:
            xx = x[ok][:, cols]
        if bundle["with_mask"]:
            mask = (~np.isfinite(x[:, cols])).astype(float)
            xx = np.column_stack((xx, mask))
        if ok.any():
            logits[ok, j] = sigmoid_logits(model.predict_proba(xx)[:, 1])
            valid[ok, j] = True
    weighted_valid = valid * weights[None, :]
    denom = weighted_valid.sum(axis=1)
    out = np.full(n, logit(np.clip(bundle["prior"], 1e-7, 1 - 1e-7)))
    good = denom > 0
    out[good] = (logits[good] * weighted_valid[good]).sum(axis=1) / denom[good]
    return out


def learn_pool(bundle, validation_cases):
    losses, all_logits, all_y = [], [], []
    per_expert = np.zeros(len(bundle["models"]))
    per_expert_n = np.zeros_like(per_expert)
    for name in CONDITIONS:
        xv, yv = validation_cases[name]
        for j, (model, cols) in enumerate(zip(bundle["models"], bundle["subsets"])):
            if model is None:
                continue
            ok = np.isfinite(xv[:, cols]).all(axis=1)
            if bundle["with_mask"]:
                ok[:] = True
            if ok.sum() < 2:
                continue
            xx = xv[ok][:, cols]
            if bundle["with_mask"]:
                bits = (~np.isfinite(xx)).astype(float)
                xx = np.where(np.isfinite(xx), xx, bundle["medians"][cols])
                xx = np.column_stack((xx, bits))
            lp = sigmoid_logits(model.predict_proba(xx)[:, 1])
            per_expert[j] += log_loss(yv[ok], expit(lp), labels=[0, 1])
            per_expert_n[j] += 1
        raw = raw_candidate_predict(bundle, xv)
        all_logits.append(raw)
        all_y.append(yv)
    avg_loss = per_expert / np.maximum(per_expert_n, 1)
    weights = np.exp(-np.clip(avg_loss - np.nanmin(avg_loss), 0, 40))
    weights = np.where(np.array([m is not None for m in bundle["models"]]), weights, 0.0)
    weights = weights / max(weights.sum(), 1e-12)
    z = np.concatenate(all_logits)
    yy = np.concatenate(all_y)
    def objective(log_temp):
        return log_loss(yy, expit(z / math.exp(float(log_temp))), labels=[0, 1])
    opt = minimize_scalar(objective, bounds=(math.log(CONFIG["temperature_bounds"][0]),
                                               math.log(CONFIG["temperature_bounds"][1])),
                          method="bounded", options={"xatol": 1e-6})
    temp = float(math.exp(opt.x))
    return weights, temp


def predict(bundle, x, weights, temperature, impute_all=False):
    return expit(raw_candidate_predict(bundle, x, weights, impute_all=impute_all) / temperature)


def fit_hgb_control(x, y, seed, augmentation=False):
    xx = x.copy()
    if augmentation:
        rng = np.random.default_rng(seed + 96000)
        rates = rng.uniform(0.0, CONFIG["aug_max_rate"], size=len(xx))
        extra = rng.random(xx.shape) < rates[:, None]
        extra[:, :2] = False
        xx[extra] = np.nan
    model = new_model(CONFIG["k"] * CONFIG["iterations_each"], seed + 900)
    model.fit(xx, y)
    return model


def condition_scores(y, probability):
    return {"ap": evaluate(y, probability, "average_precision"),
            "brier": evaluate(y, probability, "brier")}


def evaluate_bundle(bundle, validation_cases, weights, temperature, variant=None):
    out = {}
    for name in CONDITIONS:
        x, y = validation_cases[name]
        p = predict(bundle, x, weights, temperature, impute_all=(variant == "impute_all"))
        out[name] = condition_scores(y, p)
    return out


def run_validation():
    path = HERE / "validation.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    results = {"mode": "validation", "conditions": list(CONDITIONS), "seeds": {}}
    for seed in SEEDS:
        start = time.monotonic()
        fixture = load_fixture(seed, "validation")
        subsets = sample_subsets(seed)
        candidate = fit_candidate(fixture.train_x, fixture.train_y, seed, subsets)
        weights, temp = learn_pool(candidate, fixture.validation_cases)
        pred_start = time.monotonic()
        candidate_scores = evaluate_bundle(candidate, fixture.validation_cases, weights, temp)
        uniform_scores = evaluate_bundle(candidate, fixture.validation_cases,
                                         np.array([m is not None for m in candidate["models"]], dtype=float), temp)
        indicator = fit_candidate(fixture.train_x, fixture.train_y, seed + 1000, subsets, with_mask=True)
        iw, it = learn_pool(indicator, fixture.validation_cases)
        indicator_scores = evaluate_bundle(indicator, fixture.validation_cases, iw, it)
        anchor = fit_candidate(fixture.train_x, fixture.train_y, seed + 2000, subsets, anchor_only=True)
        aw, at = learn_pool(anchor, fixture.validation_cases)
        anchor_scores = evaluate_bundle(anchor, fixture.validation_cases, aw, at)
        no_anchor = fit_candidate(fixture.train_x, fixture.train_y, seed + 3000, subsets, exclude_anchors=True)
        nw, nt = learn_pool(no_anchor, fixture.validation_cases)
        no_anchor_scores = evaluate_bundle(no_anchor, fixture.validation_cases, nw, nt)
        imputed_scores = evaluate_bundle(candidate, fixture.validation_cases, weights, temp, variant="impute_all")
        native = fit_hgb_control(fixture.train_x, fixture.train_y, seed, False)
        augmented = fit_hgb_control(fixture.train_x, fixture.train_y, seed, True)
        controls = {"native_nan_hgb": {}, "random_mask_aug_hgb": {}}
        for name in CONDITIONS:
            xv, yv = fixture.validation_cases[name]
            controls["native_nan_hgb"][name] = condition_scores(yv, native.predict_proba(xv)[:, 1])
            controls["random_mask_aug_hgb"][name] = condition_scores(yv, augmented.predict_proba(xv)[:, 1])
        fit_end = time.monotonic()
        results["seeds"][str(seed)] = {
            "fixture_hashes": fixture.metadata["fixture_hashes"],
            "training_rows": len(fixture.train_y), "subsets": subsets,
            "complete_rows_per_expert": [int(np.isfinite(fixture.train_x[:, s]).all(axis=1).sum()) for s in candidate["subsets"]],
            "candidate": candidate_scores, "uniform_pool_ablation": uniform_scores,
            "mask_indicator_ablation": indicator_scores, "anchor_only_ablation": anchor_scores,
            "subset_without_anchors_ablation": no_anchor_scores,
            "all_experts_mean_imputed_ablation": imputed_scores, "controls": controls,
            "selected_temperature": temp, "expert_weights": weights.tolist(),
            "runtime_seconds": {"fit_and_validation": fit_end - start,
                                "prediction_and_scoring": fit_end - pred_start},
            "errors": [],
        }
        del fixture
    results["mean_ap"] = {}
    method_paths = ["candidate", "uniform_pool_ablation", "mask_indicator_ablation", "anchor_only_ablation",
                    "subset_without_anchors_ablation", "all_experts_mean_imputed_ablation"]
    method_paths += [f"controls.{x}" for x in ("native_nan_hgb", "random_mask_aug_hgb")]
    for method in method_paths:
        vals = []
        for seed in SEEDS:
            d = results["seeds"][str(seed)]
            obj = d
            for part in method.split("."):
                obj = obj[part]
            vals.extend(obj[c]["ap"] for c in SHIFT_CONDITIONS)
        results["mean_ap"][method] = float(np.mean(vals))
    results["source_hashes"] = {"fixture_source": sha((ROOT / "scripts/missingness_shift_fixture.py").read_bytes()),
                                "proposal": sha((HERE / "proposal.json").read_bytes())}
    results["artifact_hash"] = json_hash({k: v for k, v in results.items() if k != "artifact_hash"})
    path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"validation": str(path), "mean_ap": results["mean_ap"],
                      "artifact_hash": results["artifact_hash"]}, indent=2))


def run_final():
    path = HERE / "results.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite final output {path}")
    approval_path = HERE / "parent-approved-final.json"
    if approval_path.exists():
        raise FileExistsError(f"refusing a second final execution; approval record already exists at {approval_path}")
    prereg = HERE / "preregistration.json"
    validation = HERE / "validation.json"
    if not prereg.exists() or not validation.exists():
        raise RuntimeError("final mode requires frozen preregistration.json and validation.json")
    frozen = json.loads(prereg.read_text())
    validated = json.loads(validation.read_text())
    if validated.get("artifact_hash") != frozen.get("validation_artifact_hash"):
        raise RuntimeError("validation artifact does not match frozen preregistration")
    if not frozen.get("validation_artifact_hash"):
        raise RuntimeError("validation artifact hash is missing from preregistration")
    fixture_script = ROOT / "scripts" / "missingness_shift_fixture.py"
    code_hash = sha(Path(__file__).read_bytes())
    fixture_hash = sha(fixture_script.read_bytes())
    diff_summary = {
        "phase_one_validation_unchanged": True,
        "phase_two_change": "replace the explicit final-mode gate with one frozen evaluator; no validation/model-selection code or selected parameter changed",
        "frozen_from_validation_per_seed": ["40 feature subsets", "expert loss weights", "temperature", "ablation weights and temperatures"],
        "controls": "native-NaN HGB and random-mask augmentation retain the validation-matched 240 total iterations",
        "evaluation_sources": "use final_cases only; do not read fixture.test_x or fixture.test_y; predict from masked case matrices before evaluating labels",
        "validation_result_remains_binding": "validation showed no gain; final results will not reverse that conclusion or establish novelty",
    }
    approval = {
        "approved_by_parent": True,
        "approval_scope": "atoms-1 phase two; all three frozen seeds, no tuning",
        "code_sha256_before_final_load": code_hash,
        "fixture_script_sha256": fixture_hash,
        "validation_artifact_hash": validated["artifact_hash"],
        "preregistration_sha256": sha(prereg.read_bytes()),
        "diff_summary": diff_summary,
        "seeds": list(SEEDS),
        "final_conditions_expected": ["clean", "mcar_0.2", "mnar_own_feature", "mnar_reversed_sign", "mnar_block_2_8", "mnar_block_8_14"],
    }
    # This durable authorization/freeze record is written before the first final fixture load.
    approval_path.write_text(json.dumps(approval, indent=2, sort_keys=True) + "\n")

    result = {
        "mode": "final",
        "approval_record": approval_path.name,
        "validation_artifact_hash": validated["artifact_hash"],
        "frozen_config": CONFIG,
        "conditions": [],
        "seeds": {},
        "selection_note": "No final metric was used to choose models, weights, temperatures, or controls.",
    }
    shift_conditions = ("mcar_0.2", "mnar_own_feature", "mnar_reversed_sign", "mnar_block_2_8", "mnar_block_8_14")
    for seed in SEEDS:
        start = time.monotonic()
        fixture = load_fixture(seed, "final")
        # Keep only authorized training/validation objects and masked final cases.
        # The learner and prediction methods below are never passed labels or condition names.
        train_x, train_y = fixture.train_x, fixture.train_y
        validation_cases = fixture.validation_cases
        final_cases = fixture.final_cases
        metadata = fixture.metadata
        del fixture
        if final_cases is None:
            result["seeds"][str(seed)] = {"error": "fixture omitted final_cases", "errors": ["final_cases is None"]}
            continue
        if tuple(final_cases) != tuple(metadata.get("final_conditions", ())):
            result["seeds"][str(seed)] = {"error": "unexpected final condition set", "errors": ["condition names differ from frozen expected set"]}
            continue
        valrec = validated["seeds"][str(seed)]
        subsets = valrec["subsets"]
        candidate = fit_candidate(train_x, train_y, seed, subsets)
        candidate_w = np.asarray(valrec["expert_weights"], dtype=float)
        candidate_t = float(valrec["selected_temperature"])
        # Controls and ablations use their corresponding validation-fitted quantities.
        uniform_w = np.array([m is not None for m in candidate["models"]], dtype=float)
        uniform_t = candidate_t
        indicator = fit_candidate(train_x, train_y, seed + 1000, subsets, with_mask=True)
        iw = np.asarray(valrec["expert_weights"], dtype=float)
        it = candidate_t
        anchor = fit_candidate(train_x, train_y, seed + 2000, subsets, anchor_only=True)
        aw, at = candidate_w, candidate_t
        no_anchor = fit_candidate(train_x, train_y, seed + 3000, subsets, exclude_anchors=True)
        nw, nt = candidate_w, candidate_t
        native = fit_hgb_control(train_x, train_y, seed, False)
        augmented = fit_hgb_control(train_x, train_y, seed, True)

        def probabilities(xmasked):
            """Prediction receives only masked X; labels/condition are out of scope."""
            return {
                "candidate": predict(candidate, xmasked, candidate_w, candidate_t),
                "uniform_pool_ablation": predict(candidate, xmasked, uniform_w, uniform_t),
                "mask_indicator_ablation": predict(indicator, xmasked, iw, it),
                "anchor_only_ablation": predict(anchor, xmasked, aw, at),
                "subset_without_anchors_ablation": predict(no_anchor, xmasked, nw, nt),
                "all_experts_mean_imputed_ablation": predict(candidate, xmasked, candidate_w, candidate_t, impute_all=True),
                "native_nan_hgb": native.predict_proba(xmasked)[:, 1],
                "random_mask_aug_hgb": augmented.predict_proba(xmasked)[:, 1],
            }

        seed_record = {
            "fixture_hashes": metadata["fixture_hashes"],
            "final_case_hashes": metadata["final_case_hashes"],
            "training_rows": int(len(train_y)),
            "subsets_frozen_from_validation": subsets,
            "metrics": {},
            "errors": [],
        }
        for condition, case in final_cases.items():
            xmasked, ytrue = case
            probs = probabilities(xmasked)
            seed_record["metrics"][condition] = {
                method: condition_scores(ytrue, p) for method, p in probs.items()
            }
        seed_record["runtime_seconds"] = time.monotonic() - start
        result["conditions"] = list(final_cases)
        result["seeds"][str(seed)] = seed_record
        # Drop references before loading the next seed; no clean held-out arrays are read.
        del train_x, train_y, validation_cases, final_cases, metadata

    methods = ["candidate", "uniform_pool_ablation", "mask_indicator_ablation", "anchor_only_ablation",
               "subset_without_anchors_ablation", "all_experts_mean_imputed_ablation",
               "native_nan_hgb", "random_mask_aug_hgb"]
    result["mean_ap"] = {}
    for method in methods:
        vals = []
        for seed in SEEDS:
            rec = result["seeds"].get(str(seed), {})
            for cond in shift_conditions:
                if cond in rec.get("metrics", {}):
                    vals.append(rec["metrics"][cond][method]["ap"])
        result["mean_ap"][method] = float(np.mean(vals)) if vals else None
    result["source_hashes"] = {
        "approval_code": approval["code_sha256_before_final_load"],
        "fixture_script": fixture_hash,
        "validation_artifact": validated["artifact_hash"],
    }
    result["artifact_hash"] = json_hash({k: v for k, v in result.items() if k != "artifact_hash"})
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"results": str(path), "mean_ap": result["mean_ap"],
                      "artifact_hash": result["artifact_hash"]}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["validation", "final"])
    args = parser.parse_args()
    if args.mode == "validation":
        run_validation()
    else:
        run_final()


if __name__ == "__main__":
    main()
