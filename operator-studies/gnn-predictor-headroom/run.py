"""Development-only causal predictor headroom screen.

This screen deliberately uses only seeds 10 and 11.  It is not a held-out or
closed-loop accuracy experiment.  All predictors consume observations only at
sync endpoints; hidden intermediate trajectory values are used for scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[1]
SCREEN_PATH = ROOT / "operator-studies" / "gnn-sync-gate" / "screen.py"
SEEDS = (10, 11)
SCHEDULES = (False, True)
PARTITION_SEED = 321
SYNC = 4
WARMUP = 20
BETA_GRID = (0.05, 0.1, 0.2, 0.4, 0.8)
LAMBDA_GRID = (0.05, 0.1, 0.2, 0.4, 0.8)
MIX_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
LAG_GRID = (1, 2, 3)
RIDGE_GRID = (1e-3, 0.1, 10.0)


def _load_screen():
    spec = importlib.util.spec_from_file_location("gnn_sync_screen", SCREEN_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {SCREEN_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


screen = _load_screen()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def boundary_pairs(adj, parts: np.ndarray) -> list[tuple[int, int]]:
    ei = adj.indices().cpu().numpy()
    return sorted({(int(i), int(parts[j])) for i, j in zip(ei[0], ei[1]) if parts[i] != parts[j]})


def _pair_arrays(a: np.ndarray, pairs: list[tuple[int, int]], parts: np.ndarray):
    src = np.asarray([i for i, _ in pairs], dtype=np.int64)
    groups = {key: k for k, key in enumerate(sorted({(int(parts[i]), int(d)) for i, d in pairs}))}
    gids = np.asarray([groups[(int(parts[i]), int(d))] for i, d in pairs], dtype=np.int64)
    return src, gids


def _rel2(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum((pred - truth) ** 2, axis=1)) / (np.sqrt(np.sum(truth ** 2, axis=1)) + 1e-8)


def _metric(preds: list[np.ndarray], holds: list[np.ndarray], truths: list[np.ndarray], weight: float) -> float:
    if not truths:
        return float("nan")
    values = [_rel2(weight * p + (1.0 - weight) * h, y) for p, h, y in zip(preds, holds, truths)]
    return float(np.mean(np.concatenate(values)))


def _damped_arrays(trajectory: np.ndarray, parts: np.ndarray, pairs: list[tuple[int, int]], *, beta: float,
                   lam: float = 0.2,
                   sync: int = SYNC, warmup: int = WARMUP) -> dict[str, Any]:
    """Replay a self-fed damped trend and return stale-step arrays.

    This mirrors ``screen.replay_gate``'s predictor state and update order, but
    permits beta and returns the arrays needed for fixed/continuous mixtures.
    """
    a = np.asarray(trajectory, dtype=np.float32)
    src, _ = _pair_arrays(a, pairs, parts)
    last = a[0, src].copy()
    numer = np.zeros_like(last)
    denom = np.ones(len(src), dtype=np.float32)
    hold = a[0, src].copy()
    for t in range(1, min(warmup + 1, len(a))):
        decay = math.exp(-lam)
        slope = a[t, src] - last
        last = a[t, src].copy()
        numer = numer * decay + slope
        denom = denom * decay + 1.0
        hold = a[t, src].copy()
    preds: list[np.ndarray] = []
    holds: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    for t in range(warmup + 1, len(a)):
        at_sync = ((t - warmup) % sync) == 0
        pred = last + numer / np.maximum(denom[:, None], 1e-6) * math.exp(-beta)
        truth = a[t, src]
        if not at_sync:
            preds.append(pred.copy()); holds.append(hold.copy()); truths.append(truth.copy())
            # Self-feed the prediction; hidden truth is never consumed here.
            slope = pred - last
            last = pred.copy()
            numer = numer * math.exp(-lam) + slope
            denom = denom * math.exp(-lam) + 1.0
        else:
            # Score/use no endpoint value, then consume the newly revealed sync.
            slope = truth - last
            last = truth.copy()
            numer = numer * math.exp(-lam) + slope
            denom = denom * math.exp(-lam) + 1.0
            hold = truth.copy()
    return {"preds": preds, "holds": holds, "truths": truths,
            "predictor": _metric(preds, holds, truths, 1.0),
            "hold": _metric(preds, holds, truths, 0.0),
            "mix": {str(w): _metric(preds, holds, truths, w) for w in MIX_GRID}}


def _fit_global_ar(history: list[np.ndarray], lag: int, ridge: float):
    """Fit a global vector AR model from sync observations only."""
    if len(history) <= lag:
        return None
    # history entries are [pairs, dimensions]; each row is one pair/time.
    rows = []
    targets = []
    for t in range(lag, len(history)):
        rows.append(np.concatenate([history[t - j] for j in range(lag, 0, -1)], axis=1))
        targets.append(history[t])
    x = np.concatenate(rows, axis=0).astype(np.float64)
    y = np.concatenate(targets, axis=0).astype(np.float64)
    x = np.concatenate([np.ones((len(x), 1)), x], axis=1)
    reg = np.eye(x.shape[1], dtype=np.float64) * ridge
    reg[0, 0] = 0.0
    try:
        coef = np.linalg.solve(x.T @ x + reg, x.T @ y)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(coef).all():
        return None
    return coef


def _ridge_arrays(trajectory: np.ndarray, parts: np.ndarray, pairs: list[tuple[int, int]], *, lag: int,
                  ridge: float, sync: int = SYNC, warmup: int = WARMUP) -> dict[str, Any]:
    """Forecast stale intervals using only sync-history ridge AR fits."""
    a = np.asarray(trajectory, dtype=np.float32)
    src, _ = _pair_arrays(a, pairs, parts)
    # Sync observations are equally spaced. Do not mix the warmup's per-epoch
    # values with four-epoch sync gaps.
    history = [a[t, src].copy() for t in range(0, warmup + 1, sync)]
    coef = _fit_global_ar(history, lag, ridge)
    preds: list[np.ndarray] = []
    holds: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    fallback_steps = 0
    nonfinite_predictions = 0
    for t in range(warmup + 1, len(a)):
        at_sync = ((t - warmup) % sync) == 0
        hold = history[-1]
        if not at_sync:
            if coef is None:
                pred = hold.copy()
                fallback_steps += 1
            else:
                dt = (t - warmup - 1) % sync + 1
                # Fit predicts the next sync value. Interpolate over this
                # stale interval so the horizon remains in sync units.
                x = np.concatenate([np.concatenate(history[-lag:], axis=1)], axis=1)
                x = np.concatenate([np.ones((len(x), 1)), x], axis=1)
                next_sync = (x @ coef).astype(np.float32)
                pred = hold + (next_sync - hold) * (dt / float(sync))
                if not np.isfinite(pred).all():
                    pred = hold.copy()
                    fallback_steps += 1
                    nonfinite_predictions += 1
            preds.append(pred.copy()); holds.append(hold.copy()); truths.append(a[t, src].copy())
        else:
            # At this endpoint, forecast state is not trained on its truth until
            # after the interval is over; then the observation becomes history.
            history.append(a[t, src].copy())
            coef = _fit_global_ar(history, lag, ridge)
    return {"preds": preds, "holds": holds, "truths": truths,
            "metric": _metric(preds, holds, truths, 1.0),
            "hold": _metric(preds, holds, truths, 0.0),
            "fallback_steps": fallback_steps, "nonfinite_predictions": nonfinite_predictions}


def _result_for_trajectory(trajectory: np.ndarray, parts: np.ndarray, pairs: list[tuple[int, int]], *, baseline=None) -> dict[str, Any]:
    # Keep scalar summaries for the grid. Retain only the baseline arrays needed
    # for the pooled continuous-mixture curve; retaining every grid trace would
    # allocate multiple gigabytes on Cora.
    damped_summary: dict[str, dict[str, dict[str, Any]]] = {}
    base = baseline
    for lam in LAMBDA_GRID:
        damped_summary[str(lam)] = {}
        for beta in BETA_GRID:
            x = baseline if (baseline is not None and lam == .2 and beta == .2) else _damped_arrays(trajectory, parts, pairs, beta=beta, lam=lam)
            if lam == .2 and beta == .2:
                base = x
            damped_summary[str(lam)][str(beta)] = {"predictor": x["predictor"], "hold": x["hold"], "mix": x["mix"]}
    result: dict[str, Any] = {"damped": damped_summary}
    # Keep the curve; selection is pooled across all four development rows in
    # run(), never independently per trajectory.
    assert base is not None
    weights = np.linspace(0, 1, 101)
    result["continuous_mix_curve"] = {"weights": [float(w) for w in weights],
                                       "metrics": [_metric(base["preds"], base["holds"], base["truths"], float(w)) for w in weights],
                                       "grid_points": len(weights)}
    result["ridge"] = {}
    for lag in LAG_GRID:
        for ridge in RIDGE_GRID:
            row = _ridge_arrays(trajectory, parts, pairs, lag=lag, ridge=ridge)
            result["ridge"][f"lag{lag}_ridge{ridge:g}"] = {"metric": row["metric"], "hold": row["hold"],
                                                              "fallback_steps": row["fallback_steps"],
                                                              "nonfinite_predictions": row["nonfinite_predictions"]}
    return result


def run(data_root: Path, output: Path) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    data_files = {name: data_root / name for name in ("cora.content", "cora.cites")}
    if any(not path.is_file() for path in data_files.values()):
        raise FileNotFoundError("prepared Cora files are required")
    source_hashes = {"screen.py": sha256(SCREEN_PATH), "run.py": sha256(STUDY / "run.py"),
                     "protocol.json": sha256(ROOT / "operator-studies/gnn-sync-gate/protocol.json"),
                     "data_receipt.json": sha256(ROOT / "operator-studies/gnn-sync-gate/data-receipt.json"),
                     **{f"data/{name}": sha256(path) for name, path in data_files.items()}}
    freeze = {"schema": "gnn-predictor-headroom-freeze.v1", "seeds": list(SEEDS), "schedules": list(SCHEDULES),
              "partition_seed": PARTITION_SEED, "sync": SYNC, "warmup": WARMUP,
              "lambda_grid": list(LAMBDA_GRID), "beta_grid": list(BETA_GRID), "mix_grid": list(MIX_GRID), "lag_grid": list(LAG_GRID),
              "ridge_grid": list(RIDGE_GRID), "source_hashes": source_hashes,
              "scope": "development-only causal predictor screen; seeds 12-15 reserved; no closed-loop claim"}
    (output / "freeze.json").write_text(json.dumps(freeze, indent=2) + "\n")
    x, y, adj = screen.load_cora(data_root)
    parts = np.random.default_rng(PARTITION_SEED).integers(0, 4, size=len(y))
    pairs = boundary_pairs(adj, parts)
    rows = {}
    for seed in SEEDS:
        for lr_drop in SCHEDULES:
            trajectory, quality = screen.train_trajectory(x, y, adj, seed=seed, lr_drop=lr_drop)
            imported = screen.replay_gate(trajectory, parts, boundary_pairs=pairs, sync=SYNC, warmup=WARMUP, eta=1.0)
            baseline = _damped_arrays(trajectory, parts, pairs, beta=.2, lam=.2)
            exact_delta = abs(imported["aggregate"]["predictor"] - baseline["predictor"])
            if exact_delta > 1e-6:
                raise AssertionError(f"baseline mismatch {seed} {lr_drop}: {exact_delta}")
            ours = _result_for_trajectory(trajectory, parts, pairs, baseline=baseline)
            rows[f"seed{seed}_{'lr_drop' if lr_drop else 'constant'}"] = {
                "seed": seed, "lr_drop": lr_drop, "quality_smoke": quality,
                "trajectory_shape": list(trajectory.shape), "boundary_pairs": len(pairs),
                "imported_sni_predictor": imported["aggregate"]["predictor"],
                "imported_hold": imported["aggregate"]["hold"], "ours": ours,
                "baseline_exact_abs_delta": exact_delta,
            }
    row_values = list(rows.values())
    # Select one configuration over both schedules and both dev seeds.
    lambda_beta_mix_scores = {(lam, beta, mix): float(np.mean([r["ours"]["damped"][str(lam)][str(beta)]["mix"][str(mix)] for r in row_values]))
                               for lam in LAMBDA_GRID for beta in BETA_GRID for mix in MIX_GRID}
    selected_lambda, selected_beta, selected_mix = min(lambda_beta_mix_scores, key=lambda_beta_mix_scores.get)
    ridge_scores = {key: float(np.mean([r["ours"]["ridge"][key]["metric"] for r in row_values]))
                    for key in row_values[0]["ours"]["ridge"]}
    selected_ridge = min(ridge_scores, key=ridge_scores.get)
    weights = row_values[0]["ours"]["continuous_mix_curve"]["weights"]
    continuous_scores = {w: float(np.mean([r["ours"]["continuous_mix_curve"]["metrics"][i] for r in row_values]))
                         for i, w in enumerate(weights)}
    selected_continuous = min(continuous_scores, key=continuous_scores.get)
    schedules = {}
    for lr_drop in SCHEDULES:
        subset = [r for r in row_values if r["lr_drop"] == lr_drop]
        sni = float(np.mean([r["imported_sni_predictor"] for r in subset]))
        candidates = {
            "damped_mix": float(np.mean([r["ours"]["damped"][str(selected_lambda)][str(selected_beta)]["mix"][str(selected_mix)] for r in subset])),
            "ridge": float(np.mean([r["ours"]["ridge"][selected_ridge]["metric"] for r in subset])),
            "continuous_mix": float(np.mean([r["ours"]["continuous_mix_curve"]["metrics"][weights.index(selected_continuous)] for r in subset])),
        }
        schedules["lr_drop" if lr_drop else "constant"] = {
            "sni": sni, "selected_candidates": candidates,
            "reduction_vs_sni": {k: (sni - v) / max(sni, 1e-12) for k, v in candidates.items()},
            "five_percent_allocation_pass": {k: (sni - v) >= .05 * sni for k, v in candidates.items()},
        }
    result = {"schema": "gnn-predictor-headroom-v1", "status": "complete", "rows": rows,
              "scope": "development seeds 10/11 only; no held-out or downstream accuracy claim",
              "reserved_seeds": [12, 13, 14, 15], "source_hashes": source_hashes,
              "interpretation": "Finite causal control screen; not exhaustive DSL search or a least-squares upper bound.",
              "paid_calls": 0,
              "global_selection": {"lambda": selected_lambda, "beta": selected_beta, "mix": selected_mix, "ridge": selected_ridge,
                                   "continuous_mix_weight": selected_continuous,
                                   "continuous_mix_selection": "pooled over both seeds and schedules; dense-grid approximation"},
              "schedule_summary": schedules,
              "allocation_rule": "A configuration must reduce mean stale error versus original SNI by >=5% in both schedules before a later paid pilot is considered; this is necessary allocation evidence only."}
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.data, args.output)
    print(json.dumps({"status": result["status"], "rows": len(result["rows"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
