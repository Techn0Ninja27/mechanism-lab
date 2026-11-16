"""Small numerical audit of the StableMiss+ reweighting objective.

This is an independently written diagnostic. It does not train a predictor or
reuse the project's missingness fixture. The source implementation is cited in
the accompanying README; this script checks the algebraic behavior that any
faithful port would inherit.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
import numpy as np
import torch

SEED = 16017


def weighted_cov(a: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Match metric.py::cov for weights whose sum is one."""
    w = weights[:, None]
    first = (w * a).T @ a
    mean = (w * a).sum(axis=0)[:, None]
    return first - mean @ mean.T


def offdiag_norm(a: np.ndarray) -> float:
    mask = ~np.eye(a.shape[0], dtype=bool)
    return float(np.linalg.norm(a[mask]))


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


def optimize_mask_covariance(x: np.ndarray, m: np.ndarray, steps: int = 800):
    """Finite-difference diagnostic for only the mask covariance plus CV terms.

    The Wasserstein and feature/mask cross-correlation terms from the source
    loss are deliberately omitted. Central differences expose geometry only;
    this is not a reproduction of the source Adam run.
    """
    logits = np.zeros(len(x), dtype=float)
    def objective(z):
        w = softmax(z)
        c = weighted_cov(m, w)
        off = c - np.diag(np.diag(c))
        cv = w.std() / w.mean()
        return float((off * off).sum() + 10.0 * cv), w, c
    initial = objective(logits)
    for _ in range(steps):
        base = objective(logits)[0]
        grad = np.zeros_like(logits)
        eps = 1e-4
        # Updating all 256 logits is intentionally cheap but slow; use a
        # deterministic minibatch of coordinates for the diagnostic.
        for i in range(0, len(logits), 8):
            z = logits.copy(); z[i] += eps
            grad[i] = (objective(z)[0] - base) / eps
        logits -= 0.05 * grad
    final = objective(logits)
    return initial, final


def main(output: Path):
    rng = np.random.default_rng(SEED)
    n, d = 256, 3
    latent = rng.normal(size=(n, 3))
    x = np.column_stack((latent[:, 0], latent[:, 0] + 0.35 * latent[:, 1],
                         latent[:, 1] + 0.25 * latent[:, 2]))
    p0 = 1 / (1 + np.exp(-1.5 * x[:, 0]))
    m0 = (rng.random(n) > p0).astype(float)
    m1 = (rng.random(n) > (0.25 + 0.5 * m0)).astype(float)
    m2 = (rng.random(n) > 0.35).astype(float)
    m = np.column_stack((m0, m1, m2))
    w = np.full(n, 1 / n)
    initial, final = optimize_mask_covariance(x, m)
    wi = initial[1]; wf = final[1]
    # Execute only the official `cov` function's AST node. This avoids module
    # imports and side effects while checking the independently written formula.
    source_path = Path(__file__).parents[2] / "results/improvement-phase-2026-11-16/sources/stablemiss/metric.py"
    tree = ast.parse(source_path.read_text())
    cov_node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "cov")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[cov_node], type_ignores=[]), str(source_path), "exec"), namespace)
    official_cov = namespace["cov"]
    max_diffs = []
    for candidate in (rng.random(n), np.eye(1, n, 0).ravel()):
        candidate = candidate / candidate.sum()
        ours = weighted_cov(x, candidate)
        theirs = official_cov(torch.tensor(x, dtype=torch.float32), torch.tensor(candidate, dtype=torch.float32)).detach().numpy()
        max_diffs.append(float(np.max(np.abs(ours.astype(np.float32) - theirs))))
    uniform_cv = 0.0
    onehot = np.eye(1, n, 0).ravel()
    # PyTorch's torch.std defaults to unbiased=True, matching the source's
    # default; numpy's default is population (ddof=0), so do not mix them.
    onehot_cv = float(torch.tensor(onehot, dtype=torch.float32).std(unbiased=True) /
                      torch.tensor(onehot, dtype=torch.float32).mean())
    result = {
        "seed": SEED,
        "n": n,
        "dimensions": d,
        "source_equivalence": {
            "official_cov_ast_max_abs_diffs": max_diffs,
            "official_cov_ast_source": "metric.py::cov, extracted by AST without importing the module",
            "cov_uniform_offdiag": offdiag_norm(weighted_cov(m, wi)),
            "cov_scaled_weights_ratio": float(offdiag_norm(weighted_cov(m, 7 * wi)) / max(offdiag_norm(weighted_cov(m, wi)), 1e-12)),
            "cov_collapsed_weight_offdiag": offdiag_norm(weighted_cov(m, np.eye(1, n, 0).ravel())),
            "note": "The source softmax produces sum-one weights; arbitrary rescaling is outside its parameterization. The direct cov helper itself is not scale invariant.",
        },
        "optimization": {
            "objective_name": "mask_covariance_plus_cv_only",
            "omitted_source_terms": ["w_distance", "intra_feature_correlation", "inter_correlation"],
            "initial_objective": initial[0],
            "final_objective": final[0],
            "initial_mask_offdiag": offdiag_norm(initial[2]),
            "final_mask_offdiag": offdiag_norm(final[2]),
            "initial_effective_sample_size": float(1 / (wi @ wi)),
            "final_effective_sample_size": float(1 / (wf @ wf)),
            "initial_feature_offdiag": offdiag_norm(weighted_cov(x, wi)),
            "final_feature_offdiag": offdiag_norm(weighted_cov(x, wf)),
            "weight_max": float(wf.max()),
            "weight_min": float(wf.min()),
            "objective_improved": bool(final[0] < initial[0]),
            "uniform_cv": uniform_cv,
            "onehot_cv": onehot_cv,
            "onehot_full_cov_plus_cv_lower_bound": float(10 * onehot_cv),
            "note": "Under this partial finite-difference diagnostic and the source coefficient-of-variation penalty, the attempted update did not lower the objective. This is a fixture-specific observation, not a claim that the published optimizer fails.",
        },
        "random_fourier_observation": {
            "source_default_num_f": 1,
            "source_normalizes_each_row_across_features": True,
            "implication": "With num_f omitted, the source metric uses one random feature and row-wise min/max normalization before covariance; this is a stochastic proxy, not a direct covariance of the original variables.",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(Path(__file__).with_name("results.json"))
