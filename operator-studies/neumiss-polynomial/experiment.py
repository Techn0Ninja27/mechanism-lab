"""Development-only oracle check for a tied NeuMiss polynomial recurrence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

SEEDS = (17011, 17013, 17017)
ETA = 0.5
DEPTHS = (3, 5)
DIMENSION = 20


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def covariance(seed: int, d: int = DIMENSION) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    eig = np.linspace(0.35, 1.0, d)
    return q @ np.diag(eig) @ q.T


def masks(seed: int, n: int, d: int = DIMENSION) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        observed = rng.random(d) > 0.25
        if observed.sum() < 3:
            observed[:3] = True
        out.append(np.flatnonzero(observed))
    return out


def target_blocks(sigma: np.ndarray, patterns: list[np.ndarray]) -> list[np.ndarray]:
    return [np.linalg.inv(sigma[np.ix_(obs, obs)]) for obs in patterns]


def basis(sigma: np.ndarray, obs: np.ndarray, depth: int) -> list[np.ndarray]:
    """Matrices for A^depth + sum_{k=2..depth} c_k A^(depth-k)."""
    a = np.eye(len(obs)) - ETA * sigma[np.ix_(obs, obs)]
    powers = [np.eye(len(obs))]
    for _ in range(depth):
        powers.append(powers[-1] @ a)
    # Leading coefficient of A^depth is fixed at one; c_k multiplies A^(depth-k).
    return [powers[depth]] + [powers[depth - k] for k in range(2, depth + 1)]


def tied_operator(sigma: np.ndarray, obs: np.ndarray, depth: int) -> np.ndarray:
    return sum(basis(sigma, obs, depth))


def conventional_neumann(sigma: np.ndarray, obs: np.ndarray, depth: int) -> np.ndarray:
    a = np.eye(len(obs)) - ETA * sigma[np.ix_(obs, obs)]
    p = np.eye(len(obs))
    result = np.zeros_like(a)
    for _ in range(depth):
        result += ETA * p
        p = p @ a
    return result


def fit_global(sigma_by_seed: list[np.ndarray], patterns_by_seed: list[list[np.ndarray]], depth: int) -> np.ndarray:
    rows = []
    rhs = []
    for sigma, patterns in zip(sigma_by_seed, patterns_by_seed):
        for obs in patterns:
            bs = basis(sigma, obs, depth)
            # Every matrix entry shares the same coefficient vector.
            for i in range(len(obs)):
                for j in range(len(obs)):
                    rows.append([b[i, j] for b in bs[1:]])
                    rhs.append(np.linalg.inv(sigma[np.ix_(obs, obs)])[i, j] - bs[0][i, j])
    coeff, *_ = np.linalg.lstsq(np.asarray(rows), np.asarray(rhs), rcond=None)
    return coeff


def relative_error(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(pred - target) / max(np.linalg.norm(target), 1e-12))


def evaluate(sigma_by_seed, patterns_by_seed, coeffs_by_depth):
    rows = []
    for depth in DEPTHS:
        for split, patterns_by_seed_split in (("development", patterns_by_seed[0]), ("validation", patterns_by_seed[1])):
            for seed, sigma, patterns in zip(SEEDS, sigma_by_seed, patterns_by_seed_split):
                coeff = coeffs_by_depth[depth]
                for obs in patterns:
                    target = np.linalg.inv(sigma[np.ix_(obs, obs)])
                    bs = basis(sigma, obs, depth)
                    proposed = bs[0] + sum(c * b for c, b in zip(coeff, bs[1:]))
                    tied = sum(bs)
                    conventional = conventional_neumann(sigma, obs, depth)
                    rows.append({"depth": depth, "split": split, "seed": seed,
                                 "relative_error_proposed": relative_error(proposed, target),
                                 "relative_error_tied": relative_error(tied, target),
                                 "relative_error_conventional": relative_error(conventional, target),
                                 "pattern_size": int(len(obs))})
    return rows


def main(output: Path, plan_path: Path):
    plan = json.loads(plan_path.read_text())
    sigma = [covariance(seed) for seed in SEEDS]
    development = [masks(seed + 1000, 24) for seed in SEEDS]
    validation = [masks(seed + 2000, 24) for seed in SEEDS]
    coeffs = {str(depth): fit_global(sigma, development, depth).tolist() for depth in DEPTHS}
    rows = evaluate(sigma, (development, validation), {int(k): np.asarray(v) for k, v in coeffs.items()})
    result = {"schema_version": "neumiss-polynomial-audit.v1", "plan_hash": sha(plan_path),
              "code_sha256": sha(Path(__file__)), "source_sha256": plan["source_sha256"], "seeds": list(SEEDS), "eta": ETA,
              "dimension": DIMENSION, "depths": list(DEPTHS), "coefficients": coeffs,
              "rows": rows,
              "interpretation": "Inverse-operator approximation only; no predictive or novelty claim.",
              "gate": {"status": "descriptive", "heldout_training": False,
                       "one_global_coefficient_vector": True}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    plan_file = Path(__file__).with_name("plan.json")
    main(Path(__file__).with_name("results.json"), plan_file)
