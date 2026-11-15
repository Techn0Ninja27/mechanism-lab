"""Reproducible train/validation/final fixture for missingness-shift studies.

The fixture is deliberately a stress test, not a claim about universal MNAR
behaviour.  Training labels and values are available only for the training
split.  Validation exposes named masks; final mode is the only mode that
returns held-out test arrays and labels.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import train_test_split


SEEDS = (9401, 9403, 9407)
N_FEATURES = 20
MASK_START = 2  # columns 0 and 1 are always observed


@dataclass(frozen=True)
class Fixture:
    seed: int
    mode: str
    train_x: np.ndarray
    train_y: np.ndarray
    validation_cases: dict[str, tuple[np.ndarray, np.ndarray]]
    metadata: dict
    test_x: np.ndarray | None = None
    test_y: np.ndarray | None = None
    final_cases: dict[str, tuple[np.ndarray, np.ndarray]] | None = None


def _split(seed: int) -> tuple[np.ndarray, ...]:
    x, y = make_classification(
        n_samples=5000, n_features=20, n_informative=5, n_redundant=5,
        weights=[0.8, 0.2], shuffle=False, random_state=seed,
    )
    # 60/20/20 stratified split, with the second split stratified as well.
    train_x, hold_x, train_y, hold_y = train_test_split(
        x, y, test_size=0.4, stratify=y, random_state=seed,
    )
    val_x, test_x, val_y, test_y = train_test_split(
        hold_x, hold_y, test_size=0.5, stratify=hold_y, random_state=seed + 1,
    )
    return train_x.astype(np.float64), train_y.astype(np.int64), val_x.astype(np.float64), val_y.astype(np.int64), test_x.astype(np.float64), test_y.astype(np.int64)


def _mask_mcar(shape: tuple[int, int], rng: np.random.Generator, probability: float) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[:, MASK_START:] = rng.random((shape[0], shape[1] - MASK_START)) < probability
    return mask


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


def _mask_mar(x: np.ndarray, mean: float, std: float, rng: np.random.Generator) -> np.ndarray:
    # The anchor is observed and only controls missingness in columns 2:20.
    z = (x[:, 0] - mean) / std
    probabilities = 0.05 + 0.50 * _sigmoid(z)
    mask = np.zeros(x.shape, dtype=bool)
    mask[:, MASK_START:] = rng.random((len(x), x.shape[1] - MASK_START)) < probabilities[:, None]
    return mask


def _mask_mnar(x: np.ndarray, mean: np.ndarray, std: np.ndarray, rng: np.random.Generator, reverse: bool) -> np.ndarray:
    mask = np.zeros(x.shape, dtype=bool)
    z = (x[:, MASK_START:] - mean[MASK_START:]) / std[MASK_START:]
    if reverse:
        z = -z
    probabilities = 0.05 + 0.50 * _sigmoid(z)
    mask[:, MASK_START:] = rng.random(probabilities.shape) < probabilities
    return mask


def _mask_block(x: np.ndarray, rng: np.random.Generator, start: int, stop: int) -> np.ndarray:
    mask = np.zeros(x.shape, dtype=bool)
    rows = rng.random(len(x)) < 0.5
    mask[rows, start:stop] = True
    return mask


def _apply(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # Assignment avoids NaN * 0 propagation and leaves anchors untouched.
    out = x.copy()
    out[mask] = np.nan
    return out


def _hash_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _case(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return _apply(x, mask), y.copy()


def load_fixture(seed: int, mode: str = "validation") -> Fixture:
    """Load one deterministic fixture.

    ``validation`` never returns test arrays or labels. ``final`` explicitly
    returns them and should only be called once after model selection.
    """
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    if mode not in {"validation", "final"}:
        raise ValueError("mode must be 'validation' or 'final'")
    train_x, train_y, val_x, val_y, test_x, test_y = _split(seed)
    train_mean = np.nanmean(train_x, axis=0)
    train_std = np.nanstd(train_x, axis=0)
    train_std = np.where(train_std > 1e-12, train_std, 1.0)
    train_rng = np.random.default_rng(seed + 100)
    train_mask = _mask_mcar(train_x.shape, train_rng, 0.2)
    train_masked = _apply(train_x, train_mask)
    val_rng = np.random.default_rng(seed + 200)
    val_masks = {
        "clean": np.zeros(val_x.shape, dtype=bool),
        "mcar_0.2": _mask_mcar(val_x.shape, val_rng, 0.2),
        "mcar_0.4": _mask_mcar(val_x.shape, val_rng, 0.4),
        "mar_anchor_x0": _mask_mar(val_x, train_mean[0], train_std[0], val_rng),
    }
    validation_cases = {name: _case(val_x, val_y, mask) for name, mask in val_masks.items()}
    metadata = {
        "seed": seed, "mode": mode, "n_samples": 5000, "n_features": 20,
        "split": "60/20/20 stratified; train_test_split random states seed and seed+1",
        "informative": 5, "redundant": 5, "weights": [0.8, 0.2], "shuffle": False,
        "always_observed_columns": [0, 1], "train_missingness": {"kind": "MCAR", "probability": 0.2, "columns": "2:"},
        "validation_conditions": list(validation_cases),
        "train_only_statistics": True,
        "statistics_used_for_mask_generation_only": True,
        "test_available": mode == "final",
        "test_label_policy": "final mode only; never use for model selection",
        "fixture_hashes": {"train_clean": _hash_array(train_x), "train_masked": _hash_array(train_masked),
                           "validation_clean": _hash_array(val_x), "test_clean": _hash_array(test_x)},
        "scope_note": "Task-specific synthetic stress fixture; no universal MNAR claim.",
    }
    if mode == "validation":
        return Fixture(seed, mode, train_masked, train_y.copy(), validation_cases, metadata)
    final_rng = np.random.default_rng(seed + 300)
    final_masks = {
        "clean": np.zeros(test_x.shape, dtype=bool),
        "mcar_0.2": _mask_mcar(test_x.shape, final_rng, 0.2),
        "mnar_own_feature": _mask_mnar(test_x, train_mean, train_std, final_rng, reverse=False),
        "mnar_reversed_sign": _mask_mnar(test_x, train_mean, train_std, final_rng, reverse=True),
        "block_mcar_2_8": _mask_block(test_x, final_rng, 2, 8),
        "block_mcar_8_14": _mask_block(test_x, final_rng, 8, 14),
    }
    metadata["final_conditions"] = list(final_masks)
    final_cases = {name: _case(test_x, test_y, mask) for name, mask in final_masks.items()}
    metadata["final_case_hashes"] = {name: _hash_array(x) for name, (x, _) in final_cases.items()}
    # Final mode includes the same validation cases and exposes held-out arrays
    # separately; callers should evaluate final_cases only once.
    return Fixture(seed, mode, train_masked, train_y.copy(), validation_cases, metadata,
                   None, test_y.copy(), final_cases)


def evaluate(y_true: np.ndarray, probability: np.ndarray, metric: str = "average_precision") -> float:
    """Evaluate AP or Brier without modifying inputs."""
    y_true = np.asarray(y_true)
    probability = np.asarray(probability, dtype=float)
    if y_true.ndim != 1 or probability.ndim != 1 or len(y_true) != len(probability):
        raise ValueError("y_true and probability must be equal-length vectors")
    if not np.isfinite(probability).all():
        raise ValueError("probability contains non-finite values")
    if metric == "average_precision":
        return float(average_precision_score(y_true, probability))
    if metric == "brier":
        return float(brier_score_loss(y_true, probability))
    raise ValueError("metric must be 'average_precision' or 'brier'")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=SEEDS, default=SEEDS[0])
    parser.add_argument("--mode", choices=["validation", "final"], default="validation")
    args = parser.parse_args()
    fixture = load_fixture(args.seed, args.mode)
    print(json.dumps(fixture.metadata, indent=2))
