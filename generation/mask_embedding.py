"""Deterministic, batch-invariant embeddings for binary observation masks.

This is an engineering primitive, not a novelty claim. Frequencies are sampled
once from a seed and shared across columns, so row and column permutations have
predictable equivariance and no batch statistics enter the representation.
"""
from __future__ import annotations

import hashlib
import numpy as np


def _frequencies(num_frequencies: int, seed: int) -> np.ndarray:
    if not isinstance(num_frequencies, (int, np.integer)) or num_frequencies < 1:
        raise ValueError("num_frequencies must be a positive integer")
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    frequencies = rng.uniform(0.25, 2.0, size=int(num_frequencies))
    if not np.isfinite(frequencies).all() or np.any(frequencies == 0):
        raise RuntimeError("invalid seeded frequencies")
    return frequencies


def embed_mask(mask: np.ndarray, *, num_frequencies: int = 4, seed: int = 0) -> np.ndarray:
    """Return paired sine/cosine features for a finite binary mask.

    Output shape is ``(rows, columns, 2 * num_frequencies)``. The same seeded
    frequency vector is used for every column, preserving feature permutation
    equivariance. Constant columns remain finite and are intentionally retained.
    """
    values = np.asarray(mask, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("mask must be a non-empty two-dimensional array")
    if not np.isfinite(values).all() or not np.isin(values, [0.0, 1.0]).all():
        raise ValueError("mask must contain only finite 0/1 values")
    frequencies = _frequencies(num_frequencies, seed)
    angles = values[:, :, None] * frequencies[None, None, :]
    embedded = np.concatenate((np.sin(angles), np.cos(angles)), axis=-1)
    if not np.isfinite(embedded).all():
        raise RuntimeError("embedding produced non-finite values")
    return embedded


def contract_digest(num_frequencies: int = 4, seed: int = 0) -> str:
    values = _frequencies(num_frequencies, seed).tobytes()
    return hashlib.sha256(values).hexdigest()
