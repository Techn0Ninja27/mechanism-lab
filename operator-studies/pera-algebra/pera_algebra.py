"""Source-faithful algebraic diagnostic for PERA's rank-adaptation update.

This module is deliberately a matrix diagnostic, not a PEFT implementation. It
keeps the ordinary polynomial expansion of B separate from the coefficient
weighted Hadamard expansion of A described by arXiv:2604.11841v1.
"""
from __future__ import annotations

from itertools import combinations_with_replacement
from typing import Iterable

import torch


def pair_indices(rank: int) -> list[tuple[int, int]]:
    if rank < 1:
        raise ValueError("rank must be positive")
    return list(combinations_with_replacement(range(rank), 2))


def expand_b(B: torch.Tensor) -> torch.Tensor:
    """Return [B, b_i*b_j for i<=j] as columns."""
    if B.ndim != 2:
        raise ValueError("B must be a matrix")
    pairs = pair_indices(B.shape[1])
    terms = [B[:, i] * B[:, j] for i, j in pairs]
    return torch.cat([B, torch.stack(terms, dim=1)], dim=1)


def expand_a(A: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Return [A; h_ij*(a_i*a_j) for i<=j] as rows."""
    if A.ndim != 2:
        raise ValueError("A must be a matrix")
    rank = A.shape[0]
    pairs = pair_indices(rank)
    if h.shape != (len(pairs),):
        raise ValueError("h must contain one coefficient per i<=j pair")
    terms = [h[k] * A[i] * A[j] for k, (i, j) in enumerate(pairs)]
    return torch.cat([A, torch.stack(terms, dim=0)], dim=0)


def delta_expanded(B: torch.Tensor, A: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    return expand_b(B) @ expand_a(A, h)


def delta_sum(B: torch.Tensor, A: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Independent rank-one sum for the same i<=j expansion."""
    out = B @ A
    for k, (i, j) in enumerate(pair_indices(B.shape[1])):
        out = out + h[k] * torch.outer(B[:, i] * B[:, j], A[i] * A[j])
    return out


def nonlinear_component(B: torch.Tensor, A: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    return delta_expanded(B, A, h) - B @ A


def parameter_count(m: int, n: int, rank: int) -> int:
    return rank * (m + n) + rank * (rank + 1) // 2


def expanded_width(rank: int) -> int:
    return rank + rank * (rank + 1) // 2


def rank_bound(m: int, n: int, rank: int) -> int:
    return min(m, n, expanded_width(rank))
