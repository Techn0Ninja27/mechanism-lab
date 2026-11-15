"""Candidate-source strategies for Alien search.

The original sampler explores the coherence model's local distribution.  This
module adds a controlled uniform source and a hybrid source, while keeping the
number of attempted draws explicit so comparisons do not quietly reward one
method with a larger search budget.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import torch

from generation.generation import generate_sequences, generate_sequences_batched
from generation.token_mapping import TokenMapping


@dataclass(frozen=True)
class CandidatePool:
    """A canonical candidate pool plus accounting for the search budget."""

    sets: tuple[tuple[int, ...], ...]
    source: str
    requested_draws: int
    attempted_draws: int
    generated_records: int
    unique_sets: int
    seed: int
    k: int

    def metadata(self) -> dict:
        """Return JSON-friendly metadata suitable for a results manifest."""
        # The complete candidate list belongs in the pool artifact, not in the
        # run config. Keeping accounting separate avoids duplicating large
        # pools in every result JSON.
        return {
            "source": self.source,
            "requested_draws": self.requested_draws,
            "attempted_draws": self.attempted_draws,
            "generated_records": self.generated_records,
            "unique_sets": self.unique_sets,
            "seed": self.seed,
            "k": self.k,
        }


def canonicalize(records: Iterable[Iterable[int]], k: int) -> tuple[tuple[int, ...], ...]:
    """Deduplicate fixed-size atom records into sorted canonical tuples."""
    if k < 2:
        raise ValueError("k must be at least 2")
    result: set[tuple[int, ...]] = set()
    for record in records:
        values = tuple(sorted(int(x) for x in record))
        if len(values) == k and len(set(values)) == k:
            result.add(values)
    return tuple(sorted(result))


def uniform_sets(
    atom_ids: Iterable[int],
    *,
    num_draws: int,
    k: int,
    seed: int,
    unique_only: bool = True,
) -> CandidatePool:
    """Sample uniformly from atom sets with a deterministic local RNG.

    ``attempted_draws`` always equals ``num_draws``.  When ``unique_only`` is
    true, repeated draws are counted rather than silently retried, making the
    comparison with model sampling auditable.
    """
    ids = tuple(sorted({int(x) for x in atom_ids}))
    if num_draws < 1 or k < 2 or k > len(ids):
        raise ValueError("num_draws must be positive and k must fit atom_ids")
    rng = random.Random(seed)
    records = [tuple(sorted(rng.sample(ids, k))) for _ in range(num_draws)]
    sets = canonicalize(records, k) if unique_only else tuple(records)
    if not unique_only:
        # The caller may intentionally preserve duplicates, but this module's
        # CandidatePool still reports the number of distinct scoring targets.
        unique_count = len(set(sets))
    else:
        unique_count = len(sets)
    return CandidatePool(
        sets=sets,
        source="uniform",
        requested_draws=num_draws,
        attempted_draws=num_draws,
        generated_records=len(records),
        unique_sets=unique_count,
        seed=seed,
        k=k,
    )


def uniform_exhaustive(atom_ids: Iterable[int], *, k: int) -> CandidatePool:
    """Enumerate every combination, for a finite-space control condition."""
    ids = tuple(sorted({int(x) for x in atom_ids}))
    if k < 2 or k > len(ids):
        raise ValueError("k must be at least 2 and fit atom_ids")
    sets = tuple(combinations(ids, k))
    return CandidatePool(
        sets=sets,
        source="uniform_exhaustive",
        requested_draws=len(sets),
        attempted_draws=len(sets),
        generated_records=len(sets),
        unique_sets=len(sets),
        seed=0,
        k=k,
    )


def autoregressive_sets(
    model,
    token_mapping: TokenMapping,
    *,
    num_draws: int,
    k: int,
    seed: int,
    temperature: float = 1.0,
    top_k: int = 0,
    device: str = "cpu",
    batch_size: int = 1,
) -> CandidatePool:
    """Sample sets from the released coherence model with accounting."""
    if num_draws < 1 or k < 2:
        raise ValueError("num_draws must be positive and k must be at least 2")
    if device not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"unsupported device: {device}")

    # The upstream generators use global RNG state. Snapshot every relevant
    # backend, seed all of them, and restore in finally so caller randomness is
    # unchanged even when model execution raises.
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    mps_state = (
        torch.mps.get_rng_state()
        if hasattr(torch, "mps") and hasattr(torch.mps, "get_rng_state")
        and torch.backends.mps.is_available()
        else None
    )
    generator = generate_sequences if batch_size == 1 else generate_sequences_batched
    kwargs = {} if batch_size == 1 else {"batch_size": batch_size}
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if torch.backends.mps.is_available() and hasattr(torch.mps, "manual_seed"):
            torch.mps.manual_seed(seed)
        records = generator(
            model,
            token_mapping,
            num_samples=num_draws,
            max_length=k + 1,
            temperature=temperature,
            top_k=top_k,
            device=device,
            **kwargs,
        )
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)
    sets = canonicalize((row["clusters"] for row in records), k)
    return CandidatePool(
        sets=sets,
        source="autoregressive",
        requested_draws=num_draws,
        attempted_draws=num_draws,
        generated_records=len(records),
        unique_sets=len(sets),
        seed=seed,
        k=k,
    )


def hybrid_sets(
    model,
    token_mapping: TokenMapping,
    atom_ids: Iterable[int],
    *,
    num_draws: int,
    k: int,
    seed: int,
    autoregressive_fraction: float = 0.5,
    temperature: float = 1.0,
    top_k: int = 0,
    device: str = "cpu",
    batch_size: int = 1,
) -> CandidatePool:
    """Mix model and uniform draws under one matched draw budget."""
    if num_draws < 1:
        raise ValueError("num_draws must be positive")
    if k < 2:
        raise ValueError("k must be at least 2")
    if not 0 <= autoregressive_fraction <= 1:
        raise ValueError("autoregressive_fraction must be between 0 and 1")
    n_model = round(num_draws * autoregressive_fraction)
    n_uniform = num_draws - n_model
    model_pool = autoregressive_sets(
        model,
        token_mapping,
        num_draws=n_model,
        k=k,
        seed=seed,
        temperature=temperature,
        top_k=top_k,
        device=device,
        batch_size=batch_size,
    ) if n_model else None
    uniform_pool = uniform_sets(
        atom_ids,
        num_draws=n_uniform,
        k=k,
        seed=seed + 1,
    ) if n_uniform else None
    records = []
    if model_pool:
        records.extend(model_pool.sets)
    if uniform_pool:
        records.extend(uniform_pool.sets)
    sets = canonicalize(records, k)
    return CandidatePool(
        sets=sets,
        source=f"hybrid(ar={autoregressive_fraction:.3f})",
        requested_draws=num_draws,
        attempted_draws=(model_pool.attempted_draws if model_pool else 0)
        + (uniform_pool.attempted_draws if uniform_pool else 0),
        generated_records=(model_pool.generated_records if model_pool else 0)
        + (uniform_pool.generated_records if uniform_pool else 0),
        unique_sets=len(sets),
        seed=seed,
        k=k,
    )
