"""Contract tests comparing mask representations on fresh deterministic draws."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).parents[2]))
from generation.mask_embedding import embed_mask, contract_digest

SEEDS = (16021, 16023, 16027)
ROOT = Path(__file__).parents[2]
SOURCE = ROOT / "results/improvement-phase-2026-11-16/sources/stablemiss/metric.py"


def source_row(mask, seed=0):
    rng = np.random.default_rng(seed)
    b = rng.uniform(0, 2 * np.pi, size=mask.shape[1])
    mid = mask + b[None, :]
    mid -= mid.min(axis=1, keepdims=True)
    mid /= mid.max(axis=1, keepdims=True)
    mid *= np.pi / 2
    return np.sqrt(2.0) * (np.cos(mid) + np.sin(mid))


def rejected_column(mask, seed=0):
    rng = np.random.default_rng(seed)
    b = rng.uniform(0, 2 * np.pi, size=mask.shape[1])
    mid = mask + b[None, :]
    mid -= mid.min(axis=0, keepdims=True)
    mid /= mid.max(axis=0, keepdims=True)
    mid *= np.pi / 2
    return np.sqrt(2.0) * (np.cos(mid) + np.sin(mid))


def finite(x):
    return bool(np.isfinite(x).all())


def main(out: Path):
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        mask = (rng.random((48, 6)) > 0.35).astype(float)
        mask[:, 0] = 1.0
        mask[0, 1], mask[1, 1] = 0.0, 1.0
        candidate = embed_mask(mask, seed=seed)
        zero = embed_mask(np.zeros_like(mask), seed=seed)
        one = embed_mask(np.ones_like(mask), seed=seed)
        affine_reconstruction_error = float(np.max(np.abs(candidate - (zero + mask[..., None] * (one - zero)))))
        raw = mask.copy()
        rejected = rejected_column(mask, seed)
        source = source_row(mask, seed)
        perm = rng.permutation(len(mask)); cols = rng.permutation(mask.shape[1])
        combined = np.vstack([mask, (rng.random(mask.shape) > .5).astype(float)])
        rows.append({
            "seed": seed,
            "candidate": {
                "finite": finite(candidate),
                "distinguishes_binary_values": bool(np.linalg.norm(candidate[0, 1] - candidate[1, 1]) > 1e-8),
                "row_permutation_equivariant": bool(np.array_equal(embed_mask(mask[perm], seed=seed), candidate[perm])),
                "column_permutation_equivariant": bool(np.array_equal(embed_mask(mask[:, cols], seed=seed), candidate[:, cols])),
                "batch_composition_invariant": bool(np.array_equal(embed_mask(mask, seed=seed), embed_mask(combined, seed=seed)[:len(mask)])),
                "deterministic": bool(np.array_equal(candidate, embed_mask(mask, seed=seed))),
                "exact_affine_in_raw_binary_mask": bool(affine_reconstruction_error < 1e-12),
                "affine_reconstruction_max_abs_error": affine_reconstruction_error,
                "contract_digest": contract_digest(seed=seed),
            },
            "raw": {"finite": finite(raw), "distinguishes_binary_values": True},
            "rejected_column_cos_sin": {"finite": finite(rejected), "nan_count": int(np.isnan(rejected).sum()),
                                         "finite_var_max": float(np.nanmax(np.var(rejected, axis=0)))},
            "source_row_cos_sin": {"finite": finite(source), "batch_composition_changed": bool(not np.allclose(source, source_row(combined, seed)[:len(mask)], equal_nan=True))},
        })
    result = {"seeds": list(SEEDS), "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              "prior_art": {"title": "Random Features for Large-Scale Kernel Machines", "authors": "Rahimi and Recht", "url": "https://papers.nips.cc/paper_files/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html", "role": "standard random-feature precedent; no novelty claim"},
              "rows": rows,
              "verdict": "contracts_pass_but_candidate_is_an_affine_reparameterization_of_raw_binary_mask; raw_indicator_is_simplest_preferred_baseline",
              "novelty_established": False,
              "scope": "representation contracts only; no independence, predictive utility, or research novelty claim"}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(Path(__file__).with_name("results.json"))
