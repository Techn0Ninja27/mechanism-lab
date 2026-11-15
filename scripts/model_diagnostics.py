#!/usr/bin/env python3
"""Bounded diagnostics for the released Alien Science checkpoints.

This script does not train or modify checkpoints. It measures atom popularity,
coherence/availability score behavior on a reproducible candidate sample, and
whether the two scores are redundant. It is intentionally small enough for a
24 GB Apple Silicon machine.
"""
import argparse
import json
import pathlib
import sqlite3
import sys
from collections import Counter

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from generation.scoring import (load_author_embeddings, load_availability_model,
                                load_model, score_sets_alien)
from generation.sample import score_sets_coherence


def paper_atom_sets(path: pathlib.Path) -> list[tuple[int, ...]]:
    data = json.loads(path.read_text())
    grouped: dict[str, set[int]] = {}
    for row in data.get("atom_assignments", []):
        c, pid = row.get("cluster"), row.get("paper_id")
        if c is not None and c != -1 and pid is not None:
            grouped.setdefault(pid, set()).add(int(c))
    return [tuple(sorted(v)) for v in grouped.values() if len(v) >= 2]


def dataset_overlap(root: pathlib.Path) -> dict:
    """Count paper IDs shared by released train/validation JSONL files."""
    def ids(path: pathlib.Path) -> set[str]:
        return {json.loads(line)["paper_id"] for line in path.open()}
    train = ids(root / "data/coherence_dataset/coherence_train.jsonl")
    val = ids(root / "data/coherence_dataset/coherence_val.jsonl")
    return {"train_papers": len(train), "val_papers": len(val),
            "overlap_papers": len(train & val),
            "train_only_papers": len(train - val), "val_only_papers": len(val - train)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-random", type=int, default=512)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    ap.add_argument("--output", type=pathlib.Path, default=ROOT / "results/model-diagnostics/model_diagnostics.json")
    args = ap.parse_args()
    if args.n_random < 1:
        ap.error("--n-random must be positive")
    if args.device == "mps" and not torch.backends.mps.is_available():
        ap.error("MPS is not available")
    rng = np.random.default_rng(args.seed)
    torch.set_num_threads(4)
    cmodel, ctm = load_model(ROOT / "models/coherence_model")
    amodel, atm = load_availability_model(ROOT / "models/availability_model")
    bank, _ = load_author_embeddings(ROOT / "models/availability_model", device="cpu")
    atom_ids = sorted(map(int, ctm.raw["cluster_to_token_id"]))
    if atom_ids != sorted(map(int, atm.raw["cluster_to_token_id"])):
        raise ValueError("checkpoint atom vocabularies differ")

    paper_sets = paper_atom_sets(ROOT / "papers/clusters_80.json")
    counts = Counter(c for s in paper_sets for c in s)
    n_papers = len(paper_sets)
    random_sets = {tuple(sorted(rng.choice(atom_ids, 3, replace=False).tolist()))
                   for _ in range(args.n_random * 4)}
    random_sets = sorted(random_sets)[:args.n_random]
    real_pool = sorted({tuple(sorted(rng.choice(np.array(s), 3, replace=False).tolist()))
                        for s in paper_sets if len(s) >= 3})
    if len(real_pool) < args.n_random:
        ap.error(f"only {len(real_pool)} unique real triplets; reduce --n-random")
    real_triplets = [real_pool[i] for i in rng.choice(len(real_pool), args.n_random, replace=False)]
    source_by_set = {s: "random" for s in random_sets}
    for s in real_triplets:
        source_by_set[s] = "real" if s not in source_by_set else "both"
    candidate_sets = sorted(source_by_set)
    coh = score_sets_coherence(candidate_sets, cmodel, ctm, device=args.device, batch_size=64)
    unavail = score_sets_alien(candidate_sets, amodel, atm, bank, device=args.device,
                               batch_size=64, top_k_authors=10)
    rows = []
    for s, c, u in zip(candidate_sets, coh, unavail):
        rows.append({"cluster_ids": list(s), "coherence": float(c),
                     "unavailability": float(u),
                     "atom_frequency_sum": int(sum(counts[x] for x in s)),
                     "atom_frequency_mean": float(np.mean([counts[x] for x in s])),
                     "source": source_by_set[s]})
    c_arr, u_arr = np.asarray(coh), np.asarray(unavail)
    freq = np.asarray([sum(counts[x] for x in s) for s in candidate_sets], dtype=float)
    def corr(x, y):
        return float(np.corrcoef(x, y)[0, 1]) if np.std(x) and np.std(y) else 0.0
    source_correlations = {}
    for source in ("random", "real", "both"):
        mask = np.asarray([source_by_set[s] == source for s in candidate_sets])
        if mask.sum() >= 3:
            source_correlations[source] = {
                "count": int(mask.sum()),
                "coherence_unavailability": corr(c_arr[mask], u_arr[mask]),
                "coherence_frequency": corr(c_arr[mask], freq[mask]),
                "unavailability_frequency": corr(u_arr[mask], freq[mask]),
            }
    result = {
        "config": {"seed": args.seed, "n_random": args.n_random, "device": args.device,
                   "candidate_count": len(candidate_sets), "paper_count": n_papers,
                   "atom_count": len(atom_ids), "dataset_overlap": dataset_overlap(ROOT)},
        "correlations": {"coherence_unavailability": corr(c_arr, u_arr),
                          "coherence_frequency": corr(c_arr, freq),
                          "unavailability_frequency": corr(u_arr, freq)},
        "source_correlations": source_correlations,
        "atom_frequency": {"unique_atoms_used": len(counts),
                            "min": min(counts.values()), "max": max(counts.values()),
                            "median": float(np.median(list(counts.values())))},
        "score_summary": {"coherence_mean": float(c_arr.mean()), "coherence_std": float(c_arr.std()),
                          "unavailability_mean": float(u_arr.mean()), "unavailability_std": float(u_arr.std())},
        "top_coherence": sorted(rows, key=lambda x: x["coherence"], reverse=True)[:20],
        "top_unavailability": sorted(rows, key=lambda x: x["unavailability"], reverse=True)[:20],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["config"] | {"correlations": result["correlations"],
                                         "atom_frequency": result["atom_frequency"]}, indent=2))


if __name__ == "__main__":
    main()
