#!/usr/bin/env python3
"""Bounded regularization ablation on disjoint canonical atom sets.

This is a diagnostic for the observed tiny-corpus overfit. It never touches
the released checkpoint. Validation selects the checkpoint; test is evaluated
once at the selected step.
"""
import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generation.model import CausalTransformerLM
from scripts.train_scaling_pilot import collate, encode, load_canonical


def split_three(sets, seed):
    rows = list(sets)
    random.Random(seed).shuffle(rows)
    n = len(rows)
    n_test = max(1, n // 5)
    n_val = max(1, n // 5)
    test = rows[:n_test]
    val = rows[n_test:n_test + n_val]
    train = rows[n_test + n_val:]
    assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))
    return train, val, test


@torch.no_grad()
def nll(model, rows, pad):
    model.eval()
    total = count = 0
    for start in range(0, len(rows), 128):
        x, mask = collate(rows[start:start + 128], pad)
        logits = model(x, attention_mask=mask)
        valid = mask[:, 1:]
        total += F.cross_entropy(logits[:, :-1][valid], x[:, 1:][valid], reduction="sum").item()
        count += int(valid.sum())
    return total / max(count, 1)


def unigram_nll(train_rows, eval_rows, vocab_size):
    counts = Counter(token for row in train_rows for token in row[1:])
    denominator = sum(counts.values()) + vocab_size
    values = [-np.log((counts[token] + 1) / denominator)
              for row in eval_rows for token in row[1:]]
    return float(np.mean(values))


def run_condition(train_sets, val_rows, test_rows, mapping, *, seed, dropout,
                  weight_decay, steps, output_dir, max_seq_len):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = CausalTransformerLM(mapping.compute_vocab_size(), d_model=128,
                                nhead=4, num_layers=2, dim_feedforward=512,
                                max_seq_len=max_seq_len, dropout=dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001,
                                  weight_decay=weight_decay)
    rng = random.Random(seed + 991)
    fixed_batches = []
    for _ in range(steps):
        fixed_batches.append([
            encode(train_sets[rng.randrange(len(train_sets))], mapping, rng)
            for _ in range(64)
        ])
    initial = nll(model, val_rows, mapping.eos_token_id)
    best = (initial, 0)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir)
    (output_dir / "token_mapping.json").write_text(json.dumps(mapping.raw, indent=2))
    losses = []
    started = time.perf_counter()
    for step, batch in enumerate(fixed_batches, start=1):
        model.train()
        x, mask = collate(batch, mapping.eos_token_id)
        logits = model(x, attention_mask=mask)
        valid = mask[:, 1:]
        loss = F.cross_entropy(logits[:, :-1][valid], x[:, 1:][valid])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.detach().item())
        if step % 100 == 0 or step == steps:
            val_loss = nll(model, val_rows, mapping.eos_token_id)
            if val_loss < best[0]:
                best = (val_loss, step)
                output_dir.mkdir(parents=True, exist_ok=True)
                model.save(output_dir)
                (output_dir / "token_mapping.json").write_text(json.dumps(mapping.raw, indent=2))
    best_model = CausalTransformerLM.load(output_dir)
    test_loss = nll(best_model, test_rows, mapping.eos_token_id)
    return {
        "seed": seed, "train_sets": len(train_sets), "dropout": dropout,
        "weight_decay": weight_decay, "steps": steps,
        "initial_val_nll": initial, "best_val_nll": best[0],
        "best_step": best[1], "test_nll": test_loss,
        "unigram_test_nll": unigram_nll(
            [mapping.cluster_ids_to_token_ids(s) for s in train_sets],
            test_rows, mapping.compute_vocab_size()),
        "recent_train_nll": float(np.mean(losses[-25:])),
        "seconds": time.perf_counter() - started,
        "parameters": sum(p.numel() for p in model.parameters()),
        "checkpoint": str(output_dir),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--sizes", nargs="+", type=int, default=[500, 2000])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--output", type=Path, default=Path("results/regularization-ablation"))
    args = ap.parse_args()
    if args.steps < 100 or any(x < 1 for x in args.seed + args.sizes):
        ap.error("steps must be >=100 and sizes/seeds positive")
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[1]
    sets, mapping = load_canonical(root)
    train_all, val_sets, test_sets = split_three(sets, 2026)
    val_rng, test_rng = random.Random(777), random.Random(778)
    val_rows = [encode(s, mapping, val_rng) for s in val_sets]
    test_rows = [encode(s, mapping, test_rng) for s in test_sets]
    max_seq_len = max(map(len, sets)) + 2
    conditions = {
        "baseline": {"dropout": 0.0, "weight_decay": 0.01},
        "dropout": {"dropout": 0.2, "weight_decay": 0.01},
        "weight_decay": {"dropout": 0.0, "weight_decay": 0.1},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "split.json").write_text(json.dumps({"train": train_all, "val": val_sets, "test": test_sets}))
    summary = {
        "protocol": "Canonical atom sets are disjoint across train/validation/test. Validation selects step; test is evaluated once. Fixed batches are reused across regularizers within each seed/size. One fixed vocabulary; not temporal.",
        "split_sha256": hashlib.sha256((args.output / "split.json").read_bytes()).hexdigest(),
        "unique_sets": len(sets), "train_sets": len(train_all), "val_sets": len(val_sets),
        "test_sets": len(test_sets), "vocab_size": mapping.compute_vocab_size(),
        "results": [],
    }
    for seed in args.seed:
        for size in args.sizes:
            for name, cfg in conditions.items():
                out = args.output / f"seed{seed}-n{size}-{name}"
                result = run_condition(train_all[:min(size, len(train_all))], val_rows,
                                       test_rows, mapping, seed=seed, steps=args.steps,
                                       output_dir=out, max_seq_len=max_seq_len, **cfg)
                result["condition"] = name
                summary["results"].append(result)
                (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
                print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
