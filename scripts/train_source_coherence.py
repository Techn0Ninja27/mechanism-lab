#!/usr/bin/env python3
"""Train a small coherence scorer from the original cluster assignments only.

The output is deliberately a source-lineage artifact, not a scientific quality
claim.  Canonical paper atom sets are split as sets (70/15/15), and the test
set is evaluated exactly once after selecting the checkpoint on validation NLL.
"""
import argparse, hashlib, json, random, sys, platform
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from generation.model import CausalTransformerLM
from generation.token_mapping import TokenMapping


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_original(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    papers = defaultdict(set)
    units = []
    for row in data["atom_assignments"]:
        cid = int(row["cluster"])
        if cid < 0:
            continue
        pid, text = str(row["paper_id"]), str(row["text"])
        papers[pid].add(cid)
        units.append({"paper_id": pid, "cluster": cid,
                      "text_sha256": hashlib.sha256(text.encode()).hexdigest()})
    sets = sorted({tuple(sorted(v)) for v in papers.values() if len(v) >= 2})
    return data, papers, sets, units


def mapping_for(data):
    ids = sorted(int(k) for k in data["clusters"])
    n = len(ids)
    c2t = {cid: i for i, cid in enumerate(ids)}
    raw = {"cluster_to_token_id": {str(cid): tid for cid, tid in c2t.items()},
           "token_id_to_cluster": {str(tid): cid for cid, tid in c2t.items()},
           "cluster_to_token_str": {str(cid): f"<cluster_{cid}>" for cid in ids},
           "token_str_to_cluster": {f"<cluster_{cid}>": cid for cid in ids},
           "bos_token_id": n, "eos_token_id": n, "n_clusters": n,
           "vocab_path": "source-original", "created_at": "2026-11-16"}
    return TokenMapping(raw)


def split_sets(sets, seed):
    shuffled = list(sets)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    a, b = int(n * .70), int(n * .85)
    return shuffled[:a], shuffled[a:b], shuffled[b:]


def encode(s, mapping, rng):
    order = list(s); rng.shuffle(order)
    return mapping.cluster_ids_to_token_ids(order)


def collate(rows, pad):
    x = torch.full((len(rows), max(map(len, rows))), pad, dtype=torch.long)
    mask = torch.zeros_like(x, dtype=torch.bool)
    for i, row in enumerate(rows):
        x[i, :len(row)] = torch.tensor(row)
        mask[i, :len(row)] = True
    return x, mask


@torch.no_grad()
def nll(model, rows, pad):
    model.eval(); total = count = 0
    for start in range(0, len(rows), 128):
        x, mask = collate(rows[start:start + 128], pad)
        z = model(x, attention_mask=mask)
        valid = mask[:, 1:]
        total += F.cross_entropy(z[:, :-1][valid], x[:, 1:][valid], reduction="sum").item()
        count += int(valid.sum())
    return total / count


def baselines(train, val, vocab):
    uni = Counter(t for row in train for t in row[1:])
    pair = Counter((row[i], row[i + 1]) for row in train for i in range(len(row) - 1))
    prev = Counter(row[i] for row in train for i in range(len(row) - 1))
    u_den = sum(uni.values()) + vocab
    losses_u, losses_p = [], []
    for row in val:
        for i, target in enumerate(row[1:]):
            losses_u.append(-np.log((uni[target] + 1) / u_den))
            losses_p.append(-np.log((pair[(row[i], target)] + 1) / (prev[row[i]] + vocab)))
    return float(np.mean(losses_u)), float(np.mean(losses_p))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--clusters", type=Path, default=ROOT / "papers/clusters.json")
    p.add_argument("--output", type=Path, default=ROOT / "results/source-coherence-v1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args(argv)
    if args.max_steps < 1 or args.eval_every < 1: p.error("steps must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        p.error(f"refusing non-empty output directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    # Capture exact implementation bytes before training, never reconstruct later.
    snapshots = args.output / "source_snapshot"
    files = [Path(__file__).resolve(), ROOT / "generation/model.py", ROOT / "generation/token_mapping.py"]
    source_hashes = {}
    for source in files:
        relative = source.relative_to(ROOT)
        archived = snapshots / relative
        archived.parent.mkdir(parents=True, exist_ok=True)
        with archived.open("xb") as f:
            f.write(source.read_bytes())
        source_hashes[relative.as_posix()] = sha256(archived)
    start_record = {
        "source_hashes": source_hashes, "clusters_sha256": sha256(args.clusters),
        "seed": args.seed, "max_steps": args.max_steps, "eval_every": args.eval_every,
        "batch_size": args.batch_size, "device": "cpu", "threads": 2,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__},
        "scope": "Reproducibility rerun on an exposed split; not a fresh generalization test.",
    }
    (args.output / "run-start.json").write_text(json.dumps(start_record, indent=2))
    torch.set_num_threads(2); torch.manual_seed(args.seed)
    data, papers, sets, units = load_original(args.clusters)
    train, val, test = split_sets(sets, args.seed)
    mapping = mapping_for(data)
    print(json.dumps({"source_units": len(units), "mapped_papers": len(papers),
                      "canonical_sets": len(sets), "train_sets": len(train),
                      "val_sets": len(val), "test_sets": len(test),
                      "clusters": len(data["clusters"]) }), flush=True)
    rng = random.Random(args.seed + 1)
    val_rows = [encode(s, mapping, rng) for s in val]
    test_rows = [encode(s, mapping, rng) for s in test]
    max_len = max(map(len, sets)) + 2
    model = CausalTransformerLM(mapping.compute_vocab_size(), d_model=128, nhead=4,
                                num_layers=2, dim_feedforward=512,
                                max_seq_len=max_len, dropout=.1)
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    best = {"val_nll": float("inf"), "step": 0, "state_dict": None}
    train_rng = random.Random(args.seed + 2)
    history = []
    for step in range(1, args.max_steps + 1):
        rows = [encode(train[train_rng.randrange(len(train))], mapping, train_rng)
                for _ in range(args.batch_size)]
        x, mask = collate(rows, mapping.eos_token_id); model.train()
        z = model(x, attention_mask=mask); valid = mask[:, 1:]
        loss = F.cross_entropy(z[:, :-1][valid], x[:, 1:][valid])
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step == 1 or step % args.eval_every == 0 or step == args.max_steps:
            v = nll(model, val_rows, mapping.eos_token_id)
            history.append({"step": step, "val_nll": v})
            if v < best["val_nll"]:
                best = {"val_nll": v, "step": step,
                        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    model.load_state_dict(best.pop("state_dict"))
    args.output.mkdir(parents=True, exist_ok=True)
    ckpt = args.output / "checkpoint"
    model.save(ckpt)
    mapping_json = json.dumps(mapping.raw, indent=2)
    (args.output / "token_mapping.json").write_text(mapping_json)
    # Package the mapping beside model.pt so generation.scoring.load_model can
    # never accidentally pair this checkpoint with a different vocabulary.
    (ckpt / "token_mapping.json").write_text(mapping_json)
    (args.output / "freeze.json").write_text(json.dumps({"selected_step": int(best["step"]),
        "selected_val_nll": float(best["val_nll"]), "test_evaluation": "not yet run",
        "checkpoint_sha256": sha256(ckpt / "model.pt"),
        "mapping_sha256": sha256(ckpt / "token_mapping.json"),
        "run_start_sha256": sha256(args.output / "run-start.json")}, indent=2))
    # The checkpoint and selection record are durable before the single test pass.
    test_nll = nll(model, test_rows, mapping.eos_token_id)
    train_rows_for_base = [mapping.cluster_ids_to_token_ids(s) for s in train]
    uni, pair = baselines(train_rows_for_base, val_rows, mapping.compute_vocab_size())
    unit_path = args.output / "source_units.jsonl"
    with unit_path.open("w", encoding="utf-8") as f:
        for u in sorted(units, key=lambda x: (x["paper_id"], x["cluster"], x["text_sha256"])):
            f.write(json.dumps(u, sort_keys=True) + "\n")
    manifest = {"protocol": "original assignments; grouped canonical sets; 70/15/15; non-temporal; no availability repair",
                "script_sha256": source_hashes["scripts/train_source_coherence.py"],
                "source_hashes": source_hashes,
                "run_start_sha256": sha256(args.output / "run-start.json"),
                "seed": args.seed, "clusters_sha256": sha256(args.clusters),
                "assignment_join": {"mapped_papers": len(papers), "source_units": len(units),
                                    "canonical_sets": len(sets), "source_units_sha256": sha256(unit_path)},
                "splits": {"train_sets": [list(x) for x in train], "val_sets": [list(x) for x in val],
                           "test_sets": [list(x) for x in test]},
                "source_unit_text_hashes": "source_units.jsonl (one SHA-256 per original non-noise unit)",
                "mapping_sha256": sha256(args.output / "token_mapping.json")}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    summary = {"protocol": manifest["protocol"], "seed": args.seed,
               "counts": manifest["assignment_join"], "architecture": model.config,
               "best_step": int(best["step"]),
               "validation_history": history, "selected_val_nll": float(best["val_nll"]),
               "test_nll_once_after_selection": float(test_nll),
               "unigram_val_nll": uni, "pair_frequency_val_nll": pair,
               "checkpoint": str(ckpt)}
    # A changed source or input invalidates this run rather than silently updating provenance.
    if any(sha256(ROOT / name) != digest for name, digest in source_hashes.items()) or sha256(args.clusters) != start_record["clusters_sha256"]:
        raise RuntimeError("Source/input changed during training; preserve this run as invalid")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ["best_step", "selected_val_nll", "test_nll_once_after_selection", "unigram_val_nll", "pair_frequency_val_nll"]}, indent=2))


if __name__ == "__main__": main()
