#!/usr/bin/env python3
"""Compute apples-to-apples validation/test baselines for a frozen run.

No model is loaded or trained. Test rows use the exact RNG convention from the
training script: encode all validation sets, then all test sets with one RNG.
"""
import argparse, json, math, random
from collections import Counter
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_source_coherence import load_original, mapping_for, encode, sha256


def scores(train_sets, rows, mapping):
    vocab = mapping.compute_vocab_size()
    # Unigram: train targets only, scored on the supplied rows (including EOS).
    uni = Counter(t for s in train_sets for t in mapping.cluster_ids_to_token_ids(s)[1:])
    uden = sum(uni.values()) + vocab
    uloss = [-math.log((uni[t] + 1) / uden) for row in rows for t in row[1:]]

    # Expected adjacent-transition counts under uniformly random permutations.
    # Each atom appears first/last with probability1/k, and each ordered pair
    # is adjacent with probability1/k. Normalize CONDITIONALLY on predecessor.
    transitions = {}
    for atoms in train_sets:
        tokens = mapping.cluster_ids_to_token_ids(atoms)[1:-1]
        k = len(tokens)
        for token in tokens:
            transitions.setdefault(mapping.bos_token_id, Counter())[token] += 1.0/k
            dest = transitions.setdefault(token, Counter())
            dest[mapping.eos_token_id] += 1.0/k
            for other in tokens:
                if token != other: dest[other] += 1.0/k
    ploss=[]
    for row in rows:
        for previous,target in zip(row,row[1:]):
            counts=transitions.get(previous, Counter());den=sum(counts.values())+vocab
            ploss.append(-math.log((counts[target]+1)/den))
    normalization_errors=[abs(sum((counts[t]+1)/(sum(counts.values())+vocab) for t in range(vocab))-1) for counts in transitions.values()]
    return {"unigram_nll": float(np.mean(uloss)),
            "permutation_symmetric_conditional_pair_nll": float(np.mean(ploss)),
            "pair_alpha":1.0,"conditional_probability_max_sum_error":max(normalization_errors),
            "tokens_scored":len(uloss)}


def main(argv=None):
    p = argparse.ArgumentParser(); p.add_argument("--run", type=Path, default=ROOT / "results/source-coherence-v1")
    args = p.parse_args(argv)
    manifest = json.loads((args.run / "manifest.json").read_text())
    data, _, _, _ = load_original(ROOT / "papers/clusters.json")
    mapping = mapping_for(data)
    train = [tuple(x) for x in manifest["splits"]["train_sets"]]
    val = [tuple(x) for x in manifest["splits"]["val_sets"]]
    test = [tuple(x) for x in manifest["splits"]["test_sets"]]
    rng = random.Random(int(manifest["seed"]) + 1)
    val_rows = [encode(s, mapping, rng) for s in val]
    test_rows = [encode(s, mapping, rng) for s in test]
    out = {"protocol": "baseline-only; no model retraining; train-fit alpha=1; exact val/test RNG rows",
           "run_manifest_sha256": sha256(args.run / "manifest.json"),
           "train_sets": len(train), "val_sets": len(val), "test_sets": len(test),
           "validation": scores(train, val_rows, mapping),
           "test": scores(train, test_rows, mapping)}
    path = args.run / "baseline-audit-conditional.json"
    if path.exists():raise FileExistsError("Preserve existing baseline audit")
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__": main()
