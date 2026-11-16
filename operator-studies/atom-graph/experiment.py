"""Run the frozen atom graph link-recovery baseline."""
from __future__ import annotations

import itertools
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from generation.atom_graph import edges, evaluate, load_original, ranking, score_pairs, sha256, split_sets

STUDY = Path(__file__).resolve().parent
SOURCE = ROOT / "papers" / "clusters.json"
OUT = ROOT / "reports" / "atom-graph"
KS = (10, 50, 100)
METHODS = ("common_neighbors", "adamic_adar", "degree_product", "tfidf_cosine")


def ranked_random(candidates: list[tuple[int, int]], seed: int) -> list[tuple[int, int]]:
    out = list(candidates)
    random.Random(seed).shuffle(out)
    return out


def evaluate_split(name, train, target, definitions, atoms, train_edges):
    target_edges = edges(target)
    positives = set(target_edges) - set(train_edges)
    candidates = sorted(set(itertools.combinations(atoms, 2)) - set(train_edges))
    all_results = {}
    expected = len(positives) / len(candidates) if candidates else 0.0
    expected_result = evaluate(candidates, positives, KS)
    expected_result.update({f"expected_precision@{k}": expected for k in KS})
    all_results["random_expected"] = expected_result
    all_results["seeded_random"] = evaluate(ranked_random(candidates, 73031 + (0 if name == "validation" else 1)), positives, KS)
    for method in METHODS:
        ranked = ranking(score_pairs(atoms, train_edges, method, definitions))
        all_results[method] = evaluate(ranked, positives, KS)
    return {"split": name, "train_sets": len(train), "target_sets": len(target),
            "train_edge_count": len(train_edges), "target_unique_edge_count": len(target_edges),
            "unseen_positive_edge_count": len(positives), "candidate_count": len(candidates),
            "results": all_results}


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {OUT}")
    definitions, sets, counts = load_original(SOURCE)
    train, validation, test = split_sets(sets, 73031)
    atoms = sorted(definitions)
    train_edges = edges(train)
    OUT.mkdir(parents=True, exist_ok=True)
    source_hashes = {
        "papers/clusters.json": sha256(SOURCE),
        "operator-studies/atom-graph/plan.json": sha256(STUDY / "plan.json"),
        "operator-studies/atom-graph/experiment.py": sha256(STUDY / "experiment.py"),
        "generation/atom_graph.py": sha256(ROOT / "generation" / "atom_graph.py"),
    }
    freeze = {"source_hashes": source_hashes, "seed": 73031, "split": [len(train), len(validation), len(test)],
              "methods": list(METHODS), "ks": list(KS), "scope": "single frozen descriptive evaluation; no test tuning"}
    (OUT / "freeze.json").write_text(json.dumps(freeze, indent=2) + "\n")
    validation_result = evaluate_split("validation", train, validation, definitions, atoms, train_edges)
    test_result = evaluate_split("test", train, test, definitions, atoms, train_edges)
    if any(sha256(ROOT / path) != digest for path, digest in source_hashes.items()):
        raise RuntimeError("source changed during evaluation")
    result = {"study": "atom-graph-v1", "status": "complete", "counts": counts,
              "source_hashes": source_hashes, "split": {"seed": 73031, "train": len(train),
              "validation": len(validation), "test": len(test), "train_edge_count": len(train_edges)},
              "validation": validation_result, "test": test_result,
              "interpretation": "Undirected co-occurrence edge recovery only; not causal, useful, novel, or molecule promotion evidence.",
              "embedding_scope": "TF-IDF over all definition strings; transductive descriptive vocabulary.",
              "citation_weights": "none; every paper contributes weight one"}
    (OUT / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "validation": validation_result["results"], "test": test_result["results"]}, indent=2))


if __name__ == "__main__":
    main()
