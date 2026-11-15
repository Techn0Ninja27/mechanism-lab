#!/usr/bin/env python3
"""Audit the released Alien Science corpus without downloading anything.

The audit is deliberately descriptive: it measures coverage, concentration,
duplicate/split signals, and rough scaling inputs. It does not claim that
publication frequency equals scientific quality.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import pathlib
import sqlite3
import statistics
from typing import Iterable


def gini(values: Iterable[int]) -> float:
    xs = sorted(float(x) for x in values if x >= 0)
    if not xs or sum(xs) == 0:
        return 0.0
    n = len(xs)
    return (2 * sum((i + 1) * x for i, x in enumerate(xs)) / (n * sum(xs))) - (n + 1) / n


def hhi(values: Iterable[int]) -> float:
    xs = [float(x) for x in values if x > 0]
    total = sum(xs)
    return sum((x / total) ** 2 for x in xs) if total else 0.0


def sha(text: str) -> str:
    return hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()


def count_lines(path: pathlib.Path) -> int:
    with path.open(encoding="utf-8") as f:
        return sum(1 for _ in f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[1])
    ap.add_argument("--output", type=pathlib.Path, default=pathlib.Path("results/corpus-audit.json"))
    args = ap.parse_args()
    root = args.root
    clusters = json.loads((root / "papers/clusters_80.json").read_text(encoding="utf-8"))
    assignments = clusters["atom_assignments"]
    cluster_unit_counts = collections.Counter(int(row["cluster"]) for row in assignments if int(row["cluster"]) >= 0)
    cluster_papers = collections.defaultdict(set)
    paper_atoms = collections.defaultdict(set)
    for row in assignments:
        pid, cid = row["paper_id"], int(row["cluster"])
        if cid >= 0:
            cluster_papers[cid].add(pid)
            paper_atoms[pid].add(cid)
    atom_counts = [len(cluster_papers.get(cid, ())) for cid in map(int, clusters["clusters"])]

    db = root / "data/llm-papers.db"
    con = sqlite3.connect(db)
    paper_rows = con.execute("select paper_id, venue_year, title, abstract from papers").fetchall()
    author_counts = [n for _, n in con.execute("select author_id, count(*) from paper_authors group by author_id")]
    year_counts = collections.Counter(int(y) for _, y, _, _ in paper_rows if y is not None)
    exact_title_abstract = collections.Counter(sha((t or "") + "\n" + (a or "")) for _, _, t, a in paper_rows)

    train = root / "data/coherence_dataset/coherence_train.jsonl"
    val = root / "data/coherence_dataset/coherence_val.jsonl"
    train_records = [json.loads(line) for line in train.open(encoding="utf-8")]
    val_records = [json.loads(line) for line in val.open(encoding="utf-8")]
    train_rows = [row["input_ids"] for row in train_records]
    val_rows = [row["input_ids"] for row in val_records]
    train_exact, val_exact = set(map(tuple, train_rows)), set(map(tuple, val_rows))
    train_sets = {tuple(sorted(row[1:-1])) for row in train_rows}
    val_sets = {tuple(sorted(row[1:-1])) for row in val_rows}
    train_papers = {row["paper_id"] for row in train_records}
    val_papers = {row["paper_id"] for row in val_records}
    train_paper_perm = {(row["paper_id"], row["perm_idx"]) for row in train_records}
    val_paper_perm = {(row["paper_id"], row["perm_idx"]) for row in val_records}

    result = {
        "source": {"clusters": str(root / "papers/clusters_80.json"), "sqlite": str(db)},
        "papers": {"count": len(paper_rows), "years": dict(sorted(year_counts.items())),
                   "exact_title_abstract_duplicate_groups": sum(1 for n in exact_title_abstract.values() if n > 1),
                   "exact_duplicate_rows": sum(n - 1 for n in exact_title_abstract.values() if n > 1)},
        "assignments": {"units": len(assignments), "noise_units": sum(1 for x in assignments if int(x["cluster"]) < 0),
                        "assigned_units": sum(cluster_unit_counts.values()), "papers_with_atoms": len(paper_atoms),
                        "papers_without_atoms": len(paper_rows) - len(paper_atoms),
                        "atoms": len(clusters["clusters"]),
                        "atoms_zero_paper_coverage": sum(x == 0 for x in atom_counts),
                        "paper_atoms_mean": statistics.mean(map(len, paper_atoms.values())),
                        "paper_atoms_median": statistics.median(map(len, paper_atoms.values())),
                        "cluster_paper_coverage_min_max_median": [min(atom_counts), max(atom_counts), statistics.median(atom_counts)],
                        "cluster_paper_coverage_gini": gini(atom_counts), "cluster_paper_coverage_hhi": hhi(atom_counts)},
        "authors": {"count": len(author_counts), "papers_per_author_median": statistics.median(author_counts),
                    "papers_per_author_p95": sorted(author_counts)[int(.95 * (len(author_counts) - 1))],
                    "papers_per_author_max": max(author_counts), "author_paper_count_gini": gini(author_counts)},
        "coherence_split": {"train_rows": len(train_rows), "val_rows": len(val_rows),
                            "train_distinct_sequences": len(train_exact), "val_distinct_sequences": len(val_exact),
                            "exact_sequence_overlap": len(train_exact & val_exact),
                            "atom_set_overlap": len(train_sets & val_sets),
                            "train_papers": len(train_papers), "val_papers": len(val_papers),
                            "paper_overlap": len(train_papers & val_papers),
                            "val_papers_seen_in_train_fraction": len(val_papers & train_papers) / len(val_papers),
                            "exact_paper_perm_overlap": len(train_paper_perm & val_paper_perm),
                            "val_atom_sets": len(val_sets),
                            "warning": "The released split is row-level random: many validation papers and atom sets also occur in training. This is a valid within-corpus held-out-row diagnostic, but it is not a clean unseen-paper or future-time generalization test."},
        "size_bytes": {"papers": (root / "papers").stat().st_size if (root / "papers").is_file() else None,
                       "sqlite": db.stat().st_size, "clusters_json": (root / "papers/clusters_80.json").stat().st_size,
                       "coherence_train": train.stat().st_size, "coherence_val": val.stat().st_size},
        "scaling_interpretation": {"fixed_vocabulary": "More papers improve atom co-occurrence and author coverage only while the 273-atom vocabulary remains fixed; new mechanisms are invisible unless they map to an existing atom.",
                                   "expanded_vocabulary": "Changing clustering/coverage requires rebuilding atom assignments, coherence data, availability data and both checkpoints; model dimensions do not scale automatically.",
                                   "cheap_pilot": "Use a temporally separated <=2024 corpus, preserve paper IDs through train/validation, and compare 273-atom versus expanded vocabularies on held-out 2025 combinations before any large training run."}
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
