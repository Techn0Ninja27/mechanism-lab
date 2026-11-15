"""Temporal paper/atom splits and candidate construction.

This module keeps publication-year provenance visible.  It is a diagnostic for
the fixed released atom vocabulary, not a leakage-free historical recreation:
the atom catalogue and canonical descriptions were produced from the full
corpus.  Use it to build honest grouped/temporal evaluation inputs before
training new scorers.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Mapping


@dataclass(frozen=True)
class TemporalSplit:
    """Paper atom sets partitioned by publication year."""

    train: dict[str, tuple[int, ...]]
    validation: dict[str, tuple[int, ...]]
    test: dict[str, tuple[int, ...]]
    years: dict[str, int]
    all_atom_ids: tuple[int, ...]

    def summary(self) -> dict:
        train_triples = triples_from_papers(self.train)
        validation_triples = triples_from_papers(self.validation)
        test_triples = triples_from_papers(self.test)
        return {
            "train_papers": len(self.train),
            "validation_papers": len(self.validation),
            "test_papers": len(self.test),
            "train_atoms": len({a for s in self.train.values() for a in s}),
            "catalogue_atoms": len(self.all_atom_ids),
            "train_triples": len(train_triples),
            "validation_triples": len(validation_triples),
            "test_triples": len(test_triples),
            "validation_papers_with_triples": sum(len(s) >= 3 for s in self.validation.values()),
            "test_papers_with_triples": sum(len(s) >= 3 for s in self.test.values()),
            "validation_strict_new_triples": len(validation_triples - train_triples),
            "test_strict_new_triples": len(test_triples - train_triples),
        }


def _years_from_sqlite(db_path: str | Path) -> dict[str, int]:
    with sqlite3.connect(db_path) as db:
        rows = db.execute("SELECT paper_id, venue_year FROM papers").fetchall()
    years: dict[str, int] = {}
    for pid, year in rows:
        if year is None:
            continue
        original = str(pid)
        aliases = {original, original.replace("/", "__")}
        for alias in aliases:
            previous = years.get(alias)
            if previous is not None and previous != int(year):
                raise ValueError(f"paper ID alias collision with conflicting years: {alias}")
            years[alias] = int(year)
    return years


def load_temporal_split(
    clusters_path: str | Path,
    db_path: str | Path,
    *,
    train_end: int = 2023,
    validation_year: int = 2024,
    test_year: int = 2025,
    train_only_atoms: bool = False,
) -> TemporalSplit:
    """Load canonical atom sets and assign papers to fixed year buckets.

    Duplicate assignments within a paper are removed.  Papers with no year,
    no assigned atoms, or fewer than two atoms are omitted.  If
    ``train_only_atoms`` is true, future sets are projected onto the atom IDs
    observed in train; this is useful for a fixed-vocabulary capability check.
    """
    if not (train_end < validation_year < test_year):
        raise ValueError("expected train_end < validation_year < test_year")
    data = json.loads(Path(clusters_path).read_text(encoding="utf-8"))
    years = _years_from_sqlite(db_path)
    grouped: dict[str, set[int]] = defaultdict(set)
    catalogue = {int(cid) for cid in data["clusters"]}
    for row in data["atom_assignments"]:
        cid = int(row["cluster"])
        if cid >= 0:
            grouped[str(row["paper_id"])].add(cid)
    all_sets = {pid: tuple(sorted(atoms)) for pid, atoms in grouped.items() if len(atoms) >= 2}
    train_atoms = {
        atom for pid, atoms in all_sets.items()
        if pid in years and years[pid] <= train_end for atom in atoms
    }
    if train_only_atoms:
        all_sets = {
            pid: tuple(a for a in atoms if a in train_atoms)
            for pid, atoms in all_sets.items()
        }
        all_sets = {pid: atoms for pid, atoms in all_sets.items() if len(atoms) >= 2}

    def bucket(predicate):
        return {
            pid: atoms for pid, atoms in all_sets.items()
            if pid in years and predicate(years[pid])
        }

    return TemporalSplit(
        train=bucket(lambda y: y <= train_end),
        validation=bucket(lambda y: y == validation_year),
        test=bucket(lambda y: y == test_year),
        years=years,
        all_atom_ids=tuple(sorted(catalogue)),
    )


def triples_from_papers(papers: Mapping[str, Iterable[int]]) -> set[tuple[int, int, int]]:
    """Return unique unordered triples contained in each paper atom set."""
    result: set[tuple[int, int, int]] = set()
    for atoms in papers.values():
        values = tuple(sorted(set(int(a) for a in atoms)))
        result.update(combinations(values, 3))
    return result


def mark_future_triples(
    papers: Mapping[str, Iterable[int]],
    train_triples: set[tuple[int, int, int]],
) -> list[dict]:
    """Emit target triples with a strict-new label relative to training."""
    rows = []
    for paper_id, atoms in sorted(papers.items()):
        for triple in combinations(sorted(set(atoms)), 3):
            rows.append({
                "paper_id": paper_id,
                "triple": list(triple),
                "strictly_new": triple not in train_triples,
            })
    return rows


def uniform_candidate_triples(atom_ids: Iterable[int], n: int, seed: int = 42) -> list[dict]:
    """Draw unique uniform triples with explicit attempted/unique counts."""
    ids = tuple(sorted(set(int(a) for a in atom_ids)))
    if n < 1 or len(ids) < 3:
        raise ValueError("need positive n and at least three atom IDs")
    rng = random.Random(seed)
    records = {tuple(sorted(rng.sample(ids, 3))) for _ in range(n)}
    return [{"triple": list(t), "source": "uniform", "seed": seed, "attempts": n,
             "unique_draws": len(records)} for t in sorted(records)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Write fixed-vocabulary temporal split metadata")
    parser.add_argument("--clusters", type=Path, default=Path("papers/clusters_80.json"))
    parser.add_argument("--db", type=Path, default=Path("data/llm-papers.db"))
    parser.add_argument("--output", type=Path, default=Path("results/temporal-split.json"))
    parser.add_argument("--train-end", type=int, default=2023)
    parser.add_argument("--validation-year", type=int, default=2024)
    parser.add_argument("--test-year", type=int, default=2025)
    parser.add_argument("--train-only-atoms", action="store_true")
    args = parser.parse_args()
    split = load_temporal_split(
        args.clusters, args.db, train_end=args.train_end,
        validation_year=args.validation_year, test_year=args.test_year,
        train_only_atoms=args.train_only_atoms,
    )
    mapped_paper_ids = {
        str(row["paper_id"])
        for row in json.loads(args.clusters.read_text(encoding="utf-8"))["atom_assignments"]
        if int(row["cluster"]) >= 0
    }
    year_ids = set(split.years)
    train_triples = triples_from_papers(split.train)
    payload = {
        "protocol": "fixed vocabulary temporal diagnostic; atom catalogue was built from the full corpus",
        "config": {"train_end": args.train_end, "validation_year": args.validation_year,
                   "test_year": args.test_year, "train_only_atoms": args.train_only_atoms},
        "summary": split.summary(),
        "join_audit": {
            "mapped_papers_with_year": len(mapped_paper_ids & year_ids),
            "mapped_papers_missing_year": len(mapped_paper_ids - year_ids),
            "db_papers_with_year_not_mapped": len(year_ids - mapped_paper_ids),
            "mapped_papers_total": len(mapped_paper_ids),
        },
        "train_triples": len(train_triples),
        "validation_targets": mark_future_triples(split.validation, train_triples),
        "test_targets": mark_future_triples(split.test, train_triples),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
