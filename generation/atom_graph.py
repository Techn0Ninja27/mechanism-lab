"""Deterministic atom co-occurrence graph utilities.

The graph is an evidence index over paper atom sets. An edge means that two
atoms occurred in the same source paper; it is not a causal or semantic claim.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_original(path: Path) -> tuple[dict[int, str], list[tuple[int, ...]], dict[str, int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    definitions = {int(k): str(v.get("name", "")) for k, v in data["clusters"].items()}
    paper_atoms: dict[str, set[int]] = defaultdict(set)
    for row in data["atom_assignments"]:
        atom = int(row["cluster"])
        if atom >= 0 and atom in definitions:
            paper_atoms[str(row["paper_id"])].add(atom)
    sets = sorted({tuple(sorted(atoms)) for atoms in paper_atoms.values() if len(atoms) >= 2})
    return definitions, sets, {"source_units": sum(len(x) for x in paper_atoms.values()), "papers": len(paper_atoms), "sets": len(sets)}


def split_sets(sets: list[tuple[int, ...]], seed: int = 73031) -> tuple[list[tuple[int, ...]], ...]:
    shuffled = list(sets)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    return shuffled[: int(.70 * n)], shuffled[int(.70 * n): int(.85 * n)], shuffled[int(.85 * n):]


def edges(sets: Iterable[tuple[int, ...]]) -> Counter[tuple[int, int]]:
    out: Counter[tuple[int, int]] = Counter()
    for atoms in sets:
        out.update(itertools.combinations(atoms, 2))
    return out


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


def tfidf_vectors(definitions: dict[int, str]) -> dict[int, dict[str, float]]:
    docs = {atom: Counter(_tokenize(text)) for atom, text in definitions.items()}
    df = Counter(token for doc in docs.values() for token in doc)
    n = len(docs)
    return {atom: {token: count / max(1, sum(doc.values())) * math.log((n + 1) / (df[token] + 1))
                  for token, count in doc.items()} for atom, doc in docs.items()}


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    denom = math.sqrt(sum(x * x for x in a.values()) * sum(x * x for x in b.values()))
    return sum(a.get(k, 0.0) * v for k, v in b.items()) / denom if denom else 0.0


def score_pairs(atoms: list[int], train_edges: Counter[tuple[int, int]], method: str,
                definitions: dict[int, str]) -> dict[tuple[int, int], float]:
    neighbors: dict[int, set[int]] = defaultdict(set)
    for (a, b), weight in train_edges.items():
        neighbors[a].add(b)
        neighbors[b].add(a)
    degrees = {a: len(neighbors[a]) for a in atoms}
    vectors = tfidf_vectors(definitions) if method == "tfidf_cosine" else {}
    scores = {}
    for a, b in itertools.combinations(atoms, 2):
        if (a, b) in train_edges:
            continue
        common = neighbors[a] & neighbors[b]
        if method == "common_neighbors":
            score = float(len(common))
        elif method == "adamic_adar":
            score = sum(1 / math.log(max(2, degrees[z])) for z in common)
        elif method == "degree_product":
            score = float(degrees[a] * degrees[b])
        elif method == "tfidf_cosine":
            score = cosine(vectors[a], vectors[b])
        else:
            raise ValueError(f"unknown method: {method}")
        scores[(a, b)] = score
    return scores


def ranking(scores: dict[tuple[int, int], float]) -> list[tuple[int, int]]:
    return [pair for pair, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]


def evaluate(ranked: list[tuple[int, int]], positives: set[tuple[int, int]], ks=(10, 50, 100)) -> dict[str, float | int]:
    out = {"positive_count": len(positives), "candidate_count": len(ranked)}
    for k in ks:
        out[f"hits@{k}"] = sum(pair in positives for pair in ranked[:k])
        out[f"precision@{k}"] = out[f"hits@{k}"] / min(k, len(ranked)) if ranked else 0.0
        out[f"recall@{k}"] = out[f"hits@{k}"] / len(positives) if positives else 0.0
    return out
