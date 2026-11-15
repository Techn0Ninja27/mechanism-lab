"""Typed, content-addressed compositions of atom concepts.

An atom is a named concept with a definition and provenance.  A molecule adds
typed directed relations and an operational mechanism.  The representation is
deliberately conservative: a molecule can be useful without being a new
scientific primitive, and promotion requires held-out evidence rather than a
model's novelty assertion.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


class MoleculeError(ValueError):
    """Raised when a concept composition is malformed or unsafe."""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MoleculeError(f"{name} must be a non-empty string")
    return " ".join(value.split())


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def atom_record(atom_id: str, definition: str, provenance: Iterable[Mapping[str, Any]], *, revision: int = 1) -> dict[str, Any]:
    """Create a stable atom record; provenance is copied and sorted by content."""
    if not isinstance(atom_id, str) or not atom_id.strip():
        raise MoleculeError("atom_id must be a non-empty string")
    if not isinstance(revision, int) or revision < 1:
        raise MoleculeError("revision must be a positive integer")
    raw_refs = list(provenance)
    if any(not isinstance(x, Mapping) for x in raw_refs):
        raise MoleculeError("provenance entries must be objects")
    refs = [dict(x) for x in raw_refs]
    refs.sort(key=_json)
    return {"type": "atom", "id": atom_id, "definition": _text(definition, "definition"), "revision": revision, "provenance": refs}


def _check_atom(atom: Mapping[str, Any]) -> None:
    _text(atom.get("id"), "atom.id")
    _text(atom.get("definition"), "atom.definition")
    if not isinstance(atom.get("revision"), int) or atom["revision"] < 1:
        raise MoleculeError("atom.revision must be a positive integer")
    if not isinstance(atom.get("provenance"), list):
        raise MoleculeError("atom.provenance must be a list")


def molecule_record(atoms: Iterable[Mapping[str, Any]], relations: Iterable[Mapping[str, Any]], *,
                    mechanism: str, assumptions: Iterable[str], falsifiable_prediction: str,
                    status: str = "combination", provenance: Iterable[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """Construct and validate a molecule.

    ``status`` is one of ``known``, ``combination``, or ``candidate_primitive``;
    it records a research hypothesis and is never an automated novelty label.
    """
    if status not in {"known", "combination", "candidate_primitive"}:
        raise MoleculeError("invalid molecule status")
    atom_list = [dict(a) for a in atoms]
    if not atom_list:
        raise MoleculeError("a molecule must contain at least one atom")
    for atom in atom_list:
        _check_atom(atom)
    ids = [a["id"] for a in atom_list]
    if len(set(ids)) != len(ids):
        raise MoleculeError("duplicate atom IDs")
    atom_list.sort(key=lambda x: x["id"])
    edge_list = []
    for relation in relations:
        if not isinstance(relation, Mapping):
            raise MoleculeError("relations must be objects")
        source = _text(relation.get("source"), "relation.source")
        target = _text(relation.get("target"), "relation.target")
        kind = _text(relation.get("type"), "relation.type")
        if source not in ids or target not in ids:
            raise MoleculeError("relation references an unknown atom")
        if source == target:
            raise MoleculeError("self-relations are not allowed")
        edge = {"source": source, "target": target, "type": kind}
        if "description" in relation:
            edge["description"] = _text(relation["description"], "relation.description")
        edge_list.append(edge)
    edge_list.sort(key=_json)
    _reject_cycles(ids, edge_list)
    if isinstance(assumptions, (str, bytes)):
        raise MoleculeError("assumptions must be an iterable of strings, not a string")
    assumptions = [_text(x, "assumption") for x in assumptions]
    raw_provenance = list(provenance)
    if any(not isinstance(x, Mapping) for x in raw_provenance):
        raise MoleculeError("provenance entries must be objects")
    if not assumptions:
        raise MoleculeError("assumptions must not be empty")
    record = {"type": "molecule", "atoms": atom_list, "relations": edge_list,
              "mechanism": _text(mechanism, "mechanism"), "assumptions": sorted(assumptions),
              "falsifiable_prediction": _text(falsifiable_prediction, "falsifiable_prediction"),
              "status": status, "provenance": sorted((dict(x) for x in raw_provenance), key=_json),
              "scientific_novelty_claim": False}
    record["id"] = molecule_id(record)
    return record


def _reject_cycles(ids: Iterable[str], edges: Iterable[Mapping[str, Any]]) -> None:
    graph: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        graph[edge["source"]].append(edge["target"])
    visiting: set[str] = set(); visited: set[str] = set()
    def visit(node: str) -> None:
        if node in visiting:
            raise MoleculeError("relations must form a directed acyclic graph")
        if node in visited:
            return
        visiting.add(node)
        for child in graph[node]:
            visit(child)
        visiting.remove(node); visited.add(node)
    for node in ids:
        visit(node)


def molecule_id(molecule: Mapping[str, Any]) -> str:
    """Hash all semantic fields except the derived ID, with sorted collections."""
    data = {k: v for k, v in molecule.items() if k != "id"}
    return "mol_" + _digest(data)[:32]


def promote(molecule: Mapping[str, Any], evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Promote a molecule to a reusable macro-atom after explicit evidence.

    Promotion records tested reuse; it does not establish external scientific
    novelty.  All four evidence gates are required and truthy.
    """
    if molecule.get("type") != "molecule":
        raise MoleculeError("promotion requires a molecule")
    if molecule.get("id") != molecule_id(molecule):
        raise MoleculeError("molecule ID does not match its content")
    required = ("validation", "independent_heldout_positive", "ablation", "prior_art_review")
    missing = [key for key in required if key not in evidence]
    if missing:
        raise MoleculeError("promotion evidence missing: " + ", ".join(missing))
    for key in required:
        item = evidence[key]
        if not isinstance(item, Mapping) or item.get("passed") is not True:
            raise MoleculeError(f"promotion evidence did not pass: {key}")
        if not isinstance(item.get("artifact"), str) or not item["artifact"].strip():
            raise MoleculeError(f"promotion evidence artifact missing: {key}")
        digest = item.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdefABCDEF" for c in digest):
            raise MoleculeError(f"promotion evidence sha256 invalid: {key}")
        artifact = Path(item["artifact"])
        if not artifact.is_file():
            raise MoleculeError(f"promotion evidence artifact does not exist: {key}")
        actual = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if actual.lower() != digest.lower():
            raise MoleculeError(f"promotion evidence artifact hash mismatch: {key}")
    validation_task = evidence["validation"].get("task_id")
    heldout_task = evidence["independent_heldout_positive"].get("task_id")
    if not isinstance(validation_task, str) or not validation_task.strip() or not isinstance(heldout_task, str) or not heldout_task.strip() or validation_task == heldout_task:
        raise MoleculeError("validation and held-out evidence need distinct non-empty task_id values")
    if evidence["prior_art_review"].get("human_adjudicated") is not True:
        raise MoleculeError("prior-art review must be explicitly human-adjudicated")
    macro = {"type": "macro_atom", "definition": molecule["mechanism"],
             "decomposition": {"molecule_id": molecule.get("id") or molecule_id(molecule),
                                "atoms": [dict(a) for a in molecule["atoms"]],
                                "atom_ids": [a["id"] for a in molecule["atoms"]],
                                "relations": molecule["relations"]},
             "evidence": dict(evidence), "status": "tested_reusable_abstraction",
             "scientific_novelty_claim": False}
    macro["id"] = "macro_" + _digest({k: v for k, v in macro.items() if k != "id"})[:32]
    return macro
