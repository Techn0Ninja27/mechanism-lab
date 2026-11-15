import json
from pathlib import Path

import pytest

from generation.provenance import (ProvenanceError, assignment_records, audit_provenance,
                                   compare_paper_ids, validate_candidate_ids)


def write_json(path: Path, data: dict):
    path.write_text(json.dumps(data))


def test_audit_requires_exact_mapping_ids(tmp_path):
    clusters = {"clusters": {"1": {"name": "a"}}, "atom_assignments": [{"paper_id": "p", "text": "x", "cluster": 1}]}
    write_json(tmp_path / "c.json", clusters)
    write_json(tmp_path / "m.json", {"cluster_to_token_id": {"2": 1}})
    with pytest.raises(ProvenanceError):
        audit_provenance(tmp_path / "c.json", coherence_mapping=tmp_path / "m.json")


def test_cross_file_agreement_and_candidate_validation(tmp_path):
    a = {"clusters": {"1": {}, "2": {}}, "atom_assignments": [{"paper_id": "p", "text": "x", "cluster": 1}, {"paper_id": "q", "text": "y", "cluster": 2}]}
    b = {"clusters": {"1": {}, "2": {}}, "atom_assignments": [{"paper_id": "p", "text": "x", "cluster": 2}, {"paper_id": "q", "text": "y", "cluster": 1}]}
    write_json(tmp_path / "a.json", a); write_json(tmp_path / "b.json", b)
    report = audit_provenance(tmp_path / "a.json", canonical_clusters=tmp_path / "b.json")
    assert report["canonical_label_agreement"] == 0
    assert report["canonical_label_mismatch_warning"]
    assert report["safe_for_canonical_join"] is False
    assert validate_candidate_ids([2, 1], report) == (2, 1)
    with pytest.raises(ProvenanceError): validate_candidate_ids([3], report)


def test_require_safe_fails_closed(tmp_path):
    a = {"clusters": {"1": {}}, "atom_assignments": [{"paper_id": "p", "text": "x", "cluster": 1}]}
    b = {"clusters": {"1": {}}, "atom_assignments": [{"paper_id": "p", "text": "x", "cluster": -1}]}
    write_json(tmp_path / "a.json", a); write_json(tmp_path / "b.json", b)
    with pytest.raises(ProvenanceError): audit_provenance(tmp_path / "a.json", tmp_path / "b.json", require_safe=True)


def test_compare_paper_ids(tmp_path):
    write_json(tmp_path / "c.json", {"atom_assignments": [{"paper_id": "p", "cluster": 1}]})
    (tmp_path / "d.jsonl").write_text('{"paper_id":"p"}\n')
    assert compare_paper_ids(tmp_path / "c.json", tmp_path / "d.jsonl")["exact_match"]


def test_assignment_records_are_bounded(tmp_path):
    data = {"clusters": {"1": {}}, "atom_assignments": [{"paper_id": "p", "text": str(i), "cluster": 1} for i in range(4)]}
    write_json(tmp_path / "c.json", data)
    assert len(assignment_records(tmp_path / "c.json", [1], limit=2)) == 2
