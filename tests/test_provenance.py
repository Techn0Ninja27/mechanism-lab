import json
from pathlib import Path

import unittest
import tempfile

from generation.provenance import (ProvenanceError, assignment_records, audit_provenance,
                                   compare_paper_ids, validate_candidate_ids)


def write_json(path: Path, data: dict):
    path.write_text(json.dumps(data))


class ProvenanceTests(unittest.TestCase):
    def test_audit_requires_exact_mapping_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            clusters = {"clusters": {"1": {"name": "a"}}, "atom_assignments": [{"paper_id": "p", "text": "x", "cluster": 1}]}
            write_json(tmp_path / "c.json", clusters)
            write_json(tmp_path / "m.json", {"cluster_to_token_id": {"2": 1}})
            with self.assertRaises(ProvenanceError):
                audit_provenance(tmp_path / "c.json", coherence_mapping=tmp_path / "m.json")


    def test_cross_file_agreement_and_candidate_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            a = {"clusters": {"1": {}, "2": {}}, "atom_assignments": [{"paper_id": "p", "text": "x", "cluster": 1}, {"paper_id": "q", "text": "y", "cluster": 2}]}
            b = {"clusters": {"1": {}, "2": {}}, "atom_assignments": [{"paper_id": "p", "text": "x", "cluster": 2}, {"paper_id": "q", "text": "y", "cluster": 1}]}
            write_json(tmp_path / "a.json", a); write_json(tmp_path / "b.json", b)
            report = audit_provenance(tmp_path / "a.json", canonical_clusters=tmp_path / "b.json")
            self.assertTrue(report["canonical_label_agreement"] == 0)
            self.assertTrue(report["canonical_label_mismatch_warning"])
            self.assertTrue(report["safe_for_canonical_join"] is False)
            self.assertTrue(validate_candidate_ids([2, 1], report) == (2, 1))
            with self.assertRaises(ProvenanceError): validate_candidate_ids([3], report)


    def test_require_safe_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            a = {"clusters": {"1": {}}, "atom_assignments": [{"paper_id": "p", "text": "x", "cluster": 1}]}
            b = {"clusters": {"1": {}}, "atom_assignments": [{"paper_id": "p", "text": "x", "cluster": -1}]}
            write_json(tmp_path / "a.json", a); write_json(tmp_path / "b.json", b)
            with self.assertRaises(ProvenanceError): audit_provenance(tmp_path / "a.json", tmp_path / "b.json", require_safe=True)


    def test_compare_paper_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            write_json(tmp_path / "c.json", {"atom_assignments": [{"paper_id": "p", "cluster": 1}]})
            (tmp_path / "d.jsonl").write_text('{"paper_id":"p"}\n')
            self.assertTrue(compare_paper_ids(tmp_path / "c.json", tmp_path / "d.jsonl")["exact_match"])


    def test_assignment_records_are_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            data = {"clusters": {"1": {}}, "atom_assignments": [{"paper_id": "p", "text": str(i), "cluster": 1} for i in range(4)]}
            write_json(tmp_path / "c.json", data)
            self.assertTrue(len(assignment_records(tmp_path / "c.json", [1], limit=2)) == 2)
