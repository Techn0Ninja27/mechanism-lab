import json
import unittest
from pathlib import Path

from generation.method_lineage import inspect_pack


class MethodLineageTests(unittest.TestCase):
    def setUp(self):
        self.lineage = json.loads((Path(__file__).parents[1] / "docs/method-grounding/lineage.json").read_text())

    @staticmethod
    def pack(*sources):
        return {"schema_version": "sourcepack.v1", "sources": list(sources), "chunks": [], "limits": {}}

    def test_old_only_reports_known_successor_absent(self):
        report = inspect_pack(self.pack({"id": "old", "url": "https://github.com/marineLM/NeuMiss/blob/58d54f1815bba847f9f06fe1910c95b754d9578c/neumannS0_mlp.py", "evidence_level": "official_code"}), self.lineage)
        relation = report["coverage"]["relations"]["neumiss-sharedweights"]
        self.assertEqual(relation["status"], "older_only")
        self.assertIn("known successor absent", " ".join(report["warnings"]))

    def test_old_and_new_are_covered(self):
        report = inspect_pack(self.pack(
            {"id": "old", "url": "https://github.com/marineLM/NeuMiss/blob/58d54f1815bba847f9f06fe1910c95b754d9578c/neumannS0_mlp.py", "evidence_level": "official_code"},
            {"id": "new", "url": "https://github.com/marineLM/NeuMiss_sota/blob/7902b8dbe7114e8dc3010b5e8b132c35806a9d74/neumiss/NeuMissBlock.py", "evidence_level": "official_code"}), self.lineage)
        self.assertEqual(report["coverage"]["relations"]["neumiss-sharedweights"]["status"], "covered")

    def test_hostile_url_and_same_name_document_do_not_match(self):
        report = inspect_pack(self.pack(
            {"id": "doc", "title": "NeuMiss_sota", "url": "https://evil.example/marineLM/NeuMiss_sota"},
            {"id": "text", "title": "NeuMiss", "url": "https://example.test/NeuMiss"}), self.lineage)
        self.assertEqual(report["coverage"]["relations"]["neumiss-sharedweights"]["status"], "absent")
        self.assertEqual(report["coverage"]["relations"]["neumiss-sharedweights"]["known_successor_urls"], [])

    def test_readme_or_abstract_metadata_does_not_count_as_implementation(self):
        report = inspect_pack(self.pack(
            {"id": "readme", "url": "https://github.com/zyc-sjtu/StableMiss", "commit": "fd02c06ce24b7ae153403bcbe6a477ffc6f76fe2", "evidence_level": "official_code", "path": "README.md"},
            {"id": "paper", "url": "https://arxiv.org/abs/2106.00311", "evidence_level": "abstract"}), self.lineage)
        relation = report["coverage"]["relations"]["stablemiss-selectivecorrelation"]
        self.assertTrue(relation["older_metadata_present"])
        self.assertFalse(relation["older_present"])

    def test_matched_relation_reports_curated_claim_and_support_links(self):
        report = inspect_pack(self.pack(
            {"url": "https://github.com/marineLM/NeuMiss/blob/58d54f1815bba847f9f06fe1910c95b754d9578c/neumannS0_mlp.py", "evidence_level": "official_code"},
            {"url": "https://github.com/marineLM/NeuMiss_sota/blob/7902b8dbe7114e8dc3010b5e8b132c35806a9d74/neumiss/NeuMissBlock.py", "evidence_level": "official_code"}), self.lineage)
        relation = report["coverage"]["relations"]["neumiss-sharedweights"]
        self.assertTrue(relation["curated_relation"])
        self.assertTrue(relation["paper_reported_change"])
        self.assertTrue(any("arxiv.org" in url for url in relation["supports"]))
        self.assertFalse(relation["review_status"]["human_adjudicated"])


if __name__ == "__main__":
    unittest.main()
