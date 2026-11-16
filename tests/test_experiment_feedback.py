import hashlib
import tempfile
import unittest
from pathlib import Path

from generation.experiment_feedback import inspect_feedback


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "operator-studies/x").mkdir(parents=True)
        (self.root / "operator-studies/x/report.md").write_text("report")
        self.result = self.root / "operator-studies/x/result.json"
        self.result.write_text('{"verdict":"no_gain"}\n')
        self.pack = {"sources": [
            {"url": "https://arxiv.org/abs/2508.08256v2"},
            {"url": "https://arxiv.org/pdf/2605.02262v1.pdf"}], "chunks": []}

    def tearDown(self):
        self.tmp.cleanup()

    def record(self, **overrides):
        value = {
            "id": "trace-v1",
            "source_urls": ["https://arxiv.org/abs/2508.08256v1", "https://arxiv.org/pdf/2605.02262v7.pdf"],
            "tested_change": "fixed-budget ranking",
            "verdict": "no_support",
            "scope": "development traces only",
            "limitations": ["two exposed images"],
            "report_path": "operator-studies/x/report.md",
            "result_path": "operator-studies/x/result.json",
            "result_sha256": hashlib.sha256(self.result.read_bytes()).hexdigest(),
            "train_eligible": False,
            "novelty_established": False,
        }
        value.update(overrides)
        return value

    def registry(self, records):
        return {"schema_version": "experiment-feedback.v1", "records": records}

    def test_canonical_arxiv_versions_match_and_valid_record_is_selected(self):
        out = inspect_feedback(self.pack, self.root, self.registry([self.record()]))
        self.assertEqual(out["matched_count"], 1)
        self.assertEqual(out["records"][0]["normalized_source_ids"], ["arxiv:2508.08256", "arxiv:2605.02262"])
        self.assertTrue(out["records"][0]["artifact_sha256_verified"])

    def test_source_pair_is_required_and_unknown_fields_rejected(self):
        self.assertEqual(inspect_feedback({"sources": [self.pack["sources"][0]]}, self.root, self.registry([self.record()]))["matched_count"], 0)
        bad = self.record(extra="x")
        out = inspect_feedback(self.pack, self.root, self.registry([bad, self.record(id="valid")]))
        self.assertEqual([r["id"] for r in out["records"]], ["valid"])

    def test_tamper_flags_and_rejects_traversal_or_symlink_escape(self):
        bad = self.record(result_sha256="0" * 64)
        outside = self.root / "outside.json"
        outside.write_text("x")
        (self.root / "operator-studies/x/link.json").symlink_to(outside)
        out = inspect_feedback(self.pack, self.root, self.registry([
            bad,
            self.record(id="escape", result_path="operator-studies/x/link.json"),
            self.record(id="up", result_path="operator-studies/../outside.json")]))
        self.assertEqual(out["matched_count"], 0)
        self.assertGreaterEqual(len(out["warnings"]), 3)

    def test_invalid_promotion_flags_and_malformed_registry_are_safe(self):
        out = inspect_feedback(self.pack, self.root, self.registry([
            self.record(train_eligible=True), self.record(id="new", novelty_established=True)]))
        self.assertEqual(out["matched_count"], 0)
        out = inspect_feedback(self.pack, self.root, {"schema_version": "wrong", "records": []})
        self.assertEqual(out["records"], [])

    def test_bound_is_deterministic_and_valid_records_survive_invalid_ones(self):
        records = [self.record(id=f"v{i}") for i in range(7)]
        records.insert(2, self.record(id="bad", result_sha256="f"))
        out = inspect_feedback(self.pack, self.root, self.registry(records))
        self.assertEqual([r["id"] for r in out["records"]], ["v0", "v1", "v2", "v3", "v4"])
        self.assertTrue(any("bounded" in w for w in out["warnings"]))

    def test_malformed_pack_does_not_raise(self):
        out = inspect_feedback({}, self.root, self.registry([self.record()]))
        self.assertEqual(out["records"], [])

    def test_unhashable_record_id_does_not_hide_valid_records(self):
        out=inspect_feedback(self.pack,self.root,self.registry([self.record(id=[]),self.record(id={}),self.record(id='valid')]))
        self.assertEqual([r['id'] for r in out['records']],['valid'])

    def test_malformed_port_is_quarantined(self):
        out = inspect_feedback(self.pack, self.root, self.registry([
            self.record(source_urls=["https://arxiv.org:bad/abs/2508.08256v1", "https://arxiv.org/abs/2605.02262v1"])]))
        self.assertEqual(out["matched_count"], 0)

    def test_operator_studies_symlink_outside_root_is_rejected(self):
        record = self.record()
        external_tmp = tempfile.TemporaryDirectory()
        outside = Path(external_tmp.name) / "outside-operator"; outside.mkdir()
        (outside / "report.md").write_text("report")
        (outside / "result.json").write_text("{}")
        original = self.root / "operator-studies"
        original.rename(self.root / "operator-studies-real")
        original.symlink_to(outside, target_is_directory=True)
        out = inspect_feedback(self.pack, self.root, self.registry([record]))
        self.assertEqual(out["matched_count"], 0)
        self.assertTrue(any("operator-studies" in warning for warning in out["warnings"]))
        external_tmp.cleanup()

    def test_conflicting_duplicate_ids_reject_all_copies(self):
        out = inspect_feedback(self.pack, self.root, self.registry([
            self.record(tested_change="first"), self.record(tested_change="conflict"),
            self.record(id="valid")]))
        self.assertEqual([record["id"] for record in out["records"]], ["valid"])
        self.assertTrue(any("all copies rejected" in warning for warning in out["warnings"]))

    def test_warnings_are_bounded(self):
        out = inspect_feedback(self.pack, self.root, self.registry([
            {"id": "bad-" + ("x" * 5000)} for _ in range(100)]))
        self.assertLessEqual(len(out["warnings"]), 20)
        self.assertTrue(all(len(warning) <= 300 for warning in out["warnings"]))


if __name__ == "__main__":
    unittest.main()
