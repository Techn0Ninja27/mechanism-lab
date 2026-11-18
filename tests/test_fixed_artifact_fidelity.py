import json
import shutil
import tempfile
import unittest
from pathlib import Path

from generation.fixed_artifact_fidelity import build_report


ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "results/generation-comparison-campaign2-run/prep-draws/atoms/00/reference-slots"


class FixedArtifactFidelityTests(unittest.TestCase):
    def test_stopped_campaign_atoms(self):
        report = build_report(FIXTURE)
        self.assertEqual(report["summary"]["valid_reviewer_slots"], 23)
        self.assertEqual(report["summary"]["total_reviewer_slots"], 24)
        self.assertEqual(report["summary"]["missing_or_malformed_review_slots"], 1)
        self.assertEqual(report["summary"]["flagged_by_either"], 6)
        self.assertEqual(report["summary"]["flagged_by_both"], 2)
        self.assertEqual(report["summary"]["unknown_flagged_by_either"], 3)
        self.assertIsNone(report.get("pass"))
        self.assertNotIn("gate", report)

    def _fixture(self):
        tmp = Path(tempfile.mkdtemp())
        for slot in ("a", "b"):
            (tmp / slot).mkdir()
            for name in ("input-00.json", "review-00.json"):
                shutil.copy2(FIXTURE / slot / name, tmp / slot / name)
        self.addCleanup(shutil.rmtree, tmp)
        return tmp

    def test_missing_review_hash_is_null_and_is_missing_measurement(self):
        root = self._fixture()
        (root / "b" / "review-00.json").unlink()
        report = build_report(root)
        row = report["records"][0]
        self.assertIsNone(row["reviews"]["b"]["sha256"])
        self.assertEqual(row["reviews"]["b"]["status"], "missing")
        self.assertEqual(row["valid_reviewer_slots"], 1)
        self.assertEqual(row["missing_or_malformed_reviews"], ["b"])

    def test_contradictory_flags_are_measurements(self):
        root = self._fixture()
        path = root / "b" / "review-00.json"
        review = json.loads(path.read_text())
        review["fields"][0]["verdict"] = "unsupported"
        review["fields"][0]["evidence_ids"] = []
        path.write_text(json.dumps(review))
        report = build_report(root)
        first = report["records"][0]["fields"][0]
        self.assertEqual(first["flagged_by_slots"], ["b"])
        self.assertTrue(first["flagged_by_either"])
        self.assertFalse(first["flagged_by_both"])

    def test_differing_packets_fail_structurally(self):
        root = self._fixture()
        path = root / "b" / "input-00.json"
        packet = json.loads(path.read_text())
        packet["fields"][0]["field"] = "different"
        path.write_text(json.dumps(packet))
        with self.assertRaisesRegex(ValueError, "structural packet mismatch"):
            build_report(root)

    def test_deleted_edge_record_preserves_denominator(self):
        root = self._fixture()
        for path in root.glob("*/*.json"):
            path.unlink()
        report = build_report(root)
        self.assertEqual(report["summary"]["total_reviewer_slots"], 24)
        self.assertEqual(report["summary"]["missing_or_malformed_review_slots"], 24)

    def test_changed_source_window_fails(self):
        root = self._fixture()
        path = root / "b" / "input-00.json"
        packet = json.loads(path.read_text())
        packet["evidence"][0]["text"] += " altered"
        path.write_text(json.dumps(packet))
        with self.assertRaisesRegex(ValueError, "structural packet mismatch"):
            build_report(root)


if __name__ == "__main__":
    unittest.main()
