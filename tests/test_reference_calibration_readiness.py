import json
import shutil
import tempfile
import unittest
from pathlib import Path

from generation.reference_calibration_readiness import assess

ROOT = Path(__file__).resolve().parents[1]


class ReferenceCalibrationReadinessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "v6"
        shutil.copytree(ROOT / "results/recordwise-calibration-v6", self.root)
        self.ledger = Path(self.tmp.name) / "budget.json"
        shutil.copyfile(ROOT / "results/improvement-phase-2026-11-16/budget.json", self.ledger)

    def test_current_fixture_is_retrospective_only(self):
        result = assess(self.root, self.ledger)
        self.assertEqual(result["status"], "retrospective_parser_policy_pass")
        self.assertTrue(result["calibration_criterion_satisfied"])
        self.assertEqual(result["v2"], {"mechanically_valid": 24, "key_matching": 24, "total": 24})
        self.assertEqual(result["strict_v1"]["mechanically_valid"], 23)

    def test_assessor_does_not_rewrite_live_ledger(self):
        ledger = ROOT / "results/improvement-phase-2026-11-16/budget.json"
        before = ledger.read_bytes()
        result = assess(ROOT / "results/recordwise-calibration-v6", ledger)
        self.assertTrue(result["calibration_criterion_satisfied"])
        self.assertEqual(ledger.read_bytes(), before)

    def test_forged_passflag_does_not_bypass_checks(self):
        path = self.root / "evaluation.json"
        data = json.loads(path.read_text())
        data["calibration_passed"] = True
        path.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            assess(self.root, self.ledger)

    def test_tampered_review_and_ledger_binding_fail(self):
        path = self.root / "reviews/review-00-a.json"
        path.write_text(path.read_text() + "\n")
        with self.assertRaises(ValueError):
            assess(self.root, self.ledger)
        shutil.copytree(ROOT / "results/recordwise-calibration-v6", self.root, dirs_exist_ok=True)
        ledger = json.loads(self.ledger.read_text())
        ledger["audit_heads"]["recordwise-calibration-v6-freeze"]["last_event"]["payload"]["evaluation_sha256"] = "0" * 64
        self.ledger.write_text(json.dumps(ledger))
        with self.assertRaises(ValueError):
            assess(self.root, self.ledger)


if __name__ == "__main__":
    unittest.main()
