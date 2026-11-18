import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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

    def test_wrong_decoded_verdict_fails_even_with_bound_artifacts(self):
        import generation.reference_calibration_readiness as readiness
        original = readiness.decode_review_v2

        def wrong(packet, raw):
            decoded = original(packet, raw)
            decoded["field_items"][0]["verdict"] = (
                "unsupported" if decoded["field_items"][0]["verdict"] == "supported"
                else "supported"
            )
            return decoded

        with patch.object(readiness, "decode_review_v2", side_effect=wrong):
            with self.assertRaises(ValueError):
                assess(self.root, self.ledger)

    def test_answer_key_hash_is_required(self):
        import generation.reference_calibration_readiness as readiness
        original = readiness._sha

        def wrong_key(path):
            return "0" * 64 if Path(path).name == "answer-key.json" else original(path)

        with patch.object(readiness, "_sha", side_effect=wrong_key):
            with self.assertRaisesRegex(ValueError, "key_sha256"):
                assess(self.root, self.ledger)

    def test_evaluation_must_bind_to_freeze(self):
        path = self.root / "evaluation.json"
        data = json.loads(path.read_text())
        data["freeze_sha256"] = "0" * 64
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "freeze"):
            assess(self.root, self.ledger)

    def test_anchor_head_and_event_hash_are_checked_independently(self):
        for mode in ("head", "event"):
            ledger = json.loads(self.ledger.read_text())
            anchor = ledger["audit_heads"]["recordwise-calibration-v6-freeze"]
            if mode == "head":
                anchor["head"] = "0" * 64
            else:
                anchor["last_event"]["event_hash"] = "0" * 64
            self.ledger.write_text(json.dumps(ledger))
            with self.assertRaises(ValueError):
                assess(self.root, self.ledger)

            self.ledger.write_bytes(
                (ROOT / "results/improvement-phase-2026-11-16/budget.json").read_bytes()
            )

    def test_launch_receipt_hash_is_bound(self):
        receipt = self.root / "launch-receipts" / "v6_review_00a.json"
        receipt.write_text(receipt.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "receipt"):
            assess(self.root, self.ledger)

    def test_unbound_extra_review_file_is_rejected(self):
        extra = self.root / "reviews/review-extra.json"
        extra.write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "review|coverage|extra"):
            assess(self.root, self.ledger)

    def test_gate_rejects_retrospective_result_marked_unauthorized(self):
        from scripts import generate_causal_campaign2 as campaign
        result = {
            "status": "retrospective_parser_policy_pass",
            "v2": {"mechanically_valid": 24, "key_matching": 24, "total": 24},
            "production_authorized": False,
        }
        with patch.object(campaign, "assess", return_value=result, create=True):
            with patch("generation.reference_calibration_readiness.assess", return_value=result):
                with self.assertRaisesRegex(ValueError, "disabled"):
                    campaign.require_calibration_ready()


if __name__ == "__main__":
    unittest.main()
