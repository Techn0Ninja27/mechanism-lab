"""Adversarial lifecycle checks for campaign-2 preparation.

These tests deliberately mutate temporary artifacts.  They never use the real
campaign directory or make network calls.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import generate_causal_campaign2 as c
from scripts import run_campaign2_preparation as life
from generation.api_budget import BudgetLedger
from tests.test_campaign2_generation import FakeClient, PACK, reviews


class Campaign2LifecycleAdversarialTests(unittest.TestCase):
    def setUp(self):
        # Lifecycle events are externally anchored.  Point that anchor at a
        # disposable ledger as well as redirecting the campaign directory;
        # otherwise these mutation tests append audit heads to the developer's
        # real phase ledger on every run.
        self.live_ledger = Path(c.config()["shared_budget_path"])
        self.live_ledger_before = self.live_ledger.read_bytes() if self.live_ledger.exists() else None
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name) / "campaign"
        c.plan(PACK, self.out)
        self.canonical = patch.object(life, "CANONICAL_OUTPUT", self.out)
        self.canonical.start()
        self.addCleanup(self.canonical.stop)
        self.anchor = patch.object(
            life,
            "_anchor",
            return_value=BudgetLedger(Path(self.tmp.name) / "phase-ledger.json", 30),
        )
        self.anchor.start()
        self.addCleanup(self.anchor.stop)
        self.calibration = patch.object(c, "require_calibration_ready", return_value=None)
        self.calibration.start()
        self.addCleanup(self.calibration.stop)
        FakeClient.calls = []
        FakeClient.fail = False
        self.client = patch.object(c, "Client", FakeClient)
        self.client.start()
        self.addCleanup(self.client.stop)
        self.addCleanup(self._assert_live_ledger_unchanged)

    def _assert_live_ledger_unchanged(self):
        self.assertEqual(
            self.live_ledger.read_bytes() if self.live_ledger.exists() else None,
            self.live_ledger_before,
            "adversarial tests must not write the shared phase ledger",
        )

    def _failed_draw(self, kind="atoms", index=0):
        life.execute_draw(self.out, kind, index)
        reviews(self.out, kind, index, failed=True)
        result = life.record_review(self.out, kind, index)
        self.assertEqual(result["status"], "failed")

    def _passing_draw(self, kind="atoms", index=0):
        life.execute_draw(self.out, kind, index)
        reviews(self.out, kind, index)
        result = life.record_review(self.out, kind, index)
        self.assertEqual(result["status"], "locked")
        return result

    def test_canonical_path_and_plan_lock_prevent_spend(self):
        with self.assertRaises(ValueError):
            life.execute_draw(self.out.parent / "other", "atoms", 0)
        self.assertEqual(FakeClient.calls, [])
        with patch.object(c, "Client", side_effect=AssertionError("client called")):
            with self.assertRaises(ValueError):
                c.generate(self.out, True)
        self.assertEqual(FakeClient.calls, [])

    def test_index_bounds_and_selected_kind_are_irreversible(self):
        for index in (-1, 3, True):
            with self.assertRaises(ValueError):
                life.execute_draw(self.out, "atoms", index)
        self._passing_draw("atoms", 0)
        calls = len(FakeClient.calls)
        with self.assertRaises(ValueError):
            life.execute_draw(self.out, "atoms", 1)
        self.assertEqual(len(FakeClient.calls), calls)

    def test_review_hash_and_decision_mutations_cannot_advance(self):
        self._failed_draw("atoms", 0)
        draw = life._draw_dir(self.out, "atoms", 0)
        review = draw / "reference-slots/b/review-00.json"
        data = json.loads(review.read_text())
        data["fields"][0]["reason"] = "changed"
        review.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            life.execute_draw(self.out, "atoms", 1)
        self.assertEqual(len(FakeClient.calls), 1)

    def test_skipped_first_pass_and_wrong_draw_context_are_rejected(self):
        self._passing_draw("atoms", 0)
        with self.assertRaises(ValueError):
            life.execute_draw(self.out, "atoms", 1)

        # A fresh campaign lets us test review context without a selected lock.
        other = Path(self.tmp.name) / "context-campaign"
        c.plan(PACK, other)
        with patch.object(life, "CANONICAL_OUTPUT", other):
            life.execute_draw(other, "atoms", 0)
            reviews(other, "atoms", 0)
            path = life._draw_dir(other, "atoms", 0) / "reference-slots/a/input-00.json"
            data = json.loads(path.read_text())
            data["review_context"]["draw"] = 1
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "artifact binding|bound to this draw"):
                life.record_review(other, "atoms", 0)

    def test_uncited_quote_and_incomplete_fields_fail_before_lock(self):
        life.execute_draw(self.out, "atoms", 0)
        reviews(self.out, "atoms", 0)
        draw = life._draw_dir(self.out, "atoms", 0)
        # Altering a review before the decision must fail validation and cannot
        # create a selection lock.
        path = draw / "reference-slots/a/review-00.json"
        data = json.loads(path.read_text())
        data["fields"][0]["evidence_ids"] = ["E9999"]
        path.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            life.record_review(self.out, "atoms", 0)
        self.assertFalse((self.out / "selected" / "atoms.selection.json").exists())

    def test_provider_failure_and_exhaustion_stop_other_kind(self):
        FakeClient.fail = True
        with self.assertRaises(RuntimeError):
            life.execute_draw(self.out, "atoms", 0)
        FakeClient.fail = False
        with self.assertRaisesRegex(ValueError, "permanently stopped"):
            life.execute_draw(self.out, "distillations", 0)
        self.assertEqual(len(FakeClient.calls), 1)

        # Separate campaign: three review failures exhaust atoms and stop prose.
        exhausted = Path(self.tmp.name) / "exhausted"
        c.plan(PACK, exhausted)
        with patch.object(life, "CANONICAL_OUTPUT", exhausted):
            for index in range(3):
                life.execute_draw(exhausted, "atoms", index)
                reviews(exhausted, "atoms", index, failed=True)
                life.record_review(exhausted, "atoms", index)
            with self.assertRaisesRegex(ValueError, "permanently stopped"):
                life.execute_draw(exhausted, "distillations", 0)


if __name__ == "__main__":
    unittest.main()
