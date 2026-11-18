import json
import unittest
from pathlib import Path
from unittest.mock import patch

import tempfile
from scripts import run_campaign2_preparation as life
from scripts import generate_causal_campaign2 as c
from scripts import prepare_chunk_representations as prep
from tests.test_campaign2_generation import FakeClient
from generation.api_budget import BudgetLedger


class Campaign2DrawRecoveryTests(unittest.TestCase):
    """Recovery tests use an isolated campaign and phase ledger."""

    def setUp(self):
        live_ledger = Path(__file__).resolve().parents[1] / 'results/improvement-phase-2026-11-16/budget.json'
        snapshot = live_ledger.read_bytes() if live_ledger.exists() else None
        self.addCleanup(lambda: self.assertEqual(
            live_ledger.read_bytes() if live_ledger.exists() else None,
            snapshot, 'tests must not mutate the live phase ledger'))
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name) / 'campaign'
        c.plan(c.ROOT / 'results/gnn-useful-proposal-v1/pack.json', self.out)
        self._patches = [
            patch.object(life, 'CANONICAL_OUTPUT', self.out),
            patch.object(life, '_anchor', return_value=BudgetLedger(Path(self.tmp.name) / 'phase-ledger.json', 30)),
            patch.object(c, 'require_calibration_ready', return_value=None),
            patch.object(c, 'Client', FakeClient),
        ]
        for p in self._patches:
            p.start()
            self.addCleanup(p.stop)
        FakeClient.calls = []

    def _start_without_terminal(self, kind="atoms", index=0, with_receipt=True):
        operation_id = f"{kind}:{index}"
        life._event(self.out, "draw_started", [], operation_id=operation_id,
                    kind=kind, draw=index)
        # This is the directory execute_draw creates immediately after its
        # draw_started event, before the provider call.
        life._draw_dir(self.out, kind, index).mkdir(parents=True)
        if with_receipt:
            FakeClient(self.out, c.config(3.0)).call(
                {"id": f"prep-{kind}-d{index}"}, 0,
                c._prep_prompt(prep._pack(self.out / "sourcepack.json"), kind))
        return operation_id

    def test_complete_receipt_is_materialized_without_second_client_call(self):
        self._start_without_terminal(with_receipt=True)
        before = len(FakeClient.calls)
        with life._locked(self.out) as (out, pack):
            # _locked performs this recovery before yielding.
            self.assertTrue((out / "prep-draws/atoms/00/draw.json").is_file())
            self.assertTrue((out / "prep-draws/atoms/00/atoms.json").is_file())
        self.assertEqual(len(FakeClient.calls), before)
        events = json.loads("[" + ",".join(
            line for line in (self.out / "campaign-events.jsonl").read_text().splitlines()
        ) + "]")
        self.assertTrue(any(e["event"] == "draw_completed" and
                            e["payload"].get("recovered_after_interruption")
                            for e in events))

    def test_pending_receipt_recovers_with_reservation_retained(self):
        with patch.object(BudgetLedger, 'reconcile', return_value=None):
            self._start_without_terminal(with_receipt=True)
        ledger=life._anchor(self.out)
        self.assertTrue(all(r['status']=='pending' for r in ledger.snapshot()['reservations'].values()))
        before=len(FakeClient.calls)
        with life._locked(self.out): pass
        first=ledger.snapshot()
        with life._locked(self.out): pass
        self.assertEqual(first['reservations'],ledger.snapshot()['reservations'])
        self.assertTrue(all(r['status']=='unknown' for r in first['reservations'].values()))
        self.assertEqual(first['charged_usd'],sum(r['reserved_usd'] for r in first['reservations'].values()))
        self.assertEqual(len(FakeClient.calls),before)

    def test_receipt_copied_to_another_draw_has_no_reservation(self):
        # Create a genuine receipt for draw 0, but only draw 1 has a start event.
        FakeClient(self.out,c.config(3.0)).call({'id':'prep-atoms-d0'},0,
            c._prep_prompt(prep._pack(self.out/'sourcepack.json'),'atoms'))
        source=self.out/'calls/prep-atoms-d0-0.json'
        target=self.out/'calls/prep-atoms-d1-0.json'
        target.write_bytes(source.read_bytes())
        self._start_without_terminal(index=1,with_receipt=False)
        before=len(FakeClient.calls)
        with self.assertRaisesRegex(ValueError,'ambiguous'):
            with life._locked(self.out):pass
        self.assertTrue((self.out/'campaign-stop.json').exists())
        self.assertFalse((self.out/'prep-draws/atoms/01/draw.json').exists())
        self.assertEqual(len(FakeClient.calls),before)

    def test_started_draw_without_receipt_stops_before_any_new_call(self):
        self._start_without_terminal(with_receipt=False)
        with self.assertRaisesRegex(ValueError, "no provider receipt"):
            with life._locked(self.out):
                pass
        self.assertTrue((self.out / "campaign-stop.json").is_file())
        self.assertEqual(FakeClient.calls, [])


if __name__ == "__main__":
    unittest.main()
