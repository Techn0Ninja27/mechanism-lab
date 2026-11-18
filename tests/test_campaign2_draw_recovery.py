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

    def test_started_draw_without_receipt_stops_before_any_new_call(self):
        self._start_without_terminal(with_receipt=False)
        with self.assertRaisesRegex(ValueError, "no provider receipt"):
            with life._locked(self.out):
                pass
        self.assertTrue((self.out / "campaign-stop.json").is_file())
        self.assertEqual(FakeClient.calls, [])


if __name__ == "__main__":
    unittest.main()
