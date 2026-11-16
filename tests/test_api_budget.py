import multiprocessing as mp
import tempfile
import unittest
from pathlib import Path

from generation.api_budget import BudgetError, BudgetLedger, request_id


def _reserve(path, barrier, out):
    ledger = BudgetLedger(path, 1.0)
    barrier.wait()
    try:
        ledger.reserve("worker-" + str(out), .6)
        outq = "ok"
    except BudgetError:
        outq = "rejected"
    Path(str(path) + "." + str(out) + ".result").write_text(outq)


class BudgetTests(unittest.TestCase):
    def test_reservation_reconciliation_and_idempotence(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ledger.json"
            b = BudgetLedger(p, 2)
            b.reserve("r", 1, {"model": "cheap"})
            self.assertEqual(b.reserve("r", 1)["status"], "pending")
            with self.assertRaises(BudgetError): b.reserve("r", 1.1)
            b.mark_unknown("r")
            self.assertEqual(b.snapshot()["charged_usd"], 1)
            b.reconcile("r", .25, verified=True, receipt_id="receipt")
            b.reconcile("r", .25, verified=True, receipt_id="receipt")
            self.assertEqual(b.snapshot()["charged_usd"], .25)
            with self.assertRaises(BudgetError): b.reconcile("r", .2, verified=True)
            with self.assertRaises(BudgetError): b.reconcile("r", .1, verified=False)

    def test_rejects_bad_amounts_and_cap(self):
        with tempfile.TemporaryDirectory() as d:
            b = BudgetLedger(Path(d) / "ledger.json", 1)
            for value in (-1, float("inf"), True):
                with self.assertRaises(BudgetError): b.reserve(str(value), value)
            b.reserve("a", 1)
            with self.assertRaises(BudgetError): b.reserve("b", 0.01)

    def test_cross_process_boundary(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ledger.json"
            ctx = mp.get_context("spawn")
            barrier, jobs = ctx.Barrier(2), []
            for i in range(2):
                q = ctx.Process(target=_reserve, args=(str(p), barrier, i)); q.start(); jobs.append(q)
            for q in jobs: q.join(10); self.assertEqual(q.exitcode, 0)
            outcomes = [Path(d, f"ledger.json.{i}.result").read_text() for i in range(2)]
            self.assertEqual(outcomes.count("ok"), 1)
            self.assertEqual(outcomes.count("rejected"), 1)

    def test_request_id_is_stable(self):
        self.assertEqual(request_id({"b": 2, "a": 1}), request_id({"a": 1, "b": 2}))

    def test_verified_overage_is_retained_and_locks_ledger(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ledger.json"
            b = BudgetLedger(p, 1)
            b.reserve("r", .5)
            with self.assertRaises(BudgetError): b.reconcile("r", 1.2, verified=True)
            snap = b.snapshot()
            self.assertEqual(snap["charged_usd"], 1.2)
            self.assertEqual(snap["reservations"]["r"]["status"], "over_budget")
            self.assertEqual(b.mark_unknown("r")["status"], "over_budget")
            self.assertEqual(b.reconcile("r", 1.2, verified=True)["status"], "over_budget")
            with self.assertRaises(BudgetError): b.reconcile("r", 1.1, verified=True)
            with self.assertRaises(BudgetError): b.reserve("new", .01)


if __name__ == "__main__": unittest.main()
