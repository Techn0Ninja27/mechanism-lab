import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from generation.api_budget import BudgetError, BudgetLedger, receipt_request_id, validate_and_reconcile_receipt
from scripts.world_search import Client, digest


class ReceiptBudgetBindingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ledger_path = Path(self.tmp.name) / "ledger.json"
        self.path = Path(self.tmp.name) / "run" / "calls" / "case-0.json"
        self.request_hash = "a" * 64
        self.reserve = 0.25
        self.receipt = {"status": "complete", "request_hash": self.request_hash,
                        "finish_reason": "stop", "response_id": "resp-1",
                        "usage": {"cost": 0.10}, "content": "{}"}

    def _ledger(self):
        ledger = BudgetLedger(self.ledger_path, 1.0)
        ledger.reserve(receipt_request_id(self.path, self.request_hash), self.reserve,
                       {"request_hash": self.request_hash})
        return ledger

    def test_missing_reservation_rejected_without_creating_one(self):
        ledger = BudgetLedger(self.ledger_path, 1.0)
        with self.assertRaises(BudgetError):
            validate_and_reconcile_receipt(ledger, self.path, self.receipt,
                                           self.request_hash, self.reserve)
        self.assertFalse(self.ledger_path.exists())

    def test_pending_keeps_reserve_and_second_read_is_idempotent(self):
        ledger = self._ledger()
        first = validate_and_reconcile_receipt(ledger, self.path, self.receipt,
                                               self.request_hash, self.reserve)
        self.assertEqual(first["status"], "unknown")
        second = validate_and_reconcile_receipt(ledger, self.path, self.receipt,
                                                self.request_hash, self.reserve)
        self.assertEqual(second["status"], "unknown")
        self.assertEqual(ledger.snapshot()["charged_usd"], self.reserve)

    def test_wrong_path_and_edited_cost_rejected_after_reconcile(self):
        ledger = self._ledger()
        ledger.reconcile(receipt_request_id(self.path,self.request_hash),0.10,verified=True,receipt_id="resp-1")
        with self.assertRaises(BudgetError):
            validate_and_reconcile_receipt(ledger, self.path.with_name("other.json"),
                                           self.receipt, self.request_hash, self.reserve)
        edited = dict(self.receipt); edited["usage"] = {"cost": 0.01}
        with self.assertRaises(BudgetError):
            validate_and_reconcile_receipt(ledger, self.path, edited,
                                           self.request_hash, self.reserve)

    def test_pending_receipt_over_reserve_is_rejected_before_reconcile(self):
        ledger = self._ledger()
        edited = dict(self.receipt); edited["usage"] = {"cost": 0.30}
        with self.assertRaises(BudgetError):
            validate_and_reconcile_receipt(ledger, self.path, edited,
                                           self.request_hash, self.reserve)
        self.assertEqual(ledger.snapshot()["reservations"][receipt_request_id(self.path, self.request_hash)]["status"], "pending")

    def test_missing_receipt_retains_campaign_charge(self):
        out=Path(self.tmp.name)/'run';out.mkdir()
        missing=out/'calls/old-0.json'
        ledger=BudgetLedger(self.ledger_path,1.0)
        ledger.reserve('old',0.9,{'receipt_path':str(missing.resolve())})
        config={'model':'test','max_tokens_per_stage':[8],
                'pricing':{'prompt':1e-6,'completion':1e-6},
                'shared_budget_path':str(self.ledger_path),'shared_budget_usd':1.0,'budget_usd':0.5}
        with patch('scripts.world_search.key',return_value='not-a-key'), patch('scripts.world_search.urllib.request.urlopen') as network:
            client=Client(out,config)
            with self.assertRaisesRegex(ValueError,'run budget exhausted'):
                client.call({'id':'new'},0,'hello')
            network.assert_not_called()
        self.assertEqual(ledger.snapshot()['charged_usd'],0.9)

    def test_client_cache_hit_requires_bound_receipt(self):
        out = Path(self.tmp.name) / "run"
        path = out / "calls" / "case-0.json"
        path.parent.mkdir(parents=True)
        config = {"model":"test", "max_tokens_per_stage":[8],
                  "pricing":{"prompt":1e-6,"completion":1e-6},
                  "shared_budget_path":str(self.ledger_path),
                  "shared_budget_usd":1.0,"budget_usd":1.0,
                  "temperature":None,"reasoning":{"effort":"low"},
                  "provider_max_price":{"prompt":4,"completion":20}}
        req={"model":"test","messages":[{"role":"user","content":"hello"}],
             "max_tokens":8,"reasoning":{"effort":"low"},
             "provider":{"allow_fallbacks":False,"max_price":{"prompt":4,"completion":20}}}
        reserve=(len("hello".encode())+512)*1e-6+8e-6
        receipt={"status":"complete","request":req,"request_hash":digest(req),
                 "finish_reason":"stop","response_id":"r1","content":"{}",
                 "usage":{"cost":0.00001},"reserved_usd":reserve}
        path.write_text(json.dumps(receipt))
        client=Client(out,config)
        with self.assertRaises(BudgetError): client.call({"id":"case"},0,"hello")
        ledger=BudgetLedger(self.ledger_path,1.0)
        ledger.reserve(receipt_request_id(path,digest(req)),reserve,{"request_hash":digest(req)})
        self.assertEqual(client.call({"id":"case"},0,"hello"),"{}")
        self.assertEqual(ledger.snapshot()["charged_usd"],reserve)


if __name__ == "__main__":
    unittest.main()
