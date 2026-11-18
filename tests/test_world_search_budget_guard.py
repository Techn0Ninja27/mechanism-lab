import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from generation.api_budget import BudgetLedger, BudgetError
from scripts.world_search import Client


class WorldSearchBudgetGuardTests(unittest.TestCase):
    def test_missing_receipt_does_not_resend_reserved_request(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "run"
            ledger_path = Path(d) / "ledger.json"
            config = {
                "model": "test-model", "max_tokens_per_stage": [8],
                "pricing": {"prompt": 0.000001, "completion": 0.000001},
                "shared_budget_path": str(ledger_path), "shared_budget_usd": 10.0,
                "budget_usd": 1.0,
                "temperature": None, "reasoning": {"effort": "low"},
                "provider_max_price": {"prompt": 4, "completion": 20},
            }
            # The ledger reservation exists even though the local receipt was lost.
            with patch("scripts.world_search.key", return_value="test-key"):
                client = Client(out, config)
            req = {"model": "test-model", "messages": [{"role": "user", "content": "hello"}],
                   "max_tokens": 8, "reasoning": {"effort": "low"},
                   "provider": {"allow_fallbacks": False, "max_price": {"prompt": 4, "completion": 20}}}
            from scripts.world_search import digest
            ledger_id = digest({"path": str((out / "calls" / "case-0.json").resolve()), "request": digest(req)})
            reserve = (len("hello".encode()) + 512) * config["pricing"]["prompt"] + 8 * config["pricing"]["completion"]
            BudgetLedger(ledger_path, 10.0).reserve(ledger_id, reserve)
            with patch("scripts.world_search.urllib.request.urlopen", side_effect=AssertionError("network resend")):
                with self.assertRaises(BudgetError):
                    client.call({"id": "case"}, 0, "hello")


if __name__ == "__main__":
    unittest.main()
