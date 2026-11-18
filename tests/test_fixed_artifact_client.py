import json
import tempfile
import unittest
from pathlib import Path

from generation.api_budget import BudgetLedger, BudgetError
from generation.fixed_artifact_client import FixedArtifactClient, FixedArtifactError, SYSTEM


def req(model="anthropic/claude-opus-5.5"):
    prices = {"anthropic/claude-opus-5.5": {"prompt": 4, "completion": 20}, "anthropic/claude-fable-5.2": {"prompt": 8, "completion": 40}}
    return {"model": model, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "hello"}],
            "provider": {"only": ["anthropic"], "order": ["anthropic"], "allow_fallbacks": False, "require_parameters": True, "max_price": prices[model]},
            "max_tokens": 12288, "reasoning": {"effort": "low"}}


class FixedArtifactClientTests(unittest.TestCase):
    def make(self, transport, cap=30):
        d = tempfile.TemporaryDirectory(); root = Path(d.name)
        ledger = root / "phase.json"; BudgetLedger(ledger, cap).snapshot()
        return d, root, FixedArtifactClient(root / "receipts", ledger, transport=transport, campaign_cap_usd=24.85, phase_cap_usd=cap)

    def test_two_models_and_per_request_prices(self):
        seen = []
        d, root, c = self.make(lambda r: (seen.append(r) or {"response_id": r["model"], "resolved_model": r["model"], "finish_reason": "stop", "content": "ok", "usage": {"prompt_tokens": 1, "completion_tokens": 1}}))
        try:
            a = c.call(req(), "a.json"); b = c.call(req("anthropic/claude-fable-5.2"), "b.json")
            self.assertEqual(a["status"], "complete"); self.assertEqual(b["status"], "complete"); self.assertEqual(len(seen), 2)
            records = BudgetLedger(root / "phase.json", 30).snapshot()["reservations"]
            self.assertEqual(len(records), 2); self.assertNotEqual(*[x["reserved_usd"] for x in records.values()])
        finally: d.cleanup()

    def test_provider_failure_is_terminal_no_resend(self):
        calls = []
        def fail(r): calls.append(1); raise RuntimeError("provider down")
        d, _, c = self.make(fail)
        try:
            with self.assertRaises(RuntimeError): c.call(req(), "x.json")
            with self.assertRaises(FixedArtifactError): c.call(req(), "x.json")
            self.assertEqual(len(calls), 1)
        finally: d.cleanup()

    def test_partial_content_is_kept_and_global_stop(self):
        d, root, c = self.make(lambda r: {"response_id": "r", "resolved_model": r["model"], "finish_reason": "length", "content": "partial", "usage": {"prompt_tokens": 1, "completion_tokens": 2}})
        try:
            with self.assertRaises(FixedArtifactError): c.call(req(), "x.json")
            receipt = json.loads((root / "receipts/x.json").read_text())
            self.assertEqual(receipt["content"], "partial"); self.assertEqual(receipt["usage"]["completion_tokens"], 2)
            self.assertTrue((root / "receipts/.stop.json").exists())
        finally: d.cleanup()

    def test_deleted_receipt_does_not_release_campaign_cap(self):
        d, root, c = self.make(lambda r: {"response_id": "r", "resolved_model": r["model"], "finish_reason": "stop", "content": "ok", "usage": {"prompt_tokens": 1, "completion_tokens": 1}}, cap=30)
        try:
            expensive = {**req(), "provider": {**req()["provider"], "max_price": {"prompt": 4, "completion": 20}}}
            c.call(expensive, "x.json"); (root / "receipts/x.json").unlink()
            with self.assertRaises((BudgetError, FixedArtifactError)): c.call({**expensive, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "different"}]}, "y.json")
        finally: d.cleanup()

    def test_fabricated_cached_receipt_blocked(self):
        d, root, c = self.make(lambda r: {"response_id": "r", "resolved_model": r["model"], "finish_reason": "stop", "content": "ok", "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
        try:
            path = root / "receipts/x.json"; path.write_text(json.dumps({"status": "complete", "request_hash": "0" * 64}))
            with self.assertRaises(FixedArtifactError): c.call(req(), "x.json")
        finally: d.cleanup()

    def ok(self, payload):
        return {"response_id":"fake-provider-id", "resolved_model":payload["model"],
                "finish_reason":"stop", "content":"{}",
                "usage":{"prompt_tokens":3,"completion_tokens":2,"cost":0.001}}

    def test_complete_cache_no_transport_and_full_reserve_retained(self):
        seen=[]
        d, root, c=self.make(lambda r:(seen.append(r) or self.ok(r)))
        try:
            first=c.call(req(),"one.json"); second=c.call(req(),"one.json")
            self.assertEqual(first,second); self.assertEqual(len(seen),1)
            item=next(iter(BudgetLedger(root/"phase.json",30).snapshot()["reservations"].values()))
            self.assertEqual(item["status"],"unknown")
            self.assertEqual(item["reserved_usd"],first["reserved_usd"])
            self.assertGreater(item["reserved_usd"],0.001)
        finally:d.cleanup()

    def test_keyboard_interrupt_and_deleted_stop_still_block_other_call(self):
        def fail(r): raise KeyboardInterrupt()
        d,root,c=self.make(fail)
        try:
            with self.assertRaises(KeyboardInterrupt):c.call(req(),"one.json")
            (root/"receipts/.stop.json").unlink()
            with self.assertRaises(FixedArtifactError):c.call(req(),"two.json")
        finally:d.cleanup()

    def test_cached_request_tampering_blocks_new_request(self):
        d,root,c=self.make(self.ok)
        try:
            c.call(req(),"one.json")
            path=root/"receipts/one.json"; data=json.loads(path.read_text())
            data["request"]["messages"][1]["content"]="changed"
            path.write_text(json.dumps(data))
            with self.assertRaises(FixedArtifactError):c.call(req(),"two.json")
        finally:d.cleanup()

    def test_policy_extras_and_zero_price_rejected_before_transport(self):
        seen=[]
        d,root,c=self.make(lambda r:(seen.append(r) or self.ok(r)))
        try:
            bad=req();bad["provider"]["max_price"]["prompt"]=0
            with self.assertRaises(FixedArtifactError):c.call(bad,"one.json")
            bad=req();bad["pricing"]={"prompt":0,"completion":0}
            with self.assertRaises(FixedArtifactError):c.call(bad,"two.json")
            bad=req();bad["messages"][0]["cache_control"]={"type":"ephemeral"}
            with self.assertRaises(FixedArtifactError):c.call(bad,"three.json")
            self.assertEqual(seen,[])
        finally:d.cleanup()

    def test_reservation_failure_records_stop_without_transport(self):
        from unittest.mock import patch
        seen=[];d,root,c=self.make(lambda r:(seen.append(r) or self.ok(r)))
        try:
            with patch.object(BudgetLedger,"reserve",side_effect=KeyboardInterrupt()):
                with self.assertRaises(KeyboardInterrupt):c.call(req(),"one.json")
            self.assertTrue((root/"receipts/.stop.json").exists());self.assertEqual(seen,[])
        finally:d.cleanup()

    def test_resolved_version_and_invalid_usage_stop(self):
        for mutation in ({"resolved_model":"anthropic/claude-opus-5.5-not-the-model"}, {"usage":{}}):
            d,root,c=self.make(lambda r:{**self.ok(r),**mutation})
            try:
                with self.assertRaises(FixedArtifactError):c.call(req(),"one.json")
                self.assertTrue((root/"receipts/.stop.json").exists())
            finally:d.cleanup()

    def test_reservations_match_offline_planner(self):
        from scripts.run_fixed_artifact_campaign import _reserve_bytes,DECODERS
        d,root,c=self.make(self.ok)
        try:
            for pricing in DECODERS.values():
                payload=req(pricing["model"])
                self.assertAlmostEqual(c._reserve(payload),_reserve_bytes(len("hello".encode()),pricing))
        finally:d.cleanup()

    def test_phase_cap_denial_keeps_transport_unused(self):
        seen=[];d,root,c=self.make(lambda r:(seen.append(r) or self.ok(r)),cap=0.01)
        try:
            with self.assertRaises(BudgetError):c.call(req(),"one.json")
            self.assertEqual(seen,[])
            self.assertTrue((root/"receipts/.stop.json").exists())
        finally:d.cleanup()

    def test_provider_overcharge_is_not_hidden_by_reservation(self):
        d,root,c=self.make(lambda r:{**self.ok(r),"usage":{"prompt_tokens":3,"completion_tokens":2,"cost":1.0}})
        try:
            with self.assertRaises(BudgetError):c.call(req(),"one.json")
            item=next(iter(BudgetLedger(root/"phase.json",30).snapshot()["reservations"].values()))
            self.assertEqual(item["actual_usd"],1.0)
            self.assertTrue((root/"receipts/.stop.json").exists())
        finally:d.cleanup()

    def test_response_replacement_detected_by_external_anchor(self):
        d,root,c=self.make(self.ok)
        try:
            c.call(req(),"one.json")
            path=root/"receipts/one.json"; record=json.loads(path.read_text())
            record["content"]="schema-valid replacement"
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(FixedArtifactError,"external ledger anchor"):
                c.call(req(),"one.json")
        finally:d.cleanup()


if __name__ == "__main__": unittest.main()
