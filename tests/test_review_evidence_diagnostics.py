import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_review_evidence import diagnose, audit_paths


def review(item):
    return {"reviewer_id": "fixture", "records": [{"record_id": "r1", "field_items": [item]}]}


class EvidenceDiagnosticsTests(unittest.TestCase):
    def test_fixed_interval_and_header_are_diagnostic_only(self):
        item = {"field": "conditions[0]", "value": "fixed synchronization interval 16", "verdict": "supported",
                "reason": "generic", "support_quote": "Sync Epochs exchange fresh embeddings.\nSelected Section IV only; figures omitted; equations from HTML alttext."}
        result = diagnose(review(item)); flag = result["flagged_fields"][0]
        self.assertEqual(flag["diagnostic"], "diagnostic_not_entailment")
        self.assertIn("fixed", flag["qualifier_words_absent_from_quote"])
        self.assertIn("16", flag["numeric_tokens_absent_from_quote"])
        self.assertEqual(result["injected_header_quote_count"], 1)

    def test_qualifier_in_quote_is_not_flagged(self):
        item = {"field": "statement", "value": "Every node has a GPU", "verdict": "supported",
                "reason": "specific", "support_quote": "Every node has a GPU."}
        self.assertEqual(diagnose(review(item))["flagged_fields"], [])

    def test_reason_repetition_and_empty_overlap_warning(self):
        item = {"field": "statement", "value": "quantized throughput", "verdict": "supported",
                "reason": "boilerplate", "support_quote": "The system stores embeddings."}
        r = {"reviewer_id": "fixture", "records": [{"record_id": "r1", "field_items": [item, dict(item, field="outputs[0]")]}]}
        result = diagnose(r)
        self.assertEqual(result["reason_frequencies"][0], {"reason": "boilerplate", "count": 2})
        self.assertTrue(all(x["lexical_overlap_warning"] for x in result["flagged_fields"]))

    def test_equivalent_paraphrase_warning_is_not_a_verdict(self):
        item = {"field": "statement", "value": "The device predicts remote embeddings", "verdict": "supported",
                "reason": "specific", "support_quote": "The device forecasts embeddings from distant nodes."}
        result = diagnose(review(item)); self.assertEqual(result["record_count"], 1)
        # Token overlap is deliberately only a diagnostic; a paraphrase may
        # produce no warning and must never be converted into a verdict.
        if result["flagged_fields"]:
            self.assertEqual(result["flagged_fields"][0]["diagnostic"], "diagnostic_not_entailment")
        self.assertEqual(result["scope"].split(";")[0], "Shape and lexical diagnostics only")

    def test_redacted_output_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); source = root / "review.json"; output = root / "summary.json"
            source.write_text(json.dumps(review({"field": "statement", "value": "x", "verdict": "supported", "reason": "r", "support_quote": "x"})))
            result = audit_paths([source], output)
            self.assertTrue(output.exists()); self.assertNotIn("support_quote", output.read_text())
            with self.assertRaises(FileExistsError): audit_paths([source], output)


if __name__ == "__main__": unittest.main()
