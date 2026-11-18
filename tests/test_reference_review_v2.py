import copy
import json
import tempfile
import unittest
from pathlib import Path

from generation.evidence_reference_packets import build_evidence_packet
from generation.reference_review_v2 import decode_review_v2, reanalyse_v6
from scripts import prepare_chunk_representations as prep

ROOT = Path(__file__).resolve().parents[1]


class ReferenceReviewV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pack = prep._pack(ROOT / "results/gnn-useful-proposal-v1/pack.json")
        cls.record = json.loads((ROOT / "results/recordwise-calibration-v6/reviewer-inputs/input-00.json").read_text())

    def _output(self, packet):
        first = packet["evidence"][0]["evidence_id"]
        return {"fields": [{"field_id": f["field_id"], "verdict": "supported",
                             "reason": "specific source reason",
                             "evidence_ids": [] if f["is_unknown"] else [first]}
                            for f in packet["fields"]]}

    def test_optional_subclaim_id_is_nonsemantic_and_raw_unchanged(self):
        packet = self.record
        out = self._output(packet)
        field = out["fields"][0]
        field["claim_assessments"] = [{"subclaim_id": "claim-a", "verdict": "supported",
                                        "reason": "specific", "evidence_ids": [packet["evidence"][0]["evidence_id"]]}]
        raw = copy.deepcopy(out)
        decoded = decode_review_v2(packet, out)
        self.assertEqual(decoded["field_items"][0]["verdict"], "supported")
        self.assertEqual(out, raw)

    def test_metadata_shape_is_strict(self):
        packet = self.record
        for value in ("", 4, "x" * 101):
            out = self._output(packet)
            out["fields"][0]["claim_assessments"] = [{"subclaim_id": value,
                "verdict": "supported", "reason": "specific",
                "evidence_ids": [packet["evidence"][0]["evidence_id"]]}]
            with self.assertRaises(ValueError):
                decode_review_v2(packet, out)
        out = self._output(packet)
        out["fields"][0]["claim_assessments"] = [{"subclaim_id": "a", "extra": True,
            "verdict": "supported", "reason": "specific",
            "evidence_ids": [packet["evidence"][0]["evidence_id"]]}]
        with self.assertRaises(ValueError):
            decode_review_v2(packet, out)

    def test_semantic_and_evidence_errors_still_fail(self):
        packet = self.record
        out = self._output(packet)
        out["fields"][0]["evidence_ids"] = ["E9999"]
        with self.assertRaises(ValueError):
            decode_review_v2(packet, out)
        out = self._output(packet)
        out["fields"][0]["claim_assessments"] = [{"subclaim_id": "a",
            "verdict": "unsupported", "reason": "missing", "evidence_ids": []}]
        with self.assertRaises(ValueError):
            decode_review_v2(packet, out)

    def test_v6_reanalysis_is_separate_and_does_not_mutate_raw(self):
        root = ROOT / "results/recordwise-calibration-v6"
        before = (root / "reviews/review-11-a.json").read_bytes()
        with tempfile.TemporaryDirectory() as td:
            result = reanalyse_v6(root, Path(td) / "result.json")
            self.assertEqual(result["review_count"], 24)
            self.assertEqual(result["strict_v1_result"]["mechanically_valid"], 23)
            self.assertEqual(result["v2_result"]["mechanically_valid"], 24)
        self.assertEqual((root / "reviews/review-11-a.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
