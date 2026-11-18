import json
import tempfile
import unittest
from pathlib import Path

from generation.record_review_packets import build_record_packet, build_probe_v2, INSTRUCTIONS
from scripts import prepare_chunk_representations as prep

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "results/gnn-useful-proposal-v1/pack.json"
V1 = ROOT / "results/isolated-entailment-probe-v1"


class RecordReviewPacketTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pack = prep._pack(PACK)
        cls.atom = json.loads((V1 / "input-0.json").read_text())["record"]

    def test_scientific_fields_exclude_metadata_flags_and_header(self):
        packet = build_record_packet("atoms", self.atom, self.pack)
        fields = {x["field"] for x in packet["scientific_fields"]}
        self.assertNotIn("interpretation", fields)
        self.assertNotIn("train_eligible", fields)
        self.assertNotIn("novelty_established", fields)
        self.assertNotIn("record_id", fields)
        self.assertFalse(packet["cited_chunks"][0]["text"].startswith("Selected Section IV only"))

    def test_only_record_cited_chunks_are_included(self):
        packet = build_record_packet("atoms", self.atom, self.pack)
        self.assertEqual(packet["record_metadata"]["evidence_chunk_ids"], ["sni-method:chunk:0001"])
        self.assertEqual([x["chunk_id"] for x in packet["cited_chunks"]], ["sni-method:chunk:0001"])

    def test_instructions_define_unknowns_and_item_specific_evidence(self):
        self.assertIn("could be false while the cited text remained true", INSTRUCTIONS)
        self.assertIn("metadata", INSTRUCTIONS)
        self.assertIn("unknown item", INSTRUCTIONS)
        self.assertIn("300 characters", INSTRUCTIONS)

    def test_wrong_source_or_duplicate_chunk_rejected(self):
        bad = json.loads(json.dumps(self.atom))
        bad["evidence"][0]["chunk_id"] = "W4396628137:chunk:0001"
        with self.assertRaises(ValueError): build_record_packet("atoms", bad, self.pack)
        bad = json.loads(json.dumps(self.atom)); bad["evidence"].append(dict(bad["evidence"][0]))
        with self.assertRaises(ValueError): build_record_packet("atoms", bad, self.pack)

    def test_probe_builder_does_not_copy_answer_key(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "probe"
            manifest = build_probe_v2(V1, out, PACK)
            self.assertEqual(len(manifest["packets"]), 4)
            self.assertFalse((out / "answer-key.json").exists())
            self.assertTrue((out / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
