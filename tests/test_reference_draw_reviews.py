import json
import tempfile
import unittest
from pathlib import Path

from generation.reference_draw_reviews import build_review_set, combine_reviews, materialize_review
from scripts import prepare_chunk_representations as prep


ROOT = Path(__file__).resolve().parents[1]
PACK_PATH = ROOT / "results/gnn-useful-proposal-v1/pack.json"
V4 = ROOT / "results/recordwise-calibration-v4"


class ReferenceDrawReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pack = prep._pack(PACK_PATH)
        source = json.loads((ROOT / "results/generation-comparison-v5-plan/representations/atoms.json").read_text())
        records = []
        for i in range(12):
            value = source["atoms"][i]
            value = json.loads(json.dumps(value))
            value["id"] = f"fixture-{i:02d}"
            records.append(value)
        cls.representation = {"schema": prep.ATOM_SCHEMA, "atoms": records}

    def _make_set(self, root, slot="a"):
        return build_review_set(root, plan_sha256="a" * 64, kind="atoms",
                               draw=0, slot=slot, pack=self.pack,
                               representation=self.representation)

    def _write_outputs(self, root):
        for path in sorted(root.glob("input-*.json")):
            packet = json.loads(path.read_text())
            items = []
            first = packet["evidence"][0]["evidence_id"]
            for field in packet["fields"]:
                item = {"field_id": field["field_id"], "verdict": "supported",
                        "reason": "the cited window states this claim in " + root.name,
                        "evidence_ids": [] if field["is_unknown"] else [first]}
                if field["is_unknown"]:
                    item["absence_reason"] = "the supplied cited chunk does not specify it"
                items.append(item)
            (root / f"review-{path.stem.removeprefix('input-')}.json").write_text(
                json.dumps({"fields": items}) + "\n")

    def test_round_trip_preserves_all_evidence_and_uses_new_schema(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "slot-a"
            self._make_set(root)
            self._write_outputs(root)
            review = materialize_review(root, plan_sha256="a" * 64, kind="atoms",
                                        draw=0, slot="a", pack=self.pack,
                                        representation=self.representation,
                                        reviewer_id="reviewer-a")
            self.assertEqual(review["schema"], "reference-draw-review.v1")
            self.assertEqual(len(review["records"]), 12)
            self.assertEqual(review["legacy_compatibility"]["status"], "not_compatible")
            self.assertTrue(all("field_items" in row for row in review["records"]))
            self.assertTrue(all("evidence" in item for row in review["records"] for item in row["field_items"] if item["verdict"] == "supported"))

    def test_missing_or_tampered_packet_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "slot-b"
            self._make_set(root, slot="b")
            self._write_outputs(root)
            (root / "input-00.json").write_text((root / "input-00.json").read_text().replace("fixture-00", "changed"))
            with self.assertRaises(ValueError):
                materialize_review(root, plan_sha256="a" * 64, kind="atoms",
                                   draw=0, slot="b", pack=self.pack,
                                   representation=self.representation,
                                   reviewer_id="reviewer-b")

    def test_multiple_spans_nested_claims_and_dual_slot(self):
        with tempfile.TemporaryDirectory() as td:
            a = Path(td) / "a"; b = Path(td) / "b"
            self._make_set(a, slot="a"); self._make_set(b, slot="b")
            self._write_outputs(a); self._write_outputs(b)
            packet = json.loads((a / "input-00.json").read_text())
            output = json.loads((a / "review-00.json").read_text())
            ids = [x["evidence_id"] for x in packet["evidence"][:2]]
            output["fields"][0]["evidence_ids"] = ids
            output["fields"][0]["claim_assessments"] = [
                {"claim": "primary", "verdict": "supported", "reason": "two source windows", "evidence_ids": ids}
            ]
            (a / "review-00.json").write_text(json.dumps(output) + "\n")
            # Recompute the raw output is intentionally not a repair: this
            # test models one valid output written once before materialization.
            ra = materialize_review(a, plan_sha256="a" * 64, kind="atoms", draw=0,
                                    slot="a", pack=self.pack,
                                    representation=self.representation, reviewer_id="reviewer-a")
            rb = materialize_review(b, plan_sha256="a" * 64, kind="atoms", draw=0,
                                   slot="b", pack=self.pack,
                                   representation=self.representation, reviewer_id="reviewer-b")
            self.assertEqual(len(ra["records"][0]["field_items"][0]["evidence"]), 2)
            self.assertEqual(len(ra["records"][0]["field_items"][0]["claim_assessments"]), 1)
            combined = combine_reviews(ra, rb, plan_sha256="a" * 64, kind="atoms", draw=0)
            self.assertEqual(combined["schema"], "reference-dual-draw-review.v1")
            self.assertEqual(combined["status"], "passed")
            ra["records"][0]["field_items"] = []
            with self.assertRaisesRegex(ValueError, "empty field"):
                combine_reviews(ra, rb, plan_sha256="a" * 64, kind="atoms", draw=0)

    def test_missing_reviewer_output_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "slot-a"
            self._make_set(root); self._write_outputs(root)
            (root / "review-03.json").unlink()
            with self.assertRaises(ValueError):
                materialize_review(root, plan_sha256="a" * 64, kind="atoms", draw=0,
                                   slot="a", pack=self.pack,
                                   representation=self.representation,
                                   reviewer_id="reviewer-a")
            (root / "input-00.json").unlink()
            with self.assertRaises(ValueError):
                materialize_review(root, plan_sha256="a" * 64, kind="atoms",
                                   draw=0, slot="b", pack=self.pack,
                                   representation=self.representation,
                                   reviewer_id="reviewer-b")


if __name__ == "__main__":
    unittest.main()
