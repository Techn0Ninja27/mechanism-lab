import json
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import literature_atomization as la


PAPERS = [{"id": "p1", "title": "One", "url": "https://example/p1",
           "abstract": "We introduce a robust method for missing data."},
          {"id": "p2", "title": "Two", "url": "https://example/p2",
           "abstract": "The method improves recall but has a limitation."}]


def atom(source_id="p1", quote="robust method for missing data"):
    return {"source_id": source_id, "source_quote": quote, "statement": "A method handles missing data.",
            "kind": "method", "inputs": ["data"], "outputs": ["predictions"],
            "assumptions": [{"text": "The input is tabular.", "basis": "inferred"}]}


class LiteratureAtomizationTests(unittest.TestCase):
    def test_quote_and_source_hash_are_validated(self):
        loaded = [{"source_id": "p1", "title": "One", "url": "u", "abstract": PAPERS[0]["abstract"], "authors": []}]
        out = la.validate_atoms({"atoms": [atom()]}, loaded)
        self.assertFalse(out[0]["train_eligible"])
        self.assertEqual(len(out[0]["source_hash"]), 64)

    def test_unknown_id_and_nonquote_fail(self):
        loaded = [{"source_id": "p1", "title": "One", "url": "u", "abstract": PAPERS[0]["abstract"], "authors": []}]
        with self.assertRaises(ValueError): la.validate_atoms({"atoms": [atom("missing")]}, loaded)
        with self.assertRaises(ValueError): la.validate_atoms({"atoms": [atom(quote="invented text")]}, loaded)

    def test_plan_batches_and_bound(self):
        with tempfile.TemporaryDirectory() as d:
            inp = pathlib.Path(d) / "papers.json"; out = pathlib.Path(d) / "out"; inp.write_text(json.dumps(PAPERS))
            cfg = la.plan(inp, out, la.DEFAULT_MODEL, la.DEFAULT_MAX_TOKENS)
            self.assertEqual(cfg["batch_count"], 1); self.assertTrue((out / "plan.json").exists())

    def test_legacy_paid_runner_fails_before_key_or_network(self):
        with tempfile.TemporaryDirectory() as d:
            inp = pathlib.Path(d) / "papers.json"; out = pathlib.Path(d) / "out"; inp.write_text(json.dumps(PAPERS))
            with patch.object(la, "load_papers", side_effect=AssertionError("legacy path must not read input")):
                with self.assertRaisesRegex(RuntimeError, "atomize_budgeted.py"):
                    la.run(inp, out, la.DEFAULT_MODEL, la.DEFAULT_MAX_TOKENS)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
