from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from generation.sourcepack import SourcePackError
from scripts.prepare_matched_representations import (
    ATOM_COUNT,
    prepare,
    RENDER_EXCLUDED_PREFIX,
    render_source_context,
    source_quotas,
    validate_atoms,
    validate_distillations,
)


ROOT = Path(__file__).resolve().parents[1]
PACK_PATH = ROOT / "results/gnn-useful-proposal-v1/pack.json"


class MatchedRepresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pack = json.loads(PACK_PATH.read_text())

    def atom_fixture(self):
        chunks = render_source_context(self.pack)["chunks"]
        rows = []
        i = 0
        for source in self.pack["sources"]:
            for _ in range(source_quotas(self.pack)[source["id"]]):
                chunk = next(c for c in chunks if c["source_id"] == source["id"])
                rows.append({"id": f"atom-{i:02d}", "source_id": source["id"],
                             "statement": f"Statement {i}", "inputs": ["input"],
                             "outputs": ["output"], "conditions": ["condition"],
                             "unknowns": ["unknown"],
                             "evidence": [{"chunk_id": chunk["id"], "quote": chunk["text"][:40]}]})
                i += 1
        return {"atoms": rows}

    def distill_fixture(self):
        rows = []
        chunks = render_source_context(self.pack)["chunks"]
        i = 0
        for source in self.pack["sources"]:
            for _ in range(source_quotas(self.pack)[source["id"]]):
                chunk = next(c for c in chunks if c["source_id"] == source["id"])
                rows.append({"id": f"distill-{i:02d}", "source_id": source["id"], "summary": "Summary",
                             "unknowns": ["unknown"], "evidence": [{"chunk_id": chunk["id"], "quote": chunk["text"][:40]}]})
                i += 1
        return {"distillations": rows}

    def test_offline_plan_binds_all_sources_and_prompts(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "plan"
            manifest = prepare(PACK_PATH, out)
            self.assertEqual(manifest["status"], "offline_plan_only")
            self.assertEqual(manifest["source_count"], 5)
            self.assertEqual(manifest["paid_calls"], 0)
            prompts = json.loads((out / "prompts.json").read_text())
            self.assertEqual(set(prompts), {"atoms", "distillations"})
            self.assertTrue(all(manifest["prompts"][k]["task_blind_intent"] for k in prompts))
            self.assertEqual(manifest["atom_count"], 12)
            self.assertIn("top-level key atoms", prompts["atoms"])
            self.assertIn("top-level key distillations", prompts["distillations"])
            self.assertIn("statement", prompts["atoms"])
            self.assertIn("unknowns", prompts["distillations"])
            self.assertEqual(len(manifest["render_exclusions"]), 1)
            self.assertEqual(manifest["render_exclusions"][0]["prefix_chars"], len(RENDER_EXCLUDED_PREFIX))

    def test_existing_atoms_are_not_silently_accepted(self):
        old = ROOT / "results/ml-graph-atoms-v1/extraction/atoms.jsonl"
        rows = [json.loads(line) for line in old.read_text().splitlines() if line.strip()]
        with self.assertRaises(ValueError):
            validate_atoms({"atoms": rows[:ATOM_COUNT]}, self.pack)

    def test_atom_evidence_must_bind_to_declared_source(self):
        fixture = self.atom_fixture()
        fixture["atoms"][0]["evidence"][0]["quote"] = "not in source"
        with self.assertRaises(ValueError):
            validate_atoms(fixture, self.pack)

    def test_common_render_excludes_only_known_header_and_preserves_pack(self):
        original = json.dumps(self.pack, sort_keys=True)
        rendered = render_source_context(self.pack)
        self.assertEqual(json.dumps(self.pack, sort_keys=True), original)
        excluded = rendered["chunks"][0]
        self.assertFalse(excluded["text"].startswith(RENDER_EXCLUDED_PREFIX))
        self.assertIn("IV Design Details of SNI-GNN", excluded["text"])
        body_quote = "IV Design Details of SNI-GNN"
        fixture = self.atom_fixture()
        fixture["atoms"][0]["evidence"][0] = {"chunk_id": excluded["id"], "quote": body_quote}
        self.assertEqual(validate_atoms(fixture, self.pack)["atom_count"], ATOM_COUNT)
        fixture = self.atom_fixture()
        fixture["atoms"][0]["evidence"][0] = {"chunk_id": excluded["id"], "quote": RENDER_EXCLUDED_PREFIX.strip()}
        with self.assertRaises(ValueError):
            validate_atoms(fixture, self.pack)

    def test_atom_statement_and_source_type_are_checked(self):
        fixture = self.atom_fixture()
        fixture["atoms"][0]["statement"] = ""
        with self.assertRaises(ValueError):
            validate_atoms(fixture, self.pack)
        fixture = self.atom_fixture()
        fixture["atoms"][0]["source_id"] = []
        with self.assertRaises(ValueError):
            validate_atoms(fixture, self.pack)

    def test_extra_top_level_fields_are_rejected(self):
        fixture = self.atom_fixture()
        fixture["extra"] = "claim"
        with self.assertRaises(ValueError):
            validate_atoms(fixture, self.pack)

    def test_collectively_oversized_valid_fields_are_rejected(self):
        fixture = self.atom_fixture()
        for atom in fixture["atoms"]:
            atom["statement"] = "x" * 2400
        with self.assertRaises(ValueError):
            validate_atoms(fixture, self.pack)

    def test_atom_count_and_source_coverage_are_fixed(self):
        fixture = self.atom_fixture()
        fixture["atoms"] = fixture["atoms"][:-1]
        with self.assertRaises(ValueError):
            validate_atoms(fixture, self.pack)
        fixture = self.atom_fixture()
        for atom in fixture["atoms"]:
            atom["source_id"] = self.pack["sources"][0]["id"]
            atom["evidence"][0]["chunk_id"] = next(c for c in self.pack["chunks"] if c["source_id"] == atom["source_id"])["id"]
            atom["evidence"][0]["quote"] = next(c for c in self.pack["chunks"] if c["source_id"] == atom["source_id"])["text"][:40]
        with self.assertRaises(ValueError):
            validate_atoms(fixture, self.pack)

    def test_distillations_require_all_five_sources(self):
        fixture = self.distill_fixture()
        self.assertEqual(validate_distillations(fixture, self.pack)["source_count"], 5)
        self.assertEqual(validate_distillations(fixture, self.pack)["record_count"], 12)
        fixture["distillations"][-1]["source_id"] = fixture["distillations"][0]["source_id"]
        with self.assertRaises(ValueError):
            validate_distillations(fixture, self.pack)

    def test_plan_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "plan"
            prepare(PACK_PATH, out)
            with self.assertRaises(FileExistsError):
                prepare(PACK_PATH, out)


if __name__ == "__main__":
    unittest.main()
