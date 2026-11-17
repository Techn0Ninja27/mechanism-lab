import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.export_contract_drafts import export_run


class ContractExportTests(unittest.TestCase):
    def test_redacts_quotes_binds_summary_and_preserves_failure(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); run = root / "run"; run.mkdir()
            result = {"status": "reviewed_drafts_not_curated", "source_ids": ["paper"],
                      "drafts": [{"source_id": "paper", "contract": {"id": "step", "source_id": "paper", "interpretation": "model_proposed_not_curated",
                                  "inputs": [{"id":"input","artifact":"tensor","representation":None,"granularity":None,"phase":None,"role":"artifact","exposure":"required","conditions":[],"evidence_ids":["e"]}],
                                  "outputs": [{"id":"output","artifact":"tensor","representation":None,"granularity":None,"phase":None,"role":"artifact","exposure":"unknown","conditions":[],"evidence_ids":["e"]}],
                                  "evidence": [{"quote": "secret source span", "id": "e", "chunk_id":"paper:chunk:0001", "source_sha256":""}]}, "unknowns": ["shape missing"]}],
                      "review": {"assessments": [{"source_id": "paper", "contract_id": "step", "decision": "revise", "reason": "needs work", "issues": []}]}}
            (run / "result.json").write_text(json.dumps(result))
            from generation.sourcepack import build_source_pack
            (root / "paper.txt").write_text("secret source span")
            pack = root / "pack.json"; pack.write_text(json.dumps(build_source_pack({"sources": [{"id": "paper", "url": "https://example.test/paper", "title": "P", "path": "paper.txt", "evidence_level": "full_text"}]}, root)))
            pp = json.loads(pack.read_text()); result["drafts"][0]["contract"]["evidence"][0]["source_sha256"] = pp["sources"][0]["content_sha256"]; (run / "result.json").write_text(json.dumps(result))
            result["review"]["assessments"][0]["issues"] = [{"port_id":"input","field":"representation","reason":"Unspecified representation.","evidence":[{"chunk_id":"paper:chunk:0001","quote":"secret source span"}]}]
            (run / "result.json").write_text(json.dumps(result))
            original = (run / "result.json").read_bytes()
            out = root / "out"; export_run(run, out, pack)
            self.assertEqual((run / "result.json").read_bytes(), original)
            self.assertNotIn("secret source span", (out / "library-summary.json").read_text())
            public = json.loads((out / "result.json").read_text()); summary = json.loads((out / "library-summary.json").read_text())
            self.assertNotIn("secret source span", (out / "result.json").read_text())
            self.assertEqual(public["drafts"][0]["contract"]["evidence"][0]["quote_sha256"], hashlib.sha256(b"secret source span").hexdigest())
            self.assertEqual(summary["result_sha256"], hashlib.sha256((out / "result.json").read_bytes()).hexdigest())
            self.assertEqual(summary["status"], "reviewed_drafts_not_curated")

    def test_rejects_raw_prompt_and_refuses_mismatched_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); run = root / "run"; run.mkdir()
            (run / "result.json").write_text(json.dumps({"status": "failed", "source_ids": ["paper"], "prompt": "do not publish"}))
            from generation.sourcepack import build_source_pack
            (root / "paper.txt").write_text("source")
            pack = root / "pack.json"; pack.write_text(json.dumps(build_source_pack({"sources": [{"id": "paper", "url": "https://p", "title": "P", "path": "paper.txt", "evidence_level": "full_text"}]}, root)))
            with self.assertRaisesRegex(ValueError, "raw model"):
                export_run(run, root / "out", pack)
            (run / "result.json").write_text(json.dumps({"status": "failed", "source_ids": ["paper"], "drafts": []}))
            out = root / "out"; export_run(run, out, pack)
            (run / "result.json").write_text(json.dumps({"status": "other", "source_ids": ["paper"], "drafts": []}))
            with self.assertRaises(ValueError):
                export_run(run, out, pack)

    def test_null_review_and_failed_sources_are_retained_without_quotes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); run = root / "run"; run.mkdir()
            result = {"status": "failed", "source_ids": ["a", "b"], "drafts": [{
                "source_id": "a", "contract": None, "unknowns": ["unsupported"],
            }], "review": {"assessments": [{"source_id": "a", "contract_id": None,
                "decision": "insufficient_evidence", "reason": "No support.", "issues": []}]}}
            (run / "result.json").write_text(json.dumps(result))
            from generation.sourcepack import build_source_pack
            (root / "a.txt").write_text("x"); (root / "b.txt").write_text("y")
            pack = root / "pack.json"; pack.write_text(json.dumps(build_source_pack({"sources": [
                {"id": "a", "url": "https://a", "title": "A", "path": "a.txt", "evidence_level": "full_text"},
                {"id": "b", "url": "https://b", "title": "B", "path": "b.txt", "evidence_level": "full_text"}]}, root)))
            out = root / "out"; export_run(run, out, pack)
            summary = json.loads((out / "library-summary.json").read_text()); text = (out / "result.json").read_text()
            self.assertEqual(len(summary["records"]), 2)
            self.assertEqual(summary["records"][0]["review"]["decision"], "insufficient_evidence")
            self.assertEqual(summary["records"][1]["review"]["decision"], "not_recorded")
            self.assertNotIn('"evidence"', json.dumps(summary))

    def test_rejects_invalid_status_and_card(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); run = root / "run"; run.mkdir()
            pack = root / "pack.json"
            # Reuse a minimal valid pack fixture through the builder.
            from generation.sourcepack import build_source_pack
            (root / "paper.txt").write_text("x")
            p = build_source_pack({"sources": [{"id": "paper", "title": "P", "url": "https://p", "path": "paper.txt", "evidence_level": "full_text"}]}, root)
            pack.write_text(json.dumps(p))
            (run / "result.json").write_text(json.dumps({"status": "pending", "source_ids": ["paper"], "drafts": []}))
            with self.assertRaises(ValueError): export_run(run, root / "out", pack)
            for drafts in ([{"source_id":"other","contract":None,"unknowns":["unknown"]}],
                           [{"source_id":"paper","contract":{},"unknowns":["unknown"]}],
                           [{"source_id":"paper","contract":None,"unknowns":["unknown"]}]*2):
                (run / "result.json").write_text(json.dumps({"status":"failed","source_ids":["paper"],"drafts":drafts}))
                with self.assertRaises(ValueError): export_run(run, root / "out", pack)


if __name__ == "__main__":
    unittest.main()
