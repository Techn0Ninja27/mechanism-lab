import json
import tempfile
import unittest
from pathlib import Path

from scripts.export_research_ledger import build_ledger, export


class LedgerTests(unittest.TestCase):
    def test_exports_all_studies_and_separates_secondary_metrics(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "ledger.json"
            ledger = export(out)
            self.assertEqual(ledger["study_count"], 39)
            self.assertTrue(all(x["read_only"] is True for x in [ledger]))
            reconstruction = next(x for x in ledger["studies"] if x["id"] == "abstracts-0")
            self.assertIn("mean_ap", reconstruction["secondary_metrics"])
            self.assertEqual(reconstruction["raw_verdict"], "no_gain")
            self.assertFalse(reconstruction["novelty_established"])
            self.assertFalse(reconstruction["human_adjudicated"])
            candidate = next(x for x in ledger["studies"] if x["id"] == "digits_shift-world-2")
            self.assertEqual(candidate["audit_qualification"], "invalid_mechanism")
            self.assertIn("audit.json", candidate["evidence"]["audit"])
            self.assertTrue(all(not Path(a["path"]).is_absolute() for x in ledger["studies"] for a in x["artifact_refs"]))
            self.assertEqual(sum(x["audit_qualification"] == "clean_negative" for x in ledger["studies"] if x["cohort"] == "candidate"), 7)
            self.assertEqual(sum(x["audit_qualification"] == "invalid_mechanism" for x in ledger["studies"] if x["cohort"] == "candidate"), 3)
            self.assertTrue(all(x["secondary_metrics"].get("mean_ap", {}).get("candidate") is not None for x in ledger["studies"] if x["cohort"] == "reconstruction"))
            self.assertEqual(next(x for x in ledger["studies"] if x["id"] == "atoms-1")["failure_assessment"]["type"], "remaining_mask_information_path")
            self.assertEqual(json.loads(out.read_text())["schema_version"], "1.0")

    def test_hash_checked_feedback_is_read_only_and_non_promotional(self):
        rows = {x["id"]: x for x in build_ledger()["studies"]}
        row = rows["newton-muon-polynomial-matrix-fidelity-v1"]
        self.assertEqual(row["cohort"], "experiment_feedback")
        self.assertEqual(row["audit_qualification"], "registry_hash_and_paths_verified")
        self.assertIn("Tested change:", row["audit_note"])
        self.assertIn("Scope:", row["audit_note"])
        self.assertFalse(row["novelty_established"])
        self.assertFalse(row["human_adjudicated"])
        self.assertFalse(row["evidence"]["train_eligible"])
        self.assertEqual(row["artifact_refs"][1]["sha256"],
                         "3fcd902a467acb1570f43ff2559a217b6dfbf7af47bb1732872fd69c3bdae2c8")

    def test_feedback_hash_mismatch_is_diagnostic_and_not_a_row(self):
        from scripts import export_research_ledger as ledger_module
        original = ledger_module._read
        def fake_read(path, default):
            if str(path).endswith("experiment-feedback.json"):
                registry = original(path, default)
                registry["records"] = [dict(registry["records"][0], result_sha256="0" * 64)]
                return registry
            return original(path, default)
        ledger_module._read = fake_read
        try:
            rows, warnings = ledger_module._feedback_studies(ledger_module.ROOT)
        finally:
            ledger_module._read = original
        self.assertEqual(rows, [])
        self.assertTrue(any("hash does not match" in item for item in warnings))

    def test_feedback_missing_or_escaping_artifacts_are_diagnostic(self):
        from scripts import export_research_ledger as ledger_module
        original = ledger_module._read
        registry = original(ledger_module.ROOT / "docs/method-grounding/experiment-feedback.json", {})
        for bad_path in ("operator-studies/no-such/report.md", "operator-studies/../README.md"):
            record = dict(registry["records"][0], report_path=bad_path)
            def fake_read(path, default, record=record):
                if str(path).endswith("experiment-feedback.json"):
                    return {"schema_version": "experiment-feedback.v1", "records": [record]}
                return original(path, default)
            ledger_module._read = fake_read
            try:
                rows, warnings = ledger_module._feedback_studies(ledger_module.ROOT)
            finally:
                ledger_module._read = original
            self.assertEqual(rows, [])
            self.assertTrue(warnings)

    def test_feedback_duplicate_artifact_directory_is_not_duplicated(self):
        rows = build_ledger()["studies"]
        dirs = [x.get("artifact_directory") for x in rows if x.get("artifact_directory")]
        self.assertEqual(len(dirs), len(set(dirs)))
        self.assertFalse(any(x["id"] == "felog-rank-varint-walks-v1" for x in rows))

    def test_new_results_preserve_failure_and_frozen_provenance(self):
        rows = {x["id"]: x for x in build_ledger()["studies"]}
        utility = rows["utility-ranker"]
        self.assertEqual(utility["raw_verdict"], "not_supported")
        self.assertGreater(utility["primary_metrics"]["mean_candidate"], utility["primary_metrics"]["mean_control"])
        for sid in ("utility-ranker", "integer-composition"):
            self.assertEqual(rows[sid]["audit_qualification"], "frozen_sources_and_validation_match")
            self.assertFalse(rows[sid]["novelty_established"])

    def test_review_diagnostic_is_not_promoted_by_better_status_agreement(self):
        row=next(x for x in build_ledger()['studies'] if x['id']=='mechanism-review')
        self.assertEqual(row['audit_qualification'], 'not_independent_quality_evidence')
        self.assertFalse(row['novelty_established'])
        self.assertFalse(row['human_adjudicated'])
        self.assertEqual(row['primary_metrics']['old_reference_status_matches'],4)
        self.assertEqual(row['primary_metrics']['specialization_reference_status_matches'],5)
        self.assertIn('status contradiction',row['audit_note'])

    def test_graph_codec_failure_remains_development_evidence(self):
        row=next(x for x in build_ledger()['studies'] if x['id']=='graph-walk-codecs')
        self.assertEqual(row['raw_verdict'],'fails_development_gate')
        self.assertFalse(row['novelty_established'])
        self.assertGreater(row['primary_metrics']['amazon_resident_ratio_to_packed'],1)
        self.assertGreater(row['primary_metrics']['dblp_resident_ratio_to_packed'],1)
        self.assertIn('not_systems_validation',row['audit_qualification'])

    def test_utility_judgment_preserves_abstentions_and_retrospective_scope(self):
        row=next(x for x in build_ledger()['studies'] if x['id']=='utility-judgment-pilot')
        self.assertEqual(row['primary_metrics']['abstentions'],3)
        self.assertEqual(row['primary_metrics']['correct_answered'],1)
        self.assertFalse(row['novelty_established'])
        self.assertIn('not_ranker_validation',row['audit_qualification'])
        self.assertIn('not independent',row['audit_note'])

    def test_missing_frozen_sources_are_not_verified(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            folder = root / "reports/utility-ranker"
            folder.mkdir(parents=True)
            (folder / "final.json").write_text('{"status":"not_supported","summary":{}}')
            row = build_ledger(root)["studies"][0]
            self.assertEqual(row["raw_verdict"], "not_supported")
            self.assertEqual(row["audit_qualification"], "artifact_mismatch")
            self.assertIn("Missing frozen source hashes", row["audit_note"])

    def test_pure_builder_handles_missing_optional_fields_and_relative_paths(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "candidate-studies/x").mkdir(parents=True)
            (root / "candidate-studies/x/verdict.json").write_text('{"candidate_id":"x"}')
            (root / "candidate-studies/x/proposal.json").write_text('{}')
            (root / "candidate-studies/audit.json").write_text('{"cohort":{}}')
            (root / "reconstruction-studies").mkdir()
            (root / "reconstruction-studies/harmonized-results.json").write_text('{"comparisons":[]}')
            (root / "reconstruction-studies/manifest.json").write_text('[]')
            ledger = build_ledger(root)
            self.assertEqual(ledger["study_count"], 1)
            self.assertEqual(ledger["studies"][0]["raw_verdict"], "unknown")
            self.assertTrue(all(not Path(a["path"]).is_absolute() for a in ledger["studies"][0]["artifact_refs"]))


if __name__ == "__main__":
    unittest.main()
