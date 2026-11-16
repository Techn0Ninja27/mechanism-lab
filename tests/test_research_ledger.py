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
            self.assertEqual(ledger["study_count"], 31)
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
