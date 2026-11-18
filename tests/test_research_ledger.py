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
            self.assertEqual(ledger["study_count"], 56)
            rows = {x["id"]:x for x in ledger["studies"]}
            for sid in ("pera-algebra-diagnostic-v2", "pera-author-layer-probe-v1", "pera-rank-counterexample-v1", "pera-lean-formal-verification-v1", "pera-usefulness-comparison-v2"):
                self.assertIn(sid, rows)
                self.assertFalse(rows[sid]["novelty_established"])
                self.assertFalse(rows[sid]["evidence"]["train_eligible"])
            self.assertIn("gate_failed",rows["pera-algebra-diagnostic-v2"]["raw_verdict"])
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
            matched = rows["matched-preparation-v1"]
            self.assertEqual(matched["raw_verdict"], "preparation_stopped_quote_coverage_failure")
            self.assertEqual(matched["secondary_metrics"]["supported_records"], 2.0)
            self.assertEqual(matched["secondary_metrics"]["unsupported_records"], 22.0)
            self.assertEqual(matched["secondary_metrics"]["generation_calls"], 0.0)
            self.assertAlmostEqual(matched["secondary_metrics"]["ledger_cost_usd"], 0.218348)
            self.assertFalse(matched["novelty_established"])
            calibration = rows["reference-review-calibration-v1"]
            self.assertEqual(calibration["primary_metrics"]["mechanically_valid_reviews"], 16.0)
            self.assertEqual(calibration["primary_metrics"]["reviews_matching_key"], 16.0)
            self.assertFalse(calibration["evidence"]["train_eligible"])
            stopped = rows["campaign2-preparation-contract-v1"]
            self.assertEqual(stopped["raw_verdict"], "preparation_stopped_review_contract_failure")
            self.assertEqual(stopped["primary_metrics"]["paid_preparation_calls"], 1.0)
            self.assertEqual(stopped["primary_metrics"]["mechanically_valid_reviews"], 23.0)
            self.assertEqual(stopped["secondary_metrics"]["generation_calls"], 0.0)
            self.assertAlmostEqual(stopped["secondary_metrics"]["campaign_charge_usd"], 0.106524)
            self.assertFalse(stopped["novelty_established"])
            self.assertFalse(stopped["evidence"]["train_eligible"])
            preparation = rows["fixed-artifact-comparison-preparation-v1"]
            self.assertEqual(preparation["raw_verdict"], "preparation_snapshot_before_generation")
            self.assertEqual(preparation["primary_metrics"]["prepared_records"], 12.0)
            self.assertEqual(preparation["primary_metrics"]["mechanically_valid_reviews"], 23.0)
            self.assertEqual(preparation["primary_metrics"]["malformed_reviews"], 1.0)
            self.assertEqual(preparation["primary_metrics"]["prose_fidelity_fields"], 27.0)
            self.assertEqual(preparation["primary_metrics"]["prose_unknown_fields"], 15.0)
            self.assertEqual(preparation["primary_metrics"]["prose_flagged_either"], 1.0)
            self.assertEqual(preparation["secondary_metrics"]["prose_flagged_both"], 0.0)
            self.assertEqual(preparation["secondary_metrics"]["planned_generation_calls"], 48.0)
            self.assertEqual(preparation["secondary_metrics"]["planned_generation_slots"], 192.0)
            self.assertEqual(preparation["secondary_metrics"]["generation_calls"], 0.0)
            self.assertEqual(preparation["secondary_metrics"]["comparison_results"], 0.0)
            self.assertAlmostEqual(preparation["secondary_metrics"]["actual_upstream_inference_usd"], 0.105936)
            self.assertAlmostEqual(preparation["secondary_metrics"]["conservative_reserve_usd"], 0.3699822)
            self.assertFalse(preparation["novelty_established"])
            self.assertFalse(preparation["evidence"]["train_eligible"])
            self.assertIn("preparation only", preparation["limitations"][0])
            generation = rows["fixed-artifact-generation-v1"]
            self.assertEqual(generation["raw_verdict"], "generation_complete_evaluation_pending")
            self.assertEqual(generation["primary_metrics"]["generation_calls"], 48.0)
            self.assertEqual(generation["primary_metrics"]["candidate_slots"], 192.0)
            self.assertEqual(generation["primary_metrics"]["valid_candidates"], 139.0)
            self.assertEqual(generation["primary_metrics"]["invalid_candidates"], 53.0)
            self.assertEqual(generation["primary_metrics"]["opus_valid"], 51.0)
            self.assertEqual(generation["primary_metrics"]["fable_valid"], 88.0)
            self.assertAlmostEqual(generation["secondary_metrics"]["upstream_inference_usd"], 7.149892)
            self.assertAlmostEqual(generation["secondary_metrics"]["conservative_reserve_usd"], 23.279655)
            self.assertEqual(generation["secondary_metrics"]["final_evaluation_results"], 0.0)
            self.assertEqual(generation["secondary_metrics"]["usefulness_results"], 0.0)
            self.assertFalse(generation["novelty_established"])
            self.assertFalse(generation["evidence"]["train_eligible"])
            self.assertIn("format compliance", generation["limitations"][0])

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

    def test_headroom_feedback_projects_bounded_numeric_summary(self):
        row = next(x for x in build_ledger()["studies"] if x["id"] == "gnn-predictor-headroom-v1")
        self.assertAlmostEqual(row["primary_metrics"]["constant_lr_sni_error"], 0.006535200583991728)
        self.assertAlmostEqual(row["secondary_metrics"]["constant_lr_damped_mix_reduction_vs_sni"], 0.29894935312302723)
        self.assertAlmostEqual(row["secondary_metrics"]["lr_drop_ridge_reduction_vs_sni"], 0.03583115998576373)
        self.assertIn("development-only", row["secondary_metrics"]["scope"])
        self.assertFalse(row["novelty_established"])
        self.assertFalse(row["evidence"]["train_eligible"])

    def test_headroom_malformed_numeric_summary_is_omitted(self):
        from scripts import export_research_ledger as ledger_module
        original = ledger_module._read
        registry = original(ledger_module.ROOT / "docs/method-grounding/experiment-feedback.json", {})
        target = next(x for x in registry["records"] if x["id"] == "gnn-predictor-headroom-v1")
        result_path = str(ledger_module.ROOT / target["result_path"])
        raw_result = original(Path(result_path), {})
        raw_result["schedule_summary"]["constant"]["sni"] = float("nan")

        def fake_read(path, default):
            if str(path) == result_path:
                return raw_result
            return original(path, default)

        ledger_module._read = fake_read
        try:
            rows, warnings = ledger_module._feedback_studies(ledger_module.ROOT)
        finally:
            ledger_module._read = original
        row = next(x for x in rows if x["id"] == "gnn-predictor-headroom-v1")
        self.assertEqual(row["primary_metrics"], {})
        self.assertTrue(any("invalid numeric metric constant_lr_sni_error" in item for item in warnings))

    def test_closedloop_summary_projects_integrity_qualified_counts(self):
        from scripts import export_research_ledger as ledger_module
        record = {"id": "gnn-closedloop-gap-v1", "result_path": "operator-studies/gnn-closedloop-gap/results/summary-v2.json"}
        warnings = []
        primary, secondary = ledger_module._closedloop_metrics(ledger_module.ROOT, record, warnings)
        self.assertEqual(warnings, [])
        self.assertEqual(primary["n_seeds"], 16)
        self.assertEqual(primary["primary_comparisons"], 24)
        self.assertEqual(primary["flagged_exploratory_unadjusted_count"], 13)
        self.assertEqual(primary["flagged_simultaneous_count"], 9)
        self.assertIn("exploratory regime-selection", secondary["scope"])

    def test_closedloop_malformed_flags_or_integrity_are_omitted(self):
        from scripts import export_research_ledger as ledger_module
        original = ledger_module._read
        path = ledger_module.ROOT / "operator-studies/gnn-closedloop-gap/results/summary-v2.json"
        raw = original(path, {})
        raw["flagged_simultaneous"] = list(raw["flagged_simultaneous"]) + [list(raw["flagged_simultaneous"][0])]
        record = {"id": "gnn-closedloop-gap-v1", "result_path": str(path.relative_to(ledger_module.ROOT))}
        def fake_read(candidate, default):
            return raw if Path(candidate).resolve() == path.resolve() else original(candidate, default)
        ledger_module._read = fake_read
        try:
            primary, _ = ledger_module._closedloop_metrics(ledger_module.ROOT, record, [])
        finally:
            ledger_module._read = original
        self.assertEqual(primary, {})

        raw["integrity_checks"]["unique_keys"] = False
        ledger_module._read = fake_read
        warnings = []
        try:
            primary, _ = ledger_module._closedloop_metrics(ledger_module.ROOT, record, warnings)
        finally:
            ledger_module._read = original
        self.assertEqual(primary, {})
        self.assertTrue(any("integrity checks" in item for item in warnings))

    def test_closedloop_confirmation_projects_interval_without_promoting_it(self):
        from scripts import export_research_ledger as ledger_module
        record = {"id": "gnn-closedloop-confirm-v1", "result_path": "operator-studies/gnn-closedloop-confirm/results/interpretation.json"}
        warnings = []
        primary, secondary = ledger_module._closedloop_confirmation_metrics(ledger_module.ROOT, record, warnings)
        self.assertEqual(warnings, [])
        self.assertAlmostEqual(primary["mean_points"], 2.708202856592834)
        self.assertLess(primary["ci95_lower_points"], 0)
        self.assertGreater(primary["ci95_upper_points"], primary["mean_points"])
        self.assertTrue(secondary["positive_at_pointwise_95"].startswith("no;"))
        self.assertTrue(secondary["allocation_rule_passed"])

    def test_closedloop_confirmation_rejects_conflicting_flags_or_nan(self):
        from scripts import export_research_ledger as ledger_module
        original = ledger_module._read
        path = ledger_module.ROOT / "operator-studies/gnn-closedloop-confirm/results/interpretation.json"
        raw = original(path, {})
        record = {"id": "gnn-closedloop-confirm-v1", "result_path": str(path.relative_to(ledger_module.ROOT))}
        def fake_read(candidate, default):
            return raw if Path(candidate).resolve() == path.resolve() else original(candidate, default)
        raw["primary_positive_at_pointwise_95"] = True
        ledger_module._read = fake_read
        try:
            primary, _ = ledger_module._closedloop_confirmation_metrics(ledger_module.ROOT, record, [])
        finally:
            ledger_module._read = original
        self.assertEqual(primary, {})
        raw["primary_positive_at_pointwise_95"] = False
        raw["primary_mean_points"] = float("nan")
        warnings = []
        ledger_module._read = fake_read
        try:
            primary, _ = ledger_module._closedloop_confirmation_metrics(ledger_module.ROOT, record, warnings)
        finally:
            ledger_module._read = original
        self.assertEqual(primary, {})
        self.assertTrue(any("invalid numeric metric" in item for item in warnings))

    def test_bounded_metric_path_selector_reads_only_dict_keys(self):
        from scripts import export_research_ledger as ledger_module
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            result = root / "result.json"
            result.write_text(json.dumps({"summary": {"hold": {"mean": 1.25}, "other": 3.0}}))
            record = {"id": "selector-fixture", "result_path": "result.json", "metric_paths": {
                "primary": {"hold_mean": ["summary", "hold", "mean"]},
                "secondary": {"other": ["summary", "other"]},
            }}
            primary, secondary = ledger_module._selector_metrics(root, record, [])
            self.assertEqual(primary, {"hold_mean": 1.25})
            self.assertEqual(secondary, {"other": 3.0})

    def test_metric_path_selector_rejects_missing_bad_type_nan_and_oversized_paths(self):
        from scripts import export_research_ledger as ledger_module
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); result = root / "result.json"
            result.write_text(json.dumps({"x": {"value": 1.0}, "nan": float("nan")}))
            def check(selector):
                warnings = []
                primary, secondary = ledger_module._selector_metrics(root, {"id": "bad", "result_path": "result.json", "metric_paths": selector}, warnings)
                self.assertEqual((primary, secondary), ({}, {})); self.assertTrue(warnings)
            check({"primary": {"missing": ["x", "nope"]}})
            check({"primary": {"bad": ["x", "value", 1]}})
            check({"primary": {"nan": ["nan"]}})
            check({"primary": {str(i): ["x", "value"] for i in range(9)}})
            check({"primary": {"escape": ["../result.json"]}})

    def test_metric_path_selector_survives_feedback_validation_end_to_end(self):
        from scripts import export_research_ledger as ledger_module
        import hashlib
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); report = root / "operator-studies/demo/README.md"; result = root / "operator-studies/demo/result.json"
            report.parent.mkdir(parents=True); report.write_text("demo")
            result.write_text(json.dumps({"summary": {"hold": {"mean": 1.5}}}) + "\n")
            base = {"id": "selector-e2e", "source_urls": ["https://arxiv.org/abs/2608.06441"],
                    "tested_change": "bounded selector test", "verdict": "diagnostic", "scope": "test scope",
                    "limitations": ["not scientific evidence"], "report_path": "operator-studies/demo/README.md",
                    "result_path": "operator-studies/demo/result.json", "result_sha256": hashlib.sha256(result.read_bytes()).hexdigest(),
                    "train_eligible": False, "novelty_established": False}
            registry = {"schema_version": "experiment-feedback.v1", "records": [base]}
            (root / "docs").mkdir(); (root / "docs/method-grounding").mkdir()
            (root / "docs/method-grounding/experiment-feedback.json").write_text(json.dumps(registry))
            metadata = {"schema": "study-metrics.v1", "records": {"selector-e2e": {
                "result_sha256": base["result_sha256"],
                "metric_paths": {"primary": {"hold_mean": ["summary", "hold", "mean"]}},
            }}}
            (root / "docs/method-grounding/study-metrics.json").write_text(json.dumps(metadata))
            rows, warnings = ledger_module._feedback_studies(root)
            self.assertEqual(warnings, [])
            self.assertEqual(rows[0]["primary_metrics"], {"hold_mean": 1.5})
            metadata["records"]["selector-e2e"]["metric_paths"] = {"primary": {"missing": ["summary", "none"]}}
            (root / "docs/method-grounding/study-metrics.json").write_text(json.dumps(metadata))
            rows, warnings = ledger_module._feedback_studies(root)
            self.assertEqual(rows[0]["primary_metrics"], {})
            self.assertTrue(warnings)
            metadata["records"]["selector-e2e"]["result_sha256"] = "0" * 64
            (root / "docs/method-grounding/study-metrics.json").write_text(json.dumps(metadata))
            rows, warnings = ledger_module._feedback_studies(root)
            self.assertEqual(rows[0]["primary_metrics"], {})
            self.assertTrue(any("study-metrics result hash" in item for item in warnings))
            # The experiment-feedback record itself remains schema-valid and
            # is never extended with UI-only selector metadata.
            self.assertEqual(rows[0]["id"], "selector-e2e")

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
