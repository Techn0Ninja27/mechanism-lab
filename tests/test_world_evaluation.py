import json
import unittest

from generation.world_evaluation import (WorldEvaluationError, blind_cases,
                                          blind_cases_with_mapping,
                                          normalized_mechanism, summarize,
                                          validate_mechanism, validate_primitive_triage)


def idea(title="A"):
    return {
        "title": title, "intervention": "change X", "mechanism": "X changes Y",
        "assumptions": ["X is measurable"], "baseline": "do nothing",
        "metric": "delta Y", "falsification": "no delta Y",
        "implementation": "run the controlled test", "queries": ["X Y"],
    }


def record(case_id, arm, title="A"):
    return {"id": case_id, "arm": arm, "problem": "test problem", "idea": idea(title)}


class WorldEvaluationTests(unittest.TestCase):
    def test_schema_requires_all_mechanism_fields(self):
        with self.assertRaises(WorldEvaluationError):
            validate_mechanism({"title": "only title"})

    def test_blind_export_is_deterministic_and_hides_arm(self):
        records = [record("a", "direct"), record("b", "world", "B")]
        first = blind_cases(records, seed=7)
        second = blind_cases(records, seed=7)
        self.assertEqual(first, second)
        self.assertTrue(all("arm" not in case and "id" not in case for case in first))
        self.assertEqual([x["blind_id"] for x in first], ["case-0000", "case-0001"])

    def test_blind_mapping_is_separate_and_reproducible(self):
        records = [record("a", "direct"), record("b", "world", "B")]
        blind, mapping = blind_cases_with_mapping(records, seed=7)
        self.assertEqual(set(mapping), {x["blind_id"] for x in blind})
        self.assertEqual(set(mapping.values()), {"a", "b"})
        self.assertNotIn("arm", json.dumps(blind))

    def test_normalization_collapses_whitespace_case(self):
        a = idea("A")
        b = idea(" a ")
        b["mechanism"] = " X   changes Y "
        self.assertEqual(normalized_mechanism(a), normalized_mechanism(b))

    def test_normalization_detects_same_mechanism_different_title(self):
        a = idea("A")
        b = idea("Different title")
        b["implementation"] = "a different implementation description"
        self.assertEqual(normalized_mechanism(a), normalized_mechanism(b))

    def test_assumption_string_is_normalized_for_pilot_schema(self):
        value = idea()
        value["assumptions"] = "X is measurable"
        self.assertEqual(validate_mechanism(value)["assumptions"], ["X is measurable"])

    def test_primitive_triage_keeps_claims_separate(self):
        triage = validate_primitive_triage({
            "representation_gap": "clear", "literature_novelty": "insufficient_evidence",
            "empirical_utility": "untested", "atom_action": "propose_for_review",
            "evidence_source_ids": ["p1"]}, evidence_ids=["p1"])
        self.assertFalse(triage["human_adjudicated"])
        self.assertEqual(triage["literature_novelty"], "insufficient_evidence")

    def test_primitive_triage_rejects_unknown_evidence_and_self_admission(self):
        base = {"representation_gap": "clear", "literature_novelty": "incremental",
                "empirical_utility": "untested", "atom_action": "auto_admit", "evidence_source_ids": []}
        with self.assertRaises(WorldEvaluationError):
            validate_primitive_triage(base)
        base["atom_action"] = "propose_for_review"
        base["evidence_source_ids"] = ["missing"]
        with self.assertRaises(WorldEvaluationError):
            validate_primitive_triage(base, evidence_ids=["p1"])

    def test_summary_reports_exact_duplicates_only(self):
        result = summarize([record("1", "direct"), record("2", "direct"), record("3", "world", "B")])
        self.assertEqual(result["arms"]["direct"]["exact_duplicate_cases"], 1)
        self.assertEqual(result["arms"]["world"]["exact_duplicate_cases"], 0)
        self.assertIn("not semantic novelty", result["interpretation"])
        self.assertEqual(result["pooled"]["exact_duplicate_cases"], 2)
        self.assertEqual(result["cross_arm_exact_overlap"]["direct__world"]["shared_unique_mechanisms"], 1)

    def test_summary_preserves_invalid_record_status(self):
        bad = {"id": "x", "arm": "world", "problem": "p", "idea": idea()}
        del bad["idea"]["metric"]
        result = summarize([bad])
        self.assertEqual(result["valid_records"], 0)
        self.assertEqual(len(result["missing_or_invalid"]), 1)


if __name__ == "__main__":
    unittest.main()
