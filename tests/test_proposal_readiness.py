import copy
import unittest

from generation.proposal_readiness import evaluate_result


def proposal(**updates):
    p = {
        "title": "A bounded proposal", "hypothesis": "The change improves a metric.",
        "baseline": "Strong baseline", "operator_delta": "One operator change",
        "counterexample": "Small case where it fails", "matched_control": "Remove the operator",
        "prior_art_difference": "Scoped difference remains uncertain",
        "evidence": [{"chunk_id": "s1", "quote": "source span", "supports": "proposal premise"}],
        "mechanism_check": {"status": "nontrivial", "assumption_scope": "source regime",
            "specialized_operation": "apply operation", "remaining_effect": "effect remains",
            "simplest_baseline": "baseline", "distinguishing_test": "compare"},
        "contribution_claims": [{"id": "c1", "text": "The method claim", "role": "central", "kind": "method"}],
        "assumptions": ["assumption"],
        "data_paths": [{"input": "x", "operation": "op", "output": "y", "possible_leakage": "none"}],
        "experiment": {"development": "dev", "heldout": "heldout", "metric": "loss", "kill_threshold": "5%"},
        "decision": "investigate", "novelty_established": False,
    }
    p.update(updates)
    return p


def review(**updates):
    r = {"claim_assessments": [{"claim_id": "c1", "coverage": "unknown", "checked_chunk_ids": [], "evidence": [], "reason": "uncertain"}],
         "mechanism_check": {"status": "nontrivial", "assumption_scope": "source regime",
            "specialized_operation": "apply operation", "remaining_effect": "effect remains",
            "simplest_baseline": "baseline", "distinguishing_test": "compare"},
         "decision": "test", "failure_type": "none_found", "reason": "testable",
         "evidence": [{"chunk_id": "s1", "quote": "source span", "supports": "review premise"}],
         "strongest_counterargument": "could fail", "required_test": "run control", "novelty_established": False}
    r.update(updates)
    return r


class ReadinessTests(unittest.TestCase):
    def test_clean_test_design_candidate_requires_external_retrieval(self):
        r = evaluate_result({"status": "reviewed", "proposal": proposal(), "review": review(), "retrieved_source_count": 2})
        self.assertEqual(r["status"], "test_design_candidate")
        self.assertFalse(r["auto_execute"])

        r = evaluate_result({"status": "reviewed", "proposal": proposal(), "review": review(), "retrieved_source_count": 0})
        self.assertEqual(r["status"], "source_only_diagnostic")

        r = evaluate_result({"status": "reviewed", "portfolio": None, "proposal": proposal(), "review": review(), "retrieved_source_count": 1})
        self.assertEqual(r["status"], "test_design_candidate")

    def test_weak_control_and_unresolved_mechanism_need_revision(self):
        r = evaluate_result({"status": "reviewed", "proposal": proposal(),
            "review": review(decision="test", failure_type="weak_control",
                mechanism_check={**review()["mechanism_check"], "status": "unresolved"}),
            "retrieved_source_count": 4})
        self.assertEqual(r["status"], "needs_revision")
        self.assertTrue({x["code"] for x in r["reasons"]} & {"review_failure_type", "mechanism_unresolved"})

    def test_partial_overlap_is_a_limitation_not_an_automatic_veto(self):
        r = evaluate_result({"status": "reviewed", "proposal": proposal(),
            "review": review(claim_assessments=[{"claim_id": "c1", "coverage": "partial", "checked_chunk_ids": ["s1"],
                "evidence": [{"chunk_id": "s1", "quote": "old", "supports": "overlap"}], "reason": "partial"}]),
            "retrieved_source_count": 1})
        self.assertEqual(r["status"], "test_design_candidate")
        self.assertTrue(any("partial" in x for x in r["limitations"]))
        self.assertFalse(r["novelty_established"])

    def test_not_found_is_not_novelty_and_reject_is_not_recommended(self):
        r = evaluate_result({"status": "reviewed", "proposal": proposal(),
            "review": review(decision="reject", failure_type="prior_art_overlap"), "retrieved_source_count": 2})
        self.assertEqual(r["status"], "not_recommended")

        r = evaluate_result({"status": "reviewed", "proposal": proposal(),
            "review": review(claim_assessments=[{"claim_id": "c1", "coverage": "not_found", "checked_chunk_ids": ["s1"],
                "evidence": [], "reason": "not found"}]), "retrieved_source_count": 2})
        self.assertEqual(r["status"], "test_design_candidate")
        self.assertFalse(r["novelty_established"])

    def test_malformed_and_unreviewed_are_incomplete(self):
        self.assertEqual(evaluate_result({"status": "failed"})["status"], "incomplete")
        bad = proposal(); del bad["matched_control"]
        r = evaluate_result({"status": "reviewed", "proposal": bad, "review": review(), "retrieved_source_count": 2})
        self.assertEqual(r["status"], "incomplete")

    def test_portfolio_null_selection_keeps_each_gate(self):
        candidates = [{"id": f"cand-{i:02d}", "proposal": proposal()} for i in range(3)]
        reviews = [{"candidate_id": f"cand-{i:02d}", "review": review()} for i in range(3)]
        result = {"status": "reviewed", "retrieved_source_count": 1,
            "portfolio": {"candidates": candidates, "review": {"reviews": reviews, "selected_candidate_id": None}}}
        r = evaluate_result(result)
        self.assertEqual(r["status"], "not_selected")
        self.assertEqual(set(r["candidate_gates"]), {"cand-00", "cand-01", "cand-02"})

    def test_portfolio_unknown_selection_is_incomplete(self):
        candidates = [{"id": f"cand-{i:02d}", "proposal": proposal()} for i in range(3)]
        reviews = [{"candidate_id": f"cand-{i:02d}", "review": review()} for i in range(3)]
        r = evaluate_result({"status": "reviewed", "retrieved_source_count": 1,
            "portfolio": {"candidates": candidates, "review": {"reviews": reviews, "selected_candidate_id": "cand-99"}}})
        self.assertEqual(r["status"], "incomplete")

    def test_malformed_types_never_raise(self):
        cases = [
            {"status": ["reviewed"]},
            {"status": "reviewed", "proposal": proposal(decision=[]), "review": review(), "retrieved_source_count": 1},
            {"status": "reviewed", "proposal": proposal(), "review": review(decision=[]), "retrieved_source_count": 1},
            {"status": "reviewed", "proposal": proposal(), "review": review(failure_type={}), "retrieved_source_count": 1},
            {"status": "reviewed", "proposal": proposal(), "review": review(claim_assessments=[{"claim_id": [], "coverage": "unknown", "checked_chunk_ids": [], "evidence": [], "reason": "x"}]), "retrieved_source_count": 1},
        ]
        for value in cases:
            with self.subTest(value=value):
                self.assertEqual(evaluate_result(value)["status"], "incomplete")

    def test_malformed_portfolio_never_raises(self):
        cases = [
            {"status": "reviewed", "portfolio": {"candidates": {}, "review": {}}},
            {"status": "reviewed", "portfolio": {"candidates": [{"id": []}] * 3, "review": {}}},
            {"status": "reviewed", "portfolio": {"candidates": [{"id": f"cand-{i:02d}", "proposal": proposal()} for i in range(3)], "review": {"reviews": [{"candidate_id": "cand-00"}] * 3}}},
        ]
        for value in cases:
            with self.subTest(value=value):
                self.assertEqual(evaluate_result(value)["status"], "incomplete")

    def test_model_conflict_is_preserved_as_revision_signal(self):
        r = evaluate_result({"status": "reviewed", "proposal": proposal(),
            "review": review(decision="test", failure_type="weak_control", reason="model says test but control is weak"),
            "retrieved_source_count": 1})
        self.assertEqual(r["status"], "needs_revision")
        self.assertTrue(any("weak_control" in x["detail"] for x in r["reasons"]))


if __name__ == "__main__":
    unittest.main()
