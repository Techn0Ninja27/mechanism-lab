import copy
import unittest

from generation.proposal_portfolio import (
    PortfolioError,
    annotate_candidates,
    compare_structures,
    validate_portfolio_review,
)


def proposal(delta="route by prompt", state="state"):
    return {
        "title": "metadata may differ",
        "baseline": " FIFO   scheduler ",
        "operator_delta": delta,
        "data_paths": [{"input": "request", "operation": state, "output": "queue"}],
        "mechanism_check": {"status": "nontrivial"},
    }


class PortfolioTests(unittest.TestCase):
    def test_local_ids_and_whitespace_exact_duplicate(self):
        a = proposal()
        b = copy.deepcopy(a)
        b["title"] = "different title"
        b["baseline"] = "FIFO scheduler"
        rows = annotate_candidates([a, b, proposal("route by token")])
        self.assertEqual([r["id"] for r in rows], ["cand-00", "cand-01", "cand-02"])
        self.assertEqual(rows[1]["exact_duplicate_of"], "cand-00")
        self.assertFalse(rows[1]["auto_eligible"])
        self.assertEqual(compare_structures(rows)["exact_duplicate_groups"], [["cand-00", "cand-01"]])

    def test_state_order_is_causal_and_not_collapsed(self):
        rows = annotate_candidates([proposal(state="state A"), proposal(state="state B"), proposal("different")])
        self.assertIsNone(rows[1]["exact_duplicate_of"])

    def test_case_sensitive_math_is_not_an_exact_duplicate(self):
        rows = annotate_candidates([proposal(state="multiply A"), proposal(state="multiply a"), proposal("c")])
        self.assertIsNone(rows[1]["exact_duplicate_of"])
        self.assertTrue(all(not r["auto_eligible"] for r in rows))

    def test_unknown_semantic_status_not_auto_eligible(self):
        p = proposal(); p["mechanism_check"] = {"status": "unknown"}
        self.assertFalse(annotate_candidates([p, proposal("b"), proposal("c")])[0]["auto_eligible"])

    def test_review_requires_each_candidate_and_pair(self):
        rows = annotate_candidates([proposal("a"), proposal("b"), proposal("c")])
        reviews = [{"candidate_id": r["id"], "review": {"id": r["id"]}} for r in rows]
        pairs = []
        for i, left in enumerate(rows):
            for right in rows[i + 1:]:
                pairs.append({"candidate_ids": [left["id"], right["id"]], "relationship": "uncertain", "reason": "needs test"})
        out = validate_portfolio_review({"reviews": reviews, "comparisons": pairs, "selected_candidate_id": "cand-00", "selection_reason": "manual review"}, rows)
        self.assertFalse(out["novelty_established"])
        with self.assertRaises(PortfolioError):
            validate_portfolio_review({"reviews": reviews, "comparisons": pairs[:2], "selected_candidate_id": None, "selection_reason": "x"}, rows)

    def test_duplicate_pair_and_duplicate_selection_rejected(self):
        rows = annotate_candidates([proposal(), proposal(), proposal("c")])
        reviews = [{"candidate_id": r["id"], "review": {}} for r in rows]
        pairs = [{"candidate_ids": ["cand-00", "cand-01"], "relationship": "uncertain", "reason": "x"}] * 3
        with self.assertRaises(PortfolioError):
            validate_portfolio_review({"reviews": reviews, "comparisons": pairs, "selected_candidate_id": "cand-01", "selection_reason": "x"}, rows)


if __name__ == "__main__":
    unittest.main()
