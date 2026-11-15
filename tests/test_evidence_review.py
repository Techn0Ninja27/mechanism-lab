import unittest

from generation.evidence_review import ReviewValidationError, build_review_prompt, validate_review


EVIDENCE = [{
    "id": "p1",
    "title": "A paper",
    "url": "https://example.org/p1",
    "abstract": "We introduce a calibrated test for model predictions under distribution shift.",
}]


def review(**overrides):
    value = {
        "decision": "plausibly_distinct",
        "claimed_delta": "The proposed interface is not established by this record.",
        "matches": [{"source_id": "p1", "overlap": "calibration", "evidence_quote": "calibrated test for model predictions"}],
        "distinct_mechanism": "A held-out intervention would test the proposed interface.",
        "assumptions": ["The evaluation oracle is independent."],
        "cheapest_test": "Compare against a retrieval baseline on held-out tasks.",
        "confidence": "low",
        "missing_searches": ["Search the exact conjunction."],
        "human_adjudicated": False,
        "train_eligible": False,
    }
    value.update(overrides)
    return value


class EvidenceReviewTests(unittest.TestCase):
    def test_prompt_contains_ids_and_restrictions(self):
        prompt = build_review_prompt("A proposal", EVIDENCE)
        self.assertIn("SOURCE_ID: p1", prompt)
        self.assertIn("covered", prompt)
        self.assertIn("train_eligible", prompt)

    def test_valid_review_normalizes(self):
        out = validate_review(review(), EVIDENCE)
        self.assertEqual(out["decision"], "plausibly_distinct")
        self.assertFalse(out["train_eligible"])

    def test_quote_must_be_grounded(self):
        with self.assertRaises(ReviewValidationError):
            validate_review(review(matches=[{"source_id": "p1", "overlap": "x", "evidence_quote": "not in abstract"}]), EVIDENCE)

    def test_unknown_source_rejected(self):
        with self.assertRaises(ReviewValidationError):
            validate_review(review(matches=[{"source_id": "missing", "overlap": "x", "evidence_quote": "calibrated"}]), EVIDENCE)

    def test_covered_requires_evidence(self):
        with self.assertRaises(ReviewValidationError):
            validate_review(review(decision="covered", matches=[]), EVIDENCE)

    def test_model_cannot_self_adjudicate(self):
        with self.assertRaises(ReviewValidationError):
            validate_review(review(train_eligible=True,human_adjudicated=True), EVIDENCE)

    def test_training_requires_human_adjudication(self):
        with self.assertRaises(ReviewValidationError):
            validate_review(review(train_eligible=True), EVIDENCE)


if __name__ == "__main__":
    unittest.main()
