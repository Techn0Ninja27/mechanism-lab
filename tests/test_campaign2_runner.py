import unittest

from scripts.run_causal_campaign import _apply_campaign2_decision


class Campaign2RunnerTests(unittest.TestCase):
    def test_campaign2_report_uses_new_decision_and_preserves_legacy(self):
        arms = ("task-only", "raw", "atoms", "distill", "random")
        candidates = []
        selected = []
        reports = []
        for arm in arms:
            for i in range(6):
                call_id = f"{arm}-{i}"
                candidates.append({"call_id": call_id, "arm": arm})
                selected.append({"call_id": call_id, "candidate_id": call_id, "status": "selected"})
                positive = arm == "atoms" and i < 3
                reports.append({"call_id": call_id, "arm": arm,
                                "program_sha256": f"{arm}-program-{i if positive else 0}",
                                "better_than_control": positive,
                                "failed_final_seeds": [], "low_precision_control": False})
        candidates.append({"call_id": "linear-control-00", "arm": "linear-control"})
        selected.append({"call_id": "linear-control-00", "candidate_id": "linear-control-03", "status": "selected"})
        reports.append({"call_id": "linear-control-00", "arm": "linear-control",
                        "program_sha256": "control", "better_than_control": False,
                        "failed_final_seeds": [], "low_precision_control": False})
        result = _apply_campaign2_decision(
            {"reports": reports, "decision": {"legacy": True}},
            {"campaign": "generation-comparison-campaign2", "candidates": candidates},
            {"manifest": {"selected": selected}},
        )
        self.assertEqual(result["legacy_decision"], {"legacy": True})
        self.assertEqual(result["decision"]["decision"], "replication_candidate")
        self.assertEqual(result["report_decision"], result["decision"])

    def test_no_eligible_selection_is_added_with_inventory_arm(self):
        result = _apply_campaign2_decision(
            {"reports": [], "decision": "legacy"},
            {"campaign": "generation-comparison-campaign2",
             "candidates": [{"call_id": "atoms-0", "arm": "atoms"}]},
            {"manifest": {"selected": [{"call_id": "atoms-0", "status": "no_eligible_candidate"}]}},
        )
        self.assertEqual(result["reports"][0]["arm"], "atoms")
        self.assertEqual(result["decision"]["status"], "incomplete")


if __name__ == "__main__":
    unittest.main()
