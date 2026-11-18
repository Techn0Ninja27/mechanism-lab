import unittest
import json
import tempfile
from pathlib import Path
from generation.campaign_selection import canonical_hash
PROGRAM={"predict":{"var":"last_sync"},"update":{"var":"memory"}}
FIDELITY={"atoms":{"fields":[],"summary":{"valid_reviewer_slots":23,"total_reviewer_slots":24,"flagged_by_both":0},"representation_sha256":"fixture"}}

from generation.fixed_artifact_analysis import split_inventory, summarize_fixed_artifact


def model_rows():
    out = []
    for decoder in ("opus-5.5", "fable-5.2"):
        for arm in ("task-only", "raw", "atoms", "distill"):
            for call in range(6):
                for slot in range(4):
                    call_id = f"{decoder}-{arm}-{call}"
                    out.append({"id": f"{call_id}-cand-{slot:02d}", "decoder": decoder,
                                "arm": arm, "call_id": call_id, "slot": slot,
                                "status": "valid", "program": PROGRAM, "program_sha256": canonical_hash(PROGRAM)})
    return out


def cases():
    return [{"id": f"{decoder}-{arm}-{call}", "decoder": decoder, "arm": arm}
            for decoder in ("opus-5.5", "fable-5.2")
            for arm in ("task-only", "raw", "atoms", "distill") for call in range(6)]


def controls():
    return ([{"id": f"random-{i}", "arm": "random", "call_id": f"random-{i//4}",
               "program":PROGRAM, "program_sha256":canonical_hash(PROGRAM)} for i in range(24)] +
            [{"id": f"linear-{i}", "arm": "linear-control", "call_id": "linear-0",
              "program":PROGRAM, "program_sha256":canonical_hash(PROGRAM)} for i in range(4)])


def report_rows(decoder, *, positive_arm=None, positives=0):
    rows = []
    for arm, n in (("task-only", 6), ("raw", 6), ("atoms", 6), ("distill", 6), ("random", 6), ("linear-control", 1)):
        for i in range(n):
            rows.append({"decoder": decoder, "arm": arm, "call_id": f"{arm}-{i}",
                         "program_sha256": f"{arm}-{i}", "failed_final_seeds": [],
                         "low_precision_control": False, "better_than_control": bool(positive_arm == arm and i < positives)})
    return rows


class FixedArtifactAnalysisTests(unittest.TestCase):
    def test_split_has_96_models_and_28_shared_controls_per_decoder(self):
        result = split_inventory(model_rows(), controls(), expected_cases=cases())
        self.assertEqual(set(result), {"opus-5.5", "fable-5.2"})
        for inv in result.values():
            self.assertEqual(len(inv["candidates"]), 124)
            self.assertEqual(sum(x["arm"] == "random" for x in inv["candidates"]), 24)
            self.assertEqual(sum(x["arm"] == "linear-control" for x in inv["candidates"]), 4)
            self.assertTrue(all(x.get("decoder") is None for x in inv["candidates"] if x["arm"] in {"random", "linear-control"}))

    def test_missing_model_slot_is_rejected(self):
        with self.assertRaises(ValueError):
            split_inventory(model_rows()[:-1], controls(), expected_cases=cases())

    def test_one_condition_missing_is_incomplete(self):
        rows = report_rows("opus-5.5")[:-1]
        out = summarize_fixed_artifact({"opus-5.5": rows}, expected_decoders=("opus-5.5",),fidelity=FIDELITY)
        self.assertEqual(out["status"], "incomplete")

    def test_duplicate_ast_is_counted_once(self):
        rows = report_rows("opus-5.5", positive_arm="atoms", positives=6)
        for row in rows:
            if row["arm"] == "atoms":
                row["program_sha256"] = "same-ast"
        out = summarize_fixed_artifact({"opus-5.5": rows}, expected_decoders=("opus-5.5",),fidelity=FIDELITY)
        self.assertEqual(out["distinct_positive_programs_by_arm"]["atoms"], 1)

    def test_atoms_positive_has_fidelity_flags(self):
        rows = report_rows("opus-5.5", positive_arm="atoms", positives=1)
        out = summarize_fixed_artifact({"opus-5.5": rows}, expected_decoders=("opus-5.5",),
                                       fidelity={"atoms": {"fields": [{"field": "statement", "flagged_by_either": True}, {"field": "outputs[0]", "flagged_by_either": True}], "summary": {"valid_reviewer_slots": 2, "total_reviewer_slots": 2, "flagged_by_both": 1}, "representation_sha256": "x"}})
        atom = next(x for x in out["reports"]["opus-5.5"] if x["arm"] == "atoms" and x["better_than_control"])
        self.assertEqual([x["field"] for x in atom["fidelity_flags"]], ["statement", "outputs[0]"])

    def test_contrary_results_stay_per_decoder_and_shared_controls_are_not_third_group(self):
        rows = report_rows("opus-5.5", positive_arm="atoms", positives=3) + report_rows("fable-5.2", positive_arm="raw", positives=3)
        out = summarize_fixed_artifact(rows, expected_decoders=("opus-5.5", "fable-5.2"),fidelity=FIDELITY)
        self.assertEqual(set(out["per_decoder"]), {"opus-5.5", "fable-5.2"})
        self.assertEqual(out["per_decoder"]["opus-5.5"]["decision"], "atom_artifact_lead")
        self.assertEqual(out["per_decoder"]["fable-5.2"]["decision"], "against_this_atom_artifact")
        self.assertEqual(out["missing_decoders"], [])

    def test_split_loads_in_existing_evaluator(self):
        from scripts.run_causal_campaign import load_inventory
        result=split_inventory(model_rows(),controls(),expected_cases=cases())
        with tempfile.TemporaryDirectory() as td:
            for decoder,inventory in result.items():
                path=Path(td)/(decoder+".json");path.write_text(json.dumps(inventory))
                loaded,_=load_inventory(path)
                self.assertEqual(len(loaded["candidates"]),124)

    def test_positive_prepared_pick_cannot_omit_fidelity(self):
        with self.assertRaises(ValueError):
            summarize_fixed_artifact({"opus-5.5":{"reports":report_rows("opus-5.5",positive_arm="atoms",positives=1)}})

    def test_wrong_arm_schedule_and_multiple_linear_calls_rejected(self):
        rows=model_rows();rows[0]["arm"]="raw"
        with self.assertRaises(ValueError):split_inventory(rows,controls(),expected_cases=cases())
        control=controls();control[-1]["call_id"]="another"
        with self.assertRaises(ValueError):split_inventory(model_rows(),control,expected_cases=cases())


if __name__ == "__main__":
    unittest.main()
