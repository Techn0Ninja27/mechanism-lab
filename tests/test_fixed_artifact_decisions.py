import unittest
from generation.fixed_artifact_decisions import classify_fixed_artifact

SLOTS = {"task-only": 6, "raw": 6, "atoms": 6, "distill": 6, "random": 6, "linear-control": 1}

def rows(arm, n=6, positive=False, hashes=None, decoder="opus"):
    return [{"arm": arm, "call_id": f"{arm}-{i}", "program_sha256": (hashes[i] if hashes else f"{arm}{i}"),
             "better_than_control": positive, "failed_final_seeds": [], "low_precision_control": False, "decoder": decoder}
            for i in range(n)]

class FixedArtifactDecisionTests(unittest.TestCase):
    def full(self): return sum((rows(a, n) for a, n in SLOTS.items()), [])
    def test_atom_lead(self):
        r=self.full(); [r.remove(x) for x in list(r) if x["arm"]=="atoms"]
        r += rows("atoms", 6, True, ["a","b","c","a","b","c"])
        self.assertEqual(classify_fixed_artifact(r, expected_slots=SLOTS)["decision"], "atom_artifact_lead")
    def test_source_and_against_atom(self):
        r=self.full()
        for x in r: x["better_than_control"] = x["arm"] in {"raw","atoms","distill"}
        self.assertEqual(classify_fixed_artifact(r, expected_slots=SLOTS)["decision"], "source_artifact_lead")
        for x in r: x["better_than_control"] = x["arm"] == "raw"
        self.assertEqual(classify_fixed_artifact(r, expected_slots=SLOTS)["decision"], "against_this_atom_artifact")
    def test_boundaries_task_and_mixed(self):
        for n, expected in ((2, "attribution_blocked"), (3, "task_or_dsl_lead")):
            r=self.full()
            for x in r: x["better_than_control"] = x["arm"] == "task-only" and int(x["call_id"].rsplit('-',1)[1]) < n
            self.assertEqual(classify_fixed_artifact(r, expected_slots=SLOTS)["decision"], expected)
        r=self.full()
        for x in r: x["better_than_control"] = x["arm"] in {"raw", "atoms"}
        self.assertEqual(classify_fixed_artifact(r, expected_slots=SLOTS)["decision"], "mixed_source_result")
    def test_control_duplicate_missing_and_failure(self):
        r=self.full(); next(x for x in r if x["arm"] == "random")["better_than_control"] = True; self.assertEqual(classify_fixed_artifact(r, expected_slots=SLOTS)["decision"], "no_representation_attribution")
        r=self.full()[:-1]; self.assertEqual(classify_fixed_artifact(r, expected_slots=SLOTS)["status"], "incomplete")
        r=self.full(); r[0]["failed_final_seeds"]=[200]; self.assertEqual(classify_fixed_artifact(r, expected_slots=SLOTS)["status"], "incomplete")
    def test_decoder_results_never_pool(self):
        r=self.full(); r[0]["better_than_control"] = True; r[0]["decoder"]="fable"
        out=classify_fixed_artifact(r, expected_slots=SLOTS); self.assertEqual(out["decision"], "per_decoder"); self.assertEqual(set(out["per_decoder"]), {"opus","fable"})
    def test_duplicate_ast_not_distinct(self):
        r=self.full()
        for x in r:
            if x["arm"]=="atoms": x.update(better_than_control=True, program_sha256="same")
        self.assertNotEqual(classify_fixed_artifact(r, expected_slots=SLOTS)["decision"], "atom_artifact_lead")


class DecisionIntegrityTests(unittest.TestCase):
    def full(self): return sum((rows(a, n) for a, n in SLOTS.items()), [])
    def test_single_decoder_json_serializes(self):
        import json
        json.dumps(classify_fixed_artifact(self.full()))

    def test_expected_missing_decoder_is_incomplete(self):
        result = classify_fixed_artifact(self.full(), expected_decoders=("opus", "fable"))
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["per_decoder"]["fable"]["decision"], "incomplete")

    def test_extra_call_cannot_close(self):
        r = self.full()
        r.append({**r[0], "call_id": "extra"})
        self.assertEqual(classify_fixed_artifact(r)["status"], "incomplete")

    def test_numeric_truth_is_not_a_positive_verdict(self):
        r = self.full(); r[0]["better_than_control"] = 1
        self.assertEqual(classify_fixed_artifact(r)["status"], "incomplete")

    def test_missing_program_identity_is_incomplete(self):
        r = self.full(); r[0].pop("program_sha256")
        self.assertEqual(classify_fixed_artifact(r)["status"], "incomplete")

    def test_decoder_disagreement_not_pooled(self):
        r = self.full()
        other = [{**x, "decoder": "fable", "better_than_control": x["arm"] == "atoms"} for x in r]
        result = classify_fixed_artifact(r + other, expected_decoders=("opus", "fable"))
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["cross_decoder_interpretation"], "decoder_dependent_no_representation_reading")

if __name__ == "__main__": unittest.main()
