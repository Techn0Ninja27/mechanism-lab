import unittest
from generation.campaign2_decisions import classify_campaign

SLOTS={"task-only":6,"raw":6,"atoms":6,"distill":6,"random":6,"linear-control":1}

def rows(arm, n, *, positive=False, prefix=None, hashes=None):
    prefix=prefix or arm
    out=[]
    for i in range(n):
        out.append({"arm":arm,"call_id":f"{prefix}-{i}","program_sha256":(hashes[i] if hashes else f"{arm}{i}"),"better_than_control":positive,"failed_final_seeds":[],"low_precision_control":False})
    return out

class Campaign2Tests(unittest.TestCase):
    def full(self, extra=None):
        out=[]
        for arm,n in SLOTS.items(): out += rows(arm,n)
        return out + (extra or [])

    def test_three_distinct_source_programs_same_arm_is_replication_candidate(self):
        r=[x for x in self.full() if x["arm"]!="atoms"]+rows("atoms",6,positive=True,hashes=["a","b","c","a","b","c"])
        self.assertEqual(classify_campaign(r,expected_slots=SLOTS)["decision"],"replication_candidate")

    def test_one_and_two_task_only_positives(self):
        for count,decision in ((1,'replication_candidate'),(2,'attribution_blocked')):
            with self.subTest(count=count):
                r=self.full()
                for row in r:
                    index=int(row['call_id'].rsplit('-',1)[1])
                    row['better_than_control']=(row['arm']=='atoms' and index<3) or (row['arm']=='task-only' and index<count)
                self.assertEqual(classify_campaign(r,expected_slots=SLOTS)['decision'],decision)

    def test_two_source_programs_no_escalation(self):
        r=[x for x in self.full() if x["arm"]!="atoms"]+rows("atoms",6,positive=True,hashes=["a","b","a","b","a","b"])
        self.assertEqual(classify_campaign(r,expected_slots=SLOTS)["decision"],"no_escalation")

    def test_three_task_programs_are_task_lead(self):
        slots=SLOTS; r=[]
        r += rows("task-only",6,positive=True,hashes=["a","b","c","a","b","c"])
        for arm,n in slots.items():
            if arm != "task-only": r += rows(arm,n)
        self.assertEqual(classify_campaign(r,expected_slots=slots)["decision"],"task_or_dsl_lead")

    def test_control_positive_blocks_replication(self):
        r=[x for x in self.full() if x["arm"]!="random"]+rows("random",6,positive=True); self.assertEqual(classify_campaign(r,expected_slots=SLOTS)["decision"],"no_representation_attribution")

    def test_same_hash_is_not_two_distinct_programs(self):
        r=[x for x in self.full() if x["arm"]!="atoms"]+rows("atoms",6,positive=True,hashes=["same"]*6); self.assertEqual(classify_campaign(r,expected_slots=SLOTS)["decision"],"no_escalation")

    def test_incomplete_and_failure_cannot_close(self):
        r=self.full()[:-1]; out=classify_campaign(r); self.assertEqual(out["status"],"incomplete"); self.assertNotEqual(out["decision"],"close_benchmark")
        r=self.full(); r[0]["failed_final_seeds"]=[200]; out=classify_campaign(r); self.assertEqual(out["status"],"incomplete")

    def test_foreign_arm_is_rejected(self):
        out=classify_campaign(self.full()+rows("foreign", 1), expected_slots=SLOTS)
        self.assertEqual(out["status"], "incomplete")
        self.assertTrue(out["malformed_rows"])

    def test_all_complete_precise_no_positives_closes(self):
        self.assertEqual(classify_campaign(self.full(),expected_slots=SLOTS)["decision"],"close_benchmark")

if __name__ == "__main__": unittest.main()
