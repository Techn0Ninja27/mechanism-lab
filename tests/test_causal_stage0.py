import tempfile
import unittest
from pathlib import Path

from scripts import run_causal_stage0 as s


class Stage0Tests(unittest.TestCase):
    def test_plan_freezes_exact_seven_controls_without_training(self):
        with tempfile.TemporaryDirectory() as t:
            result = s.plan(Path(t) / "plan")
            self.assertEqual(result["controls"], ["__hold__", "gain_m05", "gain_m025", "gain_p01", "mean_sync", "aa_control", "g1_exact_detached"])
            self.assertFalse(result["training_opened"])
            self.assertFalse(result["final_execution_enabled"])
            self.assertEqual(result["seeds"], list(range(100, 116)))
            self.assertEqual(result["window"], list(range(88, 120)))

    def test_plan_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / "plan"; s.plan(path)
            with self.assertRaises(FileExistsError): s.plan(path)

    def test_controls_have_hashes_except_reference(self):
        controls = s._controls()
        self.assertEqual(len(controls), 7)
        self.assertTrue(all(isinstance(row.get("program_sha256"), str) for row in controls[:-1]))
        self.assertIsNone(controls[-1].get("program_sha256"))

    def fixture(self):
        rows=[]
        for c in s._controls():
            for seed in s.SEEDS:
                base=.5+(seed-100)*.01
                value=base+.02 if c['id']=='g1_exact_detached' else base
                rows.append({'id':c['id'],'call_id':c['call_id'],'arm':c['arm'],'seed':seed,'status':'complete',
                             'window':[{'t':t,'exact_eval':value,'own_eval':value} for t in s.WINDOW],
                             'final_train_loss':.2,'post_update_exact_eval':value})
        return rows

    def test_paired_sd_not_standalone_sd(self):
        result=s.analyse(self.fixture(),s._controls())
        self.assertEqual(result['status'],'reference_ready')
        self.assertLess(result['reference']['g1_paired_sd'],1e-12)
        self.assertEqual(result['reference']['proposed_final_seed_count'],32)

    def test_control_failure_descriptive_reference_failure_stops(self):
        rows=self.fixture();next(r for r in rows if r['id']=='gain_m05')['status']='failed'
        self.assertEqual(s.analyse(rows,s._controls())['status'],'reference_ready')
        next(r for r in rows if r['id']=='g1_exact_detached')['status']='failed'
        self.assertEqual(s.analyse(rows,s._controls())['status'],'stop_before_paid')

    def test_duplicate_keys_and_aa_window_mismatch(self):
        rows=self.fixture()
        with self.assertRaises(ValueError):s.analyse(rows+[rows[0]],s._controls())
        aa=next(r for r in rows if r['id']=='aa_control')
        aa['window'][0]['exact_eval']+=.01;aa['window'][1]['exact_eval']-=.01
        self.assertEqual(s.analyse(rows,s._controls())['status'],'stop_before_paid')


if __name__ == "__main__": unittest.main()
