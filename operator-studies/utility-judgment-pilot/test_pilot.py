import unittest
import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('utility_judgment_runner',Path(__file__).with_name('run.py'))
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
validate,score=runner.validate,runner.score

class PilotTests(unittest.TestCase):
    def test_unknowns_and_invalid_confidence(self):
        row={'winner':'abstain','p_A_better':None,'reason':'No observations','missing_evidence':['outcomes']}
        self.assertEqual(validate(row),row)
        for p in [True,float('nan'),1.1]:
            with self.assertRaises(ValueError):validate(row|{'p_A_better':p})
        with self.assertRaises(ValueError):validate(row|{'winner':'A','p_A_better':.1})
    def test_order_mapping_and_abstention(self):
        a={'case':'x','order':0,'mapping':{'A':'good','B':'bad'},'status':'complete','assessment':{'winner':'A','p_A_better':.8}}
        b={'case':'x','order':1,'mapping':{'A':'bad','B':'good'},'status':'complete','assessment':{'winner':'B','p_A_better':.2}}
        labels={'labels':{'x':{'winner':'good'}}};r=score([a,b],labels)
        self.assertTrue(r['order_checks'][0]['same_preference']);self.assertTrue(all(x['correct'] for x in r['rows']))
        self.assertAlmostEqual(r['rows'][0]['brier'],.04)
        b['assessment']={'winner':'abstain','p_A_better':None};r=score([a,b],labels)
        self.assertIsNone(r['rows'][1]['correct']);self.assertIsNone(r['rows'][1]['brier'])

if __name__=='__main__':unittest.main()
