import unittest
from simulator import JobSpec,simulate
from run import make_cases,run_case,quantile_nearest

class ScreenTests(unittest.TestCase):
    def test_backpressure_changes_only_prefill_admission(self):
        jobs=[JobSpec(i,0,3,1,3) for i in range(3)]
        a=simulate(jobs,prefill_capacity=6,decode_capacity=6,policy='fifo')
        b=simulate(jobs,prefill_capacity=6,decode_capacity=6,policy='fifo',backpressure_fraction=.5)
        self.assertEqual(a['trace'][1]['states']['2'],'prefill')
        self.assertEqual(b['trace'][1]['states']['2'],'unassigned')
        self.assertEqual(a['trace'][1]['states']['0'],b['trace'][1]['states']['0'])
    def test_deterministic_matched_cases(self):
        a=make_cases();self.assertEqual(a,make_cases());self.assertEqual(len(a),180)
        self.assertEqual(a[0]['jobs'],a[1]['jobs']);self.assertNotEqual(a[0]['prefill_capacity'],a[1]['prefill_capacity'])
        self.assertTrue(all(j['prompt']+j['true_output']<=c['decode_capacity'] for c in a for j in c['jobs']))
    def test_timeout_preserved_not_dropped(self):
        c=make_cases()[0];c['horizon']=1
        self.assertEqual(run_case(c,'largest_prompt')['status'],'incomplete_at_frozen_horizon')
    def test_tails_are_request_order_statistics(self):
        self.assertEqual(quantile_nearest(list(range(1,25)),.99),24)
        self.assertEqual(quantile_nearest(list(range(1,25)),.95),23)

if __name__=='__main__':unittest.main()
