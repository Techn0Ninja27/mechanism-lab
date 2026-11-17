import unittest
from simulator import JobSpec, PolicySpec, simulate, ORDERINGS

class PreflightTests(unittest.TestCase):
    def test_equal_prompts_order_has_no_difference(self):
        s=[JobSpec(1,0,2,1,2),JobSpec(2,0,2,1,2)]
        r=[simulate(s,prefill_capacity=4,decode_capacity=8,policy=p) for p in ORDERINGS]
        self.assertEqual(len({tuple(x['completion_times'].items()) for x in r}),1)
    def test_capacity_invariants_and_restart(self):
        s=[JobSpec(1,0,1,1,3),JobSpec(2,0,1,1,3)]
        r=simulate(s,prefill_capacity=4,decode_capacity=4,policy='fifo')
        self.assertGreaterEqual(r['restarts'],1)
        self.assertTrue(all(x['prefill_usage']<=4 and x['decode_peak_before_eos']<=4 for x in r['trace']))
    def test_admission_accounts_for_existing_next_token(self):
        s=[JobSpec(1,0,2,1,3),JobSpec(2,1,1,1,2)]
        r=simulate(s,prefill_capacity=3,decode_capacity=5,policy='fifo')
        self.assertNotIn('2:decode_admit',r['jobs']['2']['events'])
        self.assertTrue(all(x['decode_peak_before_eos']<=5 for x in r['trace']))
    def test_known_two_job_counterexample(self):
        s=[JobSpec(1,0,8,1,2),JobSpec(2,0,2,1,1)]
        large=simulate(s,prefill_capacity=10,decode_capacity=10,policy='largest_prompt')
        small=simulate(s,prefill_capacity=10,decode_capacity=10,policy='smallest_reservation')
        self.assertEqual(large['sum_flow_time'],7)
        self.assertEqual(small['sum_flow_time'],6)
    def test_policy_does_not_receive_hidden_length(self):
        self.assertNotIn('true_output',PolicySpec.__dataclass_fields__)
        traces=[]
        for length in (3,5):
            s=[JobSpec(1,0,1,1,length),JobSpec(2,0,1,1,5)]
            traces.append(simulate(s,prefill_capacity=3,decode_capacity=10,policy='largest_prompt')['trace'][:3])
        self.assertEqual(*traces)
    def test_infeasible_fixture_is_rejected_by_oracle(self):
        with self.assertRaisesRegex(ValueError,'infeasible'):
            simulate([JobSpec(1,0,4,1,4)],prefill_capacity=4,decode_capacity=5,policy='fifo')

if __name__=='__main__':unittest.main()
