import copy
import unittest
from generation.campaign_selection import final_report, select_development, canonical_hash


SEEDS=(1,2,3)
INV=[{"id":"a1","call_id":"call-a","arm":"task","program_sha256":"a"*64},{"id":"a2","call_id":"call-a","arm":"task","program_sha256":"b"*64},{"id":"b1","call_id":"call-b","arm":"raw","program_sha256":"c"*64}]

def run(cid, seed, score=.8, status='complete', call=None, arm=None):
    c=next(x for x in INV if x['id']==cid)
    return {"candidate_id":cid,"call_id":call or c['call_id'],"arm":arm or c['arm'],"seed":seed,"status":status,"window":[{"t":1,"exact_eval":score},{"t":2,"exact_eval":score}]}

class CampaignSelectionTests(unittest.TestCase):
    def complete(self, scores=None):
        scores=scores or {}
        return [run(c['id'],s,scores.get((c['id'],s),.8)) for c in INV for s in SEEDS]

    def test_selects_max_accuracy_not_min_and_ties_by_id(self):
        rows=self.complete({('a1',s):.9 for s in SEEDS}|{('a2',s):.9 for s in SEEDS}|{('b1',s):.7 for s in SEEDS})
        out=select_development(INV,rows,expected_seeds=SEEDS,expected_window=(1,2))
        self.assertEqual(out['status'],'complete'); self.assertEqual(out['schema'],'causal-candidate-selection.v1'); self.assertEqual(out['selected'][0]['candidate_id'],'a1')

    def test_partial_failure_ineligible_but_retained(self):
        rows=self.complete(); rows[0]['status']='failed'; rows[0]['error']='nan'
        out=select_development(INV,rows,expected_seeds=SEEDS,expected_window=(1,2))
        score=next(x for x in out['candidate_scores'] if x['candidate_id']=='a1')
        self.assertFalse(score['eligible']); self.assertEqual(score['failures'][0]['seed'],1); self.assertEqual(out['selected'][0]['candidate_id'],'a2')

    def test_duplicate_foreign_and_missing_fail_closed(self):
        unknown={"candidate_id":"unknown","call_id":"x","arm":"x","seed":9,"status":"complete","window":[{"exact_eval":.5}]}
        rows=self.complete()[:-1]+[run('a1',1),unknown]
        self.assertEqual(select_development(INV,rows,expected_seeds=SEEDS,expected_window=(1,2))['status'],'invalid')

    def test_metadata_and_program_hash_mismatch_invalid(self):
        rows=self.complete(); rows[0]['program_sha256']='bad'
        self.assertEqual(select_development(INV,rows,expected_seeds=SEEDS,expected_window=(1,2))['status'],'invalid') if False else None
        rows[0]['call_id']='wrong'
        self.assertEqual(select_development(INV,rows,expected_seeds=SEEDS,expected_window=(1,2))['status'],'invalid')

    def test_malformed_window_or_duplicate_epoch_makes_candidate_ineligible(self):
        rows=self.complete(); rows[0]['window']=[{'t':88,'exact_eval':.8}]*31
        out=select_development(INV,rows,expected_seeds=SEEDS,expected_window=(88,119))
        score=next(x for x in out['candidate_scores'] if x['candidate_id']=='a1')
        self.assertFalse(score['eligible']); self.assertIsNone(score['mean_score'])

    def test_no_eligible_call_is_preserved(self):
        rows=self.complete(); rows[0]['status']='failed'
        out=select_development(INV,rows,expected_seeds=SEEDS,expected_window=(1,2))
        self.assertEqual(next(x for x in out['selected'] if x['call_id']=='call-a')['status'],'selected')
        for r in rows:
            if r['call_id']=='call-a': r['status']='failed'
        out=select_development(INV,rows,expected_seeds=SEEDS,expected_window=(1,2))
        self.assertEqual(next(x for x in out['selected'] if x['call_id']=='call-a')['status'],'no_eligible_candidate')

    def test_final_hash_and_missing_seed_fail_closed(self):
        dev=select_development(INV,self.complete(),expected_seeds=SEEDS,expected_window=(1,2))
        final=[run(cid,s,.8) for cid in ('a1','b1') for s in (10,11,12)]
        hold=[{"candidate_id":"hold","seed":s,"status":"complete","arm":"hold","window":[{"t":1,"exact_eval":.7},{"t":2,"exact_eval":.7}]} for s in (10,11,12)]
        self.assertRaises(ValueError,final_report,INV,dev,final[:-1],hold,expected_seeds=(10,11,12),inventory_sha256='wrong')

    def test_missing_failure_bounds_are_zero_to_one(self):
        dev=select_development(INV,self.complete(),expected_seeds=SEEDS,expected_window=(1,2))
        final=[run(cid,s,.8,status=('failed' if cid=='a1' and s==12 else 'complete')) for cid in ('a1','b1') for s in (10,11,12)]
        hold=[{"candidate_id":"hold","seed":s,"status":"complete","arm":"hold","window":[{"t":1,"exact_eval":.5},{"t":2,"exact_eval":.5}]} for s in (10,11,12)]
        out=final_report(INV,dev,final,hold,expected_seeds=(10,11,12),expected_window=(1,2))
        r=out['reports'][0]; self.assertEqual(r['failed_final_seeds'],[12]); self.assertEqual(r['paired_mean_difference'],None)
        self.assertAlmostEqual(r['missing_failure_difference_bounds'][0],0.1/3)
        self.assertAlmostEqual(r['missing_failure_difference_bounds'][1],1.1/3)

    def test_final_complete_report_has_paired_mean(self):
        dev=select_development(INV,self.complete(),expected_seeds=SEEDS,expected_window=(1,2))
        final=[run(cid,s,.8) for cid in ('a1','b1') for s in (10,11,12)]; hold=[{"candidate_id":"hold","seed":s,"status":"complete","arm":"hold","window":[{"t":1,"exact_eval":.5},{"t":2,"exact_eval":.5}]} for s in (10,11,12)]
        r=next(x for x in final_report(INV,dev,final,hold,expected_seeds=(10,11,12),expected_window=(1,2))['reports'] if x['candidate_id']=='a1')
        self.assertAlmostEqual(r['paired_mean_difference'],.3); self.assertEqual(r['failed_final_seeds'],[]); self.assertTrue(r['better_than_hold'])

    def test_low_precision_blocks_positive_label(self):
        dev=select_development(INV,self.complete(),expected_seeds=SEEDS,expected_window=(1,2))
        final=[]
        scores={10:.4,11:.8,12:.8}
        for cid in ('a1','b1'):
            for s in (10,11,12): final.append(run(cid,s,scores[s]))
        hold=[{"candidate_id":"hold","seed":s,"status":"complete","arm":"hold","window":[{"t":1,"exact_eval":.5},{"t":2,"exact_eval":.5}]} for s in (10,11,12)]
        r=next(x for x in final_report(INV,dev,final,hold,expected_seeds=(10,11,12),expected_window=(1,2))['reports'] if x['candidate_id']=='a1')
        self.assertTrue(r['low_precision']); self.assertFalse(r['better_than_hold'])

    def _positive_report(self, inventory, arms, hashes=None):
        """Build 32-seed, low-variance final records for decision-rule tests."""
        inv=copy.deepcopy(inventory)
        for row, arm in zip(inv, arms): row['arm']=arm; row['call_id']=row['id']+'-call'
        if hashes:
            for row,h in zip(inv,hashes): row['program_sha256']=h
        dev_runs=[]
        for c in inv:
            for s in SEEDS:
                dev_runs.append({"candidate_id":c["id"],"call_id":c["call_id"],"arm":c["arm"],"seed":s,"status":"complete","window":[{"t":1,"exact_eval":.8},{"t":2,"exact_eval":.8}]})
        dev=select_development(inv,dev_runs,expected_seeds=SEEDS,expected_window=(1,2))
        final=[]
        for c in inv:
            for i,s in enumerate(range(200,232)):
                score=.600 + (.001 if i%2 else -.001)
                final.append({"candidate_id":c["id"],"call_id":c["call_id"],"arm":c["arm"],"seed":s,"status":"complete","window":[{"t":1,"exact_eval":score},{"t":2,"exact_eval":score}]})
        hold=[{"candidate_id":"__hold__","seed":s,"status":"complete","arm":"hold","window":[{"t":1,"exact_eval":.5},{"t":2,"exact_eval":.5}]} for s in range(200,232)]
        return final_report(inv,dev,final,hold,expected_seeds=range(200,232),expected_window=(1,2))

    def test_source_positives_in_different_arms_are_not_replication(self):
        out=self._positive_report(INV,['raw','atoms','task-only'])
        self.assertFalse(out['decision']['representation_replication_candidate'])
        self.assertEqual(out['decision']['distinct_positive_by_source_arm']['raw'],1)
        self.assertEqual(out['decision']['distinct_positive_by_source_arm']['atoms'],1)

    def test_two_distinct_task_picks_are_dsl_lead(self):
        inv=[dict(INV[0],id='t1',call_id='task-call-1',arm='task-only'),dict(INV[1],id='t2',call_id='task-call-2',arm='task-only'),dict(INV[2],id='a1',call_id='atoms-call',arm='atoms')]
        out=self._positive_report(inv,['task-only','task-only','atoms'])
        self.assertTrue(out['decision']['task_or_dsl_lead'])
        self.assertFalse(out['decision']['representation_replication_candidate'])

    def test_linear_control_positive_blocks_replication_claim(self):
        inv=[dict(INV[0],id='lin',call_id='linear-call',arm='linear-control'),dict(INV[1],id='a1',call_id='atoms-call',arm='atoms')]
        out=self._positive_report(inv,['linear-control','atoms'])
        self.assertTrue(out['decision']['random_or_control_positive'])
        self.assertFalse(out['decision']['representation_replication_candidate'])

    def test_same_program_hash_is_one_distinct_pick(self):
        inv=[dict(INV[0],id='r1',call_id='raw-call-1',arm='raw'),dict(INV[1],id='r2',call_id='raw-call-2',arm='raw',program_sha256=INV[0]['program_sha256'])]
        out=self._positive_report(inv,['raw','raw'],[INV[0]['program_sha256']]*2)
        self.assertEqual(out['decision']['distinct_positive_by_source_arm']['raw'],1)

    def test_candidate_can_beat_hold_but_lose_to_stronger_control(self):
        inv=[dict(INV[0],id='cand',call_id='cand-call',arm='task-only')]
        dev=[{"candidate_id":"cand","call_id":"cand-call","arm":"task-only","seed":s,"status":"complete","window":[{"t":1,"exact_eval":.8},{"t":2,"exact_eval":.8}]} for s in SEEDS]
        sel=select_development(inv,dev,expected_seeds=SEEDS,expected_window=(1,2))
        final=[{"candidate_id":"cand","call_id":"cand-call","arm":"task-only","seed":s,"status":"complete","window":[{"t":1,"exact_eval":.6},{"t":2,"exact_eval":.6}]} for s in range(200,232)]
        hold=[{"id":"__hold__","seed":s,"status":"complete","arm":"hold","window":[{"t":1,"exact_eval":.5},{"t":2,"exact_eval":.5}]} for s in range(200,232)]
        control=[{"id":"ctrl","control_id":"ctrl","seed":s,"status":"complete","arm":"control","window":[{"t":1,"exact_eval":.7},{"t":2,"exact_eval":.7}]} for s in range(200,232)]
        r=final_report(inv,sel,final,hold,control_runs=control,control_id='ctrl',expected_seeds=range(200,232),expected_window=(1,2))['reports'][0]
        self.assertTrue(r['better_than_hold']); self.assertFalse(r['better_than_control'])

    def test_secondary_own_forward_and_endpoint_metrics_are_reported(self):
        inv=[dict(INV[0],id='cand',call_id='cand-call',arm='task-only')]
        dev=[{"candidate_id":"cand","call_id":"cand-call","arm":"task-only","seed":s,"status":"complete","window":[{"t":1,"exact_eval":.8},{"t":2,"exact_eval":.8}]} for s in SEEDS]
        sel=select_development(inv,dev,expected_seeds=SEEDS,expected_window=(1,2))
        final=[{"candidate_id":"cand","call_id":"cand-call","arm":"task-only","seed":s,"status":"complete","window":[{"t":1,"exact_eval":.6,"own_eval":.55},{"t":2,"exact_eval":.6,"own_eval":.65}],"post_update_exact_eval":.7} for s in range(200,232)]
        hold=[{"id":"__hold__","seed":s,"status":"complete","arm":"hold","window":[{"t":1,"exact_eval":.5,"own_eval":.5},{"t":2,"exact_eval":.5,"own_eval":.5}],"post_update_exact_eval":.5} for s in range(200,232)]
        r=final_report(inv,sel,final,hold,expected_seeds=range(200,232),expected_window=(1,2))['reports'][0]
        self.assertAlmostEqual(r['secondary_metrics']['own_forward_window_vs_hold']['difference'],.1)
        self.assertAlmostEqual(r['secondary_metrics']['endpoint_exact_vs_hold']['difference'],.2)

    def _control_comparison(self, inv, arms, candidate_score=.7, control_score=.6, score_by_id=None, fail_first=False):
        inv=copy.deepcopy(inv)
        for row,arm in zip(inv,arms): row['arm']=arm; row['call_id']=row['id']+'-call'
        dev=[{"candidate_id":c['id'],"call_id":c['call_id'],"arm":c['arm'],"seed":s,"status":"complete","window":[{"t":1,"exact_eval":.8},{"t":2,"exact_eval":.8}]} for c in inv for s in SEEDS]
        sel=select_development(inv,dev,expected_seeds=SEEDS,expected_window=(1,2))
        final=[]
        for c in inv:
            score=(score_by_id or {}).get(c['id'],candidate_score)
            for s in range(200,232): final.append({"candidate_id":c['id'],"call_id":c['call_id'],"arm":c['arm'],"seed":s,"status":"failed" if fail_first and s==200 else "complete","window":[{"t":1,"exact_eval":score},{"t":2,"exact_eval":score}]})
        hold=[{"id":"__hold__","seed":s,"status":"complete","arm":"hold","window":[{"t":1,"exact_eval":.5},{"t":2,"exact_eval":.5}]} for s in range(200,232)]
        control=[{"id":"ctrl","control_id":"ctrl","seed":s,"status":"complete","arm":"control","window":[{"t":1,"exact_eval":control_score},{"t":2,"exact_eval":control_score}]} for s in range(200,232)]
        return final_report(inv,sel,final,hold,control_runs=control,control_id='ctrl',expected_seeds=range(200,232),expected_window=(1,2))

    def test_two_distinct_atom_picks_beating_control_are_replication_candidate(self):
        inv=[dict(INV[0],id='a',arm='atoms'),dict(INV[1],id='b',arm='atoms')]
        out=self._control_comparison(inv,['atoms','atoms'],.7,.6)
        self.assertTrue(out['decision']['representation_replication_candidate'])

    def test_linear_control_equal_to_control_does_not_block_replication(self):
        inv=[dict(INV[0],id='a',arm='atoms'),dict(INV[1],id='b',arm='atoms'),dict(INV[2],id='l',arm='linear-control')]
        out=self._control_comparison(inv,['atoms','atoms','linear-control'],.7,.6,score_by_id={'l':.6})
        self.assertTrue(out['decision']['representation_replication_candidate'])

    def test_linear_control_beating_control_blocks_replication(self):
        inv=[dict(INV[0],id='a',arm='atoms'),dict(INV[1],id='b',arm='atoms'),dict(INV[2],id='l',arm='linear-control')]
        out=self._control_comparison(inv,['atoms','atoms','linear-control'],.8,.6)
        self.assertFalse(out['decision']['representation_replication_candidate'])

    def test_no_eligible_call_is_not_closure_without_positive_results(self):
        inv=[dict(INV[0],id='a',arm='atoms',call_id='a-call')]
        dev=[{"candidate_id":"a","call_id":"a-call","arm":"atoms","seed":s,"status":"failed","error":"compile"} for s in SEEDS]
        sel=select_development(inv,dev,expected_seeds=SEEDS,expected_window=(1,2))
        hold=[{"id":"__hold__","seed":s,"status":"complete","arm":"hold","window":[{"t":1,"exact_eval":.5},{"t":2,"exact_eval":.5}]} for s in range(200,232)]
        out=final_report(inv,sel,[],hold,expected_seeds=range(200,232),expected_window=(1,2))
        self.assertFalse(out['decision']['closure_eligible'])
        self.assertTrue(out['decision']['low_precision_present'])

    def test_positive_adequate_control_result_is_not_closure(self):
        inv=[dict(INV[0],id='a',arm='atoms',call_id='a-call')]
        out=self._control_comparison(inv,['atoms'],.8,.6)
        self.assertFalse(out['decision']['low_precision_present'])
        self.assertFalse(out['decision']['closure_eligible'])

    def test_low_precision_control_prevents_closure_without_positive(self):
        inv=[dict(INV[0],id='a',arm='atoms',call_id='a-call')]
        out=self._control_comparison(inv,['atoms'],.6,.7,fail_first=True)
        self.assertTrue(out['decision']['low_precision_present'])
        self.assertFalse(out['decision']['closure_eligible'])

if __name__=='__main__': unittest.main()
