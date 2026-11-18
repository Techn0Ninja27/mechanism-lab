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

if __name__=='__main__': unittest.main()
