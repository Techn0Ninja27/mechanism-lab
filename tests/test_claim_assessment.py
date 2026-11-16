import copy,unittest
from generation.claim_assessment import assess
class ClaimAssessmentTests(unittest.TestCase):
    def setUp(self):
        self.claims=[dict(id='method',text='A method',role='central',kind='method'),dict(id='component',text='Known helper',role='supporting',kind='resource')]
        self.chunks=[dict(id='source',text='The prior method uses a helper.')]
    def row(self,cid,status='unknown'):
        return dict(claim_id=cid,coverage=status,checked_chunk_ids=[] if status=='unknown' else ['source'],
                    evidence=[dict(chunk_id='source',quote='uses a helper',supports='Known component')] if status in ('covered','partial') else [],reason='Scoped comparison')
    def test_no_evidence_and_no_match_never_yield_novelty(self):
        for status in ['unknown','not_found']:
            r=assess(self.claims,[self.row(c['id'],status) for c in self.claims],self.chunks)
            self.assertIsNone(r['novelty_estimate']);self.assertFalse(r['novelty_established']);self.assertFalse(r['train_eligible'])
        with self.assertRaises(ValueError):assess(self.claims,[self.row(c['id'],'not_found') for c in self.claims],[])
    def test_supporting_overlap_does_not_reject_central_difference(self):
        r=assess(self.claims,[self.row('method','not_found'),self.row('component','covered')],self.chunks)
        self.assertEqual(r['status'],'no_match_in_examined_evidence')
        self.assertNotIn('reject',r)
    def test_every_claim_and_attribution_are_required(self):
        rows=[self.row(c['id'],'covered') for c in self.claims]
        self.assertEqual(assess(self.claims,rows,self.chunks)['status'],'central_overlap_reported')
        for bad in [rows[:1],[rows[0],rows[0]]]:
            with self.assertRaises(ValueError):assess(self.claims,bad,self.chunks)
        rows[0]['evidence'][0]['quote']='invented'
        with self.assertRaises(ValueError):assess(self.claims,rows,self.chunks)
    def test_result_hypothesis_is_not_promoted_by_model_coverage(self):
        claims=copy.deepcopy(self.claims);claims[0]['kind']='expected_result'
        r=assess(claims,[self.row(c['id'],'covered') for c in claims],self.chunks)
        self.assertEqual(r['claims'][0]['claim']['kind'],'expected_result')
        self.assertFalse(r['novelty_established'])
