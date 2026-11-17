import copy
import unittest
from generation.contract_drafts import validate_drafts, validate_review
from tests.test_interface_contracts import ContractTests


class DraftTests(unittest.TestCase):
    def setUp(self):
        self.f = ContractTests(); self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.card = copy.deepcopy(self.f.card); self.card['interpretation'] = 'model_proposed_not_curated'
    def envelope(self, card=None, sid='s'):
        return {'records':[{'source_id':sid,'contract':card,'unknowns':['shape is not stated']}]}
    def test_valid_draft_retains_unknowns_and_is_not_curated(self):
        rows=validate_drafts(self.envelope(self.card),self.f.pack,['s'])
        self.assertEqual(rows[0]['unknowns'],['shape is not stated'])
        self.assertFalse(rows[0]['executable_compatibility_proven'])
        with self.assertRaises(ValueError):
            from generation.interface_contracts import validate_contracts
            validate_contracts({'schema':'interface-contracts.v1','contracts':[self.card]},self.f.pack)
    def test_bad_quote_hash_and_missing_record_fail(self):
        bad=copy.deepcopy(self.card);bad['evidence'][0]['quote']='invented'
        with self.assertRaises(ValueError):validate_drafts(self.envelope(bad),self.f.pack,['s'])
        with self.assertRaises(ValueError):validate_drafts({'records':[]},self.f.pack,['s'])
        bad=copy.deepcopy(self.card);bad['evidence'][0]['source_sha256']='f'*64
        with self.assertRaises(ValueError):validate_drafts(self.envelope(bad),self.f.pack,['s'])
    def test_review_accept_requires_no_issues_and_preserves_mismatch(self):
        rows=validate_drafts(self.envelope(self.card),self.f.pack,['s'])
        review={'assessments':[{'source_id':'s','contract_id':'one','decision':'accept','reason':'reviewed','issues':[]}]}
        out=validate_review(review,rows,self.f.pack);self.assertFalse(out['curated'])
        issue={'port_id':'tensor','field':'representation','reason':'unclear','evidence':[]}
        review['assessments'][0]['decision']='revise';review['assessments'][0]['issues']=[issue]
        self.assertEqual(validate_review(review,rows,self.f.pack)['assessments'][0]['decision'],'revise')
        review['assessments'][0]['decision']='accept'
        with self.assertRaises(ValueError):validate_review(review,rows,self.f.pack)
    def test_review_identity_and_evidence_fail_closed(self):
        rows=validate_drafts(self.envelope(self.card),self.f.pack,['s'])
        r={'assessments':[{'source_id':'wrong','contract_id':None,'decision':'insufficient_evidence','reason':'missing','issues':[]}]}
        with self.assertRaises(ValueError):validate_review(r,rows,self.f.pack)
        r['assessments'][0]['source_id']='s';r['assessments'][0]['issues']=[{'port_id':None,'field':'conditions','reason':'missing','evidence':[{'chunk_id':'s:1','quote':'invented'}]}]
        with self.assertRaises(ValueError):validate_review(r,rows,self.f.pack)

if __name__=='__main__': unittest.main()
