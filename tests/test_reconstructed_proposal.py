import unittest
from generation.reconstructed_proposal import validate
class ReconstructionTests(unittest.TestCase):
 def proposal(self):return {'components':['2406.16484v1','P_new'],'proposed_primitives':[{'id':'P_new','definition':'A proposed operator','source_support_ids':['arxiv:2406.16484'],'novelty_established':False}],'relations':[{'source':'2406.16484v1','target':'P_new','type':'informs','counterfactual_prediction':'Removing input changes outcome'}]}
 def test_alias_and_arbitrary_new_name(self):self.assertEqual(validate(self.proposal(),['arxiv:2406.16484'])['components'][0],'arxiv:2406.16484')
 def test_unknown_source_fails(self):
  p=self.proposal();p['proposed_primitives'][0]['source_support_ids']=['9999.12345']
  with self.assertRaises(ValueError):validate(p,['arxiv:2406.16484'])
 def test_false_novelty_claim_fails(self):
  p=self.proposal();p['proposed_primitives'][0]['novelty_established']=True
  with self.assertRaises(ValueError):validate(p,['arxiv:2406.16484'])
