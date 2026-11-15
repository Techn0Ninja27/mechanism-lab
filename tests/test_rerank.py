import unittest,numpy as np
from generation.rerank import rerank
from generation.sample import build_results_payload,_zscore
class RerankTests(unittest.TestCase):
 def setUp(self):
  self.rows=[{'cluster_ids':[0,1,2],'coherence':3.,'unavailability':1.},{'cluster_ids':[0,1,3],'coherence':2.,'unavailability':2.},{'cluster_ids':[2,3,4],'coherence':1.,'unavailability':3.}]
 def test_matches_baseline(self):
  for b in [0,.4,.7,1]:
   c=np.array([r['coherence'] for r in self.rows]);u=np.array([r['unavailability'] for r in self.rows])
   original=build_results_payload([tuple(r['cluster_ids']) for r in self.rows],c,u,_zscore(c),_zscore(u),b,3,3,{})
   new=rerank(self.rows,b,3)
   self.assertEqual([r['cluster_ids'] for r in original['top_k_sequences']],[r['cluster_ids'] for r in new['top_k_sequences']])
 def test_filter_and_pair_cap(self):
  r=rerank(self.rows,0,3,.5,1)
  self.assertEqual(len(r['top_k_sequences']),1)
  self.assertEqual(r['config']['selected'],1)
 def test_invalid(self):
  for rows in [[],self.rows+self.rows,[{**self.rows[0],'coherence':float('nan')}]]:
   with self.assertRaises(ValueError):rerank(rows)
 def test_unchanged_input(self):
  import copy
  old=copy.deepcopy(self.rows);rerank(self.rows);self.assertEqual(old,self.rows)
