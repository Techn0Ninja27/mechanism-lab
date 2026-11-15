import unittest
import numpy as np
import torch
from generation.generation import generate_sequences, generate_sequences_batched
from generation.sample import _zscore, build_results_payload

class Mapping:
    bos_token_id = 0
    eos_token_id = 0
    def token_ids_to_cluster_ids(self, row): return [x-1 for x in row if x != 0]

class TinyModel(torch.nn.Module):
    def forward(self, tokens):
        logits=torch.tensor([9.,1.,2.,3.,4.],device=tokens.device)
        return logits.expand(tokens.shape[0], tokens.shape[1], -1)

class SamplingTests(unittest.TestCase):
    def test_batch_one_matches_original(self):
        torch.manual_seed(42)
        old=generate_sequences(TinyModel(),Mapping(),17,4,1.,0,'cpu')
        torch.manual_seed(42)
        new=generate_sequences_batched(TinyModel(),Mapping(),17,4,1.,0,'cpu',1)
        self.assertEqual(old,new)
    def test_tail_and_reproducibility(self):
        def run():
            torch.manual_seed(42)
            return generate_sequences_batched(TinyModel(),Mapping(),17,4,1.,0,'cpu',8)
        rows=run(); self.assertEqual(rows,run()); self.assertEqual(len(rows),17)
        self.assertTrue(all(len(r['clusters'])==3 for r in rows))
    def test_eos_suppression_and_top_k(self):
        rows=generate_sequences_batched(TinyModel(),Mapping(),5,4,1.,1,'cpu',8)
        self.assertTrue(all(r['clusters']==[3,3,3] for r in rows))
    def test_invalid_batch(self):
        with self.assertRaises(ValueError):
            generate_sequences_batched(TinyModel(),Mapping(),batch_size=0)
    def test_ranking_extremes(self):
        c=np.array([3.,2.,1.]);u=np.array([1.,2.,3.]);sets=[(0,1,2),(0,1,3),(0,2,3)]
        for beta,expected in [(0,[0,1,2]),(1,[0,2,3])]:
            result=build_results_payload(sets,c,u,_zscore(c),_zscore(u),beta,3,3,{})
            self.assertEqual(result['top_k_sequences'][0]['cluster_ids'],expected)

if __name__=='__main__': unittest.main()
