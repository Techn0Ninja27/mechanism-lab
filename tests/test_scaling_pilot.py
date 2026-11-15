import unittest,random
from scripts.train_scaling_pilot import split_sets,encode,collate,unigram
from generation.token_mapping import TokenMapping
class PilotTests(unittest.TestCase):
    def test_set_split(self):
        rows=[(i,i+1) for i in range(20)]
        train,val=split_sets(rows,42)
        self.assertFalse(set(train)&set(val));self.assertEqual(len(train)+len(val),20)
        self.assertEqual((train,val),split_sets(rows,42))
    def test_compact_encoding(self):
        m=TokenMapping({'bos_token_id':50256,'eos_token_id':50256,'n_clusters':3,'cluster_to_token_id':{'0':0,'1':1,'2':2},'token_id_to_cluster':{'0':0,'1':1,'2':2}})
        m,_=m.compact();s=encode((0,1,2),m,random.Random(42))
        self.assertEqual(m.compute_vocab_size(),4);self.assertEqual(s[0],3);self.assertEqual(s[-1],3)
        self.assertEqual(set(s[1:-1]),{0,1,2})
    def test_padding_mask(self):
        x,m=collate([[3,0,3],[3,1,2,3]],3)
        self.assertEqual(m.sum().item(),7);self.assertFalse(m[0,-1].item())
    def test_unigram_ignores_bos(self):
        import math
        self.assertAlmostEqual(unigram([[2,0,2]],[[2,0,2]],3),-math.log(2/5))
if __name__=='__main__':unittest.main()
