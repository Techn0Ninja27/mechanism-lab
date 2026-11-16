import unittest
import numpy as np
from experiment import assign,quantize,output

class Contracts(unittest.TestCase):
    def test_quantizer_levels_and_constant(self):
        x=np.array([-3.,-1.,1.,3.]).reshape(1,4,1)
        np.testing.assert_array_equal(quantize(x,2,4),x)
        self.assertEqual(len(np.unique(quantize(x,1,4))),2)
        np.testing.assert_array_equal(quantize(np.full((1,4,2),-7.),2,4),-7.)
        np.testing.assert_array_equal(quantize(x/7,16,4),(x/7).astype(np.float16).astype(np.float64))

    def test_rank_counts_ties_and_first_window(self):
        result=assign(np.zeros(8));self.assertEqual(result.tolist(),[16,16,4,4,4,2,2,2])
        self.assertEqual(assign(np.arange(8.))[0],16)
        with self.assertRaises(AssertionError):assign(np.full(8,np.nan))

    def test_gqa_and_shared_permutation(self):
        rng=np.random.default_rng(781);q=rng.normal(size=(6,4));k=rng.normal(size=(2,8,4));v=rng.normal(size=(2,8,4))
        permutation=rng.permutation(8)
        np.testing.assert_allclose(output(q,k,v),output(q,k[:,permutation],v[:,permutation]),atol=1e-12)
        self.assertGreater(np.max(np.abs(output(q,k,v)-output(q,quantize(k,2,4),v))),1e-5)

if __name__=='__main__':unittest.main()
