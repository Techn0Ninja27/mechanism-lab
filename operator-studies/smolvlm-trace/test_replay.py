import unittest
import numpy as np
from replay import attention

class ReplayContracts(unittest.TestCase):
    def test_gqa_causal_prefix_and_decode(self):
        q=np.zeros((1,2,2,4));k=np.zeros((1,1,3,4))
        out=attention(q,k,0)
        np.testing.assert_array_equal(out[0,0],[[1.,0.,0.],[.5,.5,0.]])
        np.testing.assert_array_equal(out[0,0],out[0,1])
        decoded=attention(q[:,:,:1,:],k,2)
        np.testing.assert_allclose(decoded,np.full((1,2,1,3),1/3))

    def test_group_ratio_is_not_silently_rounded(self):
        with self.assertRaises(ValueError):attention(np.zeros((1,3,1,4)),np.zeros((1,2,2,4)),1)

if __name__=='__main__':unittest.main()
