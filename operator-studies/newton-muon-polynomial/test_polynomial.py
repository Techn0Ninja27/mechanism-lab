import unittest
import numpy as np
import torch
from polynomial import PLANS,select_plan,inverse,evaluate_matrix_polynomial

class PolynomialTests(unittest.TestCase):
    def test_rotated_spd_inverse_matches_bound(self):
        rng=np.random.default_rng(73031);q,_=np.linalg.qr(rng.normal(size=(12,12)))
        k=(q*np.geomspace(.01,1,12))@q.T;s=select_plan(k)
        self.assertIsNotNone(s['plan_index'])
        p=inverse(torch.tensor(k),s).numpy();a=k+s['delta']*np.eye(12)
        residual=np.linalg.norm(np.eye(12)-a@p,2)
        self.assertLessEqual(residual,PLANS[s['plan_index']]['s_out']+1e-5)
        reference=np.linalg.inv(a)
        self.assertLessEqual(np.linalg.norm(p-reference,'fro')/np.linalg.norm(reference,'fro'),PLANS[s['plan_index']]['s_out']+1e-5)
    def test_no_eligible_plan_is_explicit(self):
        k=np.zeros((512,512));k[0,0]=1;s=select_plan(k)
        self.assertIsNone(s['plan_index'])
        with self.assertRaises(ValueError):inverse(torch.tensor(k),s)
    def test_sparse_quartic_evaluation(self):
        r=torch.tensor([[.3,.1],[.1,.4]],dtype=torch.float64);c=[1.5,1.4,-12.9,0,32]
        expected=c[0]*torch.eye(2)+c[1]*r+c[2]*(r@r)+c[4]*torch.linalg.matrix_power(r,4)
        torch.testing.assert_close(evaluate_matrix_polynomial(r,c),expected)

if __name__=='__main__':unittest.main()
