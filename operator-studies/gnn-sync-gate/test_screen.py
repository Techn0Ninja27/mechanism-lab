import unittest, numpy as np
from screen import replay_gate

class CausalTests(unittest.TestCase):
    def test_shape_and_separate_hold(self):
        t=np.arange(12*3*2,dtype=np.float32).reshape(12,3,2); p=np.array([0,1,2]); r=replay_gate(t,p,warmup=2,sync=3)
        self.assertEqual(r['n_groups'],6); self.assertGreater(r['raw_counts']['gate'],0)
    def test_intermediate_truth_does_not_change_predictions_or_weights(self):
        t=np.ones((14,3,2),np.float32); t[5:8]+=np.array([3.,-1.],np.float32)
        a=replay_gate(t,np.array([0,1,2]),warmup=2,sync=4,debug=True)
        u=t.copy();u[3]=99
        b=replay_gate(u,np.array([0,1,2]),warmup=2,sync=4,debug=True)
        np.testing.assert_allclose(a['debug']['predictions'],b['debug']['predictions'])
        np.testing.assert_allclose(a['debug']['weights'],b['debug']['weights'])
    def test_future_truth_does_not_change_prefix(self):
        t=np.ones((14,3,2),np.float32); a=replay_gate(t,np.array([0,1,2]),warmup=2,sync=4,debug=True)
        t[11:]=17; b=replay_gate(t,np.array([0,1,2]),warmup=2,sync=4,debug=True)
        np.testing.assert_allclose(a['debug']['predictions'][:8],b['debug']['predictions'][:8])
    def test_hold_is_not_self_fed_predictor(self):
        t=np.arange(14*3*2,dtype=np.float32).reshape(14,3,2)
        r=replay_gate(t,np.array([0,1,2]),warmup=2,sync=4,debug=True)
        self.assertTrue(r['debug']['holds_are_separate'])
    def test_endpoint_scored_before_observation_and_group_update_once(self):
        t=np.arange(1,13,dtype=np.float32)[:,None,None]
        t=np.repeat(t,2,axis=1); parts=np.array([0,1])
        r=replay_gate(t,parts,warmup=1,sync=2,eta=1.,share=.01,debug=True)
        pred=np.asarray(r['debug']['predictions']); hold=np.asarray(r['debug']['holds']); w=np.asarray(r['debug']['weights'])
        # t=3 is a sync. Its frozen predictions determine the weights at t=4.
        truth=t[3,0,0]; lp=((pred[1,0,0]-truth)/(abs(truth)+1e-8))**2; lh=((hold[1,0,0]-truth)/(abs(truth)+1e-8))**2
        expected=.99*np.exp(-lp)/(np.exp(-lp)+np.exp(-lh))+.005
        self.assertAlmostEqual(w[2,0,0],expected,places=7)
        self.assertNotEqual(pred[1,0,0],pred[0,0,0])
        self.assertEqual(r['endpoint_updates_per_group'],[5,5])
        diag=r['endpoint_winner_agreement'];self.assertEqual(diag['non_tie_comparisons']+diag['ties'],10)

    def test_no_accumulated_elapsed_time_after_sync(self):
        # Identical observed constant values keep all outputs constant across
        # many sync boundaries. Varying warmup slope then exposes time handling.
        t=np.array([1.,2.,3.,4.,5.,6.,7.,8.],np.float32)[:,None,None]
        r=replay_gate(t,np.array([0]),boundary_pairs=[(0,1)],warmup=1,sync=2,debug=True)
        last=2.;n=1.;d=np.exp(-.2)+1; expected=[]
        for epoch in range(2,8):
            pred=last+n/d*np.exp(-.2);expected.append(pred)
            value=float(t[epoch,0,0]) if (epoch-1)%2==0 else pred
            n=np.exp(-.2)*n+(value-last);d=np.exp(-.2)*d+1;last=value
        np.testing.assert_allclose(np.asarray(r['debug']['predictions'])[:,0,0],expected,rtol=1e-6)

if __name__=='__main__':unittest.main()
