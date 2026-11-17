import unittest
import torch
from cost_profile import right_precondition, TinyCausalTransformer, SourceAlignedMuon, matrix_sign, NS_COEFFS

class CostScreenTests(unittest.TestCase):
    def test_block_right_precondition(self):
        g=torch.tensor([[1.,2.,3.,4.]])
        inv=torch.stack([torch.eye(2)*2,torch.eye(2)*3])
        self.assertTrue(torch.equal(right_precondition(g,inv,2),torch.tensor([[2.,4.,9.,12.]])))
    def test_tracker_shapes_and_unique_matrix_params(self):
        m=TinyCausalTransformer(16,width=8,layers=2,heads=2,seq=4)
        mods=m.matrix_modules(); self.assertEqual(len({id(x.weight) for x in mods}),len(mods)); self.assertEqual(mods[-1].block_count,4)
        self.assertEqual(mods[0].in_features,8); self.assertEqual(mods[-1].in_features,32)
        self.assertEqual(mods[-1].block_count,4)
        self.assertEqual(SourceAlignedMuon(mods,mode='newton_cholesky').states[id(mods[-1].weight)].k.shape,(4,8,8))
    def test_precondition_before_momentum(self):
        from cost_profile import TrackedLinear
        m=TrackedLinear(2,2);o=SourceAlignedMuon([m],mode='newton_cholesky')
        g=torch.tensor([[1.,2.],[3.,4.]]);inverse=torch.diag(torch.tensor([2.,3.]))
        o.states[id(m.weight)].inverse=inverse;m.weight.grad=g.clone();o.step()
        self.assertTrue(torch.equal(o.momentum[id(m.weight)],g@inverse))
        self.assertEqual(SourceAlignedMuon([m],mode='muon').states,{})

    def test_qkv_update_splits_and_weight_decay(self):
        from cost_profile import TrackedLinear
        m=TrackedLinear(2,6);o=SourceAlignedMuon([m],mode='muon');before=m.weight.detach().clone()
        grad=torch.arange(1,13,dtype=torch.float32).reshape(6,2);m.weight.grad=grad.clone()
        direction=grad+.95*grad
        expected=torch.cat([matrix_sign(x) for x in direction.chunk(3,dim=0)],dim=0)
        o.step()
        torch.testing.assert_close(m.weight,before*(1-.001*.01)-.001*(2**.5)*expected)

    def test_refresh_is_at_interval_and_clears_matrix_gradients(self):
        m=TinyCausalTransformer(16,width=8,layers=1,heads=2,seq=4); mods=m.matrix_modules(); o=SourceAlignedMuon(mods,mode='newton_cholesky',refresh=32)
        for mod in mods:
            mod.last_input=torch.ones(3,mod.in_features); mod.weight.grad=torch.ones_like(mod.weight)
        before=o.states[id(mods[0].weight)].k.clone(); o.step()
        self.assertTrue(torch.equal(before,o.states[id(mods[0].weight)].k)); self.assertEqual(o.states[id(mods[0].weight)].refresh_count,0)
        for _ in range(31):
            for mod in mods: mod.weight.grad=torch.ones_like(mod.weight)
            o.step()
        self.assertEqual(o.states[id(mods[0].weight)].refresh_count,1); self.assertIsNone(mods[0].weight.grad)

    def test_five_step_newton_schulz_and_frobenius_epsilon(self):
        x=torch.tensor([[1.,2.],[3.,4.]])
        ref=x/(x.norm()+1e-7)
        a,b,c=NS_COEFFS
        for _ in range(5):
            gram=ref@ref.T
            ref=a*ref+b*(gram@ref)+c*(gram@gram@ref)
        self.assertTrue(torch.allclose(matrix_sign(x),ref,atol=1e-6,rtol=1e-6))
    def test_batch_generation_is_deterministic(self):
        from cost_profile import make_batches
        a=make_batches('abcdef'*100,16,seq=4,batch=2,steps=2,seed=4);b=make_batches('abcdef'*100,16,seq=4,batch=2,steps=2,seed=4)
        self.assertTrue(torch.equal(a[0][0],b[0][0]))
        self.assertTrue(torch.equal(a[0][1],b[0][1]))
        text='abcdef'*100; g=torch.Generator().manual_seed(4); starts=torch.randint(0,len(text.encode())-4-1,(2,),generator=g)
        raw=torch.tensor(list(text.encode()),dtype=torch.long)
        expected=torch.stack([raw[s+1:s+5] for s in starts])
        self.assertTrue(torch.equal(a[0][1],expected))
if __name__=='__main__':unittest.main()
