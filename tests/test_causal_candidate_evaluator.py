import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from scripts import evaluate_causal_candidates as e

def hold(): return {'id':'hold','call_id':'smoke','arm':'fixture','program':{'predict':{'var':'last_sync'},'update':{'var':'memory'}}}

class EvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x,cls.y,cls.adj=e.gap.screen.load_cora(e.gap.DATA)
        cls.sp=e.gap.splits(cls.y)

    def test_hold_reproduces_full_closed_loop_baseline(self):
        path=Path(e.GAPDIR).parent/'gnn-closedloop-confirm/confirm.py'
        spec=importlib.util.spec_from_file_location('confirm_reference',path); ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
        window=tuple(range(88,120))
        a=e.run_one(self.x,self.y,self.adj,self.sp,hold(),seed=98,window=window)
        b=ref.train(self.x,self.y,self.adj,self.sp,seed=98,layers=3,parts=4,lr_drop=False,arm='hold',sync=16,coef=[0.,0.],window=window)
        self.assertEqual(a['status'],'complete')
        self.assertEqual(a['final_train_loss'],b['final_train_loss'])
        self.assertEqual(a['post_update_exact_eval'],b['post_update_exact_eval_120'])
        self.assertEqual(a['window'],[{'t':r['t'],'own_eval':r['stale_eval'],'exact_eval':r['exact_eval']} for r in b['window']])

    def test_hold_matches_preserved_confirmation_record(self):
        path=Path(e.GAPDIR).parent/'gnn-closedloop-confirm/results/raw.json'
        stored=json.loads(path.read_text())
        expected=next(r for r in stored['runs'] if r['regime']=='moderate_lead' and r['seed']==50 and r['arm']=='hold')
        actual=e.run_one(self.x,self.y,self.adj,self.sp,hold(),seed=50,window=tuple(range(88,120)))
        self.assertEqual(actual['status'],'complete')
        self.assertEqual(actual['final_train_loss'],expected['final_train_loss'])
        self.assertEqual(actual['post_update_exact_eval'],expected['post_update_exact_eval_120'])
        self.assertEqual(actual['window'],[{'t':r['t'],'own_eval':r['stale_eval'],'exact_eval':r['exact_eval']} for r in expected['window']])

    def test_runtime_failure_keeps_call_and_arm(self):
        with patch.object(e.ProgramRemote,'__call__',return_value=torch.full((len(self.y),e.gap.HID),float('nan'))):
            r=e.run_one(self.x,self.y,self.adj,self.sp,hold(),seed=98,epochs=1)
        self.assertEqual(r['status'],'failed');self.assertEqual(r['call_id'],'smoke');self.assertEqual(r['arm'],'fixture')
        self.assertNotIn('post_update_exact_eval',r)

    def test_stale_exact_input_unused_and_layer_memories_independent(self):
        prog=e.compile_program({'predict':{'var':'last_sync'},'update':{'op':'add','args':[{'var':'memory'},{'const':1}]}})
        a=e.ProgramRemote(prog,(2,3),16);b=e.ProgramRemote(prog,(2,3),16)
        for t in range(21):
            a(torch.ones(2,3)*t,t);b(torch.ones(2,3)*t,t)
        p=a(torch.ones(2,3)*999,21)
        self.assertTrue(torch.equal(p,torch.ones(2,3)*20))
        self.assertFalse(torch.equal(a.state.memory,b.state.memory))
        q=b(torch.ones(2,3)*-999,21)
        self.assertTrue(torch.equal(p,q))

    def test_selection_cannot_be_called_without_manifest(self):
        with self.assertRaises(NotImplementedError):e.select_development({'runs':[]})

if __name__=='__main__':unittest.main()
