import copy
import json
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'scripts'))
from scripts import grounded_research as g
from scripts import world_search as w
from generation.api_budget import BudgetLedger
from generation.sourcepack import build_source_pack

class GroundedResearchTests(unittest.TestCase):
    def _valid_pack(self, root):
        source = root / 'source.md'
        source.write_text('The model preserves covariance under the stated shift.\n')
        return build_source_pack({'sources': [{'id': 's1', 'title': 'Source',
            'url': 'https://example.test/source', 'path': 'source.md',
            'evidence_level': 'full_text'}]}, root, max_total_chars=1000, chunk_chars=1000)

    def test_evidence_and_novelty_fail_closed(self):
        pack = {'chunks': [{'id':'s1:1','text':'The model preserves covariance.'}]}
        p = {k:'x' for k in g.SCHEMA}
        p.update(decision='investigate', novelty_established=False,
                 evidence=[{'chunk_id':'s1:1','quote':'preserves covariance','supports':'source operator'}],
                 assumptions=['observed anchors'], data_paths=[{'input':'X','operation':'mask','output':'Xm','possible_leakage':'NaN'}],
                 experiment={k:'specified' for k in ('development','heldout','metric','kill_threshold')})
        g.validate(p,pack)
        q=copy.deepcopy(p);q['evidence'][0]['quote']='invented claim'
        with self.assertRaises(ValueError):g.validate(q,pack)
        q=copy.deepcopy(p);q['novelty_established']=True
        with self.assertRaises(ValueError):g.validate(q,pack)

    def test_paid_client_uses_global_ledger_and_reuses_receipt(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(w,'key',return_value='test-key'):
            root=pathlib.Path(tmp);ledger=root/'budget.json';out=root/'run';out.mkdir()
            cfg={'model':'test','max_tokens_per_stage':[10],'pricing':{'prompt':.000001,'completion':.000001},'budget_usd':1,
                 'shared_budget_path':str(ledger),'shared_budget_usd':1,'provider_max_price':{'prompt':1,'completion':1}}
            class Response:
                def __enter__(self):return self
                def __exit__(self,*args):pass
                def read(self):return json.dumps({'id':'receipt-1','choices':[{'message':{'content':'ok'},'finish_reason':'stop'}],
                    'usage':{'cost':.001,'cost_details':{'upstream_inference_cost':.0008}}}).encode()
            with patch.object(w.urllib.request,'urlopen',return_value=Response()) as call:
                c=w.Client(out,cfg)
                self.assertEqual(c.call({'id':'a'},0,'question'),'ok')
                self.assertEqual(c.call({'id':'a'},0,'question'),'ok')
                self.assertEqual(call.call_count,1)
                request=json.loads(call.call_args.args[0].data)
                self.assertEqual(request['provider']['max_price'],cfg['provider_max_price'])
            self.assertEqual(BudgetLedger(ledger,1).snapshot()['charged_usd'],.001)

    def test_unknown_call_keeps_reservation_and_never_retries(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(w,'key',return_value='test-key'):
            root=pathlib.Path(tmp);out=root/'run';out.mkdir();ledger=root/'budget.json'
            cfg={'model':'test','max_tokens_per_stage':[10],'pricing':{'prompt':.000001,'completion':.000001},'budget_usd':1,
                 'shared_budget_path':str(ledger),'shared_budget_usd':1}
            with patch.object(w.urllib.request,'urlopen',side_effect=TimeoutError) as call:
                c=w.Client(out,cfg)
                with self.assertRaises(TimeoutError):c.call({'id':'a'},0,'q')
                with self.assertRaises(ValueError):c.call({'id':'a'},0,'q')
                self.assertEqual(call.call_count,1)
            snap=BudgetLedger(ledger,1).snapshot()
            self.assertGreater(snap['charged_usd'],0)
            self.assertEqual(next(iter(snap['reservations'].values()))['status'],'unknown')

    def test_execute_rejects_tampered_pack_before_paid_client(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            pack = self._valid_pack(root)
            pack['chunks'][0]['text'] = 'Injected replacement text.'
            pack_path = root / 'pack.json'; pack_path.write_text(json.dumps(pack))
            question = root / 'question.txt'; question.write_text('Find one testable operator delta.')
            output = root / 'run'
            with patch.object(g, 'Client', side_effect=AssertionError('paid client must not be constructed')):
                with self.assertRaises(ValueError):
                    g.execute(pack_path, question, output, execute_paid=True)

    def test_execute_preserves_existing_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            pack_path = root / 'pack.json'; pack_path.write_text(json.dumps(self._valid_pack(root)))
            question = root / 'question.txt'; question.write_text('Find one testable operator delta.')
            output = root / 'run'; output.mkdir()
            sentinel = '{"status":"already-reviewed"}\n'
            (output / 'result.json').write_text(sentinel)
            with patch.object(g, 'Client', side_effect=AssertionError('paid client must not be constructed')):
                with self.assertRaises(FileExistsError):
                    g.execute(pack_path, question, output, execute_paid=True)
            self.assertEqual((output / 'result.json').read_text(), sentinel)

    def test_execute_refuses_changed_plan_without_paid_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            pack_path = root / 'pack.json'; pack_path.write_text(json.dumps(self._valid_pack(root)))
            question = root / 'question.txt'; question.write_text('First question.')
            output = root / 'run'
            g.execute(pack_path, question, output, execute_paid=False)
            question.write_text('Changed question.')
            with patch.object(g, 'Client', side_effect=AssertionError('paid client must not be constructed')):
                with self.assertRaises(ValueError):
                    g.execute(pack_path, question, output, execute_paid=True)
