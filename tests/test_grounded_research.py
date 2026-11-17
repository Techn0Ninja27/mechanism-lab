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

def claims():
    return [dict(id='c1',text='A proposed method',role='central',kind='method')]

def claim_reviews():
    return [dict(claim_id='c1',coverage='unknown',checked_chunk_ids=[],evidence=[],reason='Evidence insufficient')]

def mechanism_check():
    return dict(status='unresolved', assumption_scope='Source is fixed-kernel; target kernel may vary.',
                specialized_operation='dE/dt = -K(t) E', remaining_effect='Closure of the target is unknown.',
                simplest_baseline='Direct residual integration', distinguishing_test='Derive the closed target evolution.')

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
        p.update(decision='investigate', novelty_established=False, contribution_claims=claims(), mechanism_check=mechanism_check(),
                 evidence=[{'chunk_id':'s1:1','quote':'preserves covariance','supports':'source operator'}],
                 assumptions=['observed anchors'], data_paths=[{'input':'X','operation':'mask','output':'Xm','possible_leakage':'NaN'}],
                 experiment={k:'specified' for k in ('development','heldout','metric','kill_threshold')})
        g.validate(p,pack)
        q=copy.deepcopy(p);q['evidence'][0]['quote']='invented claim'
        with self.assertRaises(ValueError):g.validate(q,pack)
        q=copy.deepcopy(p);q['novelty_established']=True
        with self.assertRaises(ValueError):g.validate(q,pack)

    def test_mechanism_check_requires_derivation_but_does_not_claim_verification(self):
        pack = {'chunks':[{'id':'s1','text':'fixed kernel'}]}
        review = dict(decision='insufficient_evidence', failure_type='insufficient_evidence',
                      reason='Target closure unresolved', evidence=[dict(chunk_id='s1', quote='fixed kernel', supports='source scope')],
                      strongest_counterargument='A valid closure may exist', required_test='Derive it',
                      novelty_established=False, mechanism_check=mechanism_check(),claim_assessments=claim_reviews())
        self.assertEqual(g.validate(review, pack, review=True,proposal_claims=claims())['mechanism_check']['status'], 'unresolved')
        for field, value in [('specialized_operation', ''), ('remaining_effect', []), ('status', 'verified'),
                             ('simplest_baseline', 'x' * 3001)]:
            bad=copy.deepcopy(review); bad['mechanism_check'][field]=value
            with self.subTest(field=field), self.assertRaises(ValueError):
                g.validate(bad, pack, review=True,proposal_claims=claims())
        bad=copy.deepcopy(review); del bad['mechanism_check']['assumption_scope']
        with self.assertRaises(ValueError):g.validate(bad, pack, review=True,proposal_claims=claims())

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

    def test_network_search_plan_stays_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp);pack=root/'pack.json';pack.write_text(json.dumps(self._valid_pack(root)))
            question=root/'q.txt';question.write_text('What changes?')
            config=root/'search.json';config.write_text(json.dumps({'queries':['NeuMiss'],'allow_arxiv':True}))
            with patch.object(g,'retrieve',return_value={'records':[], 'coverage':[], 'failures':[]}) as preview, patch.object(g,'Client',side_effect=AssertionError('No paid client')):
                result=g.execute(pack,question,root/'out',search_config_path=config)
                self.assertEqual(result['status'],'planned')
                self.assertTrue(result['config']['search']['allow_arxiv'])
                self.assertFalse(preview.call_args.kwargs['allow_arxiv'])
                self.assertEqual(result['retrieval_preview']['status'],'no_local_hits')
                self.assertTrue((root/'out/planning-diagnostics.json').is_file())

    def test_empty_requested_retrieval_blocks_paid_client(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp);pack=root/'pack.json';pack.write_text(json.dumps(self._valid_pack(root)))
            question=root/'q.txt';question.write_text('What changes?')
            config=root/'search.json';config.write_text(json.dumps({'queries':['NeuMiss'],'allow_arxiv':False}))
            events=[]
            def retrieve(*args,**kwargs):
                events.append('retrieve');return {'records':[],'coverage':[{'status':'ok','records':0}], 'failures':[]}
            class Client:
                def __init__(self,*args):events.append('client')
                def call(self,*args):raise RuntimeError('mock rejection; no network')
            with patch.object(g,'retrieve',side_effect=retrieve),patch.object(g,'Client',Client):
                with self.assertRaisesRegex(ValueError,'No paid call was made'):
                    g.execute(pack,question,root/'out',True,config)
            self.assertEqual(events,['retrieve'])

    def test_retrieval_guard_allows_hits_or_explicit_source_only(self):
        g.require_retrieval_hits({'queries':['cache']},{'records':[{'source_id':'fixture'}]})
        g.require_retrieval_hits({'queries':[]},{'records':[]})

    def test_local_feedback_is_citable_and_remains_scoped(self):
        record={'id':'fixture','tested_change':'fixture scoped failure','verdict':'no_support',
                'scope':'two exposed development cases','limitations':['not a quality result'],
                'result_sha256':'a'*64,'report_path':'operator-studies/fixture/README.md'}
        feedback={'records':[record],'scope':'partial curated feedback','warnings':[]}
        contracts={'contracts':[{'id':'fixture_contract','inputs':[]}],'scope':'curated declarations only','warnings':[]}
        proposal={k:'specified' for k in g.SCHEMA}
        evidence=[{'chunk_id':'local_experiment:fixture','quote':'fixture scoped failure','supports':'previous local test only'}]
        proposal.update(decision='revise',novelty_established=False,evidence=evidence,contribution_claims=claims(),mechanism_check=mechanism_check(),
                         assumptions=['test assumption'],data_paths=[{k:'specified' for k in ('input','operation','output','possible_leakage')}],
                         experiment={k:'specified' for k in ('development','heldout','metric','kill_threshold')})
        review={k:'specified' for k in g.REVIEW_SCHEMA}
        review.update(decision='test',failure_type='none_found',evidence=evidence,novelty_established=False,mechanism_check=mechanism_check(),claim_assessments=claim_reviews())
        prompts=[]
        class Client:
            def __init__(self,*args):pass
            def call(self,identifier,stage,prompt):
                prompts.append(prompt);return json.dumps(proposal if stage==0 else review)
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp);pack=root/'pack.json';pack.write_text(json.dumps(self._valid_pack(root)))
            question=root/'q.txt';question.write_text('Test a distinct change.')
            with patch.object(g,'inspect_feedback',return_value=feedback),patch.object(g,'inspect_contracts',return_value=contracts),patch.object(g,'Client',Client):
                result=g.execute(pack,question,root/'run',True)
            self.assertEqual(result['status'],'reviewed')
            self.assertEqual(result['experiment_feedback'],feedback)
            self.assertEqual(result['source_contracts'],contracts)
            self.assertTrue(all('fixture scoped failure' in prompt for prompt in prompts))
            self.assertTrue(all('fixture_contract' in prompt for prompt in prompts))
            self.assertIn('not a ban on the source pair',prompts[0])
            self.assertFalse(result['train_eligible'])

    def test_changed_feedback_blocks_paid_execution_of_saved_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp);pack=root/'pack.json';pack.write_text(json.dumps(self._valid_pack(root)))
            question=root/'q.txt';question.write_text('Test one change.')
            with patch.object(g,'inspect_feedback',return_value={'records':[],'scope':'first','warnings':[]}):
                g.execute(pack,question,root/'run')
            with patch.object(g,'inspect_feedback',return_value={'records':[],'scope':'changed','warnings':[]}),patch.object(g,'Client',side_effect=AssertionError('No paid call')):
                with self.assertRaises(ValueError):g.execute(pack,question,root/'run',True)

    def test_changed_contracts_block_saved_paid_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp);pack=root/'pack.json';pack.write_text(json.dumps(self._valid_pack(root)))
            question=root/'q.txt';question.write_text('Test one connection.')
            with patch.object(g,'inspect_contracts',return_value={'contracts':[],'scope':'first','warnings':[]}):
                g.execute(pack,question,root/'run')
            with patch.object(g,'inspect_contracts',return_value={'contracts':[],'scope':'changed','warnings':[]}),patch.object(g,'Client',side_effect=AssertionError('No paid call')):
                with self.assertRaises(ValueError):g.execute(pack,question,root/'run',True)
