import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from generation.sourcepack import build_source_pack
from scripts.extract_contract_drafts import execute, operation_scopes

class ExtractionRunnerTests(unittest.TestCase):
    def fixture(self,root,level='full_text'):
        (root/'paper.txt').write_text('Input tensor. Returns tensor.')
        pack=build_source_pack({'sources':[{'id':'paper','title':'Paper','url':'https://example.test/paper','path':'paper.txt','evidence_level':level}]},root)
        p=root/'pack.json';p.write_text(json.dumps(pack));return p,pack
    def multi_fixture(self,root):
        (root/'paper1.txt').write_text('Input tensor. Returns tensor.')
        (root/'paper2.txt').write_text('Input graph. Returns graph.')
        pack=build_source_pack({'sources':[
            {'id':'paper1','title':'Paper 1','url':'https://example.test/paper1','path':'paper1.txt','evidence_level':'full_text'},
            {'id':'paper2','title':'Paper 2','url':'https://example.test/paper2','path':'paper2.txt','evidence_level':'full_text'}]},root)
        p=root/'pack.json';p.write_text(json.dumps(pack));return p,pack
    def test_abstract_rejected_before_client_or_plan(self):
        with tempfile.TemporaryDirectory() as d,patch('scripts.extract_contract_drafts.Client') as client:
            root=Path(d);p,_=self.fixture(root,'abstract')
            with self.assertRaisesRegex(ValueError,'Abstract-only'):execute(p,root/'out',True)
            client.assert_not_called();self.assertFalse((root/'out').exists())
    def test_offline_plan_never_constructs_client(self):
        with tempfile.TemporaryDirectory() as d,patch('scripts.extract_contract_drafts.Client') as client:
            root=Path(d);p,_=self.fixture(root);execute(p,root/'out')
            client.assert_not_called();cfg=json.loads((root/'out/plan.json').read_text())
            self.assertEqual(cfg['extraction']['shared_budget_usd'],20)
            self.assertEqual(cfg['review']['budget_usd'],1)
    def test_two_models_keep_accepted_card_unpromoted(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path,pack=self.fixture(root);c=pack['chunks'][0]
            port={'id':'tensor','artifact':'tensor','representation':None,'granularity':None,'phase':None,'role':'artifact','exposure':'required','conditions':[],'evidence_ids':['e']}
            card={'id':'step','source_id':'paper','interpretation':'model_proposed_not_curated','inputs':[port],'outputs':[{**port,'exposure':'unknown'}],
                  'evidence':[{'id':'e','chunk_id':c['id'],'source_sha256':c['source_sha256'],'quote':'Input tensor.'}]}
            responses=[{'records':[{'source_id':'paper','contract':card,'unknowns':['Representation and output exposure not specified.']}]},
                       {'assessments':[{'source_id':'paper','contract_id':'step','decision':'accept','reason':'Limited draft preserves unknowns.','issues':[]}]}]
            stages=[];models=[]
            class Client:
                def __init__(self,out,cfg):models.append(cfg['model'])
                def call(self,case,stage,prompt):stages.append(stage);return json.dumps(responses[stage])
            with patch('scripts.extract_contract_drafts.Client',Client):execute(path,root/'out',True)
            r=json.loads((root/'out/result.json').read_text())
            self.assertEqual(r['status'],'reviewed_drafts_not_curated');self.assertFalse(r['curated_registry_updated'])
            self.assertFalse(r['review']['curated']);self.assertIsNone(r['drafts'][0]['contract']['inputs'][0]['representation'])
            self.assertEqual(stages,[0,1]);self.assertNotEqual(models[0],models[1])
            with patch('scripts.extract_contract_drafts.Client') as client:
                with self.assertRaises(FileExistsError):execute(path,root/'out',True)
                client.assert_not_called()

    def test_operation_scopes_require_exact_source_coverage_and_anchors(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);_,pack=self.multi_fixture(root)
            chunks={c['source_id']:c['id'] for c in pack['chunks']}
            good={'paper1':{'operation':'tensor update','chunk_ids':[chunks['paper1']]},
                  'paper2':{'operation':'graph update','chunk_ids':[chunks['paper2']]}}
            scope_path=root/'scopes.json';scope_path.write_text(json.dumps(good))
            self.assertEqual(operation_scopes(scope_path,pack),good)
            for bad in (
                {'paper1':good['paper1']},
                {'paper1':good['paper1'],'paper2':{'operation':'graph update','chunk_ids':[chunks['paper1']]}},
                {'paper1':{'operation':'','chunk_ids':[chunks['paper1']]},'paper2':good['paper2']},
                {'paper1':{'operation':'x'*1001,'chunk_ids':[chunks['paper1']]},'paper2':good['paper2']},
                {'paper1':{'operation':'tensor update','chunk_ids':[chunks['paper1'],chunks['paper1']]},'paper2':good['paper2']},
                {'paper1':{'operation':'tensor update','chunk_ids':['missing']},'paper2':good['paper2']},
            ):
                scope_path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):operation_scopes(scope_path,pack)

    def test_scopes_are_frozen_in_plan_and_prompts(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path,pack=self.fixture(root);cid=pack['chunks'][0]['id']
            scopes={'paper':{'operation':'tensor update','chunk_ids':[cid]}}
            scope_path=root/'scopes.json';scope_path.write_text(json.dumps(scopes))
            calls=[]
            class Client:
                def __init__(self,out,cfg): pass
                def call(self,case,stage,prompt):
                    calls.append(prompt)
                    if stage == 0:
                        return json.dumps({'records':[{'source_id':'paper','contract':None,'unknowns':['The requested operation is not sufficiently described.']}]})
                    return json.dumps({'assessments':[{'source_id':'paper','contract_id':None,'decision':'insufficient_evidence','reason':'No supported operation contract.','issues':[]}]})
            with patch('scripts.extract_contract_drafts.Client',Client):
                execute(path,root/'out',True,scopes_path=scope_path)
            plan=json.loads((root/'out/plan.json').read_text())
            result=json.loads((root/'out/result.json').read_text())
            self.assertEqual(plan['operation_scopes'],scopes)
            self.assertEqual(result['requested_operations'],scopes)
            self.assertEqual(len(calls),2)
            self.assertEqual(result['status'],'reviewed_drafts_not_curated')
            self.assertTrue(all('tensor update' in prompt for prompt in calls))
            scope_path.write_text(json.dumps({'paper':{'operation':'different operation','chunk_ids':[cid]}}))
            with patch('scripts.extract_contract_drafts.Client') as client:
                with self.assertRaisesRegex(ValueError,'Plan changed'):
                    execute(path,root/'out',False,scopes_path=scope_path)
                client.assert_not_called()

if __name__=='__main__':unittest.main()
