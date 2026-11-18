import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from generation.sourcepack import build_source_pack
from scripts import generate_causal_campaign as g

class GenerationTests(unittest.TestCase):
    def fixture(self,root):
        sources=[]
        for i in range(5):
            name=f's{i}.txt';(root/name).write_text(f'Source {i} maps a recorded input to a documented output. Assumptions remain explicit.\n')
            sources.append(dict(id=f's{i}',title=f'Source {i}',url=f'https://example.test/{i}',evidence_level='full_text',path=name))
        pack=build_source_pack({'sources':sources},root)
        path=root/'pack.json';path.write_text(json.dumps(pack));return path,pack

    def payloads(self,pack):
        atoms=[];distill=[]
        quotas = {'s0': 4, 's1': 2, 's2': 2, 's3': 2, 's4': 2}
        atoms=[]
        for sid, count in quotas.items():
            c=next(x for x in pack['chunks'] if x['source_id']==sid)
            for i in range(count):
                atoms.append(dict(id=f'atom{len(atoms)}',source_id=sid,statement='A bounded statement',inputs=['input'],outputs=['output'],conditions=[],unknowns=['Not specified'],evidence=[dict(chunk_id=c['id'],quote=c['text'][:30])]))
        for sid, count in quotas.items():
            c=next(x for x in pack['chunks'] if x['source_id']==sid)
            for i in range(count):
                distill.append(dict(id=f'distill-{len(distill):02d}',source_id=sid,summary='A source summary',unknowns=['Not specified'],evidence=[dict(chunk_id=c['id'],quote=c['text'][:30])]))
        candidates=[dict(id=f'cand-{i:02d}',program={'predict':{'var':'last_sync'},'update':{'var':'memory'}},hypothesis='A hypothesis',falsifier='A falsifier') for i in range(4)]
        return {'atoms':atoms},{'distillations':distill},{'candidates':candidates}

    def test_plan_does_not_create_client_and_detects_mutated_contract(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);path,_=self.fixture(root);out=root/'campaign'
            with patch.object(g,'Client',side_effect=AssertionError('No key or network during plan')):
                p=g.plan(path,out);g.representations(out)
            self.assertEqual(len(p['cases']),12)
            frozen=json.loads((out/'random-inventory.json').read_text())
            self.assertEqual(len(frozen['random_candidates']),12)
            self.assertEqual(len(frozen['linear_controls']),4)
            p['cases'].pop();(out/'plan.json').write_text(json.dumps(p))
            with self.assertRaises(ValueError):g.verify(out)

    def test_mocked_full_generation_uses_one_budget_directory_and_retains_duplicates(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);path,pack=self.fixture(root);out=root/'campaign';g.plan(path,out)
            atoms,distill,candidates=self.payloads(pack);seen=[]
            class Fake:
                def __init__(self,folder,config):self.out=folder;seen.append((folder,config['budget_usd']))
                def call(self,case,stage,prompt):
                    payload=atoms if case['id']=='prep-atoms' else distill if case['id']=='prep-distillations' else candidates
                    content=json.dumps(payload);g.write_new(self.out/'calls'/f'{case["id"]}-{stage}.json',{'status':'complete','content':content,'reserved_usd':g.reserve(prompt)})
                    if case['id'].startswith('prep-'):
                        req=g.expected_request(prompt)
                        record=json.loads((self.out/'calls'/f'{case["id"]}-{stage}.json').read_text())
                        record.update(request=req,request_hash=g.request_hash(req))
                        (self.out/'calls'/f'{case["id"]}-{stage}.json').write_text(json.dumps(record))
                    return content
            with patch.object(g,'Client',Fake):
                self.assertEqual(g.representations(out,True)['status'],'ready')
                request_plan,prompts=g.generation_plan(out)
                self.assertEqual(len(set(p.split('<<<END TASK>>>')[0] for p in prompts.values())),1)
                self.assertIn('Source 0 maps a recorded input to a documented output.', prompts['raw'])
                self.assertNotIn('\\"chunks\\"', prompts['raw'])
                self.assertEqual({p.name for p in (out/'prompts').glob('*.txt')}, {f'{arm}.txt' for arm in g.ARMS})
                frozen_prompt=out/'prompts'/'raw.txt';frozen_text=frozen_prompt.read_text()
                frozen_prompt.write_text(frozen_text+' drift')
                with self.assertRaises(ValueError): g.generation_plan(out)
                frozen_prompt.write_text(frozen_text)
                prep_call=out/'calls'/'prep-atoms-0.json'
                original=prep_call.read_text()
                tampered=json.loads(original)
                tampered['request']['messages'][1]['content'] += ' tampered'
                prep_call.write_text(json.dumps(tampered))
                with self.assertRaises(ValueError): g.generation_plan(out)
                prep_call.write_text(original)
                result=g.generate(out,True)
            inv=json.loads((out/'inventory.json').read_text())
            self.assertEqual(result['candidates'],64)
            self.assertEqual(sum(c['arm']=='random' for c in inv['candidates']),12)
            self.assertEqual(sum(c['arm']=='linear-control' for c in inv['candidates']),4)
            self.assertEqual(seen,[(out,1.0),(out,6.0)])
            self.assertTrue(all(c['status']=='valid' for c in inv['candidates']))
            self.assertEqual(len({c['program_sha256'] for c in inv['candidates'] if c['arm']=='raw'}),1)

    def test_invalid_nonstandard_json_keeps_all_four_slots(self):
        rows=g.parse_candidates('{"candidates": NaN}',{'id':'raw-00','arm':'raw'})
        self.assertEqual(len(rows),4);self.assertTrue(all(r['status']=='invalid' for r in rows))
        self.assertTrue(all(r['call_id']=='raw-00' for r in rows))

    def test_random_sampling_is_deterministic_and_bounded(self):
        a=g.random_inventory();self.assertEqual(a,g.random_inventory())
        for r in a:self.assertLessEqual(g.compile_program(r['program']).node_count,64)

    def test_verify_rejects_prompt_drift_and_prompt_texts_rejects_path_escape(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);path,_=self.fixture(root);out=root/'campaign';g.plan(path,out)
            prompts=json.loads((out/'preparation/prompts.json').read_text())
            prompts['atoms'] += ' drift'
            (out/'preparation/prompts.json').write_text(json.dumps(prompts))
            with self.assertRaises(ValueError): g.verify(out)

        with tempfile.TemporaryDirectory() as t:
            root=Path(t);path,pack=self.fixture(root);out=root/'campaign';g.plan(path,out)
            atoms,distill,_=self.payloads(pack)
            for kind,payload in (('atoms',atoms),('distillations',distill)):
                result_path=out/'representations'/f'{kind}.json';result_path.parent.mkdir(exist_ok=True)
                result_path.write_text(json.dumps(payload))
            report={'status':'ready','plan_sha256':g.sha(out/'plan.json'),'representations':{
                'atoms':{'path':'../../outside.json','sha256':'0'*64},
                'distillations':{'path':'representations/distillations.json','sha256':g.sha(out/'representations/distillations.json')}}}
            (out/'preparation-result.json').write_text(json.dumps(report))
            with self.assertRaises(ValueError): g.prompt_texts(out)

if __name__=='__main__':unittest.main()
