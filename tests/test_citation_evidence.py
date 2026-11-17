import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from generation.citation_evidence import load_citation_evidence
from tests import test_grounded_research as fixtures
from tests.test_grounded_research import claims, claim_reviews, mechanism_check, g


def fixture(root):
    snapshot={'schema':'citation-neighborhood.v1','seed':'W1','records':[
        {'id':'W1','title':'Seed','abstract':'A cited mechanism.'},
        {'id':'W2','title':'Competitor','abstract':'Earlier competing method.'},
        {'id':'W3','title':'No abstract','abstract':''}],
        'edges':[{'source':'W1','target':'W2','type':'cites'},
                 {'source':'W1','target':'W3','type':'provider_related'}]}
    p=root/'snapshot.json';p.write_text(json.dumps(snapshot))
    config=root/'citation.json';config.write_text(json.dumps({'snapshot_path':'snapshot.json','selected_ids':['W2','W3']}))
    return config,p


class CitationEvidenceTests(unittest.TestCase):
    def test_missing_abstract_stays_omitted_and_snapshot_bound(self):
        with tempfile.TemporaryDirectory() as d:
            config,p=fixture(Path(d));r=load_citation_evidence(config)
            self.assertEqual(r['summary']['included_ids'],['W2'])
            self.assertEqual(r['summary']['omitted_ids'],['W3'])
            self.assertEqual(r['chunks'][0]['relations'][0]['type'],'cites')
            old=r['summary']['snapshot_sha256'];p.write_text(p.read_text()+'\n')
            self.assertNotEqual(load_citation_evidence(config)['summary']['snapshot_sha256'],old)

    def test_bounds_and_unknown_selection_fail(self):
        with tempfile.TemporaryDirectory() as d:
            config,p=fixture(Path(d));snapshot=json.loads(p.read_text())
            config.write_text(json.dumps({'snapshot_path':'snapshot.json','selected_ids':['W99']}))
            with self.assertRaises(ValueError):load_citation_evidence(config)
            config.write_text(json.dumps({'snapshot_path':'snapshot.json','selected_ids':['W2','W2']}))
            with self.assertRaises(ValueError):load_citation_evidence(config)
            config.write_text(json.dumps({'snapshot_path':'snapshot.json','selected_ids':['W2']}))
            snapshot['records'][1]['abstract']='x'*100001;p.write_text(json.dumps(snapshot))
            with self.assertRaises(ValueError):load_citation_evidence(config)

    def test_text_budget_and_non_neighbor_edges(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);config,p=fixture(root)
            snapshot={'schema':'citation-neighborhood.v1','seed':'W1',
                      'records':[{'id':f'W{i}','title':str(i),'abstract':'x'*5000} for i in range(1,7)],
                      'edges':[{'source':'W1','target':f'W{i}','type':'cites'} for i in range(2,7)]}
            p.write_text(json.dumps(snapshot));config.write_text(json.dumps({'snapshot_path':'snapshot.json','selected_ids':[f'W{i}' for i in range(2,7)]}))
            r=load_citation_evidence(config)
            self.assertEqual(r['summary']['included_chars'],16000)
            self.assertEqual(r['summary']['omitted_ids'],['W6'])
            self.assertTrue(all(x['truncated'] for x in r['chunks']))
            snapshot['edges'][0]['source']='W7';p.write_text(json.dumps(snapshot))
            with self.assertRaises(ValueError):load_citation_evidence(config)

    def test_offline_pipeline_cites_selected_snapshot_and_rejects_later_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);config,snapshot=fixture(root);pack=root/'pack.json'
            pack.write_text(json.dumps(fixtures.GroundedResearchTests()._valid_pack(root)))
            question=root/'question.txt';question.write_text('Compare the available methods.')
            evidence=load_citation_evidence(config);cid=evidence['chunks'][0]['id']
            proposal={k:'specified' for k in g.SCHEMA}
            ev=[{'chunk_id':cid,'quote':'Earlier competing method.','supports':'prior method only'}]
            proposal.update(decision='revise',novelty_established=False,evidence=ev,contribution_claims=claims(),mechanism_check=mechanism_check(),assumptions=['test'],data_paths=[{'input':'x','operation':'op','output':'y','possible_leakage':'none'}],experiment={k:'specified' for k in ('development','heldout','metric','kill_threshold')})
            review={k:'specified' for k in g.REVIEW_SCHEMA}
            review.update(decision='revise',failure_type='prior_art_overlap',novelty_established=False,evidence=ev,claim_assessments=claim_reviews(),mechanism_check=mechanism_check())
            prompts=[]
            class Client:
                def __init__(self,*args):pass
                def call(self,identifier,stage,prompt):
                    prompts.append(prompt);return json.dumps(proposal if stage==0 else review)
            with patch.object(g,'Client',side_effect=AssertionError('offline planning')):
                planned=g.execute(pack,question,root/'run',citation_config_path=config)
            self.assertEqual(planned['config']['citation_evidence']['omitted_ids'],['W3'])
            with patch.object(g,'Client',Client):
                result=g.execute(pack,question,root/'run',True,citation_config_path=config)
            self.assertEqual(result['status'],'reviewed')
            self.assertTrue(all('Earlier competing method.' in p for p in prompts))
            self.assertEqual(result['citation_evidence']['included_ids'],['W2'])
            g.execute(pack,question,root/'other',citation_config_path=config)
            snapshot.write_text(snapshot.read_text()+'\n')
            with patch.object(g,'Client',side_effect=AssertionError('No paid call')):
                with self.assertRaises(ValueError):g.execute(pack,question,root/'other',True,citation_config_path=config)
