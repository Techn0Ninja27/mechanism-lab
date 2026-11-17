import itertools
import unittest
import json, os, sqlite3, subprocess, sys, tempfile
from pathlib import Path
from generation.exploratory_seeds import select_seed_triplets


def catalog(n):
    return {'clusters':{str(i):{'name':str(i)} for i in range(n)},
            'atom_assignments':[{'cluster':i,'paper_id':f'p{i}','text':'source mechanism'} for i in range(n)]}


class ExploratorySeedTests(unittest.TestCase):
    def test_small_space_exact_coverage_and_exclusions(self):
        c=catalog(5);r=select_seed_triplets(c,100)
        self.assertEqual({tuple(x) for x in r['triplets']},set(itertools.combinations(range(5),3)))
        excluded=r['triplets'][:2];remaining=select_seed_triplets(c,100,excluded=excluded)
        self.assertEqual(remaining['triplets'],r['triplets'][2:])
        self.assertFalse(r['ranking_claim_allowed'])

    def test_assignment_order_does_not_change_choice_and_missing_sources_excluded(self):
        c=catalog(10);a=select_seed_triplets(c)
        c['atom_assignments'].reverse()
        self.assertEqual(select_seed_triplets(c),a)
        c['clusters']['11']={'name':'no assignments'}
        c['atom_assignments'].append({'cluster':-1,'paper_id':'p','text':'outlier'})
        self.assertEqual(select_seed_triplets(c),a)

    def test_seed_changes_sample_and_empty_input(self):
        self.assertNotEqual(select_seed_triplets(catalog(20),seed=1)['triplets'],select_seed_triplets(catalog(20),seed=2)['triplets'])
        self.assertEqual(select_seed_triplets(catalog(2))['triplets'],[])
        with self.assertRaises(ValueError):select_seed_triplets(catalog(10),count=True)

    def test_legacy_select_runs_without_any_scored_pool(self):
        source_root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);repo=root/'repo';workspace=root/'workspace'
            (repo/'papers').mkdir(parents=True);(repo/'data').mkdir();workspace.mkdir()
            (repo/'papers/clusters.json').write_text(json.dumps(catalog(5)))
            with sqlite3.connect(repo/'data/llm-papers.db') as db:
                db.execute('CREATE TABLE papers(paper_id TEXT,title TEXT,abstract TEXT,pdf_url TEXT,venue_year TEXT)')
            env=dict(os.environ,ALIEN_WORKSPACE=str(workspace),ALIEN_REPO=str(repo),PROPOSAL_OUTPUT=str(root/'out'),PYTHONPATH=str(source_root))
            subprocess.run([sys.executable,str(source_root/'local_runner/proposal_v2.py'),'select'],env=env,check=True,capture_output=True)
            result=json.loads((root/'out/selection.json').read_text())
            self.assertEqual(len(result['candidates']),6)
            self.assertEqual(result['selection']['method'],'uniform_source_backed_triplets')
            self.assertNotIn('beta',result['selection'])
            self.assertFalse(result['provenance']['ranking_claim_allowed'])
