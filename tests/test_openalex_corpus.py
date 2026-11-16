import json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from scripts.fetch_openalex_corpus import load_plan, query_plan, normalize, reconstruct, run

ROOT=Path(__file__).resolve().parents[1]
class OpenAlexCorpusTests(unittest.TestCase):
    def test_plan_is_bounded_stratified_and_balanced(self):
        plan=load_plan(ROOT/'docs/corpus-expansion/OPENALEX_BULK_PLAN.json'); slots=query_plan(plan)
        self.assertEqual(sum(x['target'] for x in slots),100000)
        self.assertEqual({x['category'] for x in slots},{'machine_learning','systems','mathematics'})
        self.assertEqual(len({x['id'] for x in slots}),len(slots)); self.assertEqual(len(slots),60)
        self.assertEqual([len(x) for x in plan['year_windows']], [2,2,2])
    def test_reconstructs_inverted_abstract(self):
        self.assertEqual(reconstruct({'hello':[1], 'world':[0]}),'world hello')
        self.assertEqual(reconstruct(None),'')
    def test_normalized_record_preserves_metrics_and_provenance(self):
        row=normalize({'id':'https://openalex.org/W1','doi':'https://doi.org/x','title':'T','publication_date':'2020-01-01','abstract_inverted_index':{'T':[0]},'cited_by_count':4,'fwci':1.2,'citation_normalized_percentile':{'value':.8},'topics':[],'type':'article','locations':[],'ids':{'arxiv':'https://arxiv.org/abs/2000.00001'}},{'id':'machine_learning:T12072:0'},'2026-11-16')
        self.assertEqual(row['abstract'],'T'); self.assertEqual(row['cited_by_count'],4); self.assertEqual(row['arxiv_id'],'https://arxiv.org/abs/2000.00001'); self.assertEqual(row['query_slot'],'machine_learning:T12072:0')
        self.assertIsNone(normalize({'id':'https://openalex.org/W2','title':'No abstract'}, {'id':'x'}, '2026-11-16'))
    def test_plan_hash_mismatch_refuses_resume(self):
        plan=ROOT/'docs/corpus-expansion/OPENALEX_BULK_PLAN.json'
        with tempfile.TemporaryDirectory() as d:
            out=Path(d); (out/'state.json').write_text(json.dumps({'plan_sha256':'bad'}))
            from scripts.fetch_openalex_corpus import run
            with self.assertRaises(ValueError): run(plan,out,max_requests=1)
    def test_page_plan_has_explicit_sort_and_bound(self):
        plan=load_plan(ROOT/'docs/corpus-expansion/OPENALEX_PAGE_PLAN.json')
        self.assertEqual(plan['pagination_mode'],'page'); self.assertEqual(plan['sort'],'publication_date:desc'); self.assertEqual(plan['max_pages_per_slot'],50)
    def test_empty_page_marks_slot_done_and_never_loops(self):
        class Response:
            status=200
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def read(self,n=-1): return b'{"meta":{"next_cursor":null},"results":[]}'
        plan=json.loads((ROOT/'docs/corpus-expansion/OPENALEX_BULK_PLAN.json').read_text())
        plan['targets']={k:1 for k in plan['targets']}; plan['topics']={k:[v[0]] for k,v in plan['topics'].items()}; plan['year_windows']=[plan['year_windows'][0]]
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'plan.json'; p.write_text(json.dumps(plan)); out=Path(d)/'out'
            with patch('scripts.fetch_openalex_corpus.urllib.request.urlopen',return_value=Response()):
                state=run(p,out,max_requests=10,interval=.2)
            self.assertEqual(state['requests'],3); self.assertEqual(state['records'],0)
            self.assertEqual(state['stop_reason'],'strata_exhausted')
            self.assertEqual(sum(state['shortfall'].values()),3)
            self.assertTrue(all(v['done'] for v in state['slots'].values()))
    def test_page_mode_advances_page_and_stops_on_short_page(self):
        class Response:
            status=200
            def __init__(self, page): self.page=page
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def read(self,n=-1):
                rows=[] if self.page > 1 else [{'id':'https://openalex.org/W1','title':'One','abstract_inverted_index':{'one':[0]},'ids':{},'topics':[]}]
                return json.dumps({'meta':{'page':self.page},'results':rows}).encode()
        plan=json.loads((ROOT/'docs/corpus-expansion/OPENALEX_PAGE_PLAN.json').read_text())
        plan['targets']={k:1 for k in plan['targets']}; plan['topics']={k:[v[0]] for k,v in plan['topics'].items()}; plan['year_windows']=[plan['year_windows'][0]]
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'plan.json'; p.write_text(json.dumps(plan)); out=Path(d)/'out'; calls=[]
            def opener(*args,**kwargs): calls.append(1); return Response(len(calls))
            with patch('scripts.fetch_openalex_corpus.urllib.request.urlopen',side_effect=opener): state=run(p,out,max_requests=10,interval=.2)
            self.assertEqual(state['requests'],3); self.assertEqual(state['records'],1); self.assertTrue(all(x['done'] for x in state['slots'].values()))
    def test_page_cannot_overshoot_target_or_duplicate_after_resume(self):
        class Response:
            status=200
            def __init__(self,number): self.number=number
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def read(self,n=-1):
                ident='W'+str(self.number)
                return json.dumps({'meta':{'next_cursor':None},'results':[{'id':'https://openalex.org/'+ident,'title':ident,'abstract_inverted_index':{ident.lower():[0]},'ids':{},'topics':[]}]}).encode()
        plan=json.loads((ROOT/'docs/corpus-expansion/OPENALEX_BULK_PLAN.json').read_text())
        plan['targets']={k:1 for k in plan['targets']}; plan['topics']={k:[v[0]] for k,v in plan['topics'].items()}; plan['year_windows']=[plan['year_windows'][0]]
        body=json.dumps({'meta':{'next_cursor':None},'results':[{'id':'https://openalex.org/W1','title':'One','abstract_inverted_index':{'one':[0]},'ids':{},'topics':[]},{'id':'https://openalex.org/W2','title':'Two','abstract_inverted_index':{'two':[0]},'ids':{},'topics':[]}]}).encode()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'plan.json'; p.write_text(json.dumps(plan)); out=Path(d)/'out'
            calls=[]
            def opener(*args,**kwargs):
                calls.append(1); return Response(len(calls))
            with patch('scripts.fetch_openalex_corpus.urllib.request.urlopen',side_effect=opener):
                first=run(p,out,max_requests=10,interval=.2)
            with (out/'records.jsonl').open() as records: line_count=sum(1 for _ in records)
            self.assertEqual(first['records'],3); self.assertEqual(line_count,3)
            with patch('scripts.fetch_openalex_corpus.urllib.request.urlopen') as call:
                second=run(p,out,max_requests=10,interval=.2)
            call.assert_not_called(); self.assertEqual(second['records'],3)
if __name__=='__main__': unittest.main()
