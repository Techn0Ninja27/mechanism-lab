"""Exposed KV prior-art recall diagnostic separating corpus and query effects."""
import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from generation.prior_art_retrieval import retrieve_prior_art

EXPECTED=['arxiv:2402.18096','arxiv:2512.19206','arxiv:2608.14191','arxiv:2610.11245']
QUERIES={
 'original':['KV cache quantization','mixed precision cache','adaptive KV retrieval'],
 'mechanism_development':['query cache precision','attention cache distortion','importance cache precision'],
}

def run(before,after):
    cells=[]
    for name,db in [('before',before),('after',after)]:
        with sqlite3.connect('file:'+str(db.resolve())+'?mode=ro',uri=True) as conn:
            ids={r[0] for r in conn.execute('SELECT canonical_id FROM works')}
        for query_name,queries in QUERIES.items():
            result=retrieve_prior_art(queries,db,limit=10,cutoff='2026-11-16')
            found=[r['canonical_id'] for r in result['records']]
            cells.append({'corpus':name,'query_set':query_name,'queries':queries,
                'corpus_sha256':hashlib.sha256(db.read_bytes()).hexdigest(),
                'covered_expected_ids':[i for i in EXPECTED if i in ids],
                'hits_at_10':sum(i in found for i in EXPECTED),
                'ranks':{i:found.index(i)+1 if i in found else None for i in EXPECTED},
                'returned':[{'id':r['canonical_id'],'title':r['title']} for r in result['records']],
                'coverage':result['coverage'],'failures':result['failures']})
    return {'date':'2026-11-16','expected':EXPECTED,'cells':cells,
            'scope':'Four exposed development sources, selected after a failed proposal review. Mechanism queries were adapted after inspecting results. This is not held-out recall, retrieval precision, novelty, or a ranker improvement.',
            'retriever_sha256':hashlib.sha256((ROOT/'generation/prior_art_retrieval.py').read_bytes()).hexdigest(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before',type=Path,required=True);p.add_argument('--after',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve existing evaluation')
    result=run(a.before,a.after)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([{'corpus':r['corpus'],'query_set':r['query_set'],'hits_at_10':r['hits_at_10']} for r in result['cells']]))
