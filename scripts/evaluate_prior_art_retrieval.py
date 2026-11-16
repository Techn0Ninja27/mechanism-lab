"""Known-source retrieval regression; diagnostic cases, not a novelty benchmark."""
import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from generation.prior_art_retrieval import retrieve_prior_art
CASES=[
 ('neumiss-name',['NeuMiss'],'arxiv:2007.01627'),
 ('neumiss-mechanism',['Neumann missingness'],'arxiv:2007.01627'),
 ('stablemiss-name',['StableMiss'],'arxiv:2305.11197'),
 ('stablemiss-mechanism',['agnostic mask distribution shift'],'arxiv:2305.11197'),
 ('imputation-title',["good imputation predict missing values"],'arxiv:2106.00311'),
 ('imputation-second-query',['learning','good imputation predict missing values'],'arxiv:2106.00311'),
 ('robust-title',['robust prediction missingness shifts'],'arxiv:2406.16484'),
 ('robust-second-query',['learning','robust prediction missingness shifts'],'arxiv:2406.16484'),
]
def run(db):
 rows=[]
 with sqlite3.connect('file:'+str(db.resolve())+'?mode=ro',uri=True) as conn:
  ids={r[0] for r in conn.execute('SELECT canonical_id FROM works')}
 for name,queries,expected in CASES:
  result=retrieve_prior_art(queries,db,limit=5,cutoff='2026-11-16')
  found=[r['canonical_id'] for r in result['records']]
  rows.append(dict(case=name,queries=queries,expected=expected,present_in_corpus=expected in ids,
                   rank=found.index(expected)+1 if expected in found else None,returned=found,failures=result['failures']))
 return dict(scope='Eight hand-selected known-source regression cases; exposed development cases, no general recall or novelty claim.',
  corpus_sha256=hashlib.sha256(db.read_bytes()).hexdigest(),retriever_sha256=hashlib.sha256((ROOT/'generation/prior_art_retrieval.py').read_bytes()).hexdigest(),
  cases=rows,hits_at_5=sum(r['rank'] is not None for r in rows),case_count=len(rows),
  corpus_covered_cases=sum(r['present_in_corpus'] for r in rows))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--db',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError('Preserve existing evaluation')
 r=run(a.db)
 with a.output.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
 print(json.dumps({k:v for k,v in r.items() if k!='cases'},indent=2))
