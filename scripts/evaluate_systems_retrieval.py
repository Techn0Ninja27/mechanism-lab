"""Six exposed prior-art cases; separates packet repair from broader expansion."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from generation.prior_art_retrieval import retrieve_prior_art
from scripts.audit_corpus_preservation import file_sha
CASES=[('fedais_name',['FedAIS'],'arxiv:2409.14655'),
       ('fedais_mechanism',['federated adaptive importance sampling'],'arxiv:2409.14655'),
       ('sat_name',['Staleness Alleviated'],'arxiv:2308.13466'),
       ('sat_mechanism',['embedding prediction staleness'],'arxiv:2308.13466'),
       ('rank_code_name',['Frequency Ordered Tokenization'],'arxiv:2602.22958'),
       ('rank_code_mechanism',['frequency tokenization compression'],'arxiv:2602.22958')]

def evaluate(paths):
    cells=[]
    for name,db in paths:
        sha=file_sha(db);rows=[]
        for case,queries,expected in CASES:
            result=retrieve_prior_art(queries,db,limit=10,cutoff='2026-11-16')
            ids=[x['canonical_id'] for x in result['records']]
            rows.append({'case':case,'queries':queries,'expected':expected,'rank':ids.index(expected)+1 if expected in ids else None,
                         'returned':ids,'failures':result['failures']})
        cells.append({'snapshot':name,'corpus_sha256':sha,'hits_at_10':sum(x['rank'] is not None for x in rows),'cases':rows})
    return {'scope':'Three known papers, six exposed queries selected after misses. Not held-out recall, precision, novelty, usefulness or proof of general retrieval improvement.',
            'cells':cells,'retriever_sha256':file_sha(ROOT/'generation/prior_art_retrieval.py'),'script_sha256':file_sha(Path(__file__))}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--before',type=Path,required=True);p.add_argument('--packet-only',type=Path,required=True);p.add_argument('--after',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve previous evaluation')
    r=evaluate([('before',a.before),('packet_only',a.packet_only),('all_ml',a.after)])
    a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps([{k:v for k,v in x.items() if k=='snapshot' or k=='hits_at_10'} for x in r['cells']]))
