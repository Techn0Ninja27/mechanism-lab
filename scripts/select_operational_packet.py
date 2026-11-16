"""Select a small, deterministic non-KV extraction packet from an existing corpus."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from generation.prior_art_retrieval import _local_search
QUERIES=['parameter efficient fine tuning','sparse neural training','test time adaptation','diffusion distillation',
         'activation checkpointing','distributed graph sampling','stochastic equality constrained optimization','randomized matrix approximation']

def main():
    p=argparse.ArgumentParser();p.add_argument('--corpus',type=Path,required=True);p.add_argument('--exclude',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('fresh packet output required')
    seen={r['source_id'] for r in json.loads(a.exclude.read_text())};rows=[]
    for query in QUERIES:
        eligible=[r for r in _local_search(a.corpus,query,20,'2026-11-16') if r['source_id'].startswith('arxiv:') and r['source_id'] not in seen and len(r['abstract'])>=400 and r['published_date']>='2023-01-01']
        if not eligible:raise ValueError('no eligible source for '+query)
        row=eligible[0];seen.add(row['source_id']);row['selection_query']=query;rows.append(row)
    a.output.mkdir(parents=True);packet=a.output/'papers.json';packet.write_text(json.dumps(rows,indent=2)+'\n')
    manifest={'date':'2026-11-17','queries':QUERIES,'rule':'First unseen arXiv-ID record in FTS top20, abstract>=400 chars, published>=2023-01-01, cutoff2026-11-16; excludes specified prior packet.',
              'corpus_sha256':hashlib.sha256(a.corpus.read_bytes()).hexdigest(),'excluded_packet_sha256':hashlib.sha256(a.exclude.read_bytes()).hexdigest(),
              'papers_sha256':hashlib.sha256(packet.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'sources':[{k:r[k] for k in ['source_id','title','url','selection_query']} for r in rows],
              'scope':'Exposed development extraction packet, not representative coverage or a novelty benchmark.'}
    (a.output/'selection.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'papers':len(rows),'papers_sha256':manifest['papers_sha256']}))
if __name__=='__main__':main()
