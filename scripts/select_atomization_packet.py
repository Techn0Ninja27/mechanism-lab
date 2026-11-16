"""Select a small ML/system/math packet from a completed evidence corpus."""
import argparse,hashlib,json,sqlite3,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from generation.prior_art_retrieval import _fts_query
QUERIES={'ml':['attention approximation','gradient compression','retrieval memory','test time adaptation','sparse training','quantization inference','mixture experts','continual learning'],
         'systems':['distributed training','inference cache'],'math':['matrix approximation','stochastic optimization']}
def select(corpus,out):
    if out.exists():raise FileExistsError('Preserve existing packet')
    c=sqlite3.connect('file:'+str(corpus.resolve())+'?mode=ro',uri=True)
    selected=[];seen=set();coverage=[]
    for group,queries in QUERIES.items():
        for query in queries:
            rows=c.execute('''SELECT w.canonical_id,w.title,w.abstract,w.landing_url,w.published_date FROM works_fts f JOIN works w ON w.source_id=f.source_id WHERE works_fts MATCH ? AND json_extract(w.bulk_metadata_json,'$.corpus_group')=? AND w.published_date>='2023-01-01' AND w.published_date<='2026-11-16' ORDER BY bm25(works_fts),w.source_id LIMIT 20''',(_fts_query(query),group)).fetchall()
            picked=[]
            for sid,title,abstract,url,date in rows:
                if sid in seen or len(abstract.encode())>16000 or not url:continue
                seen.add(sid);selected.append({'source_id':sid,'title':title,'abstract':abstract,'url':url,'published_date':date,'corpus_group':group,'query':query});picked.append(sid)
                if len(picked)==2:break
            coverage.append({'group':group,'query':query,'picked':picked})
    c.close()
    if not 20<=len(selected)<=32:raise ValueError('Insufficient bounded packet coverage')
    out.mkdir(parents=True)
    (out/'papers.json').write_text(json.dumps(selected,indent=2)+'\n')
    with corpus.open('rb') as f:source_hash=hashlib.file_digest(f,'sha256').hexdigest()
    (out/'selection.json').write_text(json.dumps({'corpus':str(corpus),'corpus_sha256':source_hash,'coverage':coverage,
      'selection':'Fixed query set; two distinct top lexical hits per query within source group, from2023 onward. No citation weighting.',
      'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
    return len(selected)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--corpus',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print('Selected',select(a.corpus,a.output))
