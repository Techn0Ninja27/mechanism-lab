"""Deterministically select an ML-first corpus from one pinned metadata snapshot."""
import argparse,collections,hashlib,heapq,json
from pathlib import Path
DATASET='secemp9/arxiv-complete'
REVISION='cee894837962fede5612cccf2a4c7cacf49b4c3a'
EXPECTED_SHA='7f7bea788af318a3f0fed1c446303c725d339c2cb189ede4412aeadd0ab9bf1f'
GROUPS={'ml':{'cs.LG','cs.AI','cs.CL','cs.CV','cs.NE','stat.ML'},
        'systems':{'cs.DC','cs.OS','cs.DB','cs.PF','cs.NI','cs.AR','cs.PL','cs.SE'},
        'math':{'stat.TH','cs.IT','cs.DS','cs.CC'}}
TARGETS={'ml':60000,'systems':20000,'math':20000}
def group(categories):
    cats=set(categories.split() if isinstance(categories,str) else categories or [])
    for name,allowed in GROUPS.items():
        if cats & allowed or name=='math' and any(c.startswith('math.') for c in cats):return name
    return None

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=73031);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve existing selection')
    h=hashlib.sha256()
    with a.input.open('rb') as f:
        for block in iter(lambda:f.read(8_388_608),b''):h.update(block)
    if h.hexdigest()!=EXPECTED_SHA:raise ValueError('Pinned metadata SHA-256 mismatch')
    import pyarrow.parquet as pq
    pf=pq.ParquetFile(a.input)
    cols=['paper_id','title','abstract','authors','categories','primary_category','first_version_date','latest_version_date','doi','license','arxiv_abs_url']
    missing=set(cols)-set(pf.schema_arrow.names)
    if missing:raise ValueError('Missing columns: '+str(missing))
    heaps={g:[] for g in TARGETS};eligible=collections.Counter();scanned=0
    for batch in pf.iter_batches(batch_size=8192,columns=cols):
        for row in batch.to_pylist():
            scanned+=1;g=group(row['categories'])
            if not g or not row['title'] or not row['abstract']:continue
            if str(row['first_version_date'])[:10]>'2026-11-16' or str(row['latest_version_date'])[:10]>'2026-11-16':continue
            eligible[g]+=1
            key=int.from_bytes(hashlib.sha256(f"{a.seed}:{row['paper_id']}".encode()).digest()[:8],'big')
            entry=(-key,str(row['paper_id']),row)
            if len(heaps[g])<TARGETS[g]:heapq.heappush(heaps[g],entry)
            elif entry[:2]>heaps[g][0][:2]:heapq.heapreplace(heaps[g],entry)
        if scanned%262144==0:print(json.dumps({'scanned':scanned,'eligible':dict(eligible)}),flush=True)
    a.output.mkdir(parents=True);year_counts=collections.Counter();n=0
    with (a.output/'records.jsonl').open('x') as f:
        for g in TARGETS:
            for _,aid,row in sorted(heaps[g],key=lambda x:x[1]):
                item={'id':aid,'title':row['title'],'abstract':row['abstract'],'authors':row['authors'],
                      'categories':row['categories'],'primary_category':row['primary_category'],
                      'published_date':str(row['first_version_date'])[:10],'updated_date':str(row['latest_version_date'])[:10],
                      'doi':row['doi'],'license':row['license'],'url':row['arxiv_abs_url'],
                      'corpus_group':g,'metadata_dataset':DATASET,'metadata_revision':REVISION,
                      'retrieval_method':'arxiv_oai_metadata_mirror','retrieved_at':'2026-11-16'}
                year_counts[g+':'+item['published_date'][:4]]+=1;n+=1
                f.write(json.dumps(item,ensure_ascii=False)+'\n')
    manifest={'dataset':DATASET,'revision':REVISION,'source_sha256':EXPECTED_SHA,'seed':a.seed,'scanned':scanned,
              'targets':TARGETS,'eligible':dict(eligible),'selected':{g:len(v) for g,v in heaps.items()},
              'total_selected':n,'years':dict(sorted(year_counts.items())),
              'selection':'Lowest seeded ID hashes within ordered ML/systems/math category groups; full metadata scan.',
              'scope':'Third-party mirror of arXiv OAI metadata harvested 2026-08-30; no citation counts or full texts.',
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'records_sha256':hashlib.sha256((a.output/'records.jsonl').read_bytes()).hexdigest()}
    (a.output/'selection-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'selected':manifest['selected'],'scanned':scanned}),flush=True)
if __name__=='__main__':main()
