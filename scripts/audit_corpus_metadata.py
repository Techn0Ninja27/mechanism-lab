"""Audit metadata coverage without interpreting influence as scientific quality."""
import argparse,collections,hashlib,json,math
from pathlib import Path
FIELDS=('citation_count','field_weighted_citation_impact','citation_normalized_percentile')
def audit(path):
    before=path.stat();digest=hashlib.sha256();n=0;groups=collections.Counter();years=collections.Counter();providers=collections.Counter()
    metrics={name:{'known':0,'missing':0,'invalid':0,'zero':0,'minimum':None,'maximum':None} for name in FIELDS}
    empty_abstracts=0;as_of_missing=0
    with path.open('rb') as f:
        for line in f:
            digest.update(line)
            if not line.strip():continue
            row=json.loads(line);n+=1
            groups[row.get('corpus_group') or 'unassigned']+=1
            years[str(row.get('published_date') or 'unknown')[:4]]+=1
            providers[row.get('retrieval_method') or 'unknown']+=1
            empty_abstracts+=int(not str(row.get('abstract') or '').strip())
            known=False
            for name in FIELDS:
                v=row.get(name);stat=metrics[name]
                if name=='citation_normalized_percentile' and isinstance(v,dict):v=v.get('value')
                if v is None:stat['missing']+=1;continue
                valid=(not isinstance(v,bool) and isinstance(v,(int,float)) and math.isfinite(v) and v>=0)
                if name=='citation_count':valid=valid and isinstance(v,int)
                if name=='citation_normalized_percentile':valid=valid and v<=1
                if not valid:stat['invalid']+=1;continue
                known=True;stat['known']+=1;stat['zero']+=int(v==0)
                stat['minimum']=v if stat['minimum'] is None else min(v,stat['minimum'])
                stat['maximum']=v if stat['maximum'] is None else max(v,stat['maximum'])
            as_of_missing+=int(known and not any(isinstance(x,dict) and x.get('as_of') for x in row.get('influence_provenance',[])))
    after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Input changed during audit; use a completed snapshot')
    return {'schema':'corpus-metadata-audit.v1','source_sha256':digest.hexdigest(),'records':n,
            'groups':dict(groups),'publication_years':dict(sorted(years.items())),
            'providers':dict(providers),'empty_abstracts':empty_abstracts,'influence_fields':metrics,
            'influence_rows_without_explicit_provenance':as_of_missing,
            'scope':'Coverage and validity only; missing is not zero. Citation metrics do not establish usefulness, feasibility, or novelty.'}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve existing audit')
    result=audit(a.input)
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'records':result['records'],'influence_fields':result['influence_fields']},indent=2))
