"""Re-rank a saved scored pool offline, with optional coherence/diversity guards."""
import argparse,json,math
from collections import Counter
from itertools import combinations
from pathlib import Path
import numpy as np

def rerank(rows,beta=.7,top_k=30,min_coherence_quantile=0.,max_pair_uses=None):
    if not rows or not 0<=beta<=1 or top_k<1 or not 0<=min_coherence_quantile<=1:
        raise ValueError('Nonempty pool, positive top-k and beta/quantile in [0,1] required')
    if max_pair_uses is not None and max_pair_uses<1:raise ValueError('max_pair_uses must be positive')
    ids=[tuple(sorted(r['cluster_ids'])) for r in rows]
    if len(set(ids))!=len(ids) or any(len(s)<2 or len(set(s))!=len(s) for s in ids):raise ValueError('Expected unique atom sets without repeated atoms')
    c=np.array([r['coherence'] for r in rows]);u=np.array([r['unavailability'] for r in rows])
    if not np.isfinite(c).all() or not np.isfinite(u).all():raise ValueError('Scores must be finite')
    def z(x):return (x-x.mean())/x.std() if x.std()>0 else np.zeros_like(x)
    scores=(1-beta)*z(c)+beta*z(u)
    threshold=float(np.quantile(c,min_coherence_quantile))
    order=sorted(range(len(rows)),key=lambda i:(-scores[i],ids[i]));selected=[];pairs=Counter()
    eligible=0;skipped=0
    for i in order:
        if c[i]<threshold:continue
        eligible+=1
        ps=list(combinations(ids[i],2))
        if max_pair_uses and any(pairs[p]>=max_pair_uses for p in ps):skipped+=1;continue
        if len(selected)<top_k:
            selected.append({**rows[i],'cluster_ids':list(ids[i]),'fusion_score':float(scores[i]),'rank_fused':len(selected)+1})
            pairs.update(ps)
    return {'config':{'beta':beta,'top_k':top_k,'min_coherence_quantile':min_coherence_quantile,'coherence_threshold':threshold,'max_pair_uses':max_pair_uses,'n_candidate_sets':len(rows),'normalization':'full original pool before filtering','eligible':eligible,'selected':len(selected)},'top_k_sequences':selected}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pool',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--beta',type=float,default=.7);p.add_argument('--top-k',type=int,default=30);p.add_argument('--min-coherence-quantile',type=float,default=0);p.add_argument('--max-pair-uses',type=int)
    a=p.parse_args();pool=json.loads(a.pool.read_text())
    result=rerank(pool['candidates'],a.beta,a.top_k,a.min_coherence_quantile,a.max_pair_uses)
    result['config']['source_config']=pool.get('config',{});result['config']['source_path']=str(a.pool)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result['config'],indent=2))
if __name__=='__main__':main()
