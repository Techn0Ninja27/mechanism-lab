"""Compare saved search sources using common normalization; no quality claims."""
import argparse,json,pathlib,itertools
from collections import Counter
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True,type=pathlib.Path);p.add_argument('--top-k',type=int,default=30);p.add_argument('--output',type=pathlib.Path,default=ROOT/'results/source-comparison.json');a=p.parse_args()
pools={str(d):json.loads((d/'k3/scored_candidates.json').read_text())['candidates'] for d in a.runs}
union={tuple(r['cluster_ids']):r for rows in pools.values() for r in rows}
C=np.array([r['coherence'] for r in union.values()]);U=np.array([r['unavailability'] for r in union.values()]); cm,cs,um,us=C.mean(),C.std(),U.mean(),U.std()
cat=json.loads((ROOT/'papers/clusters_80.json').read_text());papers={}
for r in cat['atom_assignments']:
 if r['cluster']!=-1:papers.setdefault(r['paper_id'],set()).add(r['cluster'])
rows=[]
for source,pool in pools.items():
 for beta in [0.,.4,.7]:
  selected=sorted(pool,key=lambda r:-((1-beta)*(r['coherence']-cm)/cs+beta*(r['unavailability']-um)/us))[:a.top_k]
  sets=[set(r['cluster_ids']) for r in selected];freq=Counter(c for s in sets for c in s);prob=np.array(list(freq.values()))/sum(freq.values())
  sim=np.array([[len(x&y)/len(x|y) for y in sets] for x in sets]);e=np.maximum(np.linalg.eigvalsh(sim)/len(sets),0);e=e[e>0]
  absent=[not any(s<=ps for ps in papers.values()) for s in sets]
  rows.append({'source':source,'beta':beta,'candidate_count':len(pool),'top_k':len(sets),'unique_atoms':len(freq),'effective_atoms':float(np.exp(-sum(prob*np.log(prob)))),'vendi_jaccard':float(np.exp(-sum(e*np.log(e)))),'absent_from_released_papers':sum(absent),'mean_coherence':float(np.mean([r['coherence'] for r in selected])),'mean_unavailability':float(np.mean([r['unavailability'] for r in selected])),'selected_atoms':[sorted(s) for s in sets]})
out={'normalization':'shared deduplicated union; not each source independently','union_size':len(union),'caveat':'Absence from released corpus and diversity do not establish novelty or scientific quality. Same draw count does not mean equal compute or unique-candidate count.','results':rows}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2))
for r in rows:print(r['source'],r['beta'],'unique',r['candidate_count'],'effective',round(r['effective_atoms'],1),'vendi',round(r['vendi_jaccard'],1),'absent',r['absent_from_released_papers'])
