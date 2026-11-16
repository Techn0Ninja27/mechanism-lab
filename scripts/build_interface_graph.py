#!/usr/bin/env python3
import argparse,json,hashlib
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from generation.interface_graph import load_atoms,build

def main():
 p=argparse.ArgumentParser(); root=Path(__file__).resolve().parents[1]
 p.add_argument('--atoms',default=str(root/'results/ml-graph-atoms-v1/recovery/atoms.jsonl'))
 p.add_argument('--review',default=str(root/'reports/ml-graph-atoms/review.json'))
 p.add_argument('--papers',default=str(root/'results/ml-graph-atoms-v1/papers.json'))
 p.add_argument('--out',default=str(root/'reports/interface-graph/interface-graph.json')); p.add_argument('--limit',type=int,default=20); p.add_argument('--seed',type=int,default=73031)
 p.add_argument('--selection-mode',choices=['lexical','mechanism_diverse'],default='lexical'); p.add_argument('--max-paper-uses',type=int,default=3)
 a=p.parse_args(); atoms,review,rejs=load_atoms(a.atoms,a.review,a.papers); result=build(atoms,review,rejs,seed=a.seed,limit=a.limit,selection_mode=a.selection_mode,max_paper_uses=a.max_paper_uses)
 result['input_sha256']={label:hashlib.sha256(Path(path).read_bytes()).hexdigest() for label,path in [('atoms',a.atoms),('review',a.review),('papers',a.papers)]}
 result['code_sha256']={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ['generation/interface_graph.py','scripts/build_interface_graph.py']}
 if not 1<=a.limit<=20: raise SystemExit('--limit must be in [1,20]')
 out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
 if out.exists(): raise FileExistsError('Preserve existing graph; choose a fresh --out path')
 out.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps({'atoms_retained':len(atoms),'candidate_pool_size':result['candidate_pool_size'],'out':str(out)}))
if __name__=='__main__': main()
