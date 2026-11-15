from __future__ import annotations
import argparse, hashlib, json, platform, sys, time
from pathlib import Path
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

SEEDS=(137,271,389)
LEVELS=(0.0,0.10,0.30,0.50)
AUG_RATE=0.20
METHODS=('control_clean','augmentation_value_only','candidate_mask_augmented','ablation_mask_clean')

def sha(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def split(seed):
    x,y=make_classification(n_samples=5000,n_features=20,n_informative=5,n_redundant=5,weights=[.8,.2],class_sep=1.0,random_state=seed)
    idx=np.arange(len(y)); tr,rest=train_test_split(idx,test_size=.4,random_state=seed,stratify=y)
    va,te=train_test_split(rest,test_size=.5,random_state=seed+1,stratify=y[rest])
    return x.astype(np.float32),y.astype(np.int64),{'train':np.sort(tr),'validation':np.sort(va),'test':np.sort(te)}
def corrupt(x,rate,seed):
    if rate==0: return x.copy()
    out=x.copy(); out[np.random.default_rng(seed).random(out.shape)<rate]=np.nan; return out
def matrix(x,imputer,with_mask):
    m=np.isnan(x).astype(np.float32)
    z=imputer.transform(x).astype(np.float32)
    return np.concatenate([z,m],axis=1) if with_mask else z
def fit(x,y,idx,method,seed):
    tr=x[idx]; yy=y[idx]
    imp=SimpleImputer(strategy='median').fit(tr)
    if method in ('augmentation_value_only','candidate_mask_augmented'):
        # Faithful intervention: mask training entries at the expected deployment rate.
        tr=corrupt(tr,AUG_RATE,seed+8000)
    usemask=method in ('candidate_mask_augmented','ablation_mask_clean')
    xx=matrix(tr,imp,usemask)
    clf=HistGradientBoostingClassifier(random_state=seed,max_iter=120,max_leaf_nodes=31,learning_rate=.1,l2_regularization=0.0)
    clf.fit(xx,yy)
    return imp,clf,usemask
def evaluate(mode,outpath):
    if mode not in ('validation','final'): raise ValueError(mode)
    rows=[]; splits={}; runtime={}; start=time.perf_counter()
    with threadpool_limits(limits=1):
      for seed in SEEDS:
        x,y,sp=split(seed); splits[str(seed)]={k:sha(v) for k,v in sp.items()}; ev=sp['validation' if mode=='validation' else 'test']
        for method in METHODS:
          t=time.perf_counter(); imp,clf,usemask=fit(x,y,sp['train'],method,seed); fitsec=time.perf_counter()-t
          for rate in LEVELS:
            xx=corrupt(x[ev],rate,seed+100000+int(rate*1000))
            t=time.perf_counter(); pred=clf.predict_proba(matrix(xx,imp,usemask))[:,1]; infsec=time.perf_counter()-t
            rows.append({'seed':seed,'method':method,'missing_rate':rate,'average_precision':float(average_precision_score(y[ev],pred)),'fit_seconds':fitsec,'inference_seconds':infsec})
          runtime[f'{seed}:{method}']={'fit_seconds':fitsec}
    payload={'config':{'mode':mode,'seeds':SEEDS,'metric':'average_precision','levels':LEVELS,'augmentation_rate':AUG_RATE,'test_evaluated':mode=='final','python':sys.version,'numpy':np.__version__,'sklearn':__import__('sklearn').__version__,'platform':platform.platform(),'threads':1,'hgb':{'max_iter':120,'max_leaf_nodes':31,'learning_rate':.1}},'splits':splits,'results':rows,'runtime':runtime,'elapsed_seconds':time.perf_counter()-start}
    outpath=Path(outpath)
    if mode=='final' and outpath.exists(): raise FileExistsError(f'{outpath} exists; refusing to overwrite held-out results')
    outpath.parent.mkdir(parents=True,exist_ok=True); outpath.write_text(json.dumps(payload,indent=2)); return payload
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['validation','final']); p.add_argument('--output',required=True); a=p.parse_args(); result=evaluate(a.mode,a.output); print(json.dumps({'mode':a.mode,'rows':len(result['results']),'elapsed_seconds':result['elapsed_seconds']}))
