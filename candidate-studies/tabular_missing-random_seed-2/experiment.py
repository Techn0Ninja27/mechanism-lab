"""Faithful evaluation of anchor-conditioned Ridge imputation proposal."""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

SEEDS=(137,271,389); LEVELS=(0.0,0.10,0.30,0.50); THRESHOLD=.20; ALPHA=1.0

def digest(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def split(y,seed):
    idx=np.arange(len(y)); tr,rest=train_test_split(idx,test_size=.4,random_state=seed,stratify=y)
    va,te=train_test_split(rest,test_size=.5,random_state=seed+1,stratify=y[rest])
    return {k:np.sort(v) for k,v in [('train',tr),('validation',va),('test',te)]}
def dataset(seed):
    x,y=make_classification(n_samples=5000,n_features=20,n_informative=5,n_redundant=5,weights=[.8,.2],class_sep=1.0,random_state=seed)
    return x.astype(np.float64),y.astype(np.int64),split(y,seed)
def mask(x,level,seed):
    z=x.copy()
    if level: z[np.random.default_rng(seed).random(z.shape)<level]=np.nan
    return z
class AnchorRidge:
    def __init__(self,threshold=THRESHOLD,alpha=ALPHA): self.threshold=threshold; self.alpha=alpha
    def fit(self,x,level,mask_seed):
        xstress=mask(x,level,mask_seed); rates=np.isnan(xstress).mean(axis=0)
        self.anchors_=np.flatnonzero(rates<self.threshold); self.volatile_=np.flatnonzero(rates>=self.threshold)
        self.models_={}
        if len(self.anchors_):
            for j in self.volatile_:
                ok=np.isfinite(x[:,j]) & np.isfinite(x[:,self.anchors_]).all(axis=1)
                if ok.sum()>=2:
                    m=Ridge(alpha=self.alpha).fit(x[ok][:,self.anchors_],x[ok,j]); self.models_[int(j)]=m
        return self
    def transform(self,x):
        z=x.copy(); recon=np.zeros(z.shape,dtype=bool)
        if len(self.anchors_) and np.isfinite(z[:,self.anchors_]).all(axis=1).any():
            ok=np.isfinite(z[:,self.anchors_]).all(axis=1)
            for j,m in self.models_.items():
                miss=~np.isfinite(z[:,j]) & ok
                if miss.any(): z[miss,j]=m.predict(z[miss][:,self.anchors_]); recon[miss,j]=True
        return z,recon

def fit_model(x,y,kind,seed):
    clf=HistGradientBoostingClassifier(max_iter=120,random_state=seed,max_leaf_nodes=31,l2_regularization=0.0)
    model=make_pipeline(SimpleImputer(strategy='median'),clf) if kind=='median_hgb' else clf
    model.fit(x,y); return model

def run(mode,out):
    if mode not in ('validation','final'): raise ValueError(mode)
    out=Path(out)
    if out.exists() and not args.allow_overwrite: raise FileExistsError(f'{out} exists; use --allow-overwrite explicitly')
    start=time.perf_counter(); rows=[]; split_hashes={}
    with threadpool_limits(limits=1):
      for seed in SEEDS:
        x,y,sp=dataset(seed); split_hashes[str(seed)]={k:digest(v) for k,v in sp.items()}
        ev=sp['validation' if mode=='validation' else 'test']; tr=sp['train']
        models={}; timings={}
        for name in ('median_hgb','anchor_ridge_fallback'):
            t=time.perf_counter();
            if name=='anchor_ridge_fallback':
                # Training rows are complete in this fixture; fit classifier on original data.
                models[name]=fit_model(x[tr],y[tr],name,seed)
            else: models[name]=fit_model(x[tr],y[tr],name,seed)
            timings[name]=time.perf_counter()-t
        for level in LEVELS:
            # Same independent MCAR realization for both methods, fixed per seed/condition.
            testmaskseed=seed+10000+int(level*100)
            xe=mask(x[ev],level,testmaskseed)
            prep=AnchorRidge().fit(x[tr],level,seed+20000+int(level*100))
            xc,recmask=prep.transform(xe)
            # Baseline median imputation fit on training only.
            t=time.perf_counter(); pb=models['median_hgb'].predict_proba(xe)[:,1]; timings['median_infer_'+str(level)]=time.perf_counter()-t
            t=time.perf_counter(); pc=models['anchor_ridge_fallback'].predict_proba(xc)[:,1]; timings['candidate_infer_'+str(level)]=time.perf_counter()-t
            rows.extend([
              {'seed':seed,'method':'median_hgb','condition':level,'average_precision':float(average_precision_score(y[ev],pb))},
              {'seed':seed,'method':'anchor_ridge_fallback','condition':level,'average_precision':float(average_precision_score(y[ev],pc))}])
            r2=[]
            for j in prep.volatile_:
                use=recmask[:,j]
                if use.any(): r2.append(float(r2_score(x[ev][use,j],xc[use,j])))
            rows.append({'seed':seed,'method':'diagnostic','condition':level,'anchors':prep.anchors_.tolist(),'volatile':prep.volatile_.tolist(),'ridge_models':len(prep.models_),'reconstructed_entries':int(recmask.sum()),'mean_reconstruction_r2':float(np.mean(r2)) if r2 else None,'train_seconds':timings})
    payload={'mode':mode,'seed_set':list(SEEDS),'levels':list(LEVELS),'threshold':THRESHOLD,'ridge_alpha':ALPHA,'metric':'average_precision','test_evaluated':mode=='final','split_hashes':split_hashes,'rows':rows,'elapsed_seconds':time.perf_counter()-start,'python':__import__('sys').version,'sklearn':__import__('sklearn').__version__}
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,indent=2)); print(json.dumps({'mode':mode,'rows':len(rows),'elapsed_seconds':payload['elapsed_seconds']}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['validation','final']);p.add_argument('--output',required=True);p.add_argument('--allow-overwrite',action='store_true');args=p.parse_args();run(args.mode,args.output)
