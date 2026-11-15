"""Reproducible candidate study; validation and frozen-test modes are separate."""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from threadpoolctl import threadpool_limits

SEEDS = (137, 271, 389)
CONDITIONS = ((0.,0.),(.2,0.),(.4,0.),(.6,0.),(0.,.05),(0.,.15),(0.,.30),(.4,.15),(.6,.30))

def sha(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def split(y, seed):
    ix=np.arange(len(y)); tr, rest=train_test_split(ix,test_size=.4,random_state=seed,stratify=y)
    va, te=train_test_split(rest,test_size=.5,random_state=seed+1,stratify=y[rest])
    return {"train":np.sort(tr),"validation":np.sort(va),"test":np.sort(te)}
def corruption(x, noise, drop, seed):
    rng=np.random.default_rng(seed); z=x.copy()
    if noise: z += rng.normal(0, noise*16., z.shape).astype(np.float32)
    if drop: z[rng.random(z.shape)<drop]=0
    return np.clip(z,0,16)
def phases(kind):
    if kind=="swept": return .5*(1+np.sin(2*np.pi*np.arange(8)/8))
    if kind=="uniform": return np.linspace(0,1,8)
    if kind=="constant_mid": return np.full(8,.5)
    if kind=="single_mid": return np.array([.5])
    raise ValueError(kind)
def fit(x,y,idx,kind,seed):
    raw=x[idx]; yy=y[idx]; scaler=StandardScaler().fit(raw)
    sx=scaler.transform(raw); parts=[sx]; labels=[yy]
    # Each level uses the same per-copy RNG across methods, giving paired draws.
    for j,s in enumerate(phases(kind)):
        aug=corruption(raw,.5*float(s),.15*float(s),seed+7000+j)
        parts.append(scaler.transform(aug)); labels.append(yy)
    model=SVC(kernel='rbf',C=10,gamma='scale').fit(np.vstack(parts),np.concatenate(labels))
    return scaler,model
def run(mode,out):
    if mode not in ('validation','final'): raise ValueError(mode)
    if mode=='final' and out.exists(): raise FileExistsError(f"Refusing to overwrite frozen output: {out}")
    started=time.perf_counter(); data=load_digits(); x=data.data.astype(np.float32); y=data.target.astype(np.int64)
    rows=[]; splits={}; variants=('swept','uniform','constant_mid','single_mid')
    with threadpool_limits(limits=1):
      for seed in SEEDS:
        ss=split(y,seed); splits[str(seed)]={k:sha(v) for k,v in ss.items()}
        ev=ss['validation' if mode=='validation' else 'test']; models={}
        for kind in variants:
            t=time.perf_counter(); scaler,model=fit(x,y,ss['train'],kind,seed)
            train_sec=time.perf_counter()-t; predsec=0.; vals=[]
            for n,(noise,drop) in enumerate(CONDITIONS):
                z=x[ev] if (noise==0 and drop==0) else corruption(x[ev],noise,drop,seed+20000+n)
                t=time.perf_counter(); p=model.predict(scaler.transform(z)); predsec+=time.perf_counter()-t
                vals.append(float(accuracy_score(y[ev],p)))
                rows.append({'seed':seed,'method':kind,'condition':{'noise':noise,'dropout':drop},'accuracy':vals[-1]})
            models[kind]={'mean_accuracy':float(np.mean(vals)),'train_seconds':train_sec,'inference_seconds':predsec}
        for kind,m in models.items():
            rows.append({'seed':seed,'method':kind,'condition':'aggregate','accuracy':m['mean_accuracy'],'train_seconds':m['train_seconds'],'inference_seconds':m['inference_seconds']})
    payload={'config':{'mode':mode,'seeds':SEEDS,'conditions':CONDITIONS,'variants':variants,'normalization':'pixel/16 for corruption generation; scaler fit clean training only','metric':'unweighted mean accuracy over 9 conditions','sklearn':__import__('sklearn').__version__,'numpy':np.__version__},'splits':splits,'results':rows,'elapsed_seconds':time.perf_counter()-started}
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,indent=2)); print(json.dumps({'mode':mode,'rows':len(rows),'elapsed_seconds':payload['elapsed_seconds']})); return payload
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['validation','final']); p.add_argument('--output',required=True,type=Path); a=p.parse_args(); run(a.mode,a.output)
