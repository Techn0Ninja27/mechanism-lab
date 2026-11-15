#!/usr/bin/env python3
"""Frozen, offline study of the proposed horizontal symmetry scalar."""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

HERE=Path(__file__).resolve().parent
SEEDS=(137,271,389)
CONDITIONS=((0.,0.),(.2,0.),(.4,0.),(.6,0.),(0.,.05),(0.,.15),(0.,.30),(.4,.15),(.6,.30))
METHODS=('control','flip','permutation')

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def hash_arr(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def split(y,seed):
    i=np.arange(len(y)); tr,rest=train_test_split(i,test_size=.4,random_state=seed,stratify=y)
    va,te=train_test_split(rest,test_size=.5,random_state=seed+1,stratify=y[rest])
    return {'train':np.sort(tr),'validation':np.sort(va),'test':np.sort(te)}
def corrupt(x,noise,drop,seed):
    rng=np.random.default_rng(seed); z=x.copy()
    if noise: z += rng.normal(0,noise,x.shape).astype(np.float32)
    if drop: z[rng.random(z.shape)<drop]=0.
    return np.clip(z,0.,1.)
def feature(x,kind,gamma,perm):
    if kind=='control' or kind=='no_scalar': return x
    mirrored=x.reshape(-1,8,8)[:,:,::-1].reshape(-1,64)
    other=mirrored if kind=='flip' else x[:,perm]
    s=np.exp(-gamma*np.sum((x-other)**2,axis=1,keepdims=True))
    return np.concatenate([x,s],axis=1)
def build(seed):
    d=load_digits(); x=(d.data.astype(np.float32)/16.); y=d.target.astype(np.int64); return x,y,split(y,seed)
def train_eval(seed,mode):
    x,y,sp=build(seed); ev=sp[mode]; tr=sp['train']; rng=np.random.default_rng(seed+7000)
    perm=rng.permutation(64); aug=[x[tr]]; ay=[y[tr]]
    # Proposal: three independently corrupted copies per clean training sample.
    for j in range(3):
        z=x[tr].copy(); z+=rng.normal(0,.5,z.shape).astype(np.float32)
        z=np.clip(z,0.,1.); z[rng.random(z.shape)<.10]=0.
        aug.append(z); ay.append(y[tr])
    ax=np.concatenate(aug); labels=np.concatenate(ay)
    # gamma_sym is computed from training augmentation only, before standardization.
    gamma=1./(64.*float(np.var(ax)))
    testdata=[]
    for ci,(n,d) in enumerate(CONDITIONS):
        testdata.append(corrupt(x[ev],n,d,seed+20000+ci))
    out=[]
    for method in METHODS:
        t=time.perf_counter(); xf=feature(ax,method,gamma,perm); scaler=StandardScaler().fit(xf)
        model=SVC(kernel='rbf',C=10,gamma='scale')
        model.fit(scaler.transform(xf),labels); train_s=time.perf_counter()-t
        preds=[]; infer_s=0.
        for z in testdata:
            t=time.perf_counter(); zf=feature(z,method,gamma,perm); p=model.predict(scaler.transform(zf)); infer_s+=time.perf_counter()-t
            preds.append(float(accuracy_score(y[ev],p)))
        out.append({'seed':seed,'method':method,'condition_accuracy':preds,'mean_accuracy':float(np.mean(preds)), 'train_seconds':train_s,'inference_seconds_all_conditions':infer_s})
    return {'seed':seed,'split_hashes':{k:hash_arr(v) for k,v in sp.items()},'n':{k:len(v) for k,v in sp.items()},'results':out}
def main():
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['validation','final']); p.add_argument('--output',type=Path,required=True); p.add_argument('--rerun',action='store_true'); a=p.parse_args()
    prereg=HERE/'preregistration.json'; source=HERE/'experiment.py'
    if a.mode=='final':
        if not prereg.exists(): raise SystemExit('Refusing final run without preregistration.json')
        if a.output.exists() and not a.rerun: raise SystemExit('Output exists; pass --rerun explicitly to overwrite')
    start=time.perf_counter(); rows=[train_eval(s,'validation' if a.mode=='validation' else 'test') for s in SEEDS]
    payload={'mode':a.mode,'seeds':list(SEEDS),'conditions':[{'noise':n,'dropout':d} for n,d in CONDITIONS], 'sklearn':sklearn.__version__,'source_sha256':sha(source),'elapsed_seconds':time.perf_counter()-start,'runs':rows}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(payload,indent=2))
    print(json.dumps({'mode':a.mode,'elapsed_seconds':payload['elapsed_seconds'],'rows':len(rows)},indent=2))
if __name__=='__main__': main()
