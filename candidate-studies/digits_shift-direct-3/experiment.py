#!/usr/bin/env python3
"""Gaussian pre-smoothing study for the digits_shift-direct-3 proposal."""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

SEEDS = (137, 271, 389)
SIGMAS = (0.0, 0.4, 0.6, 0.8, 1.0)
CONDITIONS = ((0.,0.),(.2,0.),(.4,0.),(.6,0.),(0.,.05),(0.,.15),(0.,.30),(.4,.15),(.6,.30))

def sha(b): return hashlib.sha256(b).hexdigest()
def arrhash(a): return sha(np.ascontiguousarray(a).tobytes())
def split(y, seed):
    ix=np.arange(len(y)); tr, rest=train_test_split(ix,test_size=.4,random_state=seed,stratify=y)
    va,te=train_test_split(rest,test_size=.5,random_state=seed+1,stratify=y[rest])
    return {'train':np.sort(tr),'validation':np.sort(va),'test':np.sort(te)}
def smooth(x,sigma):
    if sigma == 0: return x.copy()
    return np.stack([gaussian_filter(row.reshape(8,8),sigma=sigma,mode='reflect').ravel() for row in x]).astype(np.float32)
def corrupt(x,noise,drop,seed):
    rng=np.random.default_rng(seed); z=x.copy()
    if noise: z += rng.normal(0,noise*16.,size=z.shape).astype(np.float32)
    if drop: z[rng.random(z.shape)<drop]=0.
    return np.clip(z,0.,16.)
def aug(x,y,seed):
    # Proposal calls for corruption augmentation. Use independent train-only noise and dropout copies.
    rng=np.random.default_rng(seed)
    n=np.clip(x+rng.normal(0,.4*16.,x.shape).astype(np.float32),0.,16.)
    d=x.copy(); d[rng.random(d.shape)<.15]=0.
    return np.concatenate([x,n,d]),np.tile(y,3)
def fit_eval(x,y,idx,evalidx,sigma,augment,seed, conditions):
    t0=time.perf_counter(); tx=smooth(x[idx],sigma); ty=y[idx]
    if augment: tx,ty=aug(tx,ty,seed+202)
    scaler=StandardScaler().fit(tx); tx=scaler.transform(tx)
    model=SVC(kernel='rbf',C=10,gamma='scale')
    model.fit(tx,ty); train_s=time.perf_counter()-t0
    t1=time.perf_counter(); rows=[]
    for j,(noise,drop) in enumerate(conditions):
        ev=smooth(corrupt(x[evalidx],noise,drop,seed+20000+j),sigma)
        pred=model.predict(scaler.transform(ev))
        rows.append({'noise':noise,'dropout':drop,'accuracy':float(accuracy_score(y[evalidx],pred))})
    return rows,train_s,time.perf_counter()-t1
def main():
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['validation','final']); p.add_argument('--output',required=True,type=Path); p.add_argument('--prereg',type=Path); p.add_argument('--allow-rerun',action='store_true'); a=p.parse_args()
    if a.output.exists() and not a.allow_rerun: raise FileExistsError(f'{a.output} exists; pass --allow-rerun to replace')
    if a.mode=='final' and (not a.prereg or not a.prereg.exists()): raise ValueError('final mode requires frozen --prereg')
    data=load_digits(); x=data.data.astype(np.float32); y=data.target.astype(np.int64); allrows=[]; hashes={}
    if a.mode=='final': prereg=json.loads(a.prereg.read_text()); choices={str(q['seed']):q['selected_sigma'] for q in prereg['selected_validation_sigmas']}
    with threadpool_limits(limits=1):
      for seed in SEEDS:
        sp=split(y,seed); hashes[str(seed)]={k:arrhash(v) for k,v in sp.items()}
        idx=sp['train'] if a.mode=='validation' else sp['train']
        ev=sp['validation'] if a.mode=='validation' else sp['test']
        sigmas=SIGMAS if a.mode=='validation' else (choices[str(seed)],)
        for sigma in sigmas:
          rows,trt,intt=fit_eval(x,y,idx,ev,sigma,True,seed,CONDITIONS)
          allrows.append({'seed':seed,'sigma':sigma,'method':'augmented','conditions':rows,'mean_accuracy':float(np.mean([r['accuracy'] for r in rows])),'train_seconds':trt,'inference_seconds':intt})
        # Matched no-smoothing control (sigma 0), same augmentation and identical corrupted evaluation draws.
        rows,trt,intt=fit_eval(x,y,idx,ev,0.,True,seed,CONDITIONS)
        allrows.append({'seed':seed,'sigma':0.,'method':'control_augmented_no_smoothing','conditions':rows,'mean_accuracy':float(np.mean([r['accuracy'] for r in rows])),'train_seconds':trt,'inference_seconds':intt})
        # Component ablation: no augmentation; sweep validation sigmas, then use the frozen choice in final.
        ablation_sigmas=SIGMAS if a.mode=='validation' else (choices[str(seed)],)
        for sig in ablation_sigmas:
          rows,trt,intt=fit_eval(x,y,idx,ev,sig,False,seed,CONDITIONS)
          allrows.append({'seed':seed,'sigma':sig,'method':'ablation_no_augmentation','conditions':rows,'mean_accuracy':float(np.mean([r['accuracy'] for r in rows])),'train_seconds':trt,'inference_seconds':intt})
    out={'mode':a.mode,'seeds':SEEDS,'conditions':CONDITIONS,'sigma_grid':SIGMAS,'metric':'mean accuracy over 9 conditions','split_hashes':hashes,'test_evaluated':a.mode=='final','results':allrows}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2))
    print(json.dumps({'mode':a.mode,'rows':len(allrows),'elapsed_seconds':sum(r['train_seconds']+r['inference_seconds'] for r in allrows),'output':str(a.output)}))
if __name__=='__main__': main()
