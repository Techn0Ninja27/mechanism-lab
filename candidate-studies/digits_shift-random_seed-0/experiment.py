#!/usr/bin/env python3
"""Candidate study: class-conditional Fisher-porosity corruption augmentation."""
from __future__ import annotations
import argparse, hashlib, json, time, sys
from pathlib import Path
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

SEEDS=(137,271,389)
NOISE=(0.0,0.2,0.4,0.6)
DROP=(0.0,0.05,0.15,0.30)
JOINT=((0.4,0.15),(0.6,0.30))
METHODS=("clean","uniform","porosity","complement")

def sha(b): return hashlib.sha256(np.ascontiguousarray(b).tobytes()).hexdigest()
def split(y,s):
    ii=np.arange(len(y)); tr,rest=train_test_split(ii,test_size=.4,random_state=s,stratify=y)
    va,te=train_test_split(rest,test_size=.5,random_state=s+1,stratify=y[rest])
    return {"train":np.sort(tr),"validation":np.sort(va),"test":np.sort(te)}

def porosity_map(x,y,classes,mode):
    # One-vs-rest univariate Fisher score; lower score => higher porosity.
    maps=[]
    for c in classes:
        a=x[y==c]; b=x[y!=c]
        ma,mb=a.mean(0),b.mean(0)
        f=(ma-mb)**2/(a.var(0)+b.var(0)+1e-6)
        z=1/(f+1e-3)
        maps.append(z/(z.mean()+1e-12))
    w=np.asarray(maps)
    if mode=="complement":
        w=1/(w+1e-6); w/=w.mean(axis=1,keepdims=True)
    return w

def augment(x,y,weights,kind,seed):
    # Two added views per original: one noise, one dropout. Corruption severities
    # are sampled from the benchmark's predeclared train severity distribution.
    rng=np.random.default_rng(seed); classes=np.unique(y); ci={c:i for i,c in enumerate(classes)}
    w=weights[np.array([ci[c] for c in y])] if kind in ("porosity","complement") else np.ones_like(x)
    scales=rng.choice(np.array([.2,.4,.6]),size=(len(y),1))*16
    eps=rng.normal(size=x.shape).astype(np.float32)
    # preserve average per-image noise energy so only spatial allocation changes
    wn=w/np.sqrt(np.mean(w*w,axis=1,keepdims=True))
    xn=np.clip(x + eps*scales*wn,0,16)
    probs=rng.choice(np.array([.05,.15,.30]),size=(len(y),1))
    # Keep per-pixel Bernoulli probabilities valid. Capping is a necessary
    # realization of scaled dropout; realized rates are recorded in report.
    wd=w/w.mean(axis=1,keepdims=True)
    pd=np.minimum(probs*wd,1.0)
    xd=x.copy(); xd[rng.random(x.shape)<pd]=0
    return np.concatenate([x,xn,xd]),np.tile(y,3),float(pd.mean())

def corrupt(x,noise,drop,seed):
    rng=np.random.default_rng(seed); z=x.copy()
    if noise: z+=rng.normal(0,noise*16,z.shape).astype(np.float32)
    if drop: z[rng.random(z.shape)<drop]=0
    return np.clip(z,0,16)

def run(mode,out,allow_rerun=False):
    if mode not in ("validation","final"): raise ValueError(mode)
    out=Path(out)
    if mode=="final" and out.exists() and not allow_rerun: raise FileExistsError(f"refusing to overwrite frozen test output: {out}")
    started=time.perf_counter(); allrows=[]; splits={}; runtimes={}; details={}
    X=load_digits().data.astype(np.float32); Y=load_digits().target.astype(np.int64)
    with threadpool_limits(limits=1):
      for seed in SEEDS:
        sp=split(Y,seed); splits[str(seed)]={k:sha(v) for k,v in sp.items()}; ev=sp["validation" if mode=="validation" else "test"]
        xtr,ytr=X[sp["train"]],Y[sp["train"]]
        # Transformation is fitted only on the original training partition.
        scaler=StandardScaler().fit(xtr); tx=scaler.transform(xtr); ex=scaler.transform(X[ev])
        fish=porosity_map(tx,ytr,np.unique(Y),"porosity")
        inv=porosity_map(tx,ytr,np.unique(Y),"complement")
        for method in METHODS:
            t0=time.perf_counter()
            if method=="clean": xa,ya=xtr,ytr; actual=None
            else: xa,ya,actual=augment(xtr,ytr,fish if method=="porosity" else inv,method,seed+5000)
            sx=scaler.transform(xa)
            model=SVC(C=5,kernel="rbf",gamma="scale").fit(sx,ya)
            runtimes[f"{seed}:{method}"]=time.perf_counter()-t0
            details[f"{seed}:{method}"]={"n_train":int(len(ya)),"realized_aug_dropout_fraction":actual}
            for condition,(noise,drop) in enumerate([(0.,0.),*[(v,0.) for v in NOISE[1:]],*[(0.,v) for v in DROP[1:]],*JOINT]):
                z=corrupt(X[ev],noise,drop,seed+10000+condition)
                pred=model.predict(scaler.transform(z))
                allrows.append({"seed":seed,"method":method,"condition":{"noise":noise,"dropout":drop},"accuracy":float(accuracy_score(Y[ev],pred))})
    # Average conditions per split/method, then average the three paired splits.
    by={}
    for row in allrows: by.setdefault((row["seed"],row["method"]),[]).append(row["accuracy"])
    means={f"{s}:{m}":float(np.mean(by[s,m])) for s in SEEDS for m in METHODS}
    payload={"config":{"mode":mode,"seeds":SEEDS,"conditions":[{"noise":n,"dropout":d} for n,d in [(0.,0.),*[(v,0.) for v in NOISE[1:]],*[(0.,v) for v in DROP[1:]],*JOINT]],"metric":"mean accuracy over 9 conditions per split","augmentation_views_per_sample":2,"scaler_fit":"original training partition only","svc":{"C":5,"kernel":"rbf","gamma":"scale"},"test_evaluated":mode=="final","python":sys.version,"numpy":np.__version__},"splits":splits,"per_condition":allrows,"split_method_means":means,"runtime_seconds":runtimes,"augmentation_details":details,"elapsed_seconds":time.perf_counter()-started}
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,indent=2)); return payload

def main():
    p=argparse.ArgumentParser(); p.add_argument("mode",choices=["validation","final"]); p.add_argument("--output",required=True); p.add_argument("--allow-rerun",action="store_true"); a=p.parse_args()
    r=run(a.mode,a.output,a.allow_rerun); print(json.dumps({"mode":a.mode,"rows":len(r["per_condition"]),"seconds":r["elapsed_seconds"]}))
if __name__=="__main__": main()
