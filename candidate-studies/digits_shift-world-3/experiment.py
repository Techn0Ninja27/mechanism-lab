"""Offline controlled study for digits_shift-world-3."""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

SEEDS=(137,271,389)
NOISE=(0.,.2,.4,.6); DROP=(0.,.05,.15,.30)
JOINT=((.4,.15),(.6,.30))

def h(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def split(y,s):
    a=np.arange(len(y)); tr,r=train_test_split(a,test_size=.4,random_state=s,stratify=y)
    va,te=train_test_split(r,test_size=.5,random_state=s+1,stratify=y[r])
    return {'train':np.sort(tr),'validation':np.sort(va),'test':np.sort(te)}
def corrupt(x,n,d,seed):
    rng=np.random.default_rng(seed); z=x.copy()
    if n:z+=rng.normal(0,n*16.,z.shape)
    if d:z[rng.random(z.shape)<d]=0.
    return np.clip(z,0,16)
def augment(x,y,kind,seed):
    rng=np.random.default_rng(seed); copies=[]
    # Each arm receives exactly two appended copies per source row.
    if kind in ('pixel','mixed','ordinary'):
        copies.append(corrupt(x,.15,.0,seed+11))
        z=x.copy(); z[rng.random(z.shape)<.15]=0.; copies.append(z)
    if kind=='block':
        # 2x2 block placement, one block per image; replaces equal-copy pixel dropout.
        z=x.reshape(-1,8,8).copy()
        for i in range(len(z)):
            r=int(rng.integers(0,7)); c=int(rng.integers(0,7)); z[i,r:r+2,c:c+2]=0.
        copies=[x.copy(),z.reshape(-1,64)]
    return np.concatenate([x,*copies]), np.tile(y,3)
def models(x,y,idx,s):
    a,b=x[idx],y[idx]; out={}; runt={}
    configs={
      'pixel':(StandardScaler(), 'pixel'),
      'block':(PCA(n_components=32,random_state=s,svd_solver='full'), 'block'),
      'mixed':(MinMaxScaler(), 'mixed'),
      'control':(StandardScaler(), 'ordinary'),
    }
    for name,(transform,aug) in configs.items():
        xx,yy=augment(a,b,aug,s+1000)
        m=make_pipeline(transform,SVC(kernel='rbf',decision_function_shape='ovr',random_state=s))
        t=time.perf_counter(); m.fit(xx,yy); runt[name]=time.perf_counter()-t; out[name]=m
    return out,runt
def condition_set(s):
    return [(n,d) for n in NOISE for d in DROP if (n==0 or d==0)] + list(JOINT)
def evaluate(models,x,y,idx,s):
    cond=condition_set(s); data=[(n,d,corrupt(x[idx],n,d,s+20000+i)) for i,(n,d) in enumerate(cond)]
    rows={}
    for name,m in models.items():
        rows[name]=[float(accuracy_score(y[idx],m.predict(z))) for _,_,z in data]
    return cond,data,rows
def run(mode,outpath):
    start=time.perf_counter(); data=load_digits(); x=data.data.astype(np.float64); y=data.target.astype(np.int64)
    records=[]
    with threadpool_limits(limits=1):
      for s in SEEDS:
        sp=split(y,s); ms,runtime=models(x,y,sp['train'],s)
        ev=sp['validation'] if mode=='validation' else sp['test']
        cond,ds,individual=evaluate(ms,x,y,ev,s)
        if mode=='validation':
            val_margin={k:float(np.mean([np.mean(np.max(np.abs(m.decision_function(z)),axis=1)) for _,_,z in ds])) for k,m in ms.items()}
            # Weight classifier by its mean raw OVR decision margin on validation corruptions.
            w=np.array([val_margin[k] for k in ('pixel','block','mixed')]); w=w/w.sum()
            pred_scores=[]
            for _,_,z in ds:
                score=sum(wi*ms[k].decision_function(z) for wi,k in zip(w,('pixel','block','mixed')))
                pred_scores.append(float(accuracy_score(y[ev],ms['pixel'].classes_[np.argmax(score,axis=1)])))
            acc={k:float(np.mean(v)) for k,v in individual.items()}
            best=max(('pixel','block','mixed'),key=lambda k:acc[k])
            records.append({'seed':s,'splits':{k:h(v) for k,v in sp.items()},'conditions':[{'noise':n,'dropout':d} for n,d in cond], 'validation_accuracy_by_model':individual,'validation_mean':acc,'margin':val_margin,'weights':dict(zip(('pixel','block','mixed'),map(float,w))),'selected_by_val_accuracy':best,'validation_weighted_ensemble':pred_scores,'training_seconds':runtime})
        else:
            # Frozen weights/configuration are recomputed from validation only, then applied once to test.
            val=sp['validation']; _,vds,_=evaluate(ms,x,y,val,s)
            val_margin={k:float(np.mean([np.mean(np.max(np.abs(ms[k].decision_function(z)),axis=1)) for _,_,z in vds])) for k in ms}
            w=np.array([val_margin[k] for k in ('pixel','block','mixed')]);w=w/w.sum()
            pred={}
            for name,m in ms.items(): pred[name]=[float(accuracy_score(y[ev],m.predict(z))) for _,_,z in ds]
            ens=[]; uniform=[]
            for _,_,z in ds:
                scores=[ms[k].decision_function(z) for k in ('pixel','block','mixed')]
                ens.append(float(accuracy_score(y[ev],ms['pixel'].classes_[np.argmax(sum(a*b for a,b in zip(w,scores)),axis=1)])))
                uniform.append(float(accuracy_score(y[ev],ms['pixel'].classes_[np.argmax(np.mean(scores,axis=0),axis=1)])))
            valacc={k:float(np.mean(v)) for k,v in evaluate(ms,x,y,val,s)[2].items()}
            best=max(('pixel','block','mixed'),key=lambda k:valacc[k])
            records.append({'seed':s,'conditions':[{'noise':n,'dropout':d} for n,d in cond],'test_accuracy_by_model':pred,'test_margin_weighted_ensemble':ens,'test_uniform_ensemble':uniform,'test_validation_accuracy_selected':pred[best],'selected_by_validation_accuracy':best,'validation_margin_weights':dict(zip(('pixel','block','mixed'),map(float,w))),'training_seconds':runtime})
    payload={'mode':mode,'seeds':SEEDS,'primary_metric':'mean accuracy across 11 conditions (clean, marginal noise/dropout severities, joint high noise/dropout)','results':records,'elapsed_seconds':time.perf_counter()-start}
    outpath=Path(outpath)
    if outpath.exists(): raise FileExistsError(outpath)
    outpath.write_text(json.dumps(payload,indent=2)); print(json.dumps({'mode':mode,'seconds':payload['elapsed_seconds'],'seeds':len(records)}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['validation','final']);p.add_argument('--output',required=True);a=p.parse_args();run(a.mode,a.output)
