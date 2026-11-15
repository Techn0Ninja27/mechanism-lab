"""Frozen, offline evaluation for digits_shift-random_seed-3.

Run through ../scripts/with_compute_slot.py as required by AGENT_TASK.md.
"""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
SEEDS=(137,271,389)
BETA=.25
CONDITIONS=(('clean',0.,0.),('noise.2',.2,0.),('noise.4',.4,0.),('noise.6',.6,0.),('drop.05',0.,.05),('drop.15',0.,.15),('drop.30',0.,.30),('joint.4.15',.4,.15),('joint.6.30',.6,.30))
METHODS=('arc','fixed_joint','noise_axis','dropout_axis','clean_resampled')

def digest(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def split(y,s):
    ids=np.arange(len(y)); tr,rem=train_test_split(ids,test_size=.4,random_state=s,stratify=y)
    va,te=train_test_split(rem,test_size=.5,random_state=s+1,stratify=y[rem])
    return {k:np.sort(v) for k,v in [('train',tr),('validation',va),('test',te)]}
def corrupt(x,n,d,seed):
    rng=np.random.default_rng(seed); z=x.astype(np.float32,copy=True)
    if n: z += rng.normal(0,n*16.,z.shape).astype(np.float32)
    if d: z[rng.random(z.shape)<d]=0.
    return np.clip(z,0.,16.)/16.
def augment(x,y,s,method):
    rng=np.random.default_rng(s*1000+6503); copies=[]
    for k in range(4):
        if method=='arc':
            theta=rng.uniform(0.,np.pi/2,size=len(x)); sig=BETA*np.cos(theta); drop=BETA*np.sin(theta)
            z=x.astype(np.float32,copy=True)/16.
            z += rng.normal(0.,sig[:,None]*16.,z.shape).astype(np.float32)/16.
            z[rng.random(z.shape)<drop[:,None]]=0.
        elif method=='fixed_joint': z=corrupt(x,BETA,BETA,s*10000+6503+k)
        elif method=='noise_axis': z=corrupt(x,BETA,0.,s*10000+6503+k)
        elif method=='dropout_axis': z=corrupt(x,0.,BETA,s*10000+6503+k)
        elif method=='clean_resampled': z=x.astype(np.float32)/16.
        else: raise ValueError(method)
        copies.append(z)
    return np.concatenate([x.astype(np.float32)/16.,*copies]),np.tile(y,5)
def fit(x,y,idx,s,method):
    tx,ty=augment(x[idx],y[idx],s,method)
    # Proposal specifies fitting scaler on clean training rows only.
    scaler=StandardScaler().fit(x[idx].astype(np.float32)/16.)
    model=SVC(kernel='rbf',C=10,gamma='scale',random_state=s)
    model.fit(scaler.transform(tx),ty)
    return scaler,model
def run(mode,out):
    if mode not in ('validation','final'): raise ValueError(mode)
    if mode=='final':
        pre=ROOT/'preregistration.json'; val=ROOT/'validation.json'
        if not pre.exists() or not val.exists(): raise RuntimeError('final requires completed validation and preregistration')
        if out.exists(): raise FileExistsError(f'{out} exists; choose a new output path')
    started=time.perf_counter(); dat=load_digits(); x=dat.data.astype(np.float32); y=dat.target.astype(np.int64)
    rows=[]; hashes={}; fit_times={}; predict_times={}
    for s in SEEDS:
        sp=split(y,s); hashes[str(s)]={k:digest(v) for k,v in sp.items()}; ev=sp['validation' if mode=='validation' else 'test']
        for method in METHODS:
            t=time.perf_counter(); scaler,model=fit(x,y,sp['train'],s,method); fit_times[f'{s}:{method}']=time.perf_counter()-t
            for j,(name,n,d) in enumerate(CONDITIONS):
                xx=corrupt(x[ev],n,d,s*10000+9901+j)
                t=time.perf_counter(); pred=model.predict(scaler.transform(xx)); predict_times[f'{s}:{method}:{name}']=time.perf_counter()-t
                rows.append(dict(seed=s,method=method,condition=name,accuracy=float(accuracy_score(y[ev],pred))))
    payload=dict(config=dict(mode=mode,seeds=SEEDS,beta=BETA,augment_copies=4,conditions=CONDITIONS,methods=METHODS,split='stratified 60/20/20',scaler='fit on clean train only',svm='RBF SVC C=10 gamma=scale',cpu_threads=1,sklearn=__import__('sklearn').__version__),split_hashes=hashes,fit_seconds=fit_times,predict_seconds=predict_times,results=rows,elapsed_seconds=time.perf_counter()-started)
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,indent=2)); print(json.dumps({'mode':mode,'rows':len(rows),'elapsed_seconds':payload['elapsed_seconds']})); return payload
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['validation','final']); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    with threadpool_limits(limits=1): run(a.mode,a.output)
