"""Reproducible digits study for direct-0 proposal. Run via shared compute gate."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from threadpoolctl import threadpool_limits

SEEDS=(137,271,389)
TRAIN_AUG=(("noise",.2),("noise",.4),("dropout",.15),("dropout",.30))
EVAL=(('clean',0,0),('noise.2',.2,0),('noise.4',.4,0),('noise.6',.6,0),('drop.05',0,.05),('drop.15',0,.15),('drop.30',0,.30),('joint.4.15',.4,.15),('joint.6.30',.6,.30))

def digest(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def split(y,s):
    ids=np.arange(len(y)); tr,rem=train_test_split(ids,test_size=.4,random_state=s,stratify=y)
    va,te=train_test_split(rem,test_size=.5,random_state=s+1,stratify=y[rem])
    return {k:np.sort(v) for k,v in [('train',tr),('validation',va),('test',te)]}
def corrupt(x,n,d,seed):
    rng=np.random.default_rng(seed); z=x.copy()
    if n: z += rng.normal(0,n*16,z.shape).astype(np.float32)
    if d: z[rng.random(z.shape)<d]=0
    return np.clip(z,0,16)/16
def features(x,which):
    im=x.reshape(-1,8,8); h=(im[:,:,1:]-im[:,:,:-1]).reshape(-1,56); v=(im[:,1:,:]-im[:,:-1,:]).reshape(-1,56)
    m=im.reshape(-1,4,2,4,2).mean(axis=(2,4)).reshape(-1,16)
    if which=='full': return np.c_[h,v,m]
    if which=='gradients': return np.c_[h,v]
    if which=='means': return m
    if which=='raw': return x
    raise ValueError(which)
def fit(x,y,idx,method,s):
    base=x[idx]/16; ys=y[idx]; xs=[base]
    for j,(kind,lvl) in enumerate(TRAIN_AUG):
        xs.append(corrupt(x[idx],lvl if kind=='noise' else 0,lvl if kind=='dropout' else 0,s*100+j+701))
    ax=np.concatenate(xs); ay=np.tile(ys,len(xs)); z=features(ax,method)
    sc=StandardScaler().fit(z); model=SVC(kernel='rbf',C=10,gamma='scale',random_state=s)
    model.fit(sc.transform(z),ay)
    return sc,model
def run(mode,out):
    if mode not in ('validation','final'): raise ValueError(mode)
    if mode=='final' and out.exists(): raise FileExistsError(f'{out} exists; choose a new output path')
    start=time.perf_counter(); data=load_digits(); x=data.data.astype('float32'); y=data.target.astype('int64'); rows=[]; hashes={}; runtimes={}
    # All seeds and variants complete in validation before final mode is permitted by the preregistered workflow.
    for s in SEEDS:
        sp=split(y,s); hashes[str(s)]={k:digest(v) for k,v in sp.items()}; ev=sp['validation' if mode=='validation' else 'test']
        for meth in ('full','raw','gradients','means'):
            t=time.perf_counter(); sc,model=fit(x,y,sp['train'],meth,s); runtimes[f'{s}:{meth}']=time.perf_counter()-t
            for j,(name,n,d) in enumerate(EVAL):
                xx=corrupt(x[ev],n,d,s*10000+j+9901)
                pred=model.predict(sc.transform(features(xx,meth)))
                rows.append(dict(seed=s,method=meth,condition=name,accuracy=float(accuracy_score(y[ev],pred))))
    payload=dict(config=dict(mode=mode,seeds=SEEDS,train_augmentations=TRAIN_AUG,eval_conditions=EVAL,train_only_scaler=True,split='stratified 60/20/20',features='raw 64 or 56 horizontal differences + 56 vertical differences + 16 non-overlap 2x2 means',cpu_threads=1,sklearn=__import__('sklearn').__version__),split_hashes=hashes,fit_seconds=runtimes,results=rows,elapsed_seconds=time.perf_counter()-start)
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,indent=2)); print(json.dumps({'mode':mode,'rows':len(rows),'elapsed_seconds':payload['elapsed_seconds']})); return payload
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['validation','final']);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with threadpool_limits(limits=1): run(a.mode,a.output)
