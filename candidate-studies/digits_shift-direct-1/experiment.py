from __future__ import annotations
import argparse, hashlib, json, platform, sys, time
from pathlib import Path
import numpy as np
import scipy, scipy.ndimage
import sklearn
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEEDS=(137,271,389)
CONDITIONS=(('clean',0.0,0.0),('noise_0.2',0.2,0.0),('noise_0.4',0.4,0.0),('noise_0.6',0.6,0.0),('dropout_0.05',0.0,0.05),('dropout_0.15',0.0,0.15),('dropout_0.30',0.0,0.30),('joint_0.4_0.15',0.4,0.15),('joint_0.6_0.30',0.6,0.30))
VARIANTS=('baseline','blur_only','augmentation_only','blur_plus_augmentation')

def hbytes(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def hfile(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def split(y,seed):
    ix=np.arange(len(y)); tr,rest=train_test_split(ix,test_size=.4,random_state=seed,stratify=y)
    va,te=train_test_split(rest,test_size=.5,random_state=seed+1,stratify=y[rest])
    return {'train':np.sort(tr),'validation':np.sort(va),'test':np.sort(te)}
def smooth(x): return scipy.ndimage.gaussian_filter(x.reshape(-1,8,8),sigma=(0,.8,.8),mode='reflect').reshape(-1,64).astype(np.float32)
def corrupt(x,noise,drop,seed):
    rng=np.random.default_rng(seed); z=x.astype(np.float32).copy()
    if noise: z+=rng.normal(0,noise*16.0,z.shape).astype(np.float32)
    if drop: z[rng.random(z.shape)<drop]=0.0
    return np.clip(z,0,16).astype(np.float32)
def augment(x,y,seed):
    rng=np.random.default_rng(seed)
    n=np.clip(x+rng.normal(0,.15*16.0,x.shape),0,16).astype(np.float32)
    d=x.copy(); d[rng.random(d.shape)<.15]=0
    return np.concatenate([x,n,d]),np.tile(y,3)
def fit(x,y,idx,variant,seed):
    a=x[idx]; b=y[idx]
    if variant in ('blur_only','blur_plus_augmentation'): a=smooth(a)
    # The scaler sees clean training samples only, matching the proposal.
    scaler=StandardScaler().fit(a)
    if variant in ('augmentation_only','blur_plus_augmentation'): a,b=augment(a,b,seed+202)
    model=SVC(kernel='rbf',C=10,gamma='scale')
    model.fit(scaler.transform(a),b)
    return scaler,model

def run(mode,out):
    if mode not in ('validation','final'): raise ValueError(mode)
    if mode=='final' and Path(out).exists(): raise FileExistsError(f'{out} exists; choose a new output path')
    started=time.perf_counter(); rows=[]; split_hashes={}; runtimes=[]
    data=load_digits(); x=data.data.astype(np.float32); y=data.target.astype(np.int64)
    for seed in SEEDS:
        sp=split(y,seed); split_hashes[str(seed)]={k:hbytes(v) for k,v in sp.items()}
        ev=sp['validation'] if mode=='validation' else sp['test']
        # Precompute identical corruption arrays per seed and condition for all variants.
        corruptions={}
        for ci,(name,n,d) in enumerate(CONDITIONS):
            corruptions[name]=corrupt(x[ev],n,d,seed+50000+ci)
        for variant in VARIANTS:
            t=time.perf_counter(); scaler,model=fit(x,y,sp['train'],variant,seed); train_s=time.perf_counter()-t
            inf=0
            for name,n,d in CONDITIONS:
                z=corruptions[name]; z=smooth(z) if variant in ('blur_only','blur_plus_augmentation') else z
                q=time.perf_counter(); pred=model.predict(scaler.transform(z)); inf+=time.perf_counter()-q
                rows.append({'seed':seed,'variant':variant,'condition':name,'accuracy':float(accuracy_score(y[ev],pred)),'train_seconds':train_s,'inference_seconds':inf})
            runtimes.append({'seed':seed,'variant':variant,'train_seconds':train_s,'inference_seconds_all_conditions':inf})
    payload={'config':{'mode':mode,'seeds':SEEDS,'conditions':[{'name':n,'noise':a,'dropout':b} for n,a,b in CONDITIONS],'augmentation_noise_sd':.15*16,'augmentation_dropout':.15,'blur_sigma':.8,'classifier':'SVC RBF C=10 gamma=scale','split':'stratified 60/20/20; rest split seed+1','fit_preprocessing_on_training_only':True,'test_evaluated':mode=='final','python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},'split_hashes':split_hashes,'runtime':runtimes,'results':rows,'elapsed_seconds':time.perf_counter()-started,'source_sha256':hfile(__file__)}
    Path(out).write_text(json.dumps(payload,indent=2)); print(json.dumps({'mode':mode,'rows':len(rows),'elapsed_seconds':payload['elapsed_seconds'],'source_sha256':payload['source_sha256']}))
def main():
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['validation','final']); p.add_argument('--output',required=True); a=p.parse_args(); run(a.mode,a.output)
if __name__=='__main__': main()
