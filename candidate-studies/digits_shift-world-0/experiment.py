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

SEEDS=(137,271,389)
NOISE=(0.0,0.2,0.4,0.6); DROP=(0.0,0.05,0.15,0.30)
JOINT=((0.4,0.15),(0.6,0.30))
GRID=((1.0,"scale"),(10.0,"scale"),(10.0,0.01))

def h(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def split(y,s):
    ix=np.arange(len(y)); tr,r=train_test_split(ix,test_size=.4,random_state=s,stratify=y)
    va,te=train_test_split(r,test_size=.5,random_state=s+1,stratify=y[r])
    return {k:np.sort(v) for k,v in [('train',tr),('validation',va),('test',te)]}
def corrupt(x,n,d,rng):
    z=x.copy()
    if n: z+=rng.normal(0,n*16,size=z.shape).astype(np.float32)
    if d: z[rng.random(z.shape)<d]=0
    return np.clip(z,0,16)
def conditions(x,s):
    out=[]
    for n in NOISE:
        out.append((f'noise_{n}', corrupt(x,n,0,np.random.default_rng(s+1000+int(n*1000)))))
    for d in DROP:
        out.append((f'dropout_{d}',corrupt(x,0,d,np.random.default_rng(s+3000+int(d*1000)))))
    for n,d in JOINT:
        out.append((f'joint_{n}_{d}',corrupt(x,n,d,np.random.default_rng(s+5000+int(n*1000)+int(d*1000)))))
    return out
def reliability(x,seed):
    draws=[]
    # Ten independently corrupted copies per training image at each declared severity.
    for n in NOISE[1:]:
        for k in range(10): draws.append(corrupt(x,n,0,np.random.default_rng(seed+10000+int(n*1000)*10+k)))
    for d in DROP[1:]:
        for k in range(10): draws.append(corrupt(x,0,d,np.random.default_rng(seed+20000+int(d*1000)*10+k)))
    stack=np.stack(draws)
    var=stack.var(axis=(0,1)); mean=stack.mean(axis=(0,1))
    w=1/(1+var)
    collapsed=(mean<2)|(mean>14)
    return w,collapsed,var,mean
def augmented(x,y,seed,which,collapsed):
    rng=np.random.default_rng(seed+33000)
    copies=[]
    for n,d in ((.2,0),(0,.15)):
        z=x.copy()
        noise=rng.normal(0,n*16,size=z.shape).astype(np.float32)
        mask=rng.random(z.shape)<d
        if which=='candidate':
            noise[:,collapsed]=0; mask[:,collapsed]=False
        z=np.clip(z+noise,0,16); z[mask]=0
        copies.append(z)
    return np.concatenate([x,*copies]),np.tile(y,3)
def transform_fit(x,y,w,mask,scale_weight):
    z=x.copy()
    if scale_weight: z*=w
    z[:,mask]=0
    sc=StandardScaler().fit(z)
    return sc.transform(z),sc
def model_fit(x,y,params):
    with threadpool_limits(limits=1):
        m=SVC(kernel='rbf',C=params[0],gamma=params[1])
        t=time.perf_counter(); m.fit(x,y); sec=time.perf_counter()-t
    return m,sec
def evaluate(m,sc,x,y,s,w,mask,weighted):
    rows=[]; preds=[]
    with threadpool_limits(limits=1):
        t=time.perf_counter()
        for name,z in conditions(x,s):
            z=z.copy()
            if weighted: z*=w
            z[:,mask]=0
            p=m.predict(sc.transform(z)); val=float(accuracy_score(y,p)); rows.append({'condition':name,'accuracy':val}); preds.append(p)
        secs=time.perf_counter()-t
    return rows,secs
def run(mode,out):
    if mode not in ('validation','final'): raise ValueError(mode)
    if mode=='final' and not (out.parent/'preregistration.json').exists(): raise RuntimeError('final requires preregistration.json')
    started=time.perf_counter(); rows=[]; hashes={}; runtimes=[]
    data=load_digits(); X=data.data.astype(np.float32); Y=data.target.astype(np.int64)
    # Full conditions comprise clean, three noise, three dropout and two joint draws.
    for seed in SEEDS:
        sp=split(Y,seed); hashes[str(seed)]={k:h(v) for k,v in sp.items()}
        tr,ev=sp['train'],sp['validation' if mode=='validation' else 'test']
        x,y=X[tr],Y[tr]; ex,ey=X[ev],Y[ev]
        w,col,var,mean=reliability(x,seed)
        # Control is ordinary all-pixel corruption augmentation. Candidate uses same draws except
        # corruption is suppressed on the proposal's collapsed pixels.
        methods={'control':(False,np.zeros(64,bool),'control'),
                 'variance_weight_only':(True,np.zeros(64,bool),'control'),
                 'collapsed_mask_only':(False,col,'control'),
                 'candidate':(True,col,'candidate')}
        for name,(weighted,mask,augkind) in methods.items():
            ax,ay=augmented(x,y,seed,augkind,col)
            if name in ('control','variance_weight_only'):
                # control and variance-only augment all features
                pass
            if weighted: ax=ax*w
            ax[:,mask]=0
            sc=StandardScaler().fit(ax); ax=sc.transform(ax)
            for params in GRID:
                m,fitsec=model_fit(ax,ay,params)
                met,predsec=evaluate(m,sc,ex,ey,seed+60000,w,mask,weighted)
                score=float(np.mean([r['accuracy'] for r in met]))
                rows.append({'seed':seed,'method':name,'C':params[0],'gamma':params[1], 'mean_accuracy':score,'conditions':met,'fit_seconds':fitsec,'inference_seconds':predsec})
                runtimes.append(fitsec+predsec)
        # Reliability diagnostic uses only train corruption draws and held-out instability as descriptive validation data.
        unstable=np.var(np.stack([corrupt(ex,n,0,np.random.default_rng(seed+70000+int(n*1000)+k)) for k,n in enumerate((.2,.4,.6))]),axis=(0,1))
        rho=float(np.corrcoef(var,unstable)[0,1])
        rows.append({'seed':seed,'diagnostic':'train_variance_vs_validation_noise_variance_correlation','value':rho,'collapsed_pixels':int(col.sum())})
    payload={'mode':mode,'seeds':list(SEEDS),'conditions':{'noise':NOISE,'dropout':DROP,'joint':JOINT},'split_hashes':hashes,'results':rows,'elapsed_seconds':time.perf_counter()-started}
    out.parent.mkdir(parents=True,exist_ok=True)
    if mode=='final' and out.exists(): raise RuntimeError('refusing overwrite frozen held-out result')
    out.write_text(json.dumps(payload,indent=2))
    print(json.dumps({'mode':mode,'rows':len(rows),'elapsed_seconds':payload['elapsed_seconds']}))
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['validation','final']); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); run(a.mode,a.output)
