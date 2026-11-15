"""Reproducible study of digits_shift-world-1. Offline only."""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEEDS = (137, 271, 389)
NOISES = (0.0, 0.2, 0.4, 0.6)
DROPOUTS = (0.0, 0.05, 0.15, 0.30)
GRID = [(c, g) for c in (1., 10.) for g in (0.01, 0.1, 'scale')]

def h(x): return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def split(seed):
    d=load_digits(); x=d.data.astype(np.float32); y=d.target.astype(np.int64)
    i=np.arange(len(y)); tr,rest=train_test_split(i,test_size=.4,random_state=seed,stratify=y)
    va,te=train_test_split(rest,test_size=.5,random_state=seed+1,stratify=y[rest])
    return x,y,{k:np.sort(v) for k,v in [('train',tr),('validation',va),('test',te)]}

def corrupt(x, noise, drop, seed):
    rng=np.random.default_rng(seed); z=x.copy()
    if noise: z += rng.normal(0,noise*16.,z.shape).astype(np.float32)
    if drop: z[rng.random(z.shape)<drop]=0.
    return np.clip(z,0,16)

def conditions(x, y, idx, seed):
    vals=[]
    pairs=[(0.,0.),(.2,0.),(.4,0.),(.6,0.),(0.,.05),(0.,.15),(0.,.30),(.4,.15),(.6,.30)]
    for n,d in pairs:
        z=corrupt(x[idx],n,d,seed+int(n*1000)+int(d*10000))
        vals.append((f"noise={n},dropout={d}",z,y[idx]))
    return vals

def training_data(x,y,idx,anchor,seed):
    a=x[idx]; b=y[idx]
    scale=StandardScaler().fit(a); a=scale.transform(a)
    cent=np.stack([a[b==c].mean(0) for c in np.unique(b)])
    xx=[a]; yy=[b]
    # Same augmentation draw for the anchored and unanchored arms.
    for j,(noise,drop) in enumerate(((.15,0),(0,.15))):
        raw=corrupt(x[idx],noise,drop,seed + 3000+j)
        z=scale.transform(raw)
        if anchor:
            alpha=drop/.30 if drop else noise/.60
            z=(1-alpha)*z+alpha*cent[b]
        xx.append(z); yy.append(b)
    return scale,np.concatenate(xx),np.concatenate(yy),a,b

def select(x,y,idx,vidx, score_mode, anchor, seed, cv_cache):
    scale,xx,yy,base_x,base_y=training_data(x,y,idx,anchor,seed)
    valx=scale.transform(x[vidx]); valy=y[vidx]
    rows=[]
    for c,g in GRID:
        # CV is computed on the same train rows for all grid points.
        key=(c,g)
        if key not in cv_cache:
            folds=StratifiedKFold(n_splits=3,shuffle=True,random_state=seed)
            cv_cache[key]=float(cross_val_score(SVC(C=c,gamma=g,kernel='rbf'),base_x,base_y,cv=folds,n_jobs=1).mean())
        cv=cv_cache[key]
        model=SVC(C=c,gamma=g,kernel='rbf').fit(xx,yy)
        va=float(accuracy_score(valy,model.predict(valx)))
        score=va if score_mode=='validation_only' else .5*(cv+va)
        rows.append({'C':c,'gamma':g,'cv_accuracy':cv,'validation_accuracy':va,'selection_score':score})
    best=max(rows,key=lambda r:r['selection_score'])
    return scale, best, rows

def evaluate_model(x,y,indices,scale,model,pca=False):
    outputs=[]
    p=PCA(n_components=32,svd_solver='full').fit(x[indices['train']]) if pca else None
    # PCA switch threshold is fixed at bottom quintile of validation margins.
    if pca:
        vx=scale.transform(x[indices['validation']]); vm=np.sort(np.max(model.decision_function(vx),axis=1))
        threshold=float(vm[max(0,int(.2*len(vm))-1)])
    else: threshold=None
    for cname,z,truth in conditions(x,y,indices['evaluation'],indices['seed']+20000):
        q=scale.transform(z)
        if pca:
            margin=np.max(model.decision_function(q),axis=1); low=margin<=threshold
            if low.any(): q[low]=scale.transform(p.inverse_transform(p.transform(z[low])))
        outputs.append({'condition':cname,'accuracy':float(accuracy_score(truth,model.predict(q)))})
    return outputs

def run(mode,out,force=False):
    if out.exists() and not force: raise FileExistsError(f'{out} exists; pass --force to overwrite')
    if mode not in ('validation','final'): raise ValueError(mode)
    started=time.perf_counter(); allrows=[]; splits={};
    # Each seed is independent. Validation mode evaluates only validation data.
    for seed in SEEDS:
        x,y,s=split(seed); splits[str(seed)]={k:h(v) for k,v in s.items()}
        evalkey='validation' if mode=='validation' else 'test'
        ix={**s,'evaluation':s[evalkey],'seed':seed}
        arms={}; cv_cache={}
        for name,score,anchor,pca in [
            ('control','validation_only',False,False),
            ('composite_only','composite',False,False),
            ('anchor_only','validation_only',True,False),
            ('pca_only','validation_only',False,True),
            ('full','composite',True,True)]:
            train_started=time.perf_counter()
            scale,best,grid=select(x,y,s['train'],s['validation'],score,anchor,seed,cv_cache)
            # Refit selected setting on augmented data, with split-fitted scaler.
            _,xx,yy,_,_=training_data(x,y,s['train'],anchor,seed)
            model=SVC(C=best['C'],gamma=best['gamma'],kernel='rbf').fit(xx,yy)
            train_seconds=time.perf_counter()-train_started
            infer_started=time.perf_counter()
            metrics=evaluate_model(x,y,ix,scale,model,pca)
            infer_seconds=time.perf_counter()-infer_started
            arms[name]={'selected':best,'grid':grid,'metrics':metrics,'training_and_selection_seconds':train_seconds,'inference_seconds':infer_seconds}
        allrows.append({'seed':seed,'arms':arms})
    payload={'mode':mode,'seeds':list(SEEDS),'conditions':{'noise':NOISES,'dropout':DROPOUTS,'noise_scale':16},'splits':splits,'arms':allrows,'elapsed_seconds':time.perf_counter()-started}
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,indent=2)); return payload

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['validation','final']); p.add_argument('--output',type=Path,required=True); p.add_argument('--force',action='store_true'); a=p.parse_args()
    r=run(a.mode,a.output,a.force); print(json.dumps({'mode':r['mode'],'seconds':r['elapsed_seconds'],'seeds':r['seeds']}))
