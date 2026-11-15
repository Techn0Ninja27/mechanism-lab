#!/usr/bin/env python3
"""Faithful test of corruption-variance weighted PCA on sklearn digits."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from threadpoolctl import threadpool_limits

SEEDS=(137,271,389)
CONDITIONS=((0.,0.),(.2,0.),(.4,0.),(.6,0.),(0.,.05),(0.,.15),(0.,.30),(.4,.15),(.6,.30))
GRID=((1.,'scale'),(10.,'scale'),(100.,'scale'),(10.,.01),(10.,.1))

def h(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def corrupt(x,n,d,seed):
 r=np.random.default_rng(seed); z=x.copy()
 if n: z += r.normal(0,n*16.,z.shape).astype(np.float32)
 if d: z[r.random(z.shape)<d]=0.
 return np.clip(z,0,16).astype(np.float32)
def splits(y,s):
 ix=np.arange(len(y)); tr,rest=train_test_split(ix,test_size=.4,random_state=s,stratify=y)
 va,te=train_test_split(rest,test_size=.5,random_state=s+1,stratify=y[rest])
 return {k:np.sort(v) for k,v in [('train',tr),('validation',va),('test',te)]}
def make_aug(x,y,s):
 xs=[x]; ys=[y]
 # Five matched train-only copies spanning proposal's low/high noise and dropout.
 for j,(n,d) in enumerate(((.2,0),(.4,0),(.6,0),(0,.15),(0,.30))):
  xs.append(corrupt(x,n,d,s+1000+j)); ys.append(y)
 return np.concatenate(xs),np.concatenate(ys)
def pixel_weights(x,s):
 # Each pixel's sample variance across clean and 5 corrupted copies, averaged over examples.
 copies=[x]
 for j,(n,d) in enumerate(((.2,0),(.4,0),(.6,0),(0,.15),(0,.30))): copies.append(corrupt(x,n,d,s+1000+j))
 a=np.stack(copies,axis=1)
 v=np.var(a,axis=1).mean(axis=0)
 return 1./np.maximum(v,1e-8),v

def build(x,y,w,weighted=True,standardize=True):
 scale=w if weighted else np.ones(x.shape[1],dtype=np.float64)
 xw=x*scale
 ss=StandardScaler() if standardize else None
 if ss is not None: xw=ss.fit_transform(xw)
 p=PCA(n_components=32,svd_solver='full'); z=p.fit_transform(xw)
 return ss,p,scale

def transform(x,ss,p,w):
 z=x*w
 if ss is not None: z=ss.transform(z)
 return p.transform(z)
def choose(x,y,w,standardize=True):
 ss,p,scale=build(x,y,w,standardize=standardize)
 z=transform(x,ss,p,scale)
 best=None
 for c,g in GRID:
  m=SVC(C=c,gamma=g,kernel='rbf'); m.fit(z,y)
  # validation picks are computed by caller
  if best is None: best=(m,c,g,ss,p,scale)
 return best

def run(mode,out):
 t0=time.perf_counter(); X=load_digits().data.astype(np.float32); Y=load_digits().target.astype(np.int64)
 rows=[]; hashes={}; diagnostics=[]
 for s in SEEDS:
  sp=splits(Y,s); hashes[str(s)]={k:h(v) for k,v in sp.items()}
  tr,va,te=(sp[k] for k in ('train','validation','test'))
  xa,ya=make_aug(X[tr],Y[tr],s); w,var=pixel_weights(X[tr],s)
  # Candidate is exactly inverse corruption variance -> scaling -> StandardScaler -> PCA.
  # Uniform baseline differs only in weights; both use identical augmentation and PCA rank.
  for method,ww,std in [('weighted',w,True),('uniform',np.ones_like(w),True),('weighted_no_standardize',w,False)]:
   ss,p,sc=build(xa,ya,ww,standardize=std)
   ztr=transform(xa,ss,p,sc); zv=transform(X[va],ss,p,sc)
   best=None
   for c,g in GRID:
    m=SVC(C=c,gamma=g,kernel='rbf'); m.fit(ztr,ya)
    score=float(accuracy_score(Y[va],m.predict(zv)))
    if best is None or score>best[0]: best=(score,c,g,m)
   _,c,g,m=best
   evalidx=va if mode=='validation' else te
   ze=transform(X[evalidx],ss,p,sc)
   for ci,(n,d) in enumerate(CONDITIONS):
    zc=transform(corrupt(X[evalidx],n,d,s+20000+ci),ss,p,sc)
    rows.append({'seed':s,'method':method,'condition':{'noise':n,'dropout':d},'accuracy':float(accuracy_score(Y[evalidx],m.predict(zc))),'chosen_C':c,'chosen_gamma':g})
   diagnostics.append({'seed':s,'method':method,'validation_accuracy':best[0],'weights_min':float(np.min(ww)),'weights_max':float(np.max(ww)),'variance_min':float(var.min()),'variance_max':float(var.max()),'pca_components_sha':h(p.components_)})
 payload={'mode':mode,'seeds':SEEDS,'conditions':CONDITIONS,'grid':GRID,'test_evaluated':mode=='final','split_hashes':hashes,'rows':rows,'diagnostics':diagnostics,'elapsed_seconds':time.perf_counter()-t0}
 out.parent.mkdir(parents=True,exist_ok=True)
 if out.exists() and mode=='final': raise FileExistsError(f'{out} exists; refusing to overwrite frozen held-out results')
 out.write_text(json.dumps(payload,indent=2))
 print(json.dumps({'mode':mode,'rows':len(rows),'elapsed_seconds':payload['elapsed_seconds']}))
if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('mode',choices=['validation','final']); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
 with threadpool_limits(limits=1): run(a.mode,a.output)
