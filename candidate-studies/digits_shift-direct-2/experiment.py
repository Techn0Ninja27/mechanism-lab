from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

SEEDS=(137,271,389)
CONDITIONS=((0.,0.),(.2,0.),(.4,0.),(.6,0.),(0.,.05),(0.,.15),(0.,.30),(.4,.15),(.6,.30))
# Both augmented copies are fixed in advance: one noise and one dropout draw.
AUGMENT=((.4,0.),(0.,.15))
C=10.0

def sha_file(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sha_arr(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def splits(y, seed):
    ids=np.arange(len(y)); tr,rem=train_test_split(ids,test_size=.4,random_state=seed,stratify=y)
    va,te=train_test_split(rem,test_size=.5,random_state=seed+1,stratify=y[rem])
    return {"train":np.sort(tr),"validation":np.sort(va),"test":np.sort(te)}
def corrupt(x, noise, drop, seed):
    rng=np.random.default_rng(seed); z=x.copy()
    if noise: z += rng.normal(0,noise,size=z.shape).astype(np.float32)
    if drop: z[rng.random(z.shape)<drop]=0
    return np.clip(z,0,1)
def fit(x,y,idx,kind,seed,ablation=None,eta=.4,p=.15):
    raw=x[idx]/16.; yy=y[idx]
    mu=raw.mean(0); var=raw.var(0)
    if kind in ("standard_aug","standard_noaug"):
        den=np.sqrt(var); den[den<1e-12]=1
    elif kind.startswith("envelope"):
        add_noise=0 if ablation=="drop_noise_term" else eta**2
        drop_term=0 if ablation=="drop_dropout_term" else p*(1-p)*mu**2
        den=np.sqrt(var+add_noise+drop_term)
        den=np.maximum(den,1e-8)
    else: raise ValueError(kind)
    def scale(a): return (a-mu)/den
    if kind.endswith("noaug") or ablation=="no_augmentation":
        tx=raw; ty=yy
    else:
        aug=[corrupt(raw,n,p,seed+30000+j) for j,(n,p) in enumerate(AUGMENT)]
        tx=np.concatenate([raw,*aug]); ty=np.tile(yy,3)
    model=SVC(C=C,kernel="rbf",gamma="scale",cache_size=256)
    t=time.perf_counter()
    with threadpool_limits(limits=1): model.fit(scale(tx),ty)
    train_s=time.perf_counter()-t
    return (model,scale,train_s)
def eval_one(fitobj,x,y,idx,seed):
    model,scale,_=fitobj; rows=[]; t0=time.perf_counter()
    for j,(n,p) in enumerate(CONDITIONS):
        z=corrupt(x[idx]/16.,n,p,seed+50000+j)
        pred=model.predict(scale(z)); rows.append({"noise":n,"dropout":p,"accuracy":float(accuracy_score(y[idx],pred))})
    return rows,time.perf_counter()-t0

def run(mode,out,force=False):
    if mode not in ("validation","final"): raise ValueError(mode)
    out=Path(out)
    if out.exists() and not force: raise FileExistsError(f"{out} exists; pass --force for explicit rerun")
    X=load_digits().data.astype(np.float32); y=load_digits().target.astype(np.int64)
    methods=("standard_aug","envelope_aug","envelope_noaug","envelope_drop_noise","envelope_drop_dropout")
    allrows=[]; hashes={}
    for seed in SEEDS:
        sp=splits(y,seed); hashes[str(seed)]={k:sha_arr(v) for k,v in sp.items()}
        idx=sp["validation" if mode=="validation" else "test"]
        std=fit(X,y,sp["train"],"standard_aug",seed)
        std_scores,std_inf=eval_one(std,X,y,idx,seed)
        allrows.append({"seed":seed,"method":"standard_aug","conditions":std_scores,"mean_accuracy":float(np.mean([r["accuracy"] for r in std_scores])),"train_seconds":std[2],"inference_seconds":std_inf})
        for name in methods[1:]:
            abl={"envelope_drop_noise":"drop_noise_term","envelope_drop_dropout":"drop_dropout_term"}.get(name)
            scores=[]; train_s=0.; inf_s=0.
            for j,(noise,drop) in enumerate(CONDITIONS):
                kind="envelope_noaug" if name=="envelope_noaug" else "envelope_aug"
                obj=fit(X,y,sp["train"],kind,seed,abl,noise,drop)
                train_s+=obj[2]; t0=time.perf_counter()
                z=corrupt(X[idx]/16.,noise,drop,seed+50000+j)
                pred=obj[0].predict(obj[1](z)); inf_s+=time.perf_counter()-t0
                scores.append({"noise":noise,"dropout":drop,"accuracy":float(accuracy_score(y[idx],pred))})
            allrows.append({"seed":seed,"method":name,"conditions":scores,"mean_accuracy":float(np.mean([r["accuracy"] for r in scores])),"train_seconds":train_s,"inference_seconds":inf_s})
    payload={"mode":mode,"test_evaluated":mode=="final","seeds":list(SEEDS),"conditions":CONDITIONS,"augmentation":AUGMENT,"split_hashes":hashes,"script_sha256":sha_file(__file__),"results":allrows}
    out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,indent=2))
    print(json.dumps({"mode":mode,"rows":len(allrows),"output":str(out)}))

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("mode",choices=("validation","final")); ap.add_argument("--output",required=True); ap.add_argument("--force",action="store_true"); a=ap.parse_args(); run(a.mode,a.output,a.force)
