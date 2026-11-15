from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

SEEDS=(137,271,389)
LEVELS=(0.0,0.10,0.30,0.50)


def sha(b): return hashlib.sha256(np.ascontiguousarray(b).tobytes()).hexdigest()
def split_data(seed):
    x,y=make_classification(n_samples=5000,n_features=20,n_informative=5,n_redundant=5,weights=[.8,.2],class_sep=1.0,random_state=seed)
    idx=np.arange(len(y)); tr,rest=train_test_split(idx,test_size=.4,random_state=seed,stratify=y)
    va,te=train_test_split(rest,test_size=.5,random_state=seed+1,stratify=y[rest])
    return x.astype(np.float32),y.astype(np.int64),{'train':np.sort(tr),'validation':np.sort(va),'test':np.sort(te)}
def mask(x,level,seed):
    z=x.copy()
    if level: z[np.random.default_rng(seed).random(z.shape)<level]=np.nan
    return z
def encode(x,med,weights):
    m=np.isnan(x); z=np.where(m,med[None,:],x)
    return np.concatenate([z,m.astype(np.float32)*weights[None,:]],axis=1)
def mi_weights(x,y):
    v=np.nan_to_num(x,nan=np.nanmedian(x,axis=0))
    mi=mutual_info_classif(v,y,random_state=42,n_jobs=1)
    return (mi/(mi.mean()+1e-12)).astype(np.float32)
def train(seed,variant,elevated,idx,x,y):
    tx=x[idx]; ty=y[idx]; med=np.nanmedian(tx,axis=0); mi=mi_weights(tx,ty)
    # Matched augmentation draws are fixed by seed, independent of candidate variant.
    if variant=='baseline':
        xx=tx; yy=ty; weights=np.zeros(tx.shape[1],dtype=np.float32)
    else:
        rng=np.random.default_rng(seed+7000)
        # Three cumulative curriculum phases: clean, 10% (observed training regime), elevated.
        levels=[0.0,0.10,elevated] if variant!='no_curriculum' else [elevated]
        replicas=[tx]
        for phase,rate in enumerate(levels[1:] if len(levels)>1 else levels):
            replicas.append(mask(tx,rate,seed+7000+phase))
        xx=np.concatenate(replicas); yy=np.tile(ty,len(replicas))
        weights=(mi if variant=='weighted' else np.ones_like(mi))
    model=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,min_samples_leaf=20,l2_regularization=1.0,early_stopping=False,random_state=seed)
    model.fit(encode(xx,med,weights),yy)
    return model,med,weights

def run(mode,out,seedset=SEEDS):
    if mode not in ('validation','final'): raise ValueError(mode)
    start=time.perf_counter(); rows=[]; hashes={}
    for seed in seedset:
        x,y,sp=split_data(seed); ev=sp['validation' if mode=='validation' else 'test']
        hashes[str(seed)]={k:sha(v) for k,v in sp.items()}
        for variant in ('baseline','weighted','uniform','no_curriculum'):
            # Candidate rate is selected from validation only. The final mode fixes it at 0.30.
            rates=(0.30,0.50) if mode=='validation' and variant!='baseline' else ((0.30,) if variant!='baseline' else (0.30,))
            for elevated in rates:
                t0=time.perf_counter(); model,med,w=train(seed,variant,elevated,sp['train'],x,y); train_s=time.perf_counter()-t0
                for level in LEVELS:
                    masked=mask(x[ev],level,seed+100000+int(level*1000))
                    t1=time.perf_counter(); pred=model.predict_proba(encode(masked,med,w))[:,1]; infer=time.perf_counter()-t1
                    rows.append({'seed':seed,'variant':variant,'elevated_rate':elevated,'test_missing_rate':level,'average_precision':float(average_precision_score(y[ev],pred)),'train_seconds':train_s,'inference_seconds':infer})
    payload={'mode':mode,'seeds':list(seedset),'conditions':list(LEVELS),'test_evaluated':mode=='final','split_hashes':hashes,'results':rows,'elapsed_seconds':time.perf_counter()-start,'sklearn':__import__('sklearn').__version__}
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists() and mode=='final': raise FileExistsError(f'{out} exists; choose another output path (frozen result retained)')
    out.write_text(json.dumps(payload,indent=2))
    return payload
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['validation','final']);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with threadpool_limits(limits=1): r=run(a.mode,a.output)
    print(json.dumps({'mode':r['mode'],'rows':len(r['results']),'elapsed_seconds':r['elapsed_seconds']}))
