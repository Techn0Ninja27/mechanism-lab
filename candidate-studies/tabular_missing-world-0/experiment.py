"""Faithful literal-formula study for tabular_missing-world-0."""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

SEEDS = (137, 271, 389)
LEVELS = (0.0, 0.10, 0.30, 0.50)
ALPHAS = (0.0, 0.3, 0.7, 1.0)
MAX_ITER = 120

def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def arrhash(a): return sha_bytes(np.ascontiguousarray(a).tobytes())

def data(seed):
    x,y=make_classification(n_samples=5000,n_features=20,n_informative=5,n_redundant=5,
        weights=[.8,.2],class_sep=1.0,random_state=seed)
    x=x.astype(np.float32); y=y.astype(np.int64)
    idx=np.arange(len(y)); tr,rest=train_test_split(idx,test_size=.4,random_state=seed,stratify=y)
    va,te=train_test_split(rest,test_size=.5,random_state=seed+1,stratify=y[rest])
    return x,y,{"train":np.sort(tr),"validation":np.sort(va),"test":np.sort(te)}

def corrupt(x, level, seed):
    if level==0: return x.copy(),np.zeros(x.shape,dtype=bool)
    m=np.random.default_rng(seed).random(x.shape)<level
    z=x.copy(); z[m]=np.nan
    return z,m

def literal_transform(x, mask, means, alpha):
    # c_i = v_i*(1-m_i) + alpha*m_i*mu_i. Missing values become alpha*mean.
    out=np.where(mask, alpha*means[None,:], x).astype(np.float32)
    return np.concatenate([out,mask.astype(np.float32)],axis=1)

def fit_predictors(xtr,ytr,seed):
    # All arms use the same HGB capacity and clean training rows. As specified,
    # no training-time missingness augmentation is introduced.
    model=HistGradientBoostingClassifier(random_state=seed,max_iter=MAX_ITER)
    model.fit(xtr,ytr)
    return model

def run(mode, output, allow_rerun=False):
    if output.exists() and not allow_rerun: raise FileExistsError(f"refusing to overwrite {output}; pass --allow-rerun")
    started=time.perf_counter(); rows=[]; split_hashes={}; runtime={}
    for seed in SEEDS:
        x,y,s=data(seed); split_hashes[str(seed)]={k:arrhash(v) for k,v in s.items()}
        fit=x[s['train']]; yy=y[s['train']]
        means=fit.mean(axis=0)
        med=SimpleImputer(strategy='median').fit(fit)
        # Arms: declared baseline; equation candidate for each alpha; nearest
        # simpler mean-imputation+mask comparator (ablation of blending).
        arm_models={}; arm_models['median_hgb_no_mask']=fit_predictors(med.transform(fit),yy,seed)
        arm_models['mean_impute_plus_mask']=fit_predictors(np.concatenate([fit,np.zeros_like(fit)],axis=1),yy,seed)
        for a in ALPHAS:
            arm_models[f'literal_alpha_{a:g}']=fit_predictors(literal_transform(fit,np.zeros_like(fit,dtype=bool),means,a),yy,seed)
        runtime[str(seed)]={k:None for k in arm_models}
        target=s['validation'] if mode=='validation' else s['test']
        for level in LEVELS:
            xx,mask=corrupt(x[target],level,seed+50000+int(level*1000))
            for name,model in arm_models.items():
                t=time.perf_counter()
                if name=='median_hgb_no_mask': z=med.transform(np.nan_to_num(xx,nan=np.nan))
                elif name=='mean_impute_plus_mask':
                    z=np.where(mask,means[None,:],xx).astype(np.float32)
                    z=np.concatenate([z,mask.astype(np.float32)],axis=1)
                else: z=literal_transform(xx,mask,means,float(name.rsplit('_',1)[1]))
                pred=model.predict_proba(z)[:,1]
                score=float(average_precision_score(y[target],pred))
                rows.append({'seed':seed,'method':name,'missing_rate':level,'average_precision':score})
                runtime[str(seed)][name]=(runtime[str(seed)][name] or 0)+time.perf_counter()-t
    # aggregate conditions equally within seed; selection is per-seed on validation only
    by={}
    for r in rows: by.setdefault((r['seed'],r['method']),[]).append(r['average_precision'])
    aggregates=[{'seed':s,'method':m,'mean_ap':float(np.mean(vals))} for (s,m),vals in by.items()]
    selected={}
    if mode=='validation':
        for seed in SEEDS:
            options=[(next(a for a in ALPHAS if m==f'literal_alpha_{a:g}'),float(np.mean(v))) for (s,m),v in by.items() if s==seed and m.startswith('literal_alpha_')]
            selected[str(seed)]=sorted(options,key=lambda av:(-av[1],av[0]))[0][0]
    else:
        selected=json.loads((Path(__file__).parent/'validation.json').read_text())['selected_alpha_by_seed']
    payload={'mode':mode,'seeds':SEEDS,'missing_levels':LEVELS,'rows':rows,'aggregates':aggregates,
        'selected_alpha_by_seed':selected,'split_hashes':split_hashes,'runtime_seconds':runtime,
        'elapsed_seconds':time.perf_counter()-started,'training_rows':3000,'model':{'class':'HistGradientBoostingClassifier','max_iter':MAX_ITER},
        'training_augmentation':False,'test_evaluated':mode=='final'}
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(payload,indent=2))
    return payload

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['validation','final']); p.add_argument('--output',type=Path,required=True); p.add_argument('--allow-rerun',action='store_true'); a=p.parse_args()
    r=run(a.mode,a.output,a.allow_rerun); print(json.dumps({'mode':a.mode,'rows':len(r['rows']),'seconds':r['elapsed_seconds']}))
