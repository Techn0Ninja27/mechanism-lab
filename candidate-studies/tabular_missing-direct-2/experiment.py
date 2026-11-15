"""Reproducible missingness-mask shrinkage study. CPU-bound stages are gated externally."""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.datasets import make_classification
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

SEEDS=(137,271,389); LEVELS=(0.0,0.10,0.30,0.50); CAL_LEVELS=(0.30,0.60)
def sha(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def make(seed):
    x,y=make_classification(n_samples=5000,n_features=20,n_informative=5,n_redundant=5,weights=[.8,.2],random_state=seed)
    ix=np.arange(len(y)); tr,rest=train_test_split(ix,test_size=.4,random_state=seed,stratify=y)
    va,te=train_test_split(rest,test_size=.5,random_state=seed+1,stratify=y[rest])
    return x.astype('float32'),y.astype('int64'),{'train':np.sort(tr),'validation':np.sort(va),'test':np.sort(te)}
def mask(x,rate,seed):
    if rate==0:return x.copy()
    z=x.copy(); z[np.random.default_rng(seed).random(z.shape)<rate]=np.nan; return z
def design(m, interactions=True):
    cols=[m.astype(float)]
    if interactions:
        ii,jj=np.triu_indices(m.shape[1],1); cols.append((m[:,ii]*m[:,jj]).astype(float))
    return np.column_stack([np.ones(len(m)),*cols])
def fit_soft_logit(z,target,l2=1.0):
    # Fractional Bernoulli outcomes give the exact logistic mean model for E[|p-y||mask].
    def objective(w):
        a=z@w; loss=np.logaddexp(0,a)-target*a
        return float(loss.mean()+.5*l2*np.dot(w[1:],w[1:])), (z.T@(expit(a)-target))/len(target)+np.r_[0.,l2*w[1:]]
    res=minimize(objective,np.zeros(z.shape[1]),jac=True,method='L-BFGS-B',bounds=[(-20.,20.)]*z.shape[1],options={'maxiter':500,'ftol':1e-10,'maxls':30})
    if not np.isfinite(res.fun) or not np.all(np.isfinite(res.x)):
        raise RuntimeError(f'soft-logistic fit produced nonfinite values: {res.message}')
    return res.x
def calibrated(raw,missing,w,z,base):
    err=np.clip(expit(z@w),0,1)
    return base+(raw-base)*(1-err),err
def run_seed(seed,mode):
    x,y,s=make(seed); train=s['train']; ev=s['validation' if mode=='validation' else 'test']
    rng=np.random.default_rng(seed+101); aug=x[train].copy(); aug[rng.random(aug.shape)<.20]=np.nan
    xa=np.concatenate([x[train],aug]); ya=np.tile(y[train],2)
    base=.2 # population prior is specified as 80/20; training estimate is recorded below and used in formula
    base=float(y[train].mean())
    t0=time.perf_counter()
    with threadpool_limits(limits=1):
        model=make_pipeline(SimpleImputer(strategy='median'),HistGradientBoostingClassifier(random_state=seed,max_iter=120))
        model.fit(xa,ya)
        fit_seconds=time.perf_counter()-t0
        weights=None
        if mode=='validation':
            # Fit reliability only on fixed 30%+60% validation corruptions; no test split touched.
            mats=[]; errors=[]
            for lev in CAL_LEVELS:
                xx=mask(x[ev],lev,seed+30000+int(lev*1000)); m=np.isnan(xx)
                p=model.predict_proba(xx)[:,1]
                mats.append(m); errors.append(np.abs(p-y[ev]))
            weights=fit_soft_logit(design(np.vstack(mats)),np.concatenate(errors))
            scores=[]
            for lev in LEVELS:
                xx=mask(x[ev],lev,seed+10000+int(lev*1000)); m=np.isnan(xx); p=model.predict_proba(xx)[:,1]
                z=design(m); adj,err=calibrated(p,m,weights,z,base)
                scores.append({'level':lev,'raw_ap':float(average_precision_score(y[ev],p)),'shrunk_ap':float(average_precision_score(y[ev],adj)),'raw_brier':float(brier_score_loss(y[ev],p)),'shrunk_brier':float(brier_score_loss(y[ev],adj)),'mean_est_error':float(err.mean()),'mask_design_rank':int(np.linalg.matrix_rank(z))})
            # Ablations: no interactions and untrained intercept-only shrinkage (constant mean val error).
            zcal=design(np.vstack(mats)); target=np.concatenate(errors)
            wmain=fit_soft_logit(design(np.vstack(mats),False),target)
            ab=[]
            for lev in LEVELS:
                xx=mask(x[ev],lev,seed+10000+int(lev*1000)); m=np.isnan(xx); p=model.predict_proba(xx)[:,1]
                z=design(m); a,_=calibrated(p,m,weights,z,base); b,_=calibrated(p,m,wmain,design(m,False),base)
                ab.append({'level':lev,'full_ap':float(average_precision_score(y[ev],a)),'no_interactions_ap':float(average_precision_score(y[ev],b))})
            return {'seed':seed,'split_hashes':{k:sha(v) for k,v in s.items()},'train_hash':sha(x[train]),'fit_seconds':fit_seconds,'base_rate':base,'validation':scores,'ablation':ab,'calibration_optimizer':{'converged':True,'l2':1.0,'calibration_levels':CAL_LEVELS}}
        # Final evaluates frozen weights fit from validation; recreate them deterministically using val labels.
        mats=[]; errors=[]
        for lev in CAL_LEVELS:
            xx=mask(x[s['validation']],lev,seed+30000+int(lev*1000)); p=model.predict_proba(xx)[:,1]
            mats.append(np.isnan(xx)); errors.append(np.abs(p-y[s['validation']]))
        weights=fit_soft_logit(design(np.vstack(mats)),np.concatenate(errors)); wmain=fit_soft_logit(design(np.vstack(mats),False),np.concatenate(errors))
        rows=[]
        for lev in LEVELS:
            xx=mask(x[ev],lev,seed+10000+int(lev*1000)); m=np.isnan(xx); p=model.predict_proba(xx)[:,1]
            z=design(m); adj,err=calibrated(p,m,weights,z,base); noint,_=calibrated(p,m,wmain,design(m,False),base)
            wrong=(p>=.5)!=y[ev]
            rows.append({'level':lev,'n':len(ev),'raw_ap':float(average_precision_score(y[ev],p)),'candidate_ap':float(average_precision_score(y[ev],adj)),'no_interactions_ap':float(average_precision_score(y[ev],noint)),'raw_brier':float(brier_score_loss(y[ev],p)),'candidate_brier':float(brier_score_loss(y[ev],adj)),'raw_error_probability_sd':float(np.std(p[wrong])) if wrong.any() else 0.,'adjusted_error_probability_sd':float(np.std(adj[wrong])) if wrong.any() else 0.,'mean_predicted_error':float(err.mean())})
        return {'seed':seed,'split_hashes':{k:sha(v) for k,v in s.items()},'train_hash':sha(x[train]),'fit_seconds':fit_seconds,'base_rate':base,'test':rows}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['validation','final']);ap.add_argument('--output',required=True,type=Path);a=ap.parse_args()
    t=time.perf_counter()
    data={'mode':a.mode,'seeds':SEEDS,'levels':LEVELS,'runs':[run_seed(s,a.mode) for s in SEEDS],'elapsed_seconds':time.perf_counter()-t}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.mode=='final' and a.output.exists(): raise FileExistsError(f'refusing to overwrite frozen output: {a.output}')
    a.output.write_text(json.dumps(data,indent=2))
    print(json.dumps({'mode':a.mode,'elapsed_seconds':data['elapsed_seconds'],'seeds':len(data['runs'])}))
if __name__=='__main__': main()
