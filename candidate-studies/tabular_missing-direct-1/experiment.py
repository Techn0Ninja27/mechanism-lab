"""Diagnostic faithful reproduction: missingness routing on clean-only training masks."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.cluster import KMeans
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

SEEDS=(137,271,389)
LEVELS=(0.0,0.10,0.30,0.50)
def h(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def make(seed):
    x,y=make_classification(n_samples=5000,n_features=20,n_informative=5,n_redundant=5,weights=[.8,.2],random_state=seed)
    ix=np.arange(len(y)); tr,rest=train_test_split(ix,test_size=.4,random_state=seed,stratify=y)
    va,te=train_test_split(rest,test_size=.5,random_state=seed+1,stratify=y[rest])
    return x.astype('float32'),y.astype('int64'),{'train':np.sort(tr),'validation':np.sort(va),'test':np.sort(te)}
def input_features(x, masks):
    z=x.copy(); z[masks]=np.nan
    imp=SimpleImputer(strategy='median')
    return imp,z

def run(mode, out):
    if mode not in ('validation','final'): raise ValueError(mode)
    allrows=[]; splits={}; elapsed={}
    with threadpool_limits(limits=1):
      for seed in SEEDS:
        t=time.perf_counter(); x,y,s=make(seed); splits[str(seed)]={k:h(v) for k,v in s.items()}
        tr=s['train']; ev=s['validation' if mode=='validation' else 'test']
        # Deterministic half split: half A fits model/prototypes, half B is calibration-only.
        a,b=train_test_split(tr,test_size=.5,random_state=seed+2,stratify=y[tr]);
        # No missing values are specified for training, so every mask vector is zero.
        ma=np.zeros((len(a),x.shape[1]),dtype=bool)
        # Same 40-feature architecture for candidate and control.
        imputer=SimpleImputer(strategy='median').fit(x[a])
        xa=np.concatenate([imputer.transform(x[a]),ma.astype(float)],axis=1)
        xb=np.concatenate([imputer.transform(x[b]),np.zeros((len(b),x.shape[1]))],axis=1)
        model=HistGradientBoostingClassifier(random_state=seed,max_iter=120).fit(xa,y[a])
        calraw=model.predict_proba(xb)[:,1]
        # KMeans is attempted exactly on the clean-only observed masks; collapse is recorded.
        km=KMeans(n_clusters=8,random_state=seed,n_init=10).fit(ma.astype(float))
        uniq=np.unique(km.labels_); cent=km.cluster_centers_
        # Isotonic fit for clusters with >=10 positives; only one occupied cluster exists here.
        cal={}
        for cid in uniq:
            loc=km.labels_==cid
            if int(y[b][loc].sum())>=10:
                cal[int(cid)]=IsotonicRegression(out_of_bounds='clip').fit(calraw[loc],y[b][loc])
        if not cal: raise RuntimeError('No eligible calibration cluster')
        # Empty cluster centroids duplicate the zero prototype, so deterministic argmin sends all rows to first tied centroid.
        control=HistGradientBoostingClassifier(random_state=seed,max_iter=120).fit(xa,y[a])
        for level in LEVELS:
            rng=np.random.default_rng(seed+10000+int(level*1000))
            mask=rng.random((len(ev),x.shape[1]))<level if level else np.zeros((len(ev),x.shape[1]),bool)
            z=x[ev].copy(); z[mask]=np.nan
            # fit transform solely on training A
            feat=np.concatenate([imputer.transform(z),mask.astype(float)],axis=1)
            raw=control.predict_proba(feat)[:,1]
            # nearest Hamming prototype routing; tie uses lowest cluster ID, matching argmin
            d=np.sum(mask[:,None,:] != cent[None,:,:],axis=2); route=np.argmin(d,axis=1)
            pred=raw.copy()
            for cid,iso in cal.items():
                m=route==cid
                if m.any(): pred[m]=iso.predict(raw[m])
            ap0=float(average_precision_score(y[ev],raw)); ap1=float(average_precision_score(y[ev],pred))
            allrows.append({'seed':seed,'condition':{'missing':level},'n_eval':len(ev),'prototype_unique':len(uniq),'occupied_calibration_clusters':len(cal),'control_ap':ap0,'candidate_ap':ap1,'delta_ap':ap1-ap0})
        elapsed[str(seed)]=time.perf_counter()-t
    payload={'config':{'mode':mode,'seeds':SEEDS,'levels':LEVELS,'dataset':'make_classification 5000x20 (5 informative, 5 redundant, weights .8/.2); 60/20/20 stratified','clean_only_training_masks':True,'same_corruption_draws':True,'training_threads':1,'test_evaluated':mode=='final'},'splits':splits,'results':allrows,'elapsed_seconds_by_seed':elapsed}
    p=Path(out)
    if mode=='final' and p.exists(): raise FileExistsError(f'{p} exists; refusing overwrite')
    p.write_text(json.dumps(payload,indent=2))
    return payload
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('mode',choices=['validation','final']); ap.add_argument('--output',required=True); a=ap.parse_args(); d=run(a.mode,a.output); print(json.dumps({'mode':a.mode,'rows':len(d['results']),'output':a.output}))
