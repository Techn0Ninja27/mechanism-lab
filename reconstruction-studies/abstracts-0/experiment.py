"""ADMIB phase-one validation. Prediction interface receives X only."""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time, traceback
from pathlib import Path
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIXTURE = ROOT / 'scripts' / 'missingness_shift_fixture.py'
import importlib.util
spec = importlib.util.spec_from_file_location('shared_fixture', FIXTURE)
fixture_mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = fixture_mod; spec.loader.exec_module(fixture_mod)

ARMS = ('native_nan', 'native_mcar_aug', 'completed', 'completed_aug_unweighted',
        'admib', 'admib_mcar_only', 'mean_completed_aug', 'completed_aug_indicators')


def digest(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def make_masks(x, rng):
    n,p=x.shape
    mcar=rng.random((n,p))<.2; mcar[:,:2]=False
    z=(x[:,0]-np.nanmean(x[:,0]))/(np.nanstd(x[:,0])+1e-12)
    mar=rng.random((n,p)) < (.05+.5/(1+np.exp(-np.clip(z,-30,30))))[:,None]; mar[:,:2]=False
    block=np.zeros((n,p),bool); rows=rng.random(n)<.5; block[rows,2:8]=True
    mcar_replicates=[]
    for _ in range(3):
        m=rng.random((n,p))<.2; m[:,:2]=False; mcar_replicates.append(m)
    return [mcar,mar,block],mcar_replicates

def estimator():
    return HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,learning_rate=.08,
      l2_regularization=1.0,random_state=33,early_stopping=False)

def predict(model, x):
    # No target condition or labels are passed through this boundary.
    return model.predict_proba(x)[:,1]

def fit_predict(arm, train_x, train_y, val_x, seed, aug_masks, mcar_replicates, imputer):
    # Everything after this boundary uses only masked feature arrays and train labels.
    Xs=[train_x]; regimes=[np.zeros(len(train_x),int)]
    if arm == 'native_nan':
        Xs.extend([train_x,train_x,train_x]); regimes.extend([np.zeros(len(train_x),int) for _ in range(3)])
    if arm in ('native_mcar_aug',):
        for i,mm in enumerate(mcar_replicates): Xs.append(np.where(mm,np.nan,train_x)); regimes.append(np.full(len(train_x),i+1))
    if arm == 'completed':
        Xs.extend([train_x,train_x,train_x]); regimes.extend([np.zeros(len(train_x),int) for _ in range(3)])
    elif arm in ('completed_aug_unweighted','admib','admib_mcar_only','mean_completed_aug','completed_aug_indicators'):
        masks=aug_masks if arm not in ('admib_mcar_only',) else mcar_replicates
        # preserve four-copy row count for MCAR-only arm
        for i,mm in enumerate(masks): Xs.append(np.where(mm,np.nan,train_x)); regimes.append(np.full(len(train_x),i+1))
    X=np.concatenate(Xs); y=np.tile(train_y,len(Xs)); reg=np.concatenate(regimes)
    if arm in ('completed','completed_aug_unweighted','admib','admib_mcar_only','mean_completed_aug','completed_aug_indicators'):
        imp=SimpleImputer(strategy='mean') if arm=='mean_completed_aug' else imputer
        if arm=='mean_completed_aug': imp.fit(train_x)
        X=imp.transform(X)
        if arm=='completed_aug_indicators':
            # explicit mask features are derived only from the presented masked inputs
            maskbits=np.concatenate([np.isnan(z) for z in Xs]).astype(float)
            X=np.column_stack((X,maskbits))
    weights=None
    if arm in ('admib','admib_mcar_only'):
        # Estimate P(regime | completed X), then clipped inverse propensity weights.
        propensity=make_pipeline(StandardScaler(),LogisticRegression(max_iter=100,C=.5))
        propensity.fit(X,reg)
        probs=np.clip(propensity.predict_proba(X),.05,1)
        weights=np.clip(1.0/probs[np.arange(len(reg)),reg],.25,4.0)
        # equalize regime total weight; normalize to mean one
        for c in np.unique(reg): weights[reg==c] /= weights[reg==c].mean()
        weights /= weights.mean()
    model=estimator(); model.fit(X,y,sample_weight=weights)
    vx=val_x
    if arm in ('completed','completed_aug_unweighted','admib','admib_mcar_only','mean_completed_aug','completed_aug_indicators'):
        vx=imp.transform(vx)
        if arm=='completed_aug_indicators': vx=np.column_stack((vx,np.isnan(val_x).astype(float)))
    return predict(model,vx)

def main():
    pa=argparse.ArgumentParser(); pa.add_argument('--mode',choices=('validation','final'),required=True); pa.add_argument('--timeout-seconds',type=int,default=300); a=pa.parse_args()
    out=HERE/('validation.json' if a.mode=='validation' else 'results.json')
    if a.mode=='final' and out.exists(): raise SystemExit('Refusing to overwrite final output')
    if a.mode=='validation' and out.exists(): raise SystemExit('Refusing to overwrite validation output')
    start=time.monotonic(); records=[]; errors=[]
    for seed in fixture_mod.SEEDS:
      try:
        f=fixture_mod.load_fixture(seed,a.mode)
        # Split training observations for selecting a data-only imputer; both sides remain masked.
        it,iv=train_test_split(np.arange(len(f.train_y)),test_size=.2,stratify=f.train_y,random_state=seed+77)
        imp=IterativeImputer(estimator=BayesianRidge(),max_iter=5,tol=1e-3,skip_complete=True,random_state=seed)
        imp.fit(f.train_x[it])
        rng=np.random.default_rng(seed+500); aug_masks,mcar_replicates=make_masks(f.train_x,rng)
        for arm in ARMS:
          t=time.monotonic()
          cases=f.validation_cases if a.mode=='validation' else f.final_cases
          for cond,(vx,vy) in cases.items():
            # per-arm model fit is repeated per condition only to keep prediction boundary simple.
            p=fit_predict(arm,f.train_x,f.train_y,vx,seed,aug_masks,mcar_replicates,imp)
            records.append({'seed':seed,'arm':arm,'condition':cond,'ap':float(average_precision_score(vy,p)),
               'brier':float(brier_score_loss(vy,p)),'seconds':time.monotonic()-t,
               'train_x_hash':digest(f.train_x),'validation_x_hash':digest(vx)})
          if time.monotonic()-start>a.timeout_seconds: raise TimeoutError('validation compute gate exceeded')
      except Exception as e:
        errors.append({'seed':seed,'error':repr(e),'traceback':traceback.format_exc()})
        break
    payload={'mode':a.mode,'seeds':list(fixture_mod.SEEDS),'arms':list(ARMS),'records':records,'errors':errors,
       'elapsed_seconds':time.monotonic()-start,'fixture_source':str(FIXTURE),'fixture_source_sha256':hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
       'validation_hash':hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest(),
       'training_input_policy':'only fixture train_x (already masked) and train_y; no clean train values',
       'prediction_interface':'feature matrix only; no condition names or held-out labels'}
    out.write_text(json.dumps(payload,indent=2)+'\n')
    if errors: raise SystemExit('validation completed with errors; inspect validation.json')
    print(json.dumps({'records':len(records),'elapsed_seconds':payload['elapsed_seconds'],'output':str(out)}))
if __name__=='__main__': main()
