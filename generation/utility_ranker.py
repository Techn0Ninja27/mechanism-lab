"""Task-conditioned compression utility models; not a general research judge."""
from __future__ import annotations
from collections import Counter
import math
import numpy as np

OPERATORS=('delta','delta2','zigzag','varint','rle','bitpack','blockpack','raw64','for','block_for','zlib','lzma')


def signed_log(x):return math.copysign(math.log1p(abs(float(x))),float(x))
def entropy(xs):
    counts=Counter(xs);n=len(xs)
    return -sum((c/n)*math.log2(c/n) for c in counts.values()) if n else 0.0


def stream_features(xs):
    if not xs:raise ValueError('nonempty stream required')
    a=np.asarray(xs,dtype=np.float64);ds=[y-x for x,y in zip(xs,xs[1:])];delta=np.asarray(ds,dtype=np.float64)
    runs=[];length=1
    for x,y in zip(xs,xs[1:]):
        if x==y:length+=1
        else:runs.append(length);length=1
    runs.append(length);stride=max(1,len(xs)//512)
    raw={'length':len(xs),'min':min(xs),'max':max(xs),'range':max(xs)-min(xs),'mean':float(a.mean()),
         'std':float(a.std()),'variance':float(a.var()),'median_abs':float(np.median(np.abs(a))),
         'mean_abs_difference':float(np.abs(delta).mean()) if ds else 0,'std_difference':float(delta.std()) if ds else 0,
         'mean_run':float(np.mean(runs)),'max_run':max(runs),'run_count':len(runs)}
    result={k:signed_log(v) for k,v in raw.items()}
    result.update(zero_difference=sum(x==0 for x in ds)/max(1,len(ds)),symbol_entropy=entropy(xs[::stride]),
                  delta_entropy=entropy(ds[::stride]),zero_fraction=sum(x==0 for x in xs)/len(xs),
                  nonnegative_fraction=sum(x>=0 for x in xs)/len(xs))
    for bits in (8,16,32,64):result[f'fit_{bits}']=sum(-(1<<(bits-1))<=x<(1<<(bits-1)) for x in xs)/len(xs)
    return result


def feature_vector(xs, chain):
    raw=stream_features(xs);names=list(raw);values=list(raw.values())
    structure=[len(chain)]+[chain.count(op) for op in OPERATORS]
    snames=['depth']+list(OPERATORS)
    # Context interactions let a linear model change ranking with the input.
    features=values+structure+[v*s for v in values for s in structure]
    feature_names=names+['chain_'+s for s in snames]+[r+'*'+s for r in names for s in snames]
    array=np.asarray(features,dtype=np.float64)
    if not np.isfinite(array).all():raise ValueError('nonfinite raw-stream features')
    return array,feature_names


def fit_models(x,y,groups):
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.linear_model import Ridge,LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    models={}
    for alpha in (1.0,100.0):
        name=f'ridge_{alpha:g}';model=make_pipeline(StandardScaler(),Ridge(alpha=alpha));model.fit(x,y);models[name]=model
    for leaf in (3,8):
        model=ExtraTreesRegressor(n_estimators=96,max_depth=12,min_samples_leaf=leaf,max_features=.8,random_state=21001,n_jobs=2)
        model.fit(x,y);models[f'extratrees_leaf{leaf}']=model
    rng=np.random.default_rng(21003);diff=[];labels=[]
    for group in sorted(set(groups)):
        ids=np.flatnonzero(np.asarray(groups)==group)
        pairs=[(i,j) for pos,i in enumerate(ids) for j in ids[pos+1:] if abs(y[i]-y[j])>1e-8]
        if len(pairs)>30:pairs=[pairs[k] for k in rng.choice(len(pairs),30,replace=False)]
        for i,j in pairs:
            v=x[i]-x[j];label=int(y[i]>y[j]);diff.extend([v,-v]);labels.extend([label,1-label])
    scaler=StandardScaler(with_mean=False);scaled=scaler.fit_transform(np.asarray(diff))
    model=LogisticRegression(C=1.0,max_iter=500,fit_intercept=False,random_state=21003);model.fit(scaled,np.asarray(labels))
    models['pairwise_logistic']={'scaler':scaler,'model':model}
    return models


def predict(model,x):
    if isinstance(model,dict):return model['model'].decision_function(model['scaler'].transform(x))
    return model.predict(x)
