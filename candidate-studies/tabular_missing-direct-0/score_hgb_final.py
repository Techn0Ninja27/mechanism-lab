"""One-shot held-out HGB score, run only after preregistration is frozen."""
import json
from pathlib import Path
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
P=Path(__file__).resolve().parent; rows=[]
for seed in (137,271,389):
 x,y=make_classification(n_samples=5000,n_features=20,n_informative=5,n_redundant=5,weights=[.8,.2],class_sep=1.,random_state=seed);x=x.astype(np.float32);y=y.astype(np.int64);idx=np.arange(len(y))
 tr,rest=train_test_split(idx,test_size=.4,random_state=seed,stratify=y);va,te=train_test_split(rest,test_size=.5,random_state=seed+1,stratify=y[rest])
 model=HistGradientBoostingClassifier(random_state=seed,max_iter=120).fit(x[tr],y[tr]);imp=SimpleImputer(strategy='median').fit(x[tr])
 for level in (0,.1,.3,.5):
  rng=np.random.default_rng(seed+10000+int(level*100));xx=x[te].copy()
  if level: xx[rng.random(xx.shape)<level]=np.nan
  rows.append({'seed':seed,'method':'median_hgb','missing':level,'average_precision':float(average_precision_score(y[te],model.predict_proba(imp.transform(xx))[:,1]))})
(P/'hgb_final.json').write_text(json.dumps(rows,indent=2))
print('HGB held-out scores saved')
