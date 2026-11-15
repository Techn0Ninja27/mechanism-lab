"""Create fixed synthetic splits, common corruption masks, and HGB reference."""
import hashlib,json
from pathlib import Path
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
HERE=Path(__file__).resolve().parent
hashes={}; ref=[]
for seed in (137,271,389):
    x,y=make_classification(n_samples=5000,n_features=20,n_informative=5,n_redundant=5,weights=[.8,.2],class_sep=1.,random_state=seed)
    x=x.astype(np.float32);y=y.astype(np.int64); idx=np.arange(len(y))
    tr,rest=train_test_split(idx,test_size=.4,random_state=seed,stratify=y)
    va,te=train_test_split(rest,test_size=.5,random_state=seed+1,stratify=y[rest])
    med=np.nanmedian(x[tr],axis=0)
    pack={'xtr':x[tr],'ytr':y[tr],'median':med,'xval':x[va],'yval':y[va],'xtest':x[te],'ytest':y[te]}
    for split, inds in (('val',va),('test',te)):
        masks=[]
        for level in (0,.1,.3,.5):
            rng=np.random.default_rng(seed+10000+int(level*100)); mask=(rng.random(x[inds].shape)<level).astype(np.uint8); masks.append(mask)
            xx=x[inds].copy(); xx[mask.astype(bool)]=np.nan
            if level==0: xx=x[inds].copy()
            pack[f'x{split}_{level}']=xx
        pack[f'masks_{split}']=np.stack(masks)
    np.savez_compressed(HERE/f'fixture-{seed}.npz',**pack)
    hashes[str(seed)]={}
    for key in ('train','validation','test'):
        arr=np.sort({'train':tr,'validation':va,'test':te}[key]);hashes[str(seed)][key]=hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
    # Matched HGB named baseline: training data complete, median transform fit train only.
    model=HistGradientBoostingClassifier(random_state=seed,max_iter=120).fit(x[tr],y[tr])
    for split,inds in (('val',va),):
        for level in (0,.1,.3,.5):
            xx=pack[f'x{split}_{level}']; pred=model.predict_proba(SimpleImputer(strategy='median').fit(x[tr]).transform(xx))[:,1]
            ref.append({'seed':seed,'method':'median_hgb','missing':level,'average_precision':float(average_precision_score(y[inds],pred))})
(HERE/'fixture_metadata.json').write_text(json.dumps({'splits':hashes,'hgb':ref},indent=2))
print('fixtures prepared')
