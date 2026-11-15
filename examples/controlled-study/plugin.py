"""Known baseline plugin for exercising the runner, not a new algorithm."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
class MaskModel:
 def __init__(self,params,seed):self.params=params;self.seed=seed
 def fit(self,x,y):
  copies=int(self.params.get('copies',1));rate=float(self.params.get('extra_missing_rate',0));rng=np.random.default_rng(self.seed);parts=[x.copy()]
  for _ in range(copies-1):
   z=x.copy();mask=np.zeros(z.shape,dtype=bool);mask[:,2:]=rng.random(z[:,2:].shape)<rate;z[mask]=np.nan;parts.append(z)
  self.model=HistGradientBoostingClassifier(max_iter=100,random_state=self.seed).fit(np.concatenate(parts),np.tile(y,copies));return self
 def predict_proba(self,x):return self.model.predict_proba(x)
def build(params,seed):return MaskModel(params,seed)
