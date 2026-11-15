"""Controlled tabular mechanism screen; direct MLP proposal is not implemented."""
import argparse,hashlib,json,pathlib,sys,time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from scripts.world_search_benchmark import _tabular
LEVELS=(0,.1,.3,.5)
def masks(x,rates,seed):
 rng=np.random.default_rng(seed);return rng.random(x.shape)<rates

def features(x,mask,stat,indicators):
 z=np.where(mask,np.broadcast_to(stat,x.shape),x)
 return np.c_[z,mask.astype(float)] if indicators else z

def run(mode,out,selection=None):
 x,y,splits=_tabular();tr,va=splits['train'],splits['validation'];ev=splits['test'] if mode=='final' else va
 means=x[tr].mean(0);medians=np.median(x[tr],axis=0);rng=np.random.default_rng(42)
 beta_masks=[masks(x[tr],rng.beta(.5,2,x.shape[1]),100+i) for i in range(4)]
 fixed_masks=[masks(x[tr],.2,100+i) for i in range(4)]
 eval_masks=[masks(x[ev],rate,10000+i) for i,rate in enumerate(LEVELS)]
 variants=[('median_clean',medians,False,[]),('median_fixed_aug',medians,False,fixed_masks),('native_nan_beta',np.full(20,np.nan),False,beta_masks),('native_nan_beta_mask',np.full(20,np.nan),True,beta_masks),('native_nan_fixed_mask',np.full(20,np.nan),True,fixed_masks)]
 alphas=(0,.3,.7,1)
 if mode=='final':
  if not selection:raise ValueError('Need frozen validation selection')
  v=json.loads(pathlib.Path(selection).read_text());alphas=(v['selected_alpha'],)
 # Explicit extension: same missingness augmentation is supplied to every
 # soft-imputation variant. Original world text omitted training missingness.
 variants += [(f'soft_alpha_{a}',a*means,True,fixed_masks) for a in alphas]
 start=time.perf_counter();rows=[]
 for name,stat,indicators,aug in variants:
  train_masks=[np.zeros_like(x[tr],bool)]+aug
  xx=np.concatenate([features(x[tr],m,stat,indicators) for m in train_masks]);yy=np.tile(y[tr],len(train_masks))
  model=HistGradientBoostingClassifier(max_iter=180,learning_rate=.06,max_leaf_nodes=31,random_state=42).fit(xx,yy)
  vals=[float(average_precision_score(y[ev],model.predict_proba(features(x[ev],m,stat,indicators))[:,1])) for m in eval_masks]
  rows.append({'variant':name,'mean_average_precision':float(np.mean(vals)),'conditions':vals,'train_rows':len(xx)})
 best=max([r for r in rows if r['variant'].startswith('soft_')],key=lambda r:r['mean_average_precision'])
 result={'config':{'mode':mode,'test_evaluated':mode=='final','levels':LEVELS,'seed':42,'deviations':['Direct trainable-MLP imputation proposal NOT implemented; no claim about its utility.','World formula alpha*mean is tested literally: alpha0 is zero imputation, contrary to the generated prose claiming median imputation.','Soft-imputation variants add shared training missingness augmentation omitted by the original world proposal; this tests a modified composition, not exact replication.','HGB180iterations shared acrossvariants, vs300 in randomproposal; all augmented variants have five training copies.'],'success_rule':'Improve mean AP over matched simpler control; no retuning on test.'},'split_hashes':{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in splits.items()},'results':rows,'selected_alpha':float(best['variant'].split('_')[-1]),'elapsed_seconds':time.perf_counter()-start}
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2));print(json.dumps({'mode':mode,'selected_alpha':result['selected_alpha'],'results':[(r['variant'],r['mean_average_precision']) for r in rows]},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['validation','final'],default='validation');p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--selection',type=pathlib.Path);a=p.parse_args();run(a.mode,a.output,a.selection)
