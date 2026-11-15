"""Three-phase tabular missingness candidate study; all artifacts stay in this directory."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
SEEDS=(137,271,389)
LEVELS=(0.0,0.1,0.3,0.5)

def h(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def split_idx(y,seed):
 from sklearn.model_selection import train_test_split
 ix=np.arange(len(y)); tr,rest=train_test_split(ix,test_size=.4,random_state=seed,stratify=y)
 va,te=train_test_split(rest,test_size=.5,random_state=seed+1,stratify=y[rest])
 return {'train':np.sort(tr),'validation':np.sort(va),'test':np.sort(te)}
def prepare(mode):
 from sklearn.datasets import make_classification
 out=[]
 for seed in SEEDS:
  x,y=make_classification(n_samples=5000,n_features=20,n_informative=5,n_redundant=5,weights=[.8,.2],class_sep=1.0,random_state=seed)
  s=split_idx(y,seed)
  # fixed paired corruption draws across every method
  masks={str(p):np.random.default_rng(seed+10000+int(p*100)).random((len(y),20))<p for p in LEVELS}
  evalix=s['validation' if mode=='validation' else 'test']; from sklearn.impute import SimpleImputer; from sklearn.ensemble import HistGradientBoostingClassifier
  imp=SimpleImputer(strategy='median',add_indicator=True).fit(x[s['train']]); hgb=HistGradientBoostingClassifier(random_state=seed,max_iter=120); hgb.fit(imp.transform(x[s['train']]),y[s['train']]); baseline=[]
  for level in LEVELS:
   xx=x[evalix].copy(); xx[masks[str(level)][evalix]]=np.nan; baseline.append(hgb.predict_proba(imp.transform(xx))[:,1])
  np.savez(HERE/f'{mode}-seed{seed}.npz',x=x.astype('float32'),y=y.astype('int64'),baseline=np.array(baseline),**{f'idx_{k}':v for k,v in s.items()},**{f'mask_{p}':v for p,v in masks.items()})
  out.append({'seed':seed,'splits':{k:h(v) for k,v in s.items()},'x':h(x),'y':h(y)})
 return out

def train(mode):
 import torch
 from torch import nn
 torch.set_num_threads(1)
 torch.manual_seed(0)
 results=[]
 class Net(nn.Module):
  def __init__(self,nin=40):
   super().__init__(); self.enc=nn.Sequential(nn.Linear(nin,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU()); self.cls=nn.Linear(32,1); self.rec=nn.Linear(32,20)
  def forward(self,x):
   z=self.enc(x); return self.cls(z).squeeze(-1),self.rec(z)
 for seed in SEEDS:
  d=np.load(HERE/f'{mode}-seed{seed}.npz'); x=d['x']; y=d['y']; tr=d['idx_train']; evalix=d['idx_validation' if mode=='validation' else 'idx_test']
  mu=x[tr].mean(0); sd=x[tr].std(0); sd[sd<1e-6]=1; z=(x-mu)/sd
  # beta regime plus exact matched classifier-capacity ablation, uniform severity, no mask signal
  variants=('recon_beta','no_recon_beta','recon_uniform','recon_no_mask')
  for variant in variants:
   torch.manual_seed(seed+{'recon_beta':0,'no_recon_beta':0,'recon_uniform':0,'recon_no_mask':0}[variant])
   net=Net(20 if variant=='recon_no_mask' else 40); opt=torch.optim.Adam(net.parameters(),lr=.002,weight_decay=1e-4); tx=torch.tensor(z[tr]); ty=torch.tensor(y[tr],dtype=torch.float32)
   started=time.perf_counter(); rng=np.random.default_rng(seed+333)
   for ep in range(30):
    order=rng.permutation(len(tr))
    for st in range(0,len(tr),128):
     ids=order[st:st+128]; xb=tx[ids]; probs=(torch.tensor(rng.beta(2,1,len(ids)),dtype=torch.float32)*.5 if variant!='recon_uniform' else torch.tensor(rng.uniform(0,.5,len(ids)),dtype=torch.float32)); mask=torch.rand(xb.shape)<probs[:,None]
     inp=xb.clone(); inp[mask]=0
     usemask=variant!='recon_no_mask'; xx=torch.cat([inp,mask.float()],1) if usemask else inp
     logits,rec=net(xx); loss=nn.functional.binary_cross_entropy_with_logits(logits,ty[ids])
     if variant in ('recon_beta','recon_uniform','recon_no_mask'):
      loss=loss+0.2*nn.functional.mse_loss(rec[mask],xb[mask]) if mask.any() else loss
     opt.zero_grad(); loss.backward(); opt.step()
   runtime=time.perf_counter()-started
   # evaluate each condition with same fixed MCAR masks, heldout-only prediction
   for p in LEVELS:
    xm=torch.tensor(z[evalix].copy()); m=torch.tensor(d[f'mask_{p}'][evalix]); xm[m]=0
    xx=torch.cat([xm,m.float()],1) if variant!='recon_no_mask' else xm
    net.eval()
    with torch.no_grad(): pred=torch.sigmoid(net(xx)[0]).numpy()
    results.append({'seed':seed,'method':variant,'condition':p,'y':y[evalix].tolist(),'score':pred.tolist(),'train_seconds':runtime})
  for j,p in enumerate(LEVELS): results.append({'seed':seed,'method':'median_hgb','condition':p,'y':y[evalix].tolist(),'score':d['baseline'][j].tolist(),'train_seconds':None})
  # per-seed NPZ/JSON prediction artifact avoids test reruns
  for row in results[-len(variants)*len(LEVELS)-len(LEVELS):]: row.pop('y',None)
  if mode=='validation': (HERE/'validation.json').write_text(json.dumps({'mode':mode,'seeds':SEEDS,'results':results},indent=2))
  else: (HERE/'results.json').write_text(json.dumps({'mode':mode,'seeds':SEEDS,'results':results},indent=2))
 return results

def main():
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','train']);p.add_argument('mode',choices=['validation','final']);a=p.parse_args()
 if a.phase=='prepare': print(json.dumps(prepare(a.mode),indent=2))
 else: print(json.dumps({'n':len(train(a.mode)),'mode':a.mode}))
if __name__=='__main__':main()
