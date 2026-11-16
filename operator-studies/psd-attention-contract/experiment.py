"""Diagnostic: positive semidefinite does not mean entrywise nonnegative."""
import hashlib,json,platform
from pathlib import Path
import numpy as np
OUT=Path(__file__).parent/'results.json'
def run():
 rows=[]
 for seed in [88031,88033,88037]:
  x=np.random.default_rng(seed).normal(size=(24,4))
  distance=((x[:,None,:]-x[None,:,:])**2).sum(-1)
  for bandwidth in [.3,1.,3.]:
   a=np.exp(-distance/(2*bandwidth**2));values,vectors=np.linalg.eigh(a);q=vectors[:,-4:]
   approx=a@q@np.linalg.pinv(q.T@a@q)@q.T@a
   optimum=(q*values[-4:])@q.T
   assert np.allclose(approx,optimum,atol=1e-10)
   assert np.linalg.eigvalsh(approx).min()>-1e-10
   sums=approx.sum(axis=1);normalized=approx/sums[:,None]
   rows.append({'seed':seed,'bandwidth':bandwidth,'rank':4,'original_entry_min':float(a.min()),
    'approx_entry_min':float(approx.min()),'approx_eigenvalue_min':float(np.linalg.eigvalsh(approx).min()),
    'min_row_sum':float(sums.min()),'negative_entries':int((approx < -1e-10).sum()),
    'negative_normalized_weights':int((normalized < -1e-10).sum()),
    'frobenius_error':float(np.linalg.norm(a-approx)),'optimal_frobenius_error':float(np.sqrt((values[:-4]**2).sum()))})
 return {'scope':'Known mathematical contract diagnostic, not a novelty or learned-attention performance result.',
  'construction':'RBF Gram A; f(t)=t; top4 eigenspace Q; optimal rank4 Nystrom approximation.',
  'rows':rows,'cases_with_negative_entries':sum(r['negative_entries']>0 for r in rows),
  'cases':len(rows),'versions':{'python':platform.python_version(),'numpy':np.__version__},
  'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
if __name__=='__main__':
 if OUT.exists():raise FileExistsError('Preserve prior diagnostic')
 result=run();OUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
