"""Untimed Table-5 fidelity/coverage check on captured real training matrices."""
import argparse,hashlib,json,os,warnings
from pathlib import Path
import numpy as np
import torch
from polynomial import PLANS,select_plan,inverse
HERE=Path(__file__).resolve().parent


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--capture',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if a.output.exists():raise FileExistsError('Use a fresh output')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0':raise ValueError('Disable general MPS fallback explicitly')
    rows=[];modules=[];sources=[]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for width in (256,512):
            manifest=json.loads((a.capture/f'd{width}.manifest.json').read_text());path=a.capture/f'd{width}.npz'
            if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['npz_sha256']:raise ValueError('Capture hash mismatch')
            sources.append({'width':width,'npz_sha256':manifest['npz_sha256'],'source':manifest['source']})
            with np.load(path,allow_pickle=False) as arrays:
                for name,meta in manifest['arrays'].items():
                    ks=arrays[meta['K_key']];g=arrays[meta['G_key']].astype(np.float64);refs=[];approxs=[]
                    for block,k in enumerate(ks):
                        k=k.astype(np.float64);asym=float(np.linalg.norm(k-k.T,'fro')/max(np.linalg.norm(k,'fro'),1e-30))
                        if asym>1e-5:raise ValueError('Materially asymmetric covariance')
                        k=(k+k.T)*.5
                        min_eig=float(np.linalg.eigvalsh(k)[0])
                        if min_eig < -1e-6*np.linalg.norm(k,'fro'):raise ValueError('Covariance PSD check failed')
                        selection=select_plan(k);row={'width':width,'module':name,'block':block,'relative_asymmetry':asym,'minimum_eigenvalue':min_eig,**selection}
                        damped=k+selection['delta']*np.eye(width);reference=np.linalg.inv(damped);refs.append(reference)
                        if selection['plan_index'] is None:
                            row['status']='no_eligible_published_plan';approxs.append(None)
                        else:
                            tensor=torch.tensor(k,dtype=torch.float32,device='mps');approx=inverse(tensor,selection).detach().cpu().numpy().astype(np.float64);approxs.append(approx)
                            residual=float(np.linalg.norm(np.eye(width)-damped@approx,2));plan=PLANS[selection['plan_index']]
                            row.update(status='checked',residual_spectral_norm=residual,relative_inverse_frobenius=float(np.linalg.norm(approx-reference,'fro')/np.linalg.norm(reference,'fro')),published_bound=plan['s_out'],below_one_percent=residual<=.01,within_reported_bound_plus_rounding=residual<=plan['s_out']+1e-5)
                        rows.append(row)
                    if all(p is not None for p in approxs):
                        chunks=np.split(g,len(ks),axis=1);ref=np.concatenate([x@p for x,p in zip(chunks,refs)],axis=1);approx=np.concatenate([x@p for x,p in zip(chunks,approxs)],axis=1)
                        cosine=float(np.sum(ref*approx)/(np.linalg.norm(ref)*np.linalg.norm(approx)))
                        modules.append({'width':width,'module':name,'all_blocks_covered':True,'raw_preconditioned_gradient_relative_frobenius':float(np.linalg.norm(approx-ref)/np.linalg.norm(ref)),'raw_preconditioned_gradient_cosine':cosine})
                    else:modules.append({'width':width,'module':name,'all_blocks_covered':False})
    checked=[x for x in rows if x['status']=='checked'];warn=[str(x.message)[:500] for x in caught]
    result={'schema':'newton-muon-polynomial-matrix-check.v1','scope':'Untimed float32 MPS dense functional polynomial recurrence. CPU float64 selection/reference/fidelity checks; no optimized SYPP kernels, timing, momentum-update, training-quality or novelty claim.','selection':'Smallest published SYPP budget <=13 with reported residual <=.01 and epsilon <= conservative Frobenius-derived lower bound. No invented plan or fallback result.','sources':sources,'rows':rows,'modules':modules,'summary':{'covariance_blocks':len(rows),'eligible_blocks':len(checked),'uncovered_blocks':len(rows)-len(checked),'covered_modules':sum(x['all_blocks_covered'] for x in modules),'modules':len(modules),'worst_checked_residual':max((x['residual_spectral_norm'] for x in checked),default=None),'all_checked_below_one_percent':all(x['below_one_percent'] for x in checked),'all_checked_within_published_plus_rounding':all(x['within_reported_bound_plus_rounding'] for x in checked)},'warnings':warn,'internal_cpu_fallback_detected':any('fall' in x.lower() and 'cpu' in x.lower() for x in warn),'hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),HERE/'polynomial.py',HERE/'plans.json']}}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['summary']))
if __name__=='__main__':main()
