"""Post-run scale diagnostic; does not change the failed absolute-error gate."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from pera_algebra import delta_expanded, delta_sum

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def analyze(result_path, output):
    result=json.loads(result_path.read_text())
    if sha(Path(__file__).with_name('pera_algebra.py')) != result['code']['module_sha256']:
        raise ValueError('Module changed since run')
    if output.exists(): raise FileExistsError('Preserve previous diagnostic')
    rows=[]
    for arm in result['arms']:
        torch.manual_seed(arm['seed'])
        B=torch.randn(arm['m'],arm['rank'],dtype=torch.float64,requires_grad=True)
        A=torch.randn(arm['rank'],arm['n'],dtype=torch.float64,requires_grad=True)
        h=torch.randn(arm['rank']*(arm['rank']+1)//2,dtype=torch.float64,requires_grad=True)
        left=torch.autograd.grad(delta_expanded(B,A,h).square().sum(),(B,A,h))
        right=torch.autograd.grad(delta_sum(B,A,h).square().sum(),(B,A,h))
        pairs=[]
        for name,x,y in zip(('B','A','h'),left,right):
            pairs.append({'parameter':name,'max_abs_error':float((x-y).abs().max()),
                          'reference_max_abs':float(y.abs().max()),
                          'relative_l2_error':float((x-y).norm()/y.norm().clamp_min(torch.finfo(y.dtype).tiny))})
        rows.append({'seed':arm['seed'],'rank':arm['rank'],'gradients':pairs})
    record={'schema':'pera-gradient-scale-diagnostic.v1','result_sha256':sha(result_path),
            'script_sha256':sha(Path(__file__)),'rows':rows,
            'scope':'Post-run diagnostic of floating-point scale. Does not alter the original absolute tolerance or its failed status.'}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'max_relative_l2_error':max(p['relative_l2_error'] for r in rows for p in r['gradients']),
                      'max_reference_abs':max(p['reference_max_abs'] for r in rows for p in r['gradients'])}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--result',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();analyze(a.result,a.output)
