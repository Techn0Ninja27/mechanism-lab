"""Probe a hash-pinned author source locally without installing its package."""
import argparse
import ast
import hashlib
import json
import math
import platform
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def run(source, output):
    protocol = json.loads((HERE/'protocol.json').read_text())
    if sha(source) != protocol['source_sha256']:
        raise ValueError('Author snapshot differs from pinned source')
    if output.exists():
        raise FileExistsError('Preserve previous probe; use a fresh output')
    tree = ast.parse(source.read_text())
    classes = [n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in ('LoraLayer','Linear')]
    if [n.name for n in classes] != ['LoraLayer','Linear']:
        raise ValueError('Expected class definitions missing')
    # Only the two inspected class definitions are evaluated, not top-level
    # package imports, trainer code, models, or dataset loaders.
    ns = {'torch':torch,'nn':nn,'F':F,'math':math,
          'transpose':lambda w,fan: w.T if fan else w}
    exec(compile(ast.Module(body=classes,type_ignores=[]),str(source),'exec'),ns)
    rows=[]
    for bias,inject in ((False,False),(False,True),(True,True)):
        torch.manual_seed(protocol['seed'])
        layer=ns['Linear']('probe',protocol['in_features'],protocol['out_features'],
                           r_ab=protocol['rank'],init_a='zero',init_b='kaiming',bias=bias)
        # The public initializer supports Kaiming/zero, not the paper's Gaussian.
        # This probe concerns missing state and shape handling, not training.
        if inject:
            count=protocol['rank']*(protocol['rank']+1)//2
            layer.lora_A_parameter['probe']=nn.Parameter(torch.zeros(1,count))
        x=torch.randn(3,protocol['in_features'])
        row={'bias':bias,'coefficient_injected':inject,
             'coefficient_keys':list(layer.lora_A_parameter.keys())}
        try:
            y=layer(x)
            row.update(outcome='returned',shape=list(y.shape),
                       max_base_error=float((y-F.linear(x,layer.weight,layer.bias)).abs().max().detach()))
        except (AttributeError,KeyError,RuntimeError) as exc:
            row.update(outcome='exception',exception_type=type(exc).__name__,message=str(exc)[:1000])
        rows.append(row)
    checks={
        'unmodified_bias_false_forward_missing_h':rows[0]['outcome']=='exception' and 'probe' in rows[0].get('message','') and not rows[0]['coefficient_keys'],
        'injected_h_bias_false_forward_matches_base':rows[1]['outcome']=='returned' and rows[1]['max_base_error']==0,
        'injected_h_bias_true_exposes_shape_failure':rows[2]['outcome']=='exception' and rows[2].get('exception_type')=='RuntimeError',
    }
    result={'schema':'pera-author-probe-result.v1','checks':checks,'all_expected_observations':all(checks.values()),
            'rows':rows,'protocol':protocol,'protocol_sha256':sha(HERE/'protocol.json'),
            'runner_sha256':sha(Path(__file__)),'source_sha256':sha(source),
            'environment':{'python':platform.python_version(),'torch':torch.__version__,'device':'cpu','dtype':'float32'},
            'scope':protocol['scope'],'novelty_established':False,'benchmark_reproduced':False}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    return {'checks':checks,'output':str(output)}

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',type=Path,default=ROOT/'results/pera-author-code-v1/lora.py')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    print(json.dumps(run(args.source,args.output)))
