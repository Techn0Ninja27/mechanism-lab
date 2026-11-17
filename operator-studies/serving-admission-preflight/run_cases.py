import argparse, json
from pathlib import Path
from simulator import simulate, JobSpec, ORDERINGS

def run(case):
    specs = [JobSpec(**x) for x in case['jobs']]
    out = {p: simulate(specs, prefill_capacity=case['prefill_capacity'], decode_capacity=case['decode_capacity'], policy=p, horizon=case.get('horizon',100)) for p in ORDERINGS}
    return {'case':case['id'], 'outputs':out, 'scope':'Deterministic accounting preflight; true output is an EOS oracle only.'}

if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--cases',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if a.output.exists(): raise FileExistsError('refuse overwrite')
    cases=json.loads(a.cases.read_text()); result=[run(c) for c in cases]
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
