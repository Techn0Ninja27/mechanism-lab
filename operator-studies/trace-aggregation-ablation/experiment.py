"""Test a reviewer's max-aggregation suggestion using the frozen replay engine."""
import argparse,hashlib,importlib.util,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
REPLAY=HERE.parent/'trace-window-ablation/experiment.py'
BASELINE=HERE.parent/'trace-window-ablation/result.json'
NEW_ARMS=('one_bit_max','two_bit_max','exact_max')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--freeze',action='store_true');a=p.parse_args()
    protocol=HERE/'protocol.json';plan=json.loads(protocol.read_text());traces=ROOT/'results/smolvlm-trace-v3'
    for key,path in [('replay_code_sha256',REPLAY),('baseline_result_sha256',BASELINE),('trace_manifest_sha256',traces/'manifest.json')]:
        if plan[key]!=sha(path):raise ValueError('frozen dependency mismatch: '+key)
    freeze={'driver_sha256':sha(__file__),'protocol_sha256':sha(protocol),**{k:v for k,v in plan.items() if k.endswith('_sha256')}}
    if a.freeze:
        if a.output.exists():raise FileExistsError('fresh output required')
        a.output.mkdir(parents=True);(a.output/'freeze.json').write_text(json.dumps(freeze,indent=2)+'\n');return
    if (a.output/'result.json').exists():raise FileExistsError('preserve result')
    if json.loads((a.output/'freeze.json').read_text())!=freeze:raise ValueError('frozen driver changed')
    spec=importlib.util.spec_from_file_location('frozen_trace_replay',REPLAY);engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
    original_scores=engine.score_arms
    def extended_scores(z,layer,visual,text):
        scores=original_scores(z,layer,visual,text)
        q=z['prefill_q'][layer][:,text,:].astype(np.float64);k=z['prefill_k'][layer][:,visual,:].astype(np.float64)
        for name,bits in [('one_bit_max',1),('two_bit_max',2),('exact_max',None)]:
            key=k if bits is None else engine.quantize(k,bits,32)
            logits=np.einsum('htd,hkd->htk',q,engine.repeat(key,q.shape[0]))
            scores[name]=logits.reshape(q.shape[0],len(text),8,8).max(axis=-1).mean(axis=(0,1))
        return scores
    # Runtime callback only: the frozen original source file is never edited.
    engine.score_arms=extended_scores
    baseline={c['image']:c for c in json.loads(BASELINE.read_text())['cases']};cases=[]
    for entry in json.loads((traces/'manifest.json').read_text())['cases']:
        path=traces/entry['npz']
        if sha(path)!=entry['npz_sha256']:raise ValueError('trace hash mismatch')
        case=engine.run_case(path);old=baseline[case['image']]
        if any(case['primary_summary'][name]!=values for name,values in old['primary_summary'].items()):raise AssertionError('callback changed a baseline')
        case['rows']=[r for r in case['rows'] if r['arm'] in NEW_ARMS]
        case['assignments']={layer:{n:b for n,b in values.items() if n in NEW_ARMS} for layer,values in case['assignments'].items()}
        case.pop('one_bit_development_gate')
        case['one_bit_max_beats_one_bit_mean']=case['primary_summary']['one_bit_max']['relative_l2']<case['primary_summary']['one_bit_mean_logit']['relative_l2']
        cases.append(case)
    r={'date':'2026-11-17','freeze':freeze,'cases':cases,'unchanged_baselines_reproduced':True,'numpy':np.__version__,
       'one_bit_max_improves_both_images':all(c['one_bit_max_beats_one_bit_mean'] for c in cases),
       'train_eligible':False,'novelty_established':False,'scope':plan['scope']}
    (a.output/'result.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({'one_bit_max_improves_both_images':r['one_bit_max_improves_both_images'],'summaries':{c['image']:{k:v for k,v in c['primary_summary'].items() if k in NEW_ARMS or k in ('one_bit_mean_logit','two_bit_mean_logit','exact_softmax_mass')} for c in cases}},indent=2))
if __name__=='__main__':main()
