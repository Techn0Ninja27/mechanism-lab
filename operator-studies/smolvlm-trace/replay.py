"""Independently reconstruct captured eager attention in NumPy on CPU."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def attention(q,k,start):
    if q.shape[1]%k.shape[1]:raise ValueError('GQA head ratio')
    repeated=np.repeat(k.astype(np.float64),q.shape[1]//k.shape[1],axis=1)
    scores=np.einsum('lhqd,lhkd->lhqk',q.astype(np.float64),repeated)/np.sqrt(q.shape[-1])
    future=np.arange(k.shape[-2])[None,:]>np.arange(start,start+q.shape[-2])[:,None]
    scores=np.where(future[None,None,:,:],-np.inf,scores)
    scores-=scores.max(axis=-1,keepdims=True)
    probability=np.exp(scores);probability/=probability.sum(axis=-1,keepdims=True)
    return probability

def verify(run):
    manifest=json.loads((run/'manifest.json').read_text());plan=json.loads((run/'plan.json').read_text())
    if sha(run/'plan.json')!=manifest['plan_sha256']:raise ValueError('plan hash mismatch')
    if sha(run/'dependencies.lock')!=plan['dependencies_sha256']:raise ValueError('dependency hash mismatch')
    for image in plan['images']:
        if sha(run/'images'/(image['name']+'.jpg'))!=image['sha256']:raise ValueError('image hash mismatch')
    rows=[]
    for row in manifest['cases']:
        path=run/row['npz']
        if sha(path)!=row['npz_sha256']:raise ValueError('trace hash mismatch')
        with np.load(path,allow_pickle=False) as z:
            if any(not np.isfinite(z[k]).all() for k in z.files):raise ValueError('nonfinite trace')
            visual=np.flatnonzero(z['input_ids']==row['image_token_id'])
            if len(visual)!=64 or not np.array_equal(visual,z['visual_positions']):raise ValueError('visual identification')
            truth=z['prefill_attention'];actual=attention(z['prefill_q'],z['prefill_k'],0)
            error=float(np.max(np.abs(actual-truth)))
            sliced=truth[:,:,z['prompt_text_positions'],:][:,:,:,visual]
            if not np.array_equal(sliced,z['prefill_text_visual_attention']):raise ValueError('prefill slice')
            k=z['prefill_k'].copy();decode_errors=[]
            for step in range(4):
                k=np.concatenate([k,z['decode_k_new'][step]],axis=2)
                predicted=attention(z['decode_q'][step],k,len(z['input_ids'])+step)[:,:,0,:]
                recorded=z[f'decode_attention_{step}']
                decode_errors.append(float(np.max(np.abs(predicted-recorded))))
                if not np.array_equal(recorded[:,:,visual],z['decode_visual_attention'][step]):raise ValueError('decode visual slice')
            if max([error]+decode_errors)>2e-5:raise AssertionError('Q/K replay disagrees with model attention')
            rows.append({'image':row['image'],'prefill_max_abs_error':error,'decode_max_abs_error':max(decode_errors),
                         'visual_tokens':len(visual),'visual_key_variation':float(z['decode_visual_attention'].std(axis=-1).max()),
                         'npz_sha256':row['npz_sha256']})
    return {'status':'independent_numpy_attention_replay_passed','cases':rows,'verifier_sha256':sha(__file__),
            'producer_script_sha256':manifest['script_sha256'],'numpy':np.__version__,
            'scope':'Checks extraction consistency only; no intervention, predictive quality, runtime or novelty test.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve replay result')
    result=verify(a.run);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
