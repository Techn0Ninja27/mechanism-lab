"""Train and evaluate task-conditioned utility rankings on fresh codec tasks."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
import joblib
import sklearn
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from generation.integer_codecs import registry
from generation.operator_composition import compose_chain
from generation.utility_ranker import feature_vector,fit_models,predict
from fixture import task,COUNTS
HERE=Path(__file__).resolve().parent
CHAINS=[['raw64'],['raw64','zlib'],['raw64','lzma'],['for'],['for','zlib'],['for','lzma'],
        ['block_for'],['block_for','zlib'],['block_for','lzma'],['delta','for'],['delta','for','zlib'],['delta','for','lzma'],
        ['delta','block_for'],['delta','block_for','zlib'],['delta2','zigzag','varint','zlib'],
        ['delta','zigzag','varint'],['delta','zigzag','varint','zlib'],['delta','zigzag','varint','lzma'],['zigzag','rle'],['zigzag','rle','zlib']]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(path,obj):
    with Path(path).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')
def hashes():
    paths=[Path(__file__),HERE/'fixture.py',HERE/'plan.json',ROOT/'generation/utility_ranker.py',ROOT/'generation/integer_codecs.py',ROOT/'generation/operator_composition.py']
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


def dataset(split,out,allow=False):
    target=out/(split+'.npz');meta_path=out/(split+'-metadata.json')
    if target.exists() or meta_path.exists():raise FileExistsError('Dataset already exists')
    reg=registry();chains=[compose_chain(reg,c) for c in CHAINS]
    x=[];y=[];rows=[];feature_names=None;tasks=[]
    for i in range(COUNTS[split]):
        obj=task(split,i,allow_heldout=allow);xs=obj.pop('values')
        obj['input_sha256']=hashlib.sha256(json.dumps(xs).encode()).hexdigest();tasks.append(obj)
        for ci,c in enumerate(chains):
            features,feature_names=feature_vector(xs,CHAINS[ci]);x.append(features)
            row={'task':obj['id'],'family':obj['family'],'candidate':ci,'operators':CHAINS[ci]}
            try:
                payload=c.execute(xs)
                if c.decode(payload)!=xs:raise ValueError('non-lossless')
                size=len(payload)+9;gain=1-size/(8*len(xs)+9);row.update(bytes=size,gain=gain,lossless=True)
            except Exception as exc:gain=float('nan');row.update(lossless=False,error=type(exc).__name__+': '+str(exc))
            y.append(gain);rows.append(row)
        if i%20==0:print(split,i,flush=True)
    features=np.asarray(x);gains=np.asarray(y)
    np.savez_compressed(target,x=features,y=gains)
    meta={'split':split,'tasks':tasks,'rows':rows,'feature_names':feature_names,'candidate_ids':['.'.join(c) for c in CHAINS],
          'invalid_count':sum(not r['lossless'] for r in rows),'source_hashes':hashes()}
    dump(meta_path,meta)
    return features,gains,meta


def selection_scores(model,x,y,meta,fixed):
    pred=predict(model,x).reshape(-1,len(CHAINS));truth=y.reshape(-1,len(CHAINS));truth=np.where(np.isfinite(truth),truth,-np.inf)
    chosen=np.argmax(pred,axis=1);rng=np.random.default_rng(21007)
    rows=[]
    for i,obj in enumerate(meta['tasks']):
        oracle=int(np.argmax(truth[i]));pick=int(chosen[i]);random=int(rng.choice(np.flatnonzero(np.isfinite(truth[i]))))
        # A failed execution falls back to raw and is recorded rather than selected with oracle knowledge.
        failed=not np.isfinite(truth[i,pick]);actual=0 if failed else float(truth[i,pick]);best=float(truth[i,oracle])
        rows.append({'task':obj['id'],'family':obj['family'],'selected':pick,'selected_failed':failed,'oracle':oracle,
            'gain':actual,'oracle_gain':best,'regret':best-actual,'fixed_gain':float(truth[i,fixed]),
            'fixed_regret':best-float(truth[i,fixed]),'random_gain':float(truth[i,random]),'random_regret':best-float(truth[i,random])})
    summary={k:float(np.mean([r[k] for r in rows])) for k in ('gain','oracle_gain','regret','fixed_gain','fixed_regret','random_gain','random_regret')}
    summary['top1_oracle_rate']=float(np.mean([r['selected']==r['oracle'] for r in rows]));summary['tasks_beating_raw']=float(np.mean([r['gain']>0 for r in rows]))
    summary['per_family']={f:{k:float(np.mean([r[k] for r in rows if r['family']==f])) for k in ('regret','fixed_regret','random_regret','gain')} for f in sorted({r['family'] for r in rows})}
    deltas=np.asarray([r['fixed_regret']-r['regret'] for r in rows]);boot=np.random.default_rng(21009).choice(deltas,(2000,len(deltas)),replace=True).mean(axis=1)
    summary['paired_regret_reduction_vs_fixed_ci95']=np.quantile(boot,[.025,.975]).tolist()
    return {'summary':summary,'tasks':rows}


def run(mode,out):
    out.mkdir(parents=True,exist_ok=True)
    if mode=='train':
        if (out/'validation.json').exists():raise FileExistsError('Preserve validation')
        x,y,train=dataset('train',out);xv,yv,val=dataset('validation',out)
        assert not {r['stream_group'] for r in train['tasks']}&{r['stream_group'] for r in val['tasks']}
        assert not {r['input_sha256'] for r in train['tasks']}&{r['input_sha256'] for r in val['tasks']}
        valid=np.isfinite(y);groups=np.asarray([r['task'] for r in train['rows']])
        models=fit_models(x[valid],y[valid],groups[valid]);fixed=int(np.nanargmax(y.reshape(-1,len(CHAINS)).mean(axis=0)))
        results={name:selection_scores(model,xv,yv,val,fixed) for name,model in models.items()}
        chosen=min(results,key=lambda n:(results[n]['summary']['regret'],n));joblib.dump(models[chosen],out/'model.joblib')
        record={'mode':'validation','selected_model':chosen,'fixed_best_train_chain':fixed,'fixed_best_train_operators':CHAINS[fixed],
                'models':results,'source_hashes':hashes(),'model_sha256':sha(out/'model.joblib'),'feature_count':x.shape[1],
                'train_shape':list(x.shape),'validation_shape':list(xv.shape),'chain_catalog':CHAINS,
                'environment':{'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__,'joblib':joblib.__version__},
                'invalid_counts':{'train':train['invalid_count'],'validation':val['invalid_count']},'novelty_established':False}
        dump(out/'validation.json',record);print(json.dumps({'selected':chosen,'summary':results[chosen]['summary']},indent=2))
    elif mode=='freeze':
        record=json.loads((out/'validation.json').read_text())
        if record['source_hashes']!=hashes() or record['model_sha256']!=sha(out/'model.joblib'):raise ValueError('Source/model changed')
        dump(out/'freeze.json',{'source_hashes':hashes(),'model_sha256':sha(out/'model.joblib'),'validation_sha256':sha(out/'validation.json'),
            'selected_model':record['selected_model'],'fixed':record['fixed_best_train_chain'],'scope':'One heldout evaluation; no final tuning'})
    else:
        freeze=json.loads((out/'freeze.json').read_text())
        if freeze['source_hashes']!=hashes() or freeze['model_sha256']!=sha(out/'model.joblib') or freeze['validation_sha256']!=sha(out/'validation.json'):raise ValueError('Frozen artifacts changed')
        if (out/'final.json').exists():raise FileExistsError('Preserve final')
        x,y,meta=dataset('heldout',out,allow=True);model=joblib.load(out/'model.joblib')
        result=selection_scores(model,x,y,meta,freeze['fixed']);s=result['summary']
        passed=s['regret']<s['fixed_regret'] and s['regret']<s['random_regret'] and all(f['regret']<f['fixed_regret'] for f in s['per_family'].values())
        result.update(status='supported_narrow_pilot' if passed else 'not_supported',source_hashes=hashes(),model_sha256=freeze['model_sha256'],
                      selected_model=freeze['selected_model'],invalid_count=meta['invalid_count'],novelty_established=False,
                      scope='Compression utility ranking on synthetic heldout families; not general research usefulness, feasibility, or novelty prediction.')
        dump(out/'final.json',result);print(json.dumps({'status':result['status'],'summary':s},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','freeze','final']);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.mode,a.output)
