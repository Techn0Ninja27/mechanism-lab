"""Fixed-budget local attention intervention on exposed SmolVLM traces."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def finite(x):
    if not np.isfinite(x).all():raise AssertionError('nonfinite numeric value')

def quantize(x,bits,group):
    """x: KV-head, sequence, channel; min/scale per sequence group/channel."""
    finite(x)
    if bits==16:return x.astype(np.float16).astype(np.float64)
    if bits not in (1,2,4) or x.shape[1]%group:raise ValueError('invalid quantization layout')
    shape=x.shape;g=x.reshape(shape[0],shape[1]//group,group,shape[2]).astype(np.float64)
    lo=g.min(axis=2,keepdims=True).astype(np.float32).astype(np.float64)
    scale=((g.max(axis=2,keepdims=True)-lo)/(2**bits-1)).astype(np.float32).astype(np.float64)
    safe=np.where(scale>0,scale,1.)
    codes=np.clip(np.rint((g-lo)/safe),0,2**bits-1)
    out=(lo+codes*scale).reshape(shape);finite(out);return out

def assign(scores):
    finite(scores)
    if scores.shape!=(8,):raise ValueError('expected eight scores')
    order=sorted(range(1,8),key=lambda i:(-scores[i],i))
    bits=np.empty(8,dtype=np.int32);bits[0]=16
    for i,b in zip(order,[16,4,4,4,2,2,2]):bits[i]=b
    assert sorted(bits.tolist())==[2,2,2,4,4,4,16,16]
    return bits

def repeat(k,heads):
    if heads%k.shape[0]:raise ValueError('GQA mismatch')
    return np.repeat(k,heads//k.shape[0],axis=0)

def output(q,k,v):
    scores=np.einsum('hd,hkd->hk',q,repeat(k,q.shape[0]))/np.sqrt(q.shape[-1])
    scores-=scores.max(axis=-1,keepdims=True);weights=np.exp(scores);weights/=weights.sum(-1,keepdims=True)
    result=np.einsum('hk,hkd->hd',weights,repeat(v,q.shape[0]));finite(result);return result

def error(pred,truth):
    a=pred.ravel();b=truth.ravel();norm=np.linalg.norm(b)
    if norm<=1e-12 or np.linalg.norm(a)<=1e-12:raise AssertionError('degenerate output norm')
    return float(np.linalg.norm(a-b)/norm),float(np.dot(a,b)/(np.linalg.norm(a)*norm))

def score_arms(z,layer,visual,text):
    q=z['prefill_q'][layer][:,text,:].astype(np.float64)
    k=z['prefill_k'][layer][:,visual,:].astype(np.float64)
    embedding=z['aligned_input_embeddings'].astype(np.float64)
    embedding/=np.maximum(np.linalg.norm(embedding,axis=-1,keepdims=True),1e-12)
    proxy=np.einsum('td,kd->tk',embedding[text],embedding[visual]).mean(0).reshape(8,8).mean(1)
    scores={'embedding_cosine_proxy':proxy,'exact_softmax_mass':z['prefill_text_visual_attention'][layer].mean(axis=(0,1)).reshape(8,8).sum(1),
            'recency':np.arange(8,dtype=float),'position':-np.arange(8,dtype=float),
            'interleaved':np.array([8,1,5,2,7,3,6,4],dtype=float)}
    for name,bits in [('exact_mean_logit',None),('one_bit_mean_logit',1),('two_bit_mean_logit',2)]:
        key=k if bits is None else quantize(k,bits,32)
        logits=np.einsum('htd,hkd->htk',q,repeat(key,q.shape[0]))
        scores[name]=logits.mean(axis=(0,1)).reshape(8,8).mean(1)
    for seed in (73031,73033,73037):scores[f'random_{seed}']=np.random.default_rng(seed+layer).random(8)
    return scores

def run_case(path):
    rows=[];maps={};byte_rows=[]
    with np.load(path,allow_pickle=False) as z:
        visual=z['visual_positions'];text=z['prompt_text_positions']
        if len(visual)!=64:raise ValueError('fixture visual count')
        for layer in range(z['prefill_q'].shape[0]):
            k=z['prefill_k'][layer].astype(np.float64);v=z['prefill_v'][layer].astype(np.float64)
            assignments={name:assign(score) for name,score in score_arms(z,layer,visual,text).items()}
            maps[str(layer)]={n:b.tolist() for n,b in assignments.items()}
            variants={}
            for name,bits in {**assignments,'visual_fp16':np.full(8,16)}.items():
                ak=k.copy();av=v.copy()
                for w,b in enumerate(bits):
                    pos=visual[w*8:(w+1)*8]
                    ak[:,pos,:]=quantize(k[:,pos,:],int(b),8);av[:,pos,:]=quantize(v[:,pos,:],int(b),8)
                variants[name]=(ak,av)
                if layer==0:
                    h,d=k.shape[0],k.shape[-1];payload=int(sum(bits)*8*h*d*2/8)
                    metadata=int(sum(bits!=16)*h*d*2*2*4);permutation=32 if name!='visual_fp16' else 0
                    byte_rows.append({'arm':name,'visual_payload_bytes':payload,'quantizer_metadata_bytes':metadata,
                                      'permutation_bytes':permutation,'visual_persistent_bytes':payload+metadata+permutation})
            base_k=k.copy();base_v=v.copy()
            for step in range(4):
                nk=z['decode_k_new'][step,layer].astype(np.float64);nv=z['decode_v_new'][step,layer].astype(np.float64)
                base_k=np.concatenate([base_k,nk],axis=1);base_v=np.concatenate([base_v,nv],axis=1)
                q=z['decode_q'][step,layer,:,0,:].astype(np.float64);truth=output(q,base_k,base_v)
                for name,(ak,av) in list(variants.items()):
                    ak=np.concatenate([ak,nk],axis=1);av=np.concatenate([av,nv],axis=1);variants[name]=(ak,av)
                    rel,cos=error(output(q,ak,av),truth)
                    rows.append({'arm':name,'layer':layer,'step':step,'relative_l2':rel,'cosine':cos})
        summaries={}
        for name in variants:
            if len([r for r in rows if r['arm']==name])!=z['prefill_q'].shape[0]*4:raise AssertionError('unequal arm rows')
            primary=[r for r in rows if r['arm']==name and r['step']>=1]
            if len(primary)!=z['prefill_q'].shape[0]*3:raise AssertionError('missing primary rows')
            summaries[name]={'relative_l2':float(np.mean([r['relative_l2'] for r in primary])),
                             'cosine':float(np.mean([r['cosine'] for r in primary]))}
        controls=['embedding_cosine_proxy','exact_softmax_mass','exact_mean_logit','recency','position','interleaved']
        candidate=summaries['one_bit_mean_logit']
        gate=all(candidate['relative_l2']<=.95*summaries[n]['relative_l2'] and candidate['cosine']>=summaries[n]['cosine'] for n in controls)
        memory={'native_fp32_visual_bytes':64*k.shape[0]*k.shape[-1]*2*4,
                'unchanged_nonvisual_prefill_bytes':(k.shape[1]-64)*k.shape[0]*k.shape[-1]*2*4,
                'unchanged_bytes_per_generated_token':k.shape[0]*k.shape[-1]*2*4,
                'scope':'Analytical packed storage; packing not implemented. Not allocated NumPy memory or measured GPU memory. Allocator padding, kernel workspace, scoring and dequantization transient buffers excluded.'}
        return {'image':path.stem,'primary_summary':summaries,'rows':rows,'assignments':maps,'cache_accounting':byte_rows,'memory_scope':memory,
                'one_bit_development_gate':gate,'scope':'Teacher-forced local attention replay; no propagated model errors.'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--traces',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--freeze',action='store_true');a=p.parse_args();protocol=HERE/'protocol.json'
    freeze={'script_sha256':sha(__file__),'protocol_sha256':sha(protocol),'manifest_sha256':sha(a.traces/'manifest.json')}
    if a.freeze:
        if a.output.exists():raise FileExistsError('fresh freeze directory required')
        a.output.mkdir(parents=True);(a.output/'freeze.json').write_text(json.dumps(freeze,indent=2)+'\n');return
    if (a.output/'result.json').exists():raise FileExistsError('preserve result')
    if json.loads((a.output/'freeze.json').read_text())!=freeze:raise ValueError('frozen source changed')
    plan=json.loads(protocol.read_text());manifest=json.loads((a.traces/'manifest.json').read_text())
    if freeze['manifest_sha256']!=plan['source_manifest_sha256']:raise ValueError('input manifest differs from protocol')
    cases=[]
    for row in manifest['cases']:
        path=a.traces/row['npz']
        if sha(path)!=row['npz_sha256']:raise ValueError('trace digest mismatch')
        cases.append(run_case(path))
    result={'date':'2026-11-16','freeze':freeze,'cases':cases,'numpy':np.__version__,
            'train_eligible':False,'human_adjudicated':False,'novelty_established':False,
            'verdict':'development_gain_requires_heldout' if all(c['one_bit_development_gate'] for c in cases) else 'no_support_for_one_bit_in_exposed_pilot',
            'scope':plan['scope']}
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'verdict':result['verdict'],'summary':{c['image']:c['primary_summary'] for c in cases}},indent=2))
if __name__=='__main__':main()
