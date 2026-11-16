"""Frozen development codec replay. Produces byte counts, not timing claims."""
import argparse,hashlib,importlib.util,json,statistics,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
# Avoid collision with Python's built-in codecs module.
spec=importlib.util.spec_from_file_location('walk_codecs',HERE/'codecs.py');codec=importlib.util.module_from_spec(spec);spec.loader.exec_module(codec)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def array_sha(a):return hashlib.sha256(a.astype('<u4').tobytes()).hexdigest()

def walks(indptr,indices,rng,n,l):
    current=rng.integers(0,len(indptr)-1,size=n,dtype=np.uint32)
    result=np.empty((n,l),dtype=np.uint32);result[:,0]=current
    for step in range(1,l):
        degree=indptr[current+1]-indptr[current]
        # PCG64 integer sampling, with per-walk exclusive upper bounds.
        offset=rng.integers(0,degree,size=n)
        current=indices[indptr[current]+offset];result[:,step]=current
    return result

def select_hot(prob,length):
    c=1/length+1/32
    prefix=np.concatenate([[0.],np.cumsum(prob[:-1])])
    g=1-prefix+np.arange(len(prob))*c
    passes=prob-c >= .1*g
    failures=np.flatnonzero(~passes);greedy=int(failures[0]) if len(failures) else len(prob)
    good=np.flatnonzero(passes);last=int(good[-1]+1) if len(good) else 0
    return {'greedy':greedy,'global_last_passing_add':last}

def run(out,data):
    out.mkdir(parents=True,exist_ok=True)
    result_path=out/'results.json'
    if result_path.exists():raise FileExistsError('fresh output required')
    protocol=json.loads((HERE/'protocol.json').read_text())
    if sys.version_info[:3] != (3,12,14) or np.__version__!='2.5.4':raise ValueError('use pinned Python3.12.14 / NumPy2.5.4')
    sources=json.loads((data/'sources.json').read_text())
    freeze={'protocol_sha256':sha(HERE/'protocol.json'),'sources':sources,
            'code_sha256':{p.name:sha(p) for p in [HERE/'run.py',HERE/'codecs.py',HERE/'prepare.py']},
            'python':sys.version.split()[0],'numpy':np.__version__}
    plan=out/'freeze.json'
    if plan.exists() and json.loads(plan.read_text())!=freeze:raise ValueError('changed frozen inputs')
    if not plan.exists():plan.write_text(json.dumps(freeze,indent=2)+'\n')
    rows=[]
    for source in protocol['datasets']:
        sid=source['id'];arrays=np.load(data/(sid+'-csr.npz'),allow_pickle=False)
        indptr=arrays['indptr'];indices=arrays['indices'];original_ids=arrays['original_ids'];v=len(indptr)-1
        if hashlib.sha256(indptr.tobytes()+indices.tobytes()+original_ids.tobytes()).hexdigest()!=sources[sid]['csr_arrays_sha256']:raise ValueError('CSR hash mismatch')
        for seed in protocol['seeds']:
            rng=np.random.Generator(np.random.PCG64(seed));n=protocol['walks_per_round'];l=protocol['walk_length']
            first=walks(indptr,indices,rng,n,l);counts=np.bincount(first.ravel(),minlength=v)
            rank_to_id=np.lexsort((np.arange(v),-counts)).astype(np.uint32)
            prob=counts[rank_to_id]/first.size;ks=select_hot(prob,l)
            names=['raw','packed','varint']+[f'{c}:{selector}' for c in ['fabc32','fabc_packed'] for selector in ks]
            totals={name:0 for name in names};round_rows=[]
            for round_id in range(2,protocol['rounds']+1):
                sample=walks(indptr,indices,rng,n,l);stats={}
                for name in names:
                    kind,_,selector=name.partition(':');hot=rank_to_id[:ks[selector]].tolist() if selector else []
                    mapping=rank_to_id if kind=='varint' else None
                    blob=codec.encode(sample,kind,v,hot_ids=hot,rank_to_id=mapping)
                    restored=codec.decode(blob,rank_to_id=mapping)
                    if not np.array_equal(sample,restored):raise AssertionError('round-trip mismatch: '+name)
                    totals[name]+=len(blob)
                    stats[name]={'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest(),'exact_roundtrip':True}
                round_rows.append({'round':round_id,'walks_sha256':array_sha(sample),'codecs':stats})
            dictionary_bytes=rank_to_id.astype('<u4').tobytes()
            best_fabc=min(totals['fabc32:'+s] for s in ks)
            best_packed=min([totals['packed']]+[totals['fabc_packed:'+s] for s in ks])
            candidate=totals['varint']+len(dictionary_bytes)
            row={'graph':sid,'seed':seed,'nodes':v,'profile_walks_sha256':array_sha(first),
                 'profile_top_two_mass':float(prob[:2].sum()),'profile_seen_nodes':int(np.count_nonzero(counts)),
                 'hot_counts':ks,'rank_dictionary_bytes':len(dictionary_bytes),'rank_dictionary_sha256':hashlib.sha256(dictionary_bytes).hexdigest(),
                 'evaluation_positions':n*l*(protocol['rounds']-1),'payload_bytes':totals,
                 'varint_total_bytes':candidate,'best_fabc_bytes':best_fabc,'best_packed_bytes':best_packed,
                 'charged_ratio_to_fabc':candidate/best_fabc,'charged_ratio_to_packed':candidate/best_packed,
                 'resident_ratio_to_fabc':totals['varint']/best_fabc,'resident_ratio_to_packed':totals['varint']/best_packed,
                 'rounds':round_rows}
            rows.append(row)
            print(json.dumps({k:row[k] for k in ['graph','seed','hot_counts','charged_ratio_to_packed','resident_ratio_to_packed']}),flush=True)
    summary={}
    for source in protocol['datasets']:
        selected=[r for r in rows if r['graph']==source['id']]
        metrics={k:statistics.median(r[k] for r in selected) for k in ['charged_ratio_to_fabc','charged_ratio_to_packed','resident_ratio_to_fabc','resident_ratio_to_packed','profile_top_two_mass']}
        metrics['pass_gate']=metrics['charged_ratio_to_fabc']<=.85 and metrics['charged_ratio_to_packed']<=.90
        summary[source['id']]=metrics
    result={'status':'passes_development_gate' if all(r['pass_gate'] for r in summary.values()) else 'fails_development_gate',
            'scope':protocol['scope'],'summary':summary,'rows':rows,'novelty_established':False,'train_eligible':False,'human_adjudicated':False,
            'freeze_sha256':sha(plan),'all_roundtrips_exact':True}
    result_path.write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--data',type=Path,required=True);a=p.parse_args();r=run(a.output,a.data);print(json.dumps({'status':r['status'],'summary':r['summary']}))
