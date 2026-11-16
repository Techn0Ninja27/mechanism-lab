"""Frozen validation selection and one final pass for typed integer compositions."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
from pathlib import Path
import statistics
import struct
import sys
import time
import zlib

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from generation.integer_codecs import registry
from generation.operator_composition import compose_chain
from fixture import load,SEEDS

HERE=Path(__file__).resolve().parent
HEADER=struct.Struct('<2sBHI')
BASELINE_CHAINS=[('raw64',),('raw64','zlib'),('raw64','lzma'),
    ('delta','zigzag','varint'),('delta','zigzag','varint','zlib'),
    ('delta2','zigzag','varint','zlib'),('block_for',),('delta','block_for'),('delta','block_for','zlib')]
BLOCK_OPTIONS=[('raw64',),('raw64','zlib'),('delta','zigzag','varint'),
    ('delta2','zigzag','varint'),('zigzag','rle'),('for',),('delta','for')]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def mean(xs):return statistics.mean(xs)


def library():
    reg=registry();chains=reg.enumerate_chains('signed_ints',output_type='bytes',max_depth=4,max_candidates=4096)
    chains=sorted(chains,key=lambda c:tuple(o.operator_id for o in c.operators))
    methods={'.'.join(o.operator_id for o in c.operators):c for c in chains}
    return reg,methods


REG,METHODS=library()
BLOCKS=[compose_chain(REG,ids) for ids in BLOCK_OPTIONS]
METHOD_IDS=sorted(METHODS)+['known_adaptive_blocks','known_constant_blocks']


def block_encode(values,constant_only=False):
    out=bytearray(struct.pack('<I',len(values)))
    for start in range(0,len(values),128):
        block=values[start:start+128]
        choices=[4] if constant_only and len(set(block))==1 else [0] if constant_only else list(range(len(BLOCKS)))
        payloads=[(len(payload:=BLOCKS[i].execute(block)),i,payload) for i in choices]
        _,kind,payload=min(payloads)
        out.extend(struct.pack('<BI',kind,len(payload)));out.extend(payload)
    return bytes(out)


def block_decode(data):
    if len(data)<4:raise ValueError('missing block count')
    n=struct.unpack('<I',data[:4])[0]
    if n>1_000_000:raise ValueError('count too large')
    pos=4;out=[]
    while len(out)<n:
        if pos+5>len(data):raise ValueError('truncated block header')
        kind,size=struct.unpack('<BI',data[pos:pos+5]);pos+=5
        if kind>=len(BLOCKS) or pos+size>len(data):raise ValueError('invalid block')
        block=BLOCKS[kind].decode(data[pos:pos+size]);pos+=size
        if len(block)!=min(128,n-len(out)):raise ValueError('wrong block length')
        out.extend(block)
    if pos!=len(data):raise ValueError('trailing bytes')
    return out


def encode(name,values):
    if name=='known_adaptive_blocks':payload=block_encode(values)
    elif name=='known_constant_blocks':payload=block_encode(values,True)
    else:payload=METHODS[name].execute(values)
    return HEADER.pack(b'ML',1,METHOD_IDS.index(name),len(values))+payload


def decode(data):
    if len(data)<HEADER.size:raise ValueError('truncated frame')
    magic,version,index,count=HEADER.unpack(data[:HEADER.size])
    if magic!=b'ML' or version!=1 or index>=len(METHOD_IDS) or count>1_000_000:raise ValueError('invalid frame')
    name=METHOD_IDS[index];payload=data[HEADER.size:]
    xs=block_decode(payload) if name.startswith('known_') else METHODS[name].decode(payload)
    if len(xs)!=count:raise ValueError('wrong frame count')
    return xs


def measure(name,xs):
    payload=encode(name,xs)
    if decode(payload)!=xs:raise ValueError('not lossless')
    enc=[];dec=[]
    for _ in range(3):
        t=time.perf_counter();current=encode(name,xs);enc.append(time.perf_counter()-t)
        if current!=payload:raise ValueError('nondeterministic encoding')
        t=time.perf_counter();decoded=decode(current);dec.append(time.perf_counter()-t)
        if decoded!=xs:raise ValueError('not lossless')
    return {'bytes':len(payload),'bytes_per_sample':len(payload)/len(xs),'payload_sha256':hashlib.sha256(payload[HEADER.size:]).hexdigest(),
            'lossless':True,'encode_seconds_median':statistics.median(enc),'decode_seconds_median':statistics.median(dec),
            'encode_seconds_range':[min(enc),max(enc)],'decode_seconds_range':[min(dec),max(dec)]}


def sources():
    paths=[Path(__file__),HERE/'fixture.py',HERE/'plan.json',ROOT/'generation/operator_composition.py',ROOT/'generation/integer_codecs.py',ROOT/'generation/molecules.py']
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


def run(mode):
    target=HERE/('validation.json' if mode=='validation' else 'results.json')
    if target.exists():raise FileExistsError('Preserve existing results')
    baseline_names=['.'.join(x) for x in BASELINE_CHAINS]+['known_adaptive_blocks','known_constant_blocks']
    frozen=None
    if mode=='final':
        frozen=json.loads((HERE/'freeze.json').read_text())
        if frozen['source_hashes']!=sources() or frozen['validation_sha256']!=sha(HERE/'validation.json'):raise ValueError('Frozen artifacts changed')
        names=sorted(set([frozen['candidate']]+baseline_names))
    else:names=METHOD_IDS
    record={'status':'running','source_hashes':sources(),'mode':mode,'rows':[],'errors':[],
            'environment':{'python':platform.python_version(),'platform':platform.platform(),'zlib_runtime':zlib.ZLIB_RUNTIME_VERSION},
            'common_header_bytes':HEADER.size,'catalog':METHOD_IDS,'novelty_established':False}
    save(target,record)
    try:
        splits=('development','validation') if mode=='validation' else ('final',)
        for split in splits:
            for seed in SEEDS[split]:
                xs=list(load(split,seed,allow_final=mode=='final'))
                for name in names:
                    row={'split':split,'seed':seed,'method':name,'input_sha256':hashlib.sha256(json.dumps(xs).encode()).hexdigest()}
                    try:row.update(measure(name,xs))
                    except Exception as exc:row.update(lossless=False,error=type(exc).__name__+': '+str(exc));record['errors'].append(row)
                    record['rows'].append(row)
        def aggregate(name,split):
            rows=[r for r in record['rows'] if r['method']==name and r['split']==split]
            return mean([r['bytes_per_sample'] for r in rows]) if len(rows)==3 and all(r['lossless'] for r in rows) else float('inf')
        split='validation' if mode=='validation' else 'final'
        if mode=='validation':
            candidate=min(METHODS,key=lambda n:(aggregate(n,split),n))
            control=min(baseline_names,key=lambda n:(aggregate(n,split),n))
            per_seed=[min((r for r in record['rows'] if r['split']==split and r['seed']==seed and r['method'] in METHODS and r['lossless']),key=lambda r:(r['bytes_per_sample'],r['method']))['method'] for seed in SEEDS[split]]
            stable=all(n==candidate for n in per_seed)
            passed=stable and aggregate(candidate,split)<=.99*aggregate(control,split)
            record['selection']={'candidate':candidate,'control':control,'per_seed_winners':per_seed,'stable':stable,'validation_passed':passed}
        else:
            candidate=frozen['candidate'];control=frozen['control']
            passed=frozen['validation_passed'] and aggregate(candidate,split)<=.99*aggregate(control,split)
        record['summary']={'candidate':candidate,'control':control,'candidate_bytes_per_sample':aggregate(candidate,split),'control_bytes_per_sample':aggregate(control,split),
                           'candidate_to_control_ratio':aggregate(candidate,split)/aggregate(control,split),'status':'supported_pilot' if passed else 'no_gain'}
        record['status']='complete'
    except Exception as exc:record['status']='failed';record['error']=type(exc).__name__+': '+str(exc);raise
    finally:target.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps(record['summary'],indent=2))


def freeze():
    validation=json.loads((HERE/'validation.json').read_text())
    if validation['status']!='complete' or validation['source_hashes']!=sources():raise ValueError('Validation or source mismatch')
    selection=validation['selection']
    save(HERE/'freeze.json',{'source_hashes':sources(),'validation_sha256':sha(HERE/'validation.json'),**selection,
                           'scope':'One final descriptive run; failed validation cannot be reversed.'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['validation','freeze','final']);a=p.parse_args()
    freeze() if a.mode=='freeze' else run(a.mode)
