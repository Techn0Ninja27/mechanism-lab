#!/usr/bin/env python3
"""Small, seeded data/optimization pilot; never replaces released checkpoints."""
import argparse,json,random,sys,time,hashlib
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from generation.model import CausalTransformerLM
from generation.token_mapping import TokenMapping

def load_canonical(root):
    mapping=TokenMapping.load(root/'data/coherence_dataset/token_mapping.json')
    groups=set()
    for name in ['coherence_train.jsonl','coherence_val.jsonl']:
        with (root/'data/coherence_dataset'/name).open() as f:
            for line in f:
                atoms=tuple(sorted(set(mapping.token_ids_to_cluster_ids(json.loads(line)['input_ids']))))
                if len(atoms)>=2:groups.add(atoms)
    mapping,_=mapping.compact()
    return sorted(groups),mapping

def split_sets(sets,seed):
    ordered=list(sets);random.Random(seed).shuffle(ordered)
    n=max(1,len(ordered)//5)
    val,train=ordered[:n],ordered[n:]
    assert not set(val)&set(train)
    return train,val

def encode(atoms,mapping,rng):
    order=list(atoms);rng.shuffle(order)
    return mapping.cluster_ids_to_token_ids(order)

def collate(seqs,pad):
    x=torch.full((len(seqs),max(map(len,seqs))),pad,dtype=torch.long)
    mask=torch.zeros_like(x,dtype=torch.bool)
    for i,row in enumerate(seqs):x[i,:len(row)]=torch.tensor(row);mask[i,:len(row)]=True
    return x,mask

@torch.no_grad()
def evaluate(model,seqs,pad):
    model.eval();total=count=0
    for start in range(0,len(seqs),128):
        x,m=collate(seqs[start:start+128],pad);z=model(x,attention_mask=m)
        valid=m[:,1:];total+=F.cross_entropy(z[:,:-1][valid],x[:,1:][valid],reduction='sum').item();count+=valid.sum().item()
    return total/count

def unigram(train,val,vocab):
    counts=Counter(t for row in train for t in row[1:]);den=sum(counts.values())+vocab
    return float(-np.mean([np.log((counts[t]+1)/den) for row in val for t in row[1:]]))

def run(train,val,mapping,steps,seed,out,max_len):
    torch.manual_seed(seed);rng=random.Random(seed)
    model=CausalTransformerLM(mapping.compute_vocab_size(),d_model=128,nhead=4,num_layers=2,dim_feedforward=512,max_seq_len=max_len,dropout=0)
    opt=torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=.01)
    initial=evaluate(model,val,mapping.eos_token_id);model.train();start=time.perf_counter();losses=[]
    for step in range(steps):
        rows=[encode(train[rng.randrange(len(train))],mapping,rng) for _ in range(64)]
        x,m=collate(rows,mapping.eos_token_id);z=model(x,attention_mask=m);valid=m[:,1:]
        loss=F.cross_entropy(z[:,:-1][valid],x[:,1:][valid]);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(loss.item())
    train_seconds=time.perf_counter()-start;heldout=evaluate(model,val,mapping.eos_token_id)
    out.mkdir(parents=True,exist_ok=True);model.save(out);(out/'token_mapping.json').write_text(json.dumps(mapping.raw,indent=2))
    base_train=[mapping.cluster_ids_to_token_ids(s) for s in train]
    return {'training_sets':len(train),'steps':steps,'examples_seen':steps*64,'initial_nll':initial,'heldout_nll':heldout,'unigram_nll':unigram(base_train,val,mapping.compute_vocab_size()),'recent_train_nll':float(np.mean(losses[-25:])),'train_seconds':train_seconds,'parameters':sum(p.numel() for p in model.parameters()),'checkpoint':str(out),'seed':seed}

def main():
    p=argparse.ArgumentParser();p.add_argument('--steps',nargs='+',type=int,default=[100,400]);p.add_argument('--sizes',nargs='+',type=int,default=[500,2000,8000]);p.add_argument('--seed',type=int,default=42);p.add_argument('--output',type=Path,default=Path('results/retraining-corrected'));a=p.parse_args()
    if any(x<1 for x in a.steps+a.sizes):p.error('Steps/sizes must be positive')
    torch.set_num_threads(4);root=Path(__file__).resolve().parents[1];sets,mapping=load_canonical(root);train,val_sets=split_sets(sets,a.seed)
    rng=random.Random(a.seed+1);val=[encode(s,mapping,rng) for s in val_sets];max_len=max(map(len,sets))+2
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    split={'train_sets':[list(s) for s in train],'val_sets':[list(s) for s in val_sets]};(out/'split.json').write_text(json.dumps(split))
    summary={'seed':a.seed,'vocabulary_size':mapping.compute_vocab_size(),'unique_sets':len(sets),'train_sets':len(train),'heldout_sets':len(val_sets),'max_seq_len':max_len,'split_sha256':hashlib.sha256((out/'split.json').read_bytes()).hexdigest(),'protocol':'Disjoint canonical sets; randomized training order each draw; fixed randomized validation order; all-target NLL including EOS; fixed steps at each data size; one seed; CPU4threads. Not temporal, not quality evaluation.','results':[]}
    for steps in a.steps:
        for n in a.sizes:
            r=run(train[:min(n,len(train))],val,mapping,steps,a.seed,out/f'n{n}-steps{steps}',max_len);summary['results'].append(r);print(json.dumps(r),flush=True);(out/'summary.json').write_text(json.dumps(summary,indent=2))
if __name__=='__main__':main()
