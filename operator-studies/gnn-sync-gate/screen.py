"""Causal offline screening for a sync-revealed predictor/hold gate.

This is a trajectory replay, not a distributed-training or accuracy benchmark.
Only embeddings observed at synchronization epochs update the gate.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

def load_cora(root: str | Path):
    root=Path(root); rows=[]; labels=[]; ids={}
    for line in (root/'cora.content').read_text().splitlines():
        p=line.split(); ids[p[0]]=len(rows); rows.append([float(x) for x in p[1:-1]]); labels.append(p[-1])
    classes={x:i for i,x in enumerate(sorted(set(labels)))}
    y=np.array([classes[x] for x in labels],dtype=np.int64); x=np.asarray(rows,np.float32)
    x/=np.maximum(x.sum(1,keepdims=True),1.0)
    edges=set()
    for line in (root/'cora.cites').read_text().splitlines():
        a,b=line.split()[:2]
        if a in ids and b in ids:
            u,v=ids[a],ids[b]; edges.update(((u,v),(v,u)))
    n=len(x); edges.update((i,i) for i in range(n)); edges=sorted(edges)
    ei=np.asarray(edges,np.int64).T
    deg=np.bincount(ei[0],minlength=n).astype(np.float32); w=1/np.sqrt(deg[ei[0]]*deg[ei[1]])
    adj=torch.sparse_coo_tensor(torch.from_numpy(ei),torch.from_numpy(w),(n,n)).coalesce()
    return torch.from_numpy(x),torch.from_numpy(y),adj

def split_labels(y, seed=123):
    g=np.random.default_rng(seed); train=[]
    for c in np.unique(y.numpy()):
        ix=np.flatnonzero(y.numpy()==c); g.shuffle(ix); train.extend(ix[:20])
    remain=np.setdiff1d(np.arange(len(y)),np.asarray(train)); g.shuffle(remain)
    return np.asarray(train),remain[:500]

def train_trajectory(x,y,adj,seed=0,lr_drop=False,epochs=120):
    torch.manual_seed(seed); np.random.seed(seed)
    w1=torch.nn.Parameter(torch.randn(x.shape[1],16)*0.05); w2=torch.nn.Parameter(torch.randn(16,int(y.max())+1)*0.05)
    opt=torch.optim.Adam([w1,w2],lr=.01,weight_decay=.0005); tr,va=split_labels(y)
    out=[]; losses=[]
    for t in range(epochs):
        if lr_drop and t==60:
            for q in opt.param_groups:q['lr']*=.1
        h=torch.relu(torch.sparse.mm(adj,x@w1)); z=torch.sparse.mm(adj,h@w2)
        out.append(h.detach().cpu().numpy().astype(np.float32)); losses.append(float(torch.nn.functional.cross_entropy(z[tr],y[tr]).detach()))
        opt.zero_grad(); torch.nn.functional.cross_entropy(z[tr],y[tr]).backward(); opt.step()
    return np.stack(out), {'train_loss_final':losses[-1], 'val_accuracy':float((z[va].argmax(1)==y[va]).float().mean()), 'epochs':epochs}

def _predict(last, numer, denom, dt):
    return last+(numer/np.maximum(denom[:,None],1e-6))*dt*np.exp(-.2*dt)

def _observe(last, numer, denom, value, dt):
    """SNI-style exponentially weighted slope update, including self-feed."""
    decay=np.exp(-.2*max(1,dt)); slope=(value-last)/max(1,dt)
    return value.copy(), numer*decay+slope, denom*decay+1.

def replay_gate(trajectory, partitions, *, boundary_pairs=None, sync=4, warmup=20, eta=1.0, share=.01, mixture=.5, debug=False):
    """Return causal aggregate metrics. trajectory shape [T,N,D]."""
    a=np.asarray(trajectory,np.float32); p=np.asarray(partitions); T,N,D=a.shape; P=int(p.max())+1
    pairs=sorted(boundary_pairs) if boundary_pairs is not None else [(i,d) for i in range(N) for d in range(P) if d!=p[i]]
    group_keys=sorted({(int(p[i]),int(d)) for i,d in pairs}); groups={key:k for k,key in enumerate(group_keys)}; G=len(groups)
    M=len(pairs); src=np.asarray([i for i,d in pairs]); gids=np.asarray([groups[(int(p[i]),int(d))] for i,d in pairs])
    last=a[0,src].copy(); numer=np.zeros((M,D),np.float32); denom=np.ones(M,np.float32); hold=a[0,src].copy(); logw=np.zeros((G,2),np.float64)
    # Warmup observations are real observations and establish a nonzero trend.
    for t in range(1,min(warmup+1,T)):
        dt=1; last,numer,denom=_observe(last,numer,denom,a[t,src],dt); hold=a[t,src].copy()
    losses={k:[] for k in ('gate','predictor','hold','mixture')}; pred_trace=[]; hold_trace=[]; weight_trace=[]
    pending=[]; update_counts=np.zeros(G,dtype=int); agreement=ties=nonties=0
    def rel2(pred, truth): return np.sqrt(np.sum((pred-truth)**2,axis=1))/(np.sqrt(np.sum(truth**2,axis=1))+1e-8)
    def feedback(pred, hold_value, truth):
        nonlocal logw
        sq=np.stack([rel2(pred,truth)**2,rel2(hold_value,truth)**2],axis=1); grouped=np.zeros((G,2),np.float64)
        for g in range(G):
            ix=(gids==g); grouped[g]=sq[ix].mean(axis=0)
        # endpoint has two columns supplied; aggregate all pairs before update.
        scores=logw-eta*grouped
        scores-=scores.max(1,keepdims=True)
        z=np.exp(scores); z/=z.sum(1,keepdims=True); logw=np.log((1-share)*z+share/2)
    for t in range(warmup+1,T):
        at_sync=((t-warmup)%sync)==0
        dt=1  # state is updated every epoch, including real sync epochs
        pred=_predict(last,numer,denom,max(1,dt)); h=hold.copy(); truth=a[t,src]
        q=logw[gids]-logw[gids].max(1,keepdims=True); ww=np.exp(q); ww/=ww.sum(1,keepdims=True)
        used=ww[:,0,None]*pred+ww[:,1,None]*h
        if not at_sync:
            losses['gate'].extend(rel2(used,truth).tolist()); losses['predictor'].extend(rel2(pred,truth).tolist()); losses['hold'].extend(rel2(h,truth).tolist()); losses['mixture'].extend(rel2(mixture*pred+(1-mixture)*h,truth).tolist())
            pending.append(np.sign(rel2(pred,truth)-rel2(h,truth)))  # evaluator-only
        else:
            # Freeze and score endpoint before consuming fresh synchronization truth.
            if pending:
                feedback(pred,h,truth)
                ep_pred,ep_hold=rel2(pred,truth),rel2(h,truth)
                ew=np.sign(ep_pred-ep_hold)
                for sw in pending:
                    valid=(ew!=0)&(sw!=0)
                    ties += int(np.sum(~valid)); nonties += int(np.sum(valid))
                    agreement += int(np.sum((ew==sw)&valid))
                update_counts += 1
                pending=[]
            last,numer,denom=_observe(last,numer,denom,truth,1); hold=truth.copy()
        if not at_sync:
            # SNI's self-feeding state receives the prediction, never the gate
            # mixture and never the unavailable fresh remote embedding.
            last,numer,denom=_observe(last,numer,denom,pred,1)
        if debug:
            pred_trace.append(pred.copy()); hold_trace.append(h.copy()); weight_trace.append(ww.copy())
    result={'n_pairs':M,'n_groups':G,'group_keys':[list(x) for x in group_keys],'aggregate':{k:float(np.mean(v)) if v else None for k,v in losses.items()},'raw_counts':{k:len(v) for k,v in losses.items()},'eta':eta,'sync':sync,'warmup':warmup,'endpoint_updates_per_group':update_counts.tolist(),'endpoint_winner_agreement':{'agreements':agreement,'non_tie_comparisons':nonties,'ties':ties,'rate':(agreement/nonties if nonties else None)}}
    if debug: result['debug']={'predictions':np.asarray(pred_trace).tolist(),'holds':np.asarray(hold_trace).tolist(),'weights':np.asarray(weight_trace).tolist(),'holds_are_separate':bool(np.any(np.asarray(pred_trace)!=np.asarray(hold_trace)))}
    return result

def run_all(data_root, output=None):
    """Run the frozen four-seed/two-schedule screening and tune on seeds 0/1."""
    x,y,adj=load_cora(data_root); parts=np.random.default_rng(321).integers(0,4,size=len(y)); ei=adj.indices().cpu().numpy(); boundary=sorted({(int(i),int(parts[j])) for i,j in zip(ei[0],ei[1]) if parts[i]!=parts[j]})
    runs={}
    for seed in range(4):
        for drop in (False,True):
            h,quality=train_trajectory(x,y,adj,seed,drop); key=f'seed{seed}_'+('lr_drop' if drop else 'constant')
            runs[key]={'seed':seed,'lr_drop':drop,'quality_smoke':quality,'trajectory_shape':list(h.shape),'boundary_pairs':len(boundary),'gates':{str(e):replay_gate(h,parts,boundary_pairs=boundary,eta=e) for e in (.1,1.,10.)},'mixtures':{str(m):replay_gate(h,parts,boundary_pairs=boundary,mixture=m)['aggregate'] for m in (0,.25,.5,.75,1)}}
    dev=[r for k,r in runs.items() if r['seed'] in (0,1)]
    eta_scores={str(e):float(np.mean([r['gates'][str(e)]['aggregate']['gate'] for r in dev])) for e in (.1,1.,10.)}; mix_scores={str(m):float(np.mean([r['mixtures'][str(m)]['mixture'] for r in dev])) for m in (0,.25,.5,.75,1)}
    selected_eta=min(eta_scores,key=eta_scores.get); selected_mix=min(mix_scores,key=mix_scores.get); screen={}
    for drop in (False,True):
        rows=[r for r in runs.values() if r['seed'] in (2,3) and r['lr_drop']==drop]
        g=float(np.mean([r['gates'][selected_eta]['aggregate']['gate'] for r in rows]))
        controls={m:float(np.mean([r['mixtures'][m]['mixture'] for r in rows])) for m in set(['0','1',selected_mix])}
        best=min(controls.values())
        screen['lr_drop' if drop else 'constant']={'selected_gate':g,'selected_mixture':selected_mix,'fixed_controls':controls,'best_fixed_control':best,'relative_reduction_vs_best_fixed':(best-g)/max(best,1e-12),'go_rule_pass':(best-g)>=.05*best}
    result={'schema':'gnn-sync-gate-suite.v1','scope':'offline synchronous trajectory replay; development seeds 0/1 tune, screening seeds 2/3; no downstream accuracy/hardware claim','partition_seed':321,'boundary_pairs':len(boundary),'development_selection':{'eta_scores':eta_scores,'mixture_scores':mix_scores,'selected_eta':selected_eta,'selected_mixture':selected_mix},'runs':runs,'screening':screen}
    if output: Path(output).parent.mkdir(parents=True,exist_ok=True); Path(output).write_text(json.dumps(result,indent=2)+'\n')
    return result

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',required=True); ap.add_argument('--output',required=True); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--lr-drop',action='store_true'); ap.add_argument('--suite',action='store_true'); a=ap.parse_args()
    if a.suite: run_all(a.data,a.output); return
    x,y,adj=load_cora(a.data); h,quality=train_trajectory(x,y,adj,a.seed,a.lr_drop); rng=np.random.default_rng(321); parts=rng.integers(0,4,size=len(y)); result={'schema':'gnn-sync-gate-screen.v1','scope':'offline synchronous trajectory replay; no downstream accuracy or hardware claim','seed':a.seed,'lr_drop':a.lr_drop,'quality_smoke':quality,'trajectory_shape':list(h.shape),'partitions_seed':321,'controls':{},'implementation':'Fixed Share is a known delayed-feedback expert rule; this screen tests application utility only.'}
    ei=adj.indices().cpu().numpy(); boundary=sorted({(int(i),int(parts[j])) for i,j in zip(ei[0],ei[1]) if parts[i]!=parts[j]})
    result['boundary_pairs']=len(boundary)
    for eta in (.1,1.,10.): result['controls'][f'gate_eta_{eta}']=replay_gate(h,parts,boundary_pairs=boundary,eta=eta)
    result['controls']['fixed_mixture_0']=replay_gate(h,parts,boundary_pairs=boundary,mixture=0.)['aggregate']; result['controls']['fixed_mixture_25']=replay_gate(h,parts,boundary_pairs=boundary,mixture=.25)['aggregate']; result['controls']['fixed_mixture_50']=replay_gate(h,parts,boundary_pairs=boundary,mixture=.5)['aggregate']; result['controls']['fixed_mixture_75']=replay_gate(h,parts,boundary_pairs=boundary,mixture=.75)['aggregate']; result['controls']['fixed_mixture_100']=replay_gate(h,parts,boundary_pairs=boundary,mixture=1.)['aggregate']
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); Path(a.output).write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__': main()
