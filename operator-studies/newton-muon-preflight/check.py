"""Exact algebra preflight; no training, timing, or utility claim."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

def psd(seed, n):
    rng=np.random.default_rng(seed); a=rng.normal(size=(n,n)); return a@a.T+0.25*np.eye(n)

def lowrank_inverse(k, rank, damping):
    eig,vec=np.linalg.eigh(k); order=np.argsort(eig)[::-1]; eig,vec=eig[order],vec[:,order]; n=k.shape[0]
    if rank==n: return vec@np.diag(1/(eig+damping))@vec.T,None
    rho=float((np.trace(k)-eig[:rank].sum())/(n-rank)); c=1/(rho+damping); v=vec[:,:rank]; d=1/(eig[:rank]+damping)-c
    return c*np.eye(n)+(v*d)@v.T,rho

def approximated_k(k, rank):
    eig,vec=np.linalg.eigh(k); order=np.argsort(eig)[::-1]; eig,vec=eig[order],vec[:,order]; n=k.shape[0]
    if rank==n: return k.copy(),None
    rho=float((np.trace(k)-eig[:rank].sum())/(n-rank)); v=vec[:,:rank]
    return rho*np.eye(n)+(v*(eig[:rank]-rho))@v.T,rho

def polar(x):
    u,_,vt=np.linalg.svd(x,full_matrices=False); return u@vt

def momentum_direction(grads, scales, beta):
    m=np.zeros_like(grads[0],dtype=float); out=[]
    for g,c in zip(grads,scales): m=beta*m+(1-beta)*(c*g); out.append(polar(m))
    return out

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); p.add_argument('--seed',type=int,default=73031); a=p.parse_args()
    rows=[]
    for n,seed in ((4,a.seed),(7,a.seed+1)):
        k=psd(seed,n); delta=.2*float(np.trace(k))/n
        for rank in (0,1,n-1,n):
            approx,rho=lowrank_inverse(k,rank,delta); k_approx,rho_check=approximated_k(k,rank); explicit=np.linalg.inv(k_approx+delta*np.eye(n)); rows.append({'n':n,'rank':rank,'damping':delta,'rho':rho,'max_abs_error_vs_explicit_inverse':float(np.max(np.abs(approx-explicit))),'exact_branch':rank==n})
    grads=[np.array([[1.,0.]]),np.array([[0.,1.]])]; beta=.9; varying=momentum_direction(grads,[1.,4.],beta); constant=momentum_direction(grads,[3.,3.],beta); baseline=momentum_direction(grads,[1.,1.],beta)
    vg=float(np.max(np.abs(varying[1]-constant[1]))); cg=float(np.max(np.abs(constant[1]-baseline[1])))
    result={'schema':'newton-muon-preflight.v1','scope':'Seeded algebra and momentum identities only; no training, timing, MPS, utility, novelty, or real-data claim.','seed':a.seed,'inverse_checks':rows,'inverse_max_error':max(x['max_abs_error_vs_explicit_inverse'] for x in rows),'momentum_check':{'beta':beta,'gradients':[g.tolist() for g in grads],'varying_scales':[1.,4.],'constant_scales':[3.,3.],'varying_scale_polar_gap':vg,'constant_scale_polar_gap':cg,'interpretation':'r=0 is scalar preconditioning at each step; varying positive scalars before momentum can change polar direction, while one constant scalar does not.'},'cost_notes':{'full_apply':'For G in R^(m x n), dense G P costs O(m n^2).','lowrank_apply':'The rank-r formula costs O(m n r) plus O(m n) for cG; apply savings require 2r<n up to constants.','refresh':'Dense eigendecomposition is O(n^3); iterative subspace extraction adds iteration-count and matrix-multiply cost. K formation/storage remain O(n^2) unless separately changed.','gate':'Accounting gates, not measured speed or utility results.'},'passed':all(x['max_abs_error_vs_explicit_inverse']<1e-12 for x in rows) and cg<1e-12 and vg>0.1}
    if not result['passed']: raise AssertionError('Algebra preflight failed')
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'inverse_max_error':result['inverse_max_error'],'varying_scale_polar_gap':vg,'constant_scale_polar_gap':cg}))
if __name__=='__main__': main()
