"""Functional Table-5 inverse recurrence; generic dense products, not SYPP kernels."""
import json
from pathlib import Path
import numpy as np
import torch

PLANS=json.loads(Path(__file__).with_name('plans.json').read_text())['plans']


def select_plan(k, gamma=.2, budget=13, tolerance=.01):
    k=np.asarray(k,dtype=np.float64)
    if k.ndim!=2 or k.shape[0]!=k.shape[1] or not np.isfinite(k).all():raise ValueError('Finite square covariance required')
    n=k.shape[0];delta=float(gamma*np.trace(k)/n)
    if delta<=0:raise ValueError('Positive trace damping required')
    # Frobenius norm bounds largest eigenvalue for PSD K. Small relative slack
    # is conservative; PSD is a covariance assumption checked by the caller.
    upper=float(np.linalg.norm(k,'fro'))*1.001
    alpha=1/(upper+delta);epsilon=alpha*delta
    eligible=[(i,p) for i,p in enumerate(PLANS) if p['epsilon']<=epsilon and p['total_sypp']<=budget and p['s_out']<=tolerance]
    selected=min(eligible,key=lambda x:(x[1]['total_sypp'],x[1]['s_out'],x[0])) if eligible else None
    return {'plan_index':selected[0] if selected else None,'alpha':alpha,'delta':delta,'epsilon_bound':epsilon,'spectral_upper_bound':upper,'tolerance':tolerance,'sypp_budget':budget}


def evaluate_matrix_polynomial(r, coefficients):
    eye=torch.eye(r.shape[-1],dtype=r.dtype,device=r.device)
    powers={0:eye,1:r}
    def power(degree):
        if degree not in powers:
            if degree%2==0:
                half=power(degree//2);powers[degree]=half@half
            else:powers[degree]=power(degree-1)@r
        return powers[degree]
    return sum((coefficient*power(i) for i,coefficient in enumerate(coefficients) if coefficient),torch.zeros_like(r))


def inverse(k, selection):
    if selection['plan_index'] is None:raise ValueError('No eligible published plan; do not invent coefficients')
    plan=PLANS[selection['plan_index']]
    eye=torch.eye(k.shape[-1],dtype=k.dtype,device=k.device)
    alpha=selection['alpha'];a=k+selection['delta']*eye;r=eye-alpha*a;x=eye
    for step,poly in enumerate(plan['polynomials']):
        q=evaluate_matrix_polynomial(r,poly);x=x@q
        if step+1<len(plan['polynomials']):r=eye-(eye-r)@q
    return alpha*x
