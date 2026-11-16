"""Post-hoc strong polynomial controls for the frozen NeuMiss audit."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
from experiment import DEPTHS, DIMENSION, ETA, SEEDS, covariance, masks, relative_error
A, B = 0.35, 1.0
RICHARDSON_ETA = 2.0 / (A + B)
def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
def direct_recurrence(row, matrix, coeff):
    h = row @ matrix
    for c in coeff:
        h = h @ matrix + c * row
    return h
def neumann_inverse(sigma, depth, eta):
    a = np.eye(len(sigma)) - eta * sigma
    power = np.eye(len(sigma)); out = np.zeros_like(sigma)
    for _ in range(depth + 1):
        out += eta * power; power = power @ a
    return out
def chebyshev_inverse(sigma, degree):
    values, vectors = np.linalg.eigh(sigma)
    x = (A + B - 2.0 * values) / (B - A); x0 = (A + B) / (B - A)
    order = degree + 1
    q = np.cos(order * np.arccos(np.clip(x, -1.0, 1.0))) / np.cosh(order * np.arccosh(x0))
    return (vectors * ((1.0 - q) / values)) @ vectors.T
def candidate(block, depth, frozen):
    coeff = np.asarray(frozen['coefficients'][str(depth)], dtype=float)
    a = np.eye(len(block)) - ETA * block; powers = [np.eye(len(block))]
    for _ in range(depth): powers.append(powers[-1] @ a)
    return powers[depth] + sum(c * powers[depth-k] for c, k in zip(coeff, range(2, depth + 1)))
def main(output: Path):
    root = Path(__file__).resolve().parent; plan = json.loads((root/'plan.json').read_text()); frozen = json.loads((root/'results.json').read_text())
    sigmas = [covariance(s) for s in SEEDS]; development = [masks(s+1000,24) for s in SEEDS]; validation = [masks(s+2000,24) for s in SEEDS]
    rng = np.random.default_rng(17019); checks=[]
    for depth in DEPTHS:
        for _ in range(4):
            raw=rng.normal(size=(6,6)); mat=raw/np.linalg.norm(raw,2)*.3; row=rng.normal(size=6); coeff=rng.normal(size=depth-1)
            powers=[np.linalg.matrix_power(mat,depth)]+[np.linalg.matrix_power(mat,depth-k) for k in range(2,depth+1)]
            expected=row@powers[0]+sum(c*(row@b) for c,b in zip(coeff,powers[1:]))
            checks.append(float(np.max(np.abs(direct_recurrence(row,mat,coeff)-expected))))
    rows=[]
    for depth in DEPTHS:
        for split, patterns_by_seed in (("development",development),("validation",validation)):
            for seed,sigma,patterns in zip(SEEDS,sigmas,patterns_by_seed):
                for obs in patterns:
                    block=sigma[np.ix_(obs,obs)]; target=np.linalg.inv(block)
                    rows.append({'depth':depth,'split':split,'seed':seed,'candidate':relative_error(candidate(block,depth,frozen),target),'neumann_fixed_eta_same_degree':relative_error(neumann_inverse(block,depth,ETA),target),'richardson_optimal_eta_same_degree':relative_error(neumann_inverse(block,depth,RICHARDSON_ETA),target),'chebyshev_known_bounds_same_degree':relative_error(chebyshev_inverse(block,depth),target),'pattern_size':int(len(obs))})
    result={'schema_version':'neumiss-polynomial-strong-controls.v1','frozen_v1_results_sha256':sha(root/'results.json'),'frozen_v1_plan_sha256':sha(root/'plan.json'),'code_sha256':sha(Path(__file__)),'source_sha256':plan['source_sha256'],'design_bounds':[A,B],'fixed_eta':ETA,'richardson_eta':RICHARDSON_ETA,'degree_convention':'All controls use degree L, hence L+1 polynomial terms. The original v1 conventional comparator used L terms and is retained unchanged.','matvec_cost':{'candidate':'L shared A matvecs plus downstream Wc','neumann_controls':'L+1 A matvecs for degree L','chebyshev':'degree-L eigendecomposition oracle; runtime is not claimed'},'recurrence_max_abs_error':max(checks),'recurrence_checks':checks,'rows':rows,'interpretation':'Post-hoc operator comparison only; no training, predictive utility, or novelty claim.','no_validation_tuning':True}
    output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
if __name__=='__main__': main(Path(__file__).with_name('results-strong-controls.json'))
