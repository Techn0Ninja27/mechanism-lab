"""Cheap falsification of grounded-03's proposed fixed point; no training."""
import hashlib
import json
from pathlib import Path
import numpy as np


def error(a, b):
    return float(np.linalg.norm(a-b)/np.linalg.norm(b))


def main():
    rows=[]
    for seed in (18011,18013,18017):
        rng=np.random.default_rng(seed)
        q,_=np.linalg.qr(rng.normal(size=(5,5)))
        sigma=q@np.diag(np.linspace(.8,1.4,5))@q.T
        # Fresh observation patterns, all quantities here are an oracle diagnostic.
        observed=np.flatnonzero(rng.random(5)>.25)
        if len(observed)<2:observed=np.array([0,1])
        s=sigma[np.ix_(observed,observed)];eye=np.eye(len(s))
        c=float(np.linalg.eigvalsh(s).max());w=eye-s/c
        d=np.diag(np.diag(eye-w));wrong=np.linalg.solve(d,w)
        target=np.linalg.inv(s)
        wrong_fixed=np.linalg.solve(eye-wrong,eye)
        z=np.zeros_like(s)
        for _ in range(200):z=wrong@z+eye
        diagonal=np.diag(np.diag(s));inv_diagonal=np.linalg.inv(diagonal)
        half=np.diag(1/np.sqrt(np.diag(s)));standard=half@s@half
        eig=np.linalg.eigvalsh(standard);eta=2/(eig.min()+eig.max())
        jacobi=np.zeros_like(s);standardized=np.zeros_like(s)
        discrepancies=[]
        for _ in range(20):
            jacobi=jacobi+eta*inv_diagonal@(eye-s@jacobi)
            standardized=standardized+eta*(half-standard@standardized)
            discrepancies.append(float(np.max(np.abs(jacobi-half@standardized))))
        # Allow a best global readout scale to rule out an irrelevant unit mismatch.
        scale=float(np.sum(wrong_fixed*target)/np.sum(wrong_fixed**2))
        h=rng.normal(size=len(s));p=np.diag((rng.random(len(s))>.2).astype(float))
        folded=np.linalg.solve(d,w)
        rows.append({'seed':seed,'observed_features':observed.tolist(),
            'wrong_iteration_spectral_radius':float(max(abs(np.linalg.eigvals(wrong)))),
            'wrong_iterate_vs_own_fixed_point':error(z,wrong_fixed),
            'wrong_fixed_point_vs_inverse':error(wrong_fixed,target),
            'wrong_fixed_point_best_global_scale_error':error(scale*wrong_fixed,target),
            'correct_jacobi_vs_inverse':error(jacobi,target),
            'jacobi_standardization_max_abs_difference':max(discrepancies),
            'diagonal_fold_into_weight_max_abs_difference':float(np.max(np.abs(np.linalg.solve(d,p@w@h)-p@folded@h)))})
    result={'seeds':[18011,18013,18017], 'rows':rows,
        'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'status':'reject_formula_as_written', 'novelty_established':False,
        'scope':'Algebraic/numerical fixed-point diagnostic, no predictor trained and no utility claim. Fixed diagonal folding is a representational equivalence, not identical training dynamics.'}
    Path(__file__).with_name('results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
