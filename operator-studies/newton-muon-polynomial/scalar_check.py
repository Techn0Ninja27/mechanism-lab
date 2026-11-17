"""Scalar residual-map replay of the published Newton--Muon Table 5 plans."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent

def evaluate(poly, r):
    out = np.zeros_like(r, dtype=np.float64)
    for coefficient in reversed(poly): out = out * r + coefficient
    return out

def replay(plan, grid):
    residual = grid.copy()
    peak = float(np.max(np.abs(residual)))
    per_step = []
    for poly in plan["polynomials"]:
        q = evaluate(poly, residual)
        residual = 1.0 - (1.0 - residual) * q
        peak = max(peak, float(np.max(np.abs(residual))))
        per_step.append(float(np.max(np.abs(residual))))
    return per_step, peak

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--points", type=int, default=1_000_001)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("Preserve existing result; choose a new output")
    if type(args.points) is not int or args.points < 1001 or args.points > 5_000_001: raise ValueError("points must be 1001..5000001")
    plans_path = HERE / "plans.json"; payload = json.loads(plans_path.read_text())
    grid_by_epsilon = {str(e): np.linspace(0.0, 1.0-e, args.points, dtype=np.float64) for e in sorted({p['epsilon'] for p in payload['plans']})}
    rows=[]
    for i, plan in enumerate(payload['plans']):
        steps, peak = replay(plan, grid_by_epsilon[str(plan['epsilon'])])
        max_final=steps[-1]
        rows.append({'index':i,'epsilon':plan['epsilon'],'total_sypp':plan['total_sypp'],'published_s_out':plan['s_out'],'numeric_final_residual':max_final,'numeric_peak_residual':peak,'final_within_published_plus_1e-5':max_final <= plan['s_out'] + 1e-5,'peak_within_one':peak <= 1.0 + 1e-12,'step_maxima':steps,'polynomial_count':len(plan['polynomials']),'sypp_sum':sum(plan['sypp'])})
    result={'schema':'newton-muon-polynomial-scalar-check.v1','scope':'Float64 scalar residual-map replay on a dense initial interval. This checks transcription and numerical behavior only; it is not a matrix proof, source-certificate verification, or optimizer performance result.','source':payload['source'],'points':args.points,'plans_checked':len(rows),'rows':rows,'all_finals_within_published_plus_1e-5':all(r['final_within_published_plus_1e-5'] for r in rows),'all_peaks_within_one':all(r['peak_within_one'] for r in rows),'plans_sha256':hashlib.sha256(plans_path.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'plans':len(rows),'all_finals':result['all_finals_within_published_plus_1e-5'],'all_peaks':result['all_peaks_within_one']}))
if __name__=='__main__': main()
