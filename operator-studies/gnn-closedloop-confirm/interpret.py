"""Preserve historical labels while making their limited meaning explicit."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def interpret(summary_path, output):
    raw_path = summary_path.with_name('raw.json')
    summary_bytes = summary_path.read_bytes()
    s = json.loads(summary_bytes)
    raw_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    if s.get('schema') != 'gnn-closedloop-confirm-summary.v1' or s.get('raw_sha256') != raw_hash:
        raise ValueError('Unexpected summary or changed raw result')
    checks = s.get('integrity_checks')
    if not isinstance(checks, dict) or len(checks) != 7 or any(v is not True for v in checks.values()):
        raise ValueError('Original integrity checks did not all pass')
    primary = s['regimes']['moderate_lead']['comparisons']['PRIMARY_G1_minus_hold_window_exact']
    mean, se = primary['mean_pt'], primary['se_pt']
    low, high = primary['pointwise_ci95_pt']
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in (mean,se,low,high)) or not low <= mean <= high or se < 0:
        raise ValueError('Malformed primary comparison')
    allocation = mean > .5 and mean > 2*se
    if primary['passes_rule'] is not allocation or s['decision']['lead_confirmed_exact_forward'] is not allocation:
        raise ValueError('Legacy label does not match its allocation rule')
    out = {'schema':'gnn-confirmation-interpretation.v1',
           'status':'allocation_signal_not_scientific_confirmation',
           'source_summary_sha256':hashlib.sha256(summary_bytes).hexdigest(),
           'source_raw_sha256':raw_hash,
           'primary_mean_points':mean,'primary_se_points':se,
           'primary_pointwise_ci95_points':[low,high],
           'allocation_rule_passed':allocation,
           'primary_positive_at_pointwise_95':low > 0,
           'historical_label':'lead_confirmed_exact_forward means only >0.5 points and >2 SE; it is not a 95% confirmation flag.',
           'scope':'Eight fresh seeds in one selected Cora regime; fixed comparison, pointwise interval, no new ML method.',
           'limitations':['The interval is not a simultaneous or equivalence bound.',
                          'Own-forward versus exact-forward difference is a descriptive evaluation decomposition, not a causal attribution.',
                          'Unclamped extrapolators were tested; projection and bounded forecasts remain missing controls.'],
           'auto_execute':False,'train_eligible':False,'novelty_established':False}
    with output.open('x') as f: json.dump(out,f,indent=2);f.write('\n')
    return out


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=interpret(a.summary,a.output);print(json.dumps({'allocation_rule_passed':r['allocation_rule_passed'],'primary_positive_at_pointwise_95':r['primary_positive_at_pointwise_95']}))
