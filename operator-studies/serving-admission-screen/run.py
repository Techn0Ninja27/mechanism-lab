"""Frozen synthetic development screen; never treats trace metrics as GPU speed."""
import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from simulator import JobSpec, simulate

HERE=Path(__file__).resolve().parent
ARMS=('fifo','largest_prompt','smallest_reservation','smallest_prompt','smallest_prefill','fifo_backpressure')


def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()


def make_cases():
    cases=[]
    for arrival_kind in ('staggered','bursts'):
        for association in ('independent','positive','negative'):
            for seed in range(10):
                rng=random.Random(91000+seed)
                jobs=[]; arrival=0
                for i in range(24):
                    p=rng.randint(1,8); raw=rng.randint(1,12); noise=rng.randint(-2,2)
                    d=raw if association=='independent' else max(1,min(12,(2*p if association=='positive' else 18-2*p)+noise))
                    arrival += rng.choice((0,1,1,2,3))
                    jobs.append(dict(id=i,arrival=arrival if arrival_kind=='staggered' else (i//8)*12,
                                     prompt=p,prefill_slots=math.ceil(p/2),true_output=d,lower_bound=1))
                for capacity in (8,16,32):
                    cases.append(dict(id=f'{arrival_kind}-{association}-s{seed}-p{capacity}',
                                      arrival_kind=arrival_kind,association=association,seed=seed,
                                      prefill_capacity=capacity,decode_capacity=24,horizon=1000,jobs=jobs))
    return cases


def quantile_nearest(values,q):
    return sorted(values)[math.ceil(q*len(values))-1]


def run_case(case,arm):
    specs=[JobSpec(**s) for s in case['jobs']]
    try:
        r=simulate(specs,prefill_capacity=case['prefill_capacity'],decode_capacity=case['decode_capacity'],
                   policy='fifo' if arm=='fifo_backpressure' else arm,horizon=case['horizon'],
                   backpressure_fraction=.5 if arm=='fifo_backpressure' else None)
    except RuntimeError as exc:
        if 'horizon exhausted' not in str(exc): raise
        return {'status':'incomplete_at_frozen_horizon','arm':arm}
    flow=[r['completion_times'][s.id]-s.arrival for s in specs]
    blocked=0
    for slot in r['trace']:
        unassigned=[s.prompt for s in specs if slot['states'][str(s.id)]=='unassigned']
        blocked += bool(unassigned and slot['prefill_usage']+min(unassigned)>case['prefill_capacity'])
    return {'status':'completed','arm':arm,'mean_flow_time':statistics.mean(flow),
            'sum_completion_time':r['sum_completion_time'],'p95_flow_time':quantile_nearest(flow,.95),
            'p99_flow_time':quantile_nearest(flow,.99),'max_flow_time':max(flow),'restarts':r['restarts'],
            'prefill_capacity_blocked_slots':blocked,'slots':len(r['trace']),'trace_sha256':digest(r)}


def summarize(rows):
    groups=[]
    for arrival in ('staggered','bursts'):
        for association in ('independent','positive','negative'):
            for capacity in (8,16,32):
                group=[r for r in rows if (r['arrival_kind'],r['association'],r['prefill_capacity'])==(arrival,association,capacity)]
                complete=all(a['status']=='completed' for r in group for a in r['arms'])
                ratios=[];tails=[]
                if complete:
                    for r in group:
                        arms={a['arm']:a for a in r['arms']};c=arms['largest_prompt']
                        # Best comparator per metric on this trace; a conservative
                        # comparison envelope, not a realizable oracle scheduler.
                        controls=[v for k,v in arms.items() if k!='largest_prompt']
                        ratios.append(c['mean_flow_time']/min(a['mean_flow_time'] for a in controls))
                        tails.append(c['p99_flow_time']/min(a['p99_flow_time'] for a in controls))
                med=statistics.median(ratios) if complete else None
                tail=statistics.median(tails) if complete else None
                groups.append({'arrival_kind':arrival,'association':association,'prefill_capacity':capacity,
                               'traces':len(group),'all_arms_completed':complete,
                               'median_mean_flow_ratio_to_best_control':med,'median_p99_ratio_to_best_control':tail,
                               'candidate_gate':complete and med<=.95 and tail<=1.05,
                               'primary_group':capacity==8})
    passed=all(g['candidate_gate'] for g in groups if g['primary_group'])
    return {'status':'eligible_for_broader_screen' if passed else 'fails_frozen_local_development_gate',
            'groups':groups,'case_count':len(rows),'arm_runs':sum(len(r['arms']) for r in rows),
            'incomplete_arm_runs':sum(a['status']!='completed' for r in rows for a in r['arms']),
            'novelty_established':False,'train_eligible':False,
            'scope':'Synthetic development screen only; no untouched efficacy test, serving throughput, hardware speed, or novelty claim.'}


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if a.output.exists():raise FileExistsError('Use a fresh output directory')
    cases=make_cases();frozen=json.loads((HERE/'freeze.json').read_text())
    if digest(cases)!=frozen['case_digest']:raise ValueError('Case generation changed')
    for name,h in frozen['code_hashes'].items():
        if hashlib.sha256((HERE/name).read_bytes()).hexdigest()!=h:raise ValueError('Frozen code changed')
    a.output.mkdir(parents=True);(a.output/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
    rows=[]
    for case in cases:
        rows.append({k:v for k,v in case.items() if k!='jobs'}|{'arms':[run_case(case,arm) for arm in ARMS]})
    (a.output/'raw.json').write_text(json.dumps(rows,indent=2)+'\n')
    result=summarize(rows);result['raw_sha256']=digest(rows);(a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='groups'}))
