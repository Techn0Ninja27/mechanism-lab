"""Read-only secondary metric replay of the preserved development screen."""
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev

ROOT=Path(__file__).resolve().parents[2]
path=ROOT/'results/generation-comparison-stage0-v1/result.json'
raw=json.loads(path.read_text())
rows={(r['id'],r['seed']):r for r in raw['runs']}
assert len(rows)==len(raw['runs'])==112
seeds=list(range(100,116))
metrics={'window_exact':lambda r:mean(x['exact_eval'] for x in r['window']),
         'window_own':lambda r:mean(x['own_eval'] for x in r['window']),
         'endpoint_exact':lambda r:r['post_update_exact_eval']}
result={}
for arm in sorted({a for a,s in rows}):
    result[arm]={}
    for name,metric in metrics.items():
        differences=[100*(metric(rows[arm,s])-metric(rows['__hold__',s])) for s in seeds]
        result[arm][name]={'mean_difference_pp':mean(differences),'paired_sd_pp':stdev(differences),
                           'paired_se_pp':stdev(differences)/len(seeds)**.5,'positive_seeds':sum(x>0 for x in differences),'n':len(seeds)}
print(json.dumps({'raw_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'units':'percentage points','metrics':result},indent=2))
