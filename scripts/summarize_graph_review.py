"""Validate and summarize blinded machine labels; never infer scientific validity."""
import argparse,collections,hashlib,json
from pathlib import Path
LABELS={'direct_interface_supported','requires_specific_bridge','unsupported_connection','non_operational_producer','insufficient_information'}

def summarize(folder):
    packet_path=folder/'packet.json';packet=json.loads(packet_path.read_text());key=json.loads((folder/'key.json').read_text())
    manifest=json.loads((folder/'manifest.json').read_text())
    digest=hashlib.sha256(packet_path.read_bytes()).hexdigest()
    if digest!=manifest['packet_sha256'] or hashlib.sha256((folder/'key.json').read_bytes()).hexdigest()!=manifest['key_sha256']:
        raise ValueError('packet/key integrity mismatch')
    ids=[r['id'] for r in packet['items']]
    if len(ids)!=len(set(ids)) or {r['id'] for r in key}!=set(ids) or len(key)!=len(ids):raise ValueError('packet/key IDs')
    judges={}
    for name in ['a','b']:
        obj=json.loads((folder/f'reviewer-{name}.json').read_text())
        if obj['packet_sha256']!=digest:raise ValueError('review belongs to another packet')
        rows=obj['reviews']
        if len(rows)!=len(ids) or {r['id'] for r in rows}!=set(ids):raise ValueError('missing or duplicated reviewed IDs')
        for r in rows:
            if r.get('label') not in LABELS or r.get('confidence') not in {'low','medium','high'}:raise ValueError('review labels')
            for field,limit in [('reason',400),('missing_contract_or_bridge',500)]:
                if not isinstance(r.get(field),str) or not r[field].strip() or len(r[field])>limit:raise ValueError('invalid explanation')
        judges[name]={r['id']:r for r in rows}
    cohorts={}
    for group in sorted({m['group'] for r in key for m in r['memberships']}):
        selected=[r['id'] for r in key if any(m['group']==group for m in r['memberships'])]
        if len(selected)!=8:raise ValueError('expected eight exposed cases per cohort')
        cohorts[group]={'cases':len(selected),'ids':selected,
            'reviewer_counts':{j:dict(collections.Counter(rows[i]['label'] for i in selected)) for j,rows in judges.items()},
            'both_require_specific_bridge':[i for i in selected if all(rows[i]['label']=='requires_specific_bridge' for rows in judges.values())]}
    disagreements=[{'id':i,'a':judges['a'][i]['label'],'b':judges['b'][i]['label']} for i in ids if judges['a'][i]['label']!=judges['b'][i]['label']]
    return {'date':'2026-11-16','packet_sha256':digest,'unique_cases':len(ids),'exact_agreement':len(ids)-len(disagreements),
            'cohorts':cohorts,'disagreements':disagreements,'novelty_established':False,'train_eligible':False,
            'scope':'Two separate native-agent reviews, same model family, no external novelty search. Overlapping exposed top-eight cohorts; not independent samples, human labels, a usefulness benchmark or general retrieval-quality evidence.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve summary')
    result=summarize(a.input);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
