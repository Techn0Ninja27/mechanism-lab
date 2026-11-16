"""Validate saved extraction atoms independently; no edits to quotes or API calls."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.literature_atomization import load_papers,batches,validate_atoms,_parse_json

def recover(papers_path,run,out):
    if out.exists():raise FileExistsError('Preserve existing recovery')
    plan=json.loads((run/'plan.json').read_text())
    if hashlib.sha256(papers_path.read_bytes()).hexdigest()!=plan['input_sha256']:raise ValueError('Input differs from original run')
    validator=ROOT/'scripts/literature_atomization.py'
    if hashlib.sha256(validator.read_bytes()).hexdigest()!=plan['source_hashes']['scripts/literature_atomization.py']:raise ValueError('Original validator changed')
    papers=load_papers(papers_path);accepted=[];rejected=[];receipts=[]
    for i,batch in enumerate(batches(papers)):
        path=run/'calls'/f'atom-batch-{i:03d}-0.json'
        if not path.exists():
            rejected.append({'batch':i,'error':'No saved call'});continue
        raw=path.read_bytes();record=json.loads(raw);receipts.append({'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()})
        if record.get('status')!='complete':
            rejected.append({'batch':i,'error':'Call incomplete; no retry'});continue
        try:
            response=_parse_json(record['content'])
            if not isinstance(response,dict) or not isinstance(response.get('atoms'),list):raise ValueError('No atoms list')
        except Exception as exc:
            rejected.append({'batch':i,'error':str(exc)});continue
        for j,atom in enumerate(response['atoms']):
            try:
                validated=validate_atoms({'atoms':[atom]},batch)[0]
                validated['extraction_location']={'batch':i,'atom_index':j}
                accepted.append(validated)
            except Exception as exc:rejected.append({'batch':i,'atom_index':j,'source_id':atom.get('source_id') if isinstance(atom,dict) else None,'error':str(exc)})
    out.mkdir(parents=True)
    with (out/'atoms.jsonl').open('x') as f:
        for atom in accepted:f.write(json.dumps(atom,ensure_ascii=False)+'\n')
    summary={'atoms':len(accepted),'covered_papers':len({a['source_id'] for a in accepted}),'rejected':rejected,
      'source_receipts':receipts,'original_plan_sha256':hashlib.sha256((run/'plan.json').read_bytes()).hexdigest(),
      'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'new_api_calls':0,'quotes_modified':False,'scope':'Per-atom structural recovery; semantic review and training eligibility unchanged.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');return summary
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--papers',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=recover(a.papers,a.run,a.output);print(json.dumps({k:v for k,v in r.items() if k!='source_receipts'},indent=2))
