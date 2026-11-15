"""Cached literature retrieval and blinded model triage; no novelty ground truth."""
import argparse,concurrent.futures,json,pathlib,random,sys,time,urllib.request,urllib.parse,fcntl
from world_search import Client
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from generation.evidence_review import build_review_prompt,validate_review
from generation.world_evaluation import blind_cases_with_mapping,summarize

def retrieve(case,out):
 target=out/'evidence'/f'{case["id"]}.json';target.parent.mkdir(exist_ok=True)
 if target.exists():return json.loads(target.read_text())
 records={};logs=[]
 for query in case['proposal']['queries']:
  url='https://api.openalex.org/works?'+urllib.parse.urlencode({'search':query,'per-page':4,'select':'id,title,doi,publication_date,abstract_inverted_index,primary_location'})
  try:
   with urllib.request.urlopen(url,timeout=30) as r:data=json.load(r)
   for w in data.get('results',[]):
    inv=w.get('abstract_inverted_index')
    if not inv:continue
    text=' '.join(word for _,word in sorted((i,word) for word,pos in inv.items() for i in pos))
    records[w['id']]={'id':w['id'].split('/')[-1],'title':w['title'],'abstract':text,'url':w.get('doi') or (w.get('primary_location') or {}).get('landing_page_url') or w['id'],'date':w.get('publication_date')}
   logs.append({'query':query,'count':len(data.get('results',[]))})
  except Exception as exc:logs.append({'query':query,'error':type(exc).__name__})
  time.sleep(.2)
 result={'records':list(records.values()),'queries':logs,'coverage':'Two query strings, top4 each, abstracts only. Absence is not novelty.'};target.write_text(json.dumps(result,indent=2));return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=pathlib.Path,required=True);a=p.parse_args();out=a.run/'evaluation';out.mkdir(exist_ok=True)
 with (out/'run.lock').open('w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  records=[json.loads(f.read_text()) for f in sorted((a.run/'candidates').glob('*.json'))];valid=[r for r in records if r['status']=='complete']
  blind, blind_mapping = blind_cases_with_mapping(valid, seed=42)
  (out/'blind-cases.json').write_text(json.dumps(blind,indent=2))
  # Keep source IDs separate from the payload supplied to reviewers. This
  # permits paired analysis without leaking arm/source metadata to the judge.
  (out/'blind-id-map.json').write_text(json.dumps(blind_mapping,indent=2,sort_keys=True))
  blind_by_source = {source_id: next(item for item in blind if item['blind_id'] == blind_id)
                     for blind_id, source_id in blind_mapping.items()}
  config={'model':'deepseek/deepseek-v3.2','max_tokens_per_stage':[2048],'budget_usd':.25,'pricing':{'prompt':.00000026,'completion':.00000038}}
  client=Client(out,config)
  def one(case):
   target=out/f'{case["id"]}-review.json'
   if target.exists():return
   evidence=retrieve(case,out)
   prompt=build_review_prompt(json.dumps(blind_by_source[case['id']]['idea']),evidence['records'])
   result={'id':case['id'],'human_adjudicated':False,'train_eligible':False}
   try:
    raw=client.call({'id':case['id']},0,prompt).strip()
    if raw.startswith('```'):raw='\n'.join(raw.splitlines()[1:-1])
    result.update(valid=True,review=validate_review(json.loads(raw),evidence['records']))
   except Exception as exc:result.update(valid=False,error=str(exc)[:300])
   target.write_text(json.dumps(result,indent=2));print(case['id'],result.get('review',{}).get('decision',result.get('error')),flush=True)
  with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(one,valid))
  reviews=[json.loads(f.read_text()) for f in out.glob('*-review.json')]
  summary=summarize(records,[r['review'] for r in reviews if r.get('valid')]);summary['review_validation_failures']=sum(not r.get('valid') for r in reviews)
  (out/'summary.json').write_text(json.dumps(summary,indent=2))
if __name__=='__main__':main()
