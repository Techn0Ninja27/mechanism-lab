"""Run bounded Haiku evidence comparisons; keep all labels unadjudicated."""
import os
import concurrent.futures,json,pathlib,sys,time,hashlib,urllib.request,urllib.error
from run import key
ROOT=pathlib.Path(os.environ.get('ALIEN_WORKSPACE', pathlib.Path(__file__).resolve().parent));OUT=ROOT/os.environ.get('PROPOSAL_OUTPUT', 'results/proposals-v3');sys.path.insert(0,os.environ.get('ALIEN_REPO', str(ROOT/'upstream')))
from generation.evidence_review import build_review_prompt,validate_review,ReviewValidationError

def main(mode):
 data=json.loads((OUT/'selection.json').read_text())
 def one(r):
  evidence=json.loads((OUT/'retrieval'/f'{r["id"]}-evidence.json').read_text())['records']
  directory=OUT/('control-reviews' if mode=='controls' else 'reviews');directory.mkdir(exist_ok=True)
  target=directory/(r['id']+'.json')
  if target.exists():return
  if mode=='controls':
   if not evidence:return
   proposal=evidence[0]['abstract']
  else:
   path=OUT/r['id']/'proposal.md'
   if not path.exists():return
   proposal=path.read_text()
  prompt=build_review_prompt(proposal,evidence)
  (directory/(r['id']+'-prompt.txt')).write_text(prompt)
  req=urllib.request.Request('https://openrouter.ai/api/v1/chat/completions',data=json.dumps({'model':'anthropic/claude-haiku-5.5','messages':[{'role':'user','content':prompt}],'temperature':0,'max_tokens':4096}).encode(),headers={'Authorization':'Bearer '+key(),'Content-Type':'application/json'})
  try:
   with urllib.request.urlopen(req,timeout=180) as response:d=json.load(response)
  except urllib.error.HTTPError as e:raise RuntimeError(f'OpenRouter HTTP {e.code}') from None
  choice=d['choices'][0];text=choice['message'].get('content') or ''
  record={'case':r['id'],'model':d.get('model'),'usage':d.get('usage'),'request_id':d.get('id'),'finish_reason':choice.get('finish_reason'),'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'evidence_ids':[e['id'] for e in evidence],'raw_response':text,'train_eligible':False,'expected_decision':'covered' if mode=='controls' else None}
  try:
   if choice.get('finish_reason')=='length':raise ValueError('Truncated response')
   cleaned=text.strip()
   if cleaned.startswith('```'):cleaned='\n'.join(cleaned.splitlines()[1:-1])
   record['review']=validate_review(json.loads(cleaned),evidence);record['valid']=True
  except (ValueError,ReviewValidationError) as e:record['valid']=False;record['validation_error']=str(e)
  target.write_text(json.dumps(record,indent=2));print(mode,r['id'],'valid=',record['valid'],record.get('review',{}).get('decision'),flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:list(ex.map(one,data['candidates']))
if __name__=='__main__':
 import fcntl
 with (OUT/'review.lock').open('w') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise SystemExit('Review worker already running')
  main(sys.argv[1])
