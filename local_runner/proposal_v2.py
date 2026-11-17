"""Select fresh candidates and decode evidence-bounded research proposals."""
import os
import concurrent.futures,hashlib,json,pathlib,re,sqlite3,sys,time,urllib.request,urllib.error,ssl
try:
 import certifi
 HTTPS_CONTEXT=ssl.create_default_context(cafile=certifi.where())
except ImportError:
 HTTPS_CONTEXT=ssl.create_default_context()
from run import key
ROOT=pathlib.Path(os.environ.get('ALIEN_WORKSPACE', pathlib.Path(__file__).resolve().parent));UP=pathlib.Path(os.environ.get('ALIEN_REPO', ROOT/'upstream'));sys.path.insert(0,str(UP))
from generation.exploratory_seeds import select_seed_triplets
OUT=ROOT/os.environ.get('PROPOSAL_OUTPUT', 'results/proposals-v3');OUT.mkdir(parents=True,exist_ok=True)
MAX_TOKENS=int(os.environ.get('DECODE_MAX_TOKENS', '16384'))
MODEL=os.environ.get('DECODE_MODEL', 'anthropic/claude-opus-5.5')
if not 1024 <= MAX_TOKENS <= 16384: raise ValueError('Output budget must be between 1024 and 16384')
def select():
 sourcepath=UP/'papers/clusters.json';cat=json.loads(sourcepath.read_text())
 reviewed=set()
 for p in (ROOT.parents[1]/'notes/2026-11-16-alien-science-review').glob('*.md'):
  m=re.search(r'(\d+)-(\d+)-(\d+)\.md$',p.name)
  if m:reviewed.add(tuple(sorted(map(int,m.groups()))))
 selection=select_seed_triplets(cat,6,seed=int(os.environ.get('PROPOSAL_SEED','45')),excluded=reviewed)
 selected=[{'cluster_ids':ids} for ids in selection['triplets']]
 db=sqlite3.connect(UP/'data/llm-papers.db');db.row_factory=sqlite3.Row
 for i,r in enumerate(selected,1):
  r={k:v for k,v in r.items() if k=='cluster_ids'};selected[i-1]=r
  r['id']=f'{i:02d}-'+ '-'.join(map(str,r['cluster_ids']));r['atoms']=[]
  for cid in r['cluster_ids']:
   assignments=[x for x in cat['atom_assignments'] if x['cluster']==cid];papers=[]
   for pid in sorted({x['paper_id'] for x in assignments})[:3]:
    row=db.execute('select title,abstract,pdf_url,venue_year from papers where paper_id=?',(pid.replace('__','/') if pid.startswith('conf__') else pid,)).fetchone()
    if row:papers.append(dict(row))
   r['atoms'].append({'id':cid,'description':cat['clusters'][str(cid)]['name'],'source_units':[{'paper_id':x['paper_id'],'text':x['text']} for x in assignments[:3]],'sample_supporting_papers':papers})
 (OUT/'selection.json').write_text(json.dumps({'provenance':{'decode_allowed':True,'ranking_claim_allowed':False,'mode':'uniform unranked baseline; original assignment evidence','source_file':'papers/clusters.json','source_sha256':hashlib.sha256((UP/'papers/clusters.json').read_bytes()).hexdigest(),'warning':'Do not interpret seed selection as semantically validated scorer ranking; clusters_80 assignments failed lineage audit.'},'source':str(sourcepath),'selection':{k:v for k,v in selection.items() if k!='triplets'},'excluded_reviewed_triplets':[list(s) for s in sorted(reviewed)],'candidates':selected},indent=2))
 print('\n'.join(r['id']+' '+str([a['description'][:100] for a in r['atoms']]) for r in selected))
def decode():
 data=json.loads((OUT/'selection.json').read_text())
 if not data.get('provenance', {}).get('decode_allowed', False):
  raise SystemExit('Provenance gate failed: repair and validate atom/scorer ID lineage before paid decoding')
 missing=[r['id'] for r in data['candidates'] if not (OUT/'prior-art'/f'{r["id"]}.md').exists()]
 if missing:raise SystemExit('Prior-art notes missing: '+','.join(missing))
 lessons=(ROOT/'results/proposal-v2-review-lessons.md').read_text()
 def one(r):
  directory=OUT/r['id'];directory.mkdir(exist_ok=True)
  evidence=(OUT/'prior-art'/f'{r["id"]}.md');background=evidence.read_text() if evidence.exists() else 'No targeted live literature review is supplied yet. Novelty unresolved.'
  prompt='''Design ONE narrowly specified, falsifiable research proposal from these three atoms. This is a new proposal-generation protocol, not a replication of the authors original decoder. Use the supplied evidence and lessons. These are unranked seed concepts; do not claim they are top-ranked, novel, or predicted breakthroughs. The original scorer-to-description lineage failed validation. Source units are extracted interpretations, not verbatim paper quotes; verify claims against supplied paper abstracts. Do not manufacture references, measurements, guarantees, or claims of novelty. Treat canonical atom summaries as broad hypotheses; source abstracts/units may narrow them. Components being individually known does not establish that their conjunction contributes anything. If you cannot identify a plausible nontrivial causal interface, explicitly mark REJECT rather than dress up a pipeline.
Return 900-1400 words of Markdown with: precise title; one-sentence hypothesis; what intervention changes which measurable outcome; closest supplied prior art and exact proposed difference (or uncertainty); explicit equations/algorithm only where necessary; compatibility assumptions; cheap phase-zero falsification experiment with named baseline/control/ablation, dataset construction and heldout scheme, metric, proposed preregistered pass/fail rule; compute estimate labeled estimate with assumptions; strongest objection and failure cases; decision (investigate/revise/reject) and why. Separate source-backed facts from your proposed mechanisms. No unsupported causal guarantees. Cite only supplied URLs, accurately. No unsafe exploit instructions: security-related designs must remain defensive and use benign controlled tasks. The novelty review is bounded, not exhaustive.
'''+ '\nLESSONS FROM PREVIOUS REVIEWS:\n'+lessons+'\nCANDIDATE AND SOURCE MATERIAL:\n'+json.dumps(r,ensure_ascii=False)+'\nTARGETED PRIOR ART NOTES:\n'+background
  (directory/'prompt.txt').write_text(prompt)
  if (directory/'proposal.md').exists():return
  request=urllib.request.Request('https://openrouter.ai/api/v1/chat/completions',data=json.dumps({'model':MODEL,'messages':[{'role':'user','content':prompt}],'reasoning':{'effort':'low'},'max_tokens':MAX_TOKENS}).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+key()})
  try:
   with urllib.request.urlopen(request,timeout=240,context=HTTPS_CONTEXT) as response:d=json.load(response)
  except urllib.error.HTTPError as e:raise RuntimeError(f'OpenRouter HTTP {e.code}; body omitted') from None
  (directory/f'response-{time.time_ns()}.json').write_text(json.dumps(d,indent=2))
  c=d['choices'][0];receipt={'id':d.get('id'),'model':d.get('model'),'usage':d.get('usage'),'finish_reason':c.get('finish_reason'),'max_tokens':MAX_TOKENS,'reasoning_effort':'low','prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()}
  (directory/f'receipt-{time.time_ns()}.json').write_text(json.dumps(receipt,indent=2))
  text=c['message'].get('content')
  if c.get('finish_reason')=='content_filter':raise RuntimeError('Provider refusal; no automatic fallback: '+r['id'])
  if not text or c.get('finish_reason')=='length':raise RuntimeError('Incomplete response: '+r['id'])
  (directory/'proposal.md').write_text(text);print('Completed',r['id'],flush=True)
 def safe_one(r):
  try:one(r)
  except Exception as e:
   (OUT/r['id']/'error.json').write_text(json.dumps({'error':str(e),'type':type(e).__name__},indent=2));print('Failed',r['id'],str(e),flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:list(ex.map(safe_one,data['candidates']))
if __name__=='__main__':
 import fcntl
 with (OUT/'pipeline.lock').open('w') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise SystemExit('Proposal pipeline is already running; no duplicate calls started')
  select() if sys.argv[1]=='select' else decode()
