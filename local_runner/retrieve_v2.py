"""Retrieve bounded, cached OpenAlex abstracts for six specific comparisons."""
import os
import json,pathlib,urllib.request,urllib.parse,time
ROOT=pathlib.Path(os.environ.get('ALIEN_WORKSPACE', pathlib.Path(__file__).resolve().parent));OUT=ROOT/'results/proposals-v2';queries={
'01-29-85-254':['machine unlearning knowledge conflicts','unlearning code execution verification'],
'02-28-59-175':['model editing collapse perplexity','sequential knowledge editing retention'],
'03-63-93-168':['image forgery detection demographic bias','deepfake detection dataset fairness'],
'04-178-227-232':['activation steering low rank subspace','agent trajectory learning steering vectors'],
'05-125-194-250':['intrinsic dimension adaptive computation early exit','language model layer geometry halting'],
'06-3-63-86':['physics informed dataset curation verification','scientific claims verification physical constraints']}
def abstract(inv):
 if not inv:return ''
 pairs=[(p,w) for w,positions in inv.items() for p in positions]
 return ' '.join(w for p,w in sorted(pairs))
(OUT/'retrieval').mkdir(exist_ok=True)
for id,qs in queries.items():
 records={};logs=[]
 for q in qs:
  url='https://api.openalex.org/works?'+urllib.parse.urlencode({'search':q,'per-page':8,'select':'id,title,doi,publication_date,abstract_inverted_index,primary_location'})
  cache=OUT/'retrieval'/(id+'-'+str(qs.index(q))+'.json')
  if cache.exists():data=json.loads(cache.read_text())
  else:
   try:
    with urllib.request.urlopen(url,timeout=45) as r:data=json.load(r)
   except Exception as e:
    logs.append({'query':q,'error':type(e).__name__});continue
   cache.write_text(json.dumps(data));time.sleep(.25)
  logs.append({'query':q,'url':url,'returned':len(data.get('results',[]))})
  for w in data.get('results',[]):
   text=abstract(w.get('abstract_inverted_index'))
   if text:records[w['id']]={'id':w['id'].split('/')[-1],'title':w['title'],'url':w.get('doi') or (w.get('primary_location') or {}).get('landing_page_url') or w['id'],'date':w['publication_date'],'abstract':text}
 (OUT/'retrieval'/(id+'-evidence.json')).write_text(json.dumps({'queries':logs,'records':list(records.values()),'coverage':'Two lexical searches, top8 each, abstract-only. Missing retrieval is not evidence of novelty.'},indent=2))
 print(id,len(records),'abstracts',flush=True)
