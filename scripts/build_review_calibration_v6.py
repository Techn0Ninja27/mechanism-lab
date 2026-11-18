"""Offline cue-balanced development fixture. No model calls."""
import copy,hashlib,json,random,re
from pathlib import Path
from scripts.prepare_chunk_representations import _pack,render_source_context
from generation.evidence_reference_packets import build_evidence_packet,REVIEW_INSTRUCTIONS
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/recordwise-calibration-v6'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 if OUT.exists():raise FileExistsError(OUT)
 # Preserve the abandoned draft; reuse only its explicit source claims.
 draft=ROOT/'results/recordwise-calibration-v6-draft'
 old=[json.loads((draft/f'input-{i:02d}.json').read_text()) for i in range(6)]
 records=[]
 for p in old:
  vals={x['field']:x['value'] for x in p['fields']}
  records.append({'id':'temporary','source_id':p['record_metadata']['source_id'],'statement':vals['statement'],'inputs':[vals['inputs[0]']],'outputs':[vals['outputs[0]']],'conditions':[vals['conditions[0]']],'unknowns':[],'evidence':[{'chunk_id':p['record_metadata']['evidence_chunk_ids'][0]}]})
 records[1].update(statement='SNI-GNN uses a linear-trend predictor running on the SmartNIC.',inputs=['historical embeddings with their reception timestamps'],outputs=['During stale epochs the SmartNIC predicts the next embedding with the linear-trend model and supplies it to the GPU for aggregation.'],conditions=['Training uses a fixed synchronization interval.'])
 records[4]['inputs']=['GNN embedding evolution modeled as a temporal graph']
 negative=copy.deepcopy(records)
 changes=[('statement','SNI-GNN predicts current remote embeddings without changing the optimizer update order.','The cited chunk says nothing about preserving optimizer update order.'),('conditions[0]','Training uses a fixed synchronization interval.','This claim is stated in chunk 0002, but absent from this record\'s cited chunk 0001.'),('outputs[0]','cached embeddings remain accurate when the graph topology changes','The abstract reports cache selection and aggregate accuracy, not robustness to topology changes.'),('conditions[0]','workers use a shared quantization scale','The abstract states quantization-aware communication but does not specify scale sharing.'),('inputs[0]','the predictor receives ground-truth future embeddings during deployment','Modeling embedding evolution does not establish access to future ground truth at deployment.'),('statement','Sift uses channel-wise historical embeddings while preserving the order of message arrivals.','The abstract does not state preservation of message arrival order.')]
 # Original fixed-interval miss, paired with same text supported by chunk 0002.
 negative[1]=copy.deepcopy(records[0]);negative[1]['conditions']=[changes[1][1]]
 entries=[]
 for i,rec in enumerate(records):entries.append((rec,None,None))
 for i,(field,value,reason) in enumerate(changes):
  if '[' in field: negative[i][field.split('[')[0]][0]=value
  else:negative[i][field]=value
  entries.append((negative[i],field,reason))
 rng=random.Random(2026111906);rng.shuffle(entries)
 packpath=ROOT/'results/gnn-useful-proposal-v1/pack.json';pack=_pack(packpath);chunks={c['id']:c['text'] for c in render_source_context(pack)['chunks']}
 OUT.mkdir();inputs=OUT/'reviewer-inputs';inputs.mkdir();specs=[];keys=[];predictions=[]
 for i,(rec,bad,reason) in enumerate(entries):
  rec['id']=hashlib.sha256(f'packet-{rng.getrandbits(128)}'.encode()).hexdigest()[:12]
  packet=build_evidence_packet('atoms',rec,pack);path=inputs/f'input-{i:02d}.json';path.write_text(json.dumps(packet,ensure_ascii=False,indent=2)+'\n')
  specs.append({'path':str(path.relative_to(OUT)),'sha256':sha(path),'opaque_id':rec['id']})
  fields={f['field']:('unsupported' if f['field']==bad else 'supported') for f in packet['fields']}
  keys.append({'opaque_id':rec['id'],'expected_class':'negative' if bad else 'positive','fields':fields,'negative_field':bad,'negative_rationale':reason})
  claim=' '.join(f['value'] for f in packet['fields']);source=' '.join(chunks[e['chunk_id']] for e in rec['evidence']);nums=lambda text:set(re.findall(r'(?<!\w)\d+(?:\.\d+)?',text))
  predictions.append({'opaque_id':rec['id'],'actual_positive':bad is None,'all_supported':True,'cue':not bool(re.search(r'\b(exactly|fixed|exclusively|never)\b',claim,re.I)),'absent_number':not bool(nums(claim)-nums(source))})
 keypath=OUT/'answer-key.json';keypath.write_text(json.dumps({'status':'draft_requires_independent_key_audit','records':keys},indent=2)+'\n')
 scores={name:{'correct_records':sum(r[name]==r['actual_positive'] for r in predictions),'record_accuracy':sum(r[name]==r['actual_positive'] for r in predictions)/12} for name in ['all_supported','cue','absent_number']}
 scores['adequate']=all(x['record_accuracy']<=.75 for x in scores.values());scores['predictions']=predictions;scores['reviewer_calls']=0
 (OUT/'baseline-scores.json').write_text(json.dumps(scores,indent=2)+'\n')
 (inputs/'reviewer-instructions.md').write_text(REVIEW_INSTRUCTIONS+'\n')
 s=chunks['sni-method:chunk:0002'];phrase='During stale epochs (between syncs) the SmartNIC predicts the next embedding using the linear-trend model, and supplies the predicted embeddings to the GPU for aggregation.';start=s.index(phrase);end=start+len(phrase)
 boundary={'chunk_id':'sni-method:chunk:0002','start':start,'end':end,'exact_text':phrase,'single_window_contains_whole_sentence':False,'covering_windows':[[2250,2550],[2500,2800]]}
 assert not any(a<=start and a+300>=end for a in range(0,len(s),250))
 (OUT/'boundary-case.json').write_text(json.dumps(boundary,indent=2)+'\n')
 (OUT/'manifest.json').write_text(json.dumps({'schema':'recordwise-calibration-v6-manifest.v1','status':'draft_requires_independent_key_audit','pack_sha256':sha(packpath),'builder_sha256':sha(Path(__file__)),'instructions_sha256':sha(inputs/'reviewer-instructions.md'),'answer_key_sha256':sha(keypath),'packet_count':12,'packets':specs,'baseline_passed':scores['adequate'],'reviewer_calls':0},indent=2)+'\n')
 print(json.dumps({k:v for k,v in scores.items() if k!='predictions'},indent=2))
if __name__=='__main__':main()
