"""Offline mechanical scorer for recordwise reviewer calibration."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any, Mapping
VERDICTS={"supported","unsupported","contradicted","indeterminate"}
def _sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def _fields(p:Mapping[str,Any])->dict[str,Any]:
 rows=p.get("scientific_fields")
 if not isinstance(rows,list) or not rows:raise ValueError("packet fields missing")
 out={}
 for x in rows:
  if not isinstance(x,Mapping) or not isinstance(x.get("field"),str) or x["field"] in out:raise ValueError("bad packet field")
  out[x["field"]]=x.get("value")
 return out
def _chunks(p):
 rows=p.get("cited_chunks")
 if not isinstance(rows,list) or not rows:raise ValueError("packet chunks missing")
 out={}
 for x in rows:
  if not isinstance(x,Mapping) or not isinstance(x.get("chunk_id"),str) or not isinstance(x.get("text"),str) or x["chunk_id"] in out:raise ValueError("bad packet chunk")
  out[x["chunk_id"]]=x["text"]
 return out
def validate_review(packet:Mapping[str,Any],review:Mapping[str,Any])->dict[str,Any]:
 packet_rows={x["field"]:x for x in packet.get("scientific_fields",[]) if isinstance(x,Mapping) and isinstance(x.get("field"),str)}
 expected,chunks=_fields(packet),_chunks(packet); rows=review.get("field_items")
 if not isinstance(rows,list):raise ValueError("field_items required")
 seen=set();norm=[]
 for x in rows:
  f=x.get("field") if isinstance(x,Mapping) else None
  if f not in expected or f in seen or x.get("value")!=expected[f]:raise ValueError("field coverage/value mismatch")
  v,reason=x.get("verdict"),x.get("reason")
  if v not in VERDICTS or not isinstance(reason,str) or not reason.strip():raise ValueError("bad verdict/reason")
  ev=x.get("evidence",[])
  if not isinstance(ev,list):raise ValueError("evidence must be list")
  checked=[];used=set()
  for s in ev:
   if not isinstance(s,Mapping) or not isinstance(s.get("chunk_id"),str) or not isinstance(s.get("quote"),str):raise ValueError("bad evidence")
   k=(s["chunk_id"],s["quote"])
   if k in used or s["chunk_id"] not in chunks or not s["quote"] or len(s["quote"])>300 or s["quote"] not in chunks[s["chunk_id"]]:raise ValueError("evidence not bound")
   used.add(k);checked.append({"chunk_id":s["chunk_id"],"quote":s["quote"]})
  unknown=bool(packet_rows[f].get("is_unknown",False))
  if v=="supported" and not checked and not (unknown and isinstance(x.get("absence_reason"),str) and x["absence_reason"].strip()):raise ValueError("supported item lacks evidence")
  if v=="supported" and unknown and (checked or not isinstance(x.get("absence_reason"),str) or not x["absence_reason"].strip()):raise ValueError("unknown needs absence reason")
  assessments=x.get("claim_assessments")
  if assessments is not None:
   if not isinstance(assessments,list) or not assessments:raise ValueError("claim assessments malformed")
   for a in assessments:
    if not isinstance(a,Mapping) or a.get("verdict") not in VERDICTS:raise ValueError("claim assessment verdict malformed")
    ae=a.get("evidence",[])
    if not isinstance(ae,list):raise ValueError("subclaim evidence malformed")
    if a.get("verdict")=="supported" and not ae:raise ValueError("supported subclaim lacks evidence")
    for span in ae:
     if not isinstance(span,Mapping) or span.get("chunk_id") not in chunks or not isinstance(span.get("quote"),str) or not span['quote'] or len(span['quote'])>300 or span['quote'] not in chunks[span['chunk_id']]:raise ValueError("subclaim evidence not bound")
   if v=="supported" and any(a.get("verdict")!="supported" for a in assessments):raise ValueError("field passes with failed subclaim")
  seen.add(f);norm.append({"field":f,"verdict":v,"reason":reason,"evidence":checked})
 if seen!=set(expected):raise ValueError("incomplete review")
 counts={v:sum(x["verdict"]==v for x in norm) for v in VERDICTS}
 return {"items":norm,"counts":counts,"all_supported":counts["supported"]==len(norm)}
def score_directory(folder:str|Path)->dict[str,Any]:
 root=Path(folder);kp=root/"answer-key.json"
 if not kp.is_file():raise ValueError("answer key required")
 key=json.loads(kp.read_text());paths=sorted(root.glob("input-*.json"));packets=[]
 if not paths:raise ValueError("no calibration packets")
 for path in paths:
  packet=json.loads(path.read_text());pid=packet.get("opaque_id",path.stem.removeprefix("input-")); expected=next((x.get("fields",x.get("expected_fields")) for x in key.get("records",[]) if x.get("packet_id")==pid),None)
  if not isinstance(expected,Mapping):raise ValueError(f"missing answer key {pid}")
  reviews=sorted(root.glob(f"review-{path.stem.removeprefix('input-')}-*.json")) or sorted(root.glob(f"review-*-{path.stem.removeprefix('input-')}.json"))
  if len(reviews)!=2:raise ValueError(f"two reviews required {pid}")
  rs=[]
  for rp in reviews:
   d=json.loads(rp.read_text());valid=validate_review(packet,d);vm={x["field"]:x["verdict"] for x in valid["items"]}
   rs.append({"path":rp.name,"sha256":_sha(rp),"reviewer_id":d.get("reviewer_id"),"matches_answer_key":all(vm.get(k)==v for k,v in expected.items()),"counts":valid["counts"]})
  packets.append({"packet_id":pid,"input_sha256":_sha(path),"reviewers":rs,"both_match":all(x["matches_answer_key"] for x in rs)})
 return {"schema":"recordwise-calibration-score.v1","packet_count":len(packets),"review_count":2*len(packets),"packets":packets,"all_reviewers_match":all(x["both_match"] for x in packets),"artifact_sha256":{"answer-key.json":_sha(kp)}}

def score_v4(folder: str | Path, review_folder: str | Path | None = None) -> dict[str, Any]:
 root=Path(folder);review_root=Path(review_folder) if review_folder is not None else root;manifest=json.loads((root/'manifest.json').read_text());key_path=root/'answer-key.json';key=json.loads(key_path.read_text());instruction=root/'reviewer-instructions.md'
 if manifest.get('schema')!='recordwise-calibration-v4-manifest.v1' or manifest.get('packet_count')!=8:raise ValueError('wrong v4 manifest')
 if _sha(key_path)!=manifest.get('answer_key_sha256') or _sha(instruction)!=manifest.get('instructions_sha256'):raise ValueError('frozen hash mismatch')
 records={x['opaque_id']:x for x in key.get('records',[])};fields={x['opaque_id']:x for x in key.get('field_items',[])}
 if len(records)!=8 or set(records)!=set(fields):raise ValueError('answer key coverage mismatch')
 output=[]
 specs=manifest.get('packets',[])
 if len(specs)!=8 or len({x.get('opaque_id') for x in specs})!=8 or {x.get('opaque_id') for x in specs}!=set(records):raise ValueError('packet manifest coverage mismatch')
 for spec in specs:
  path=root/spec['path'];pid=spec['opaque_id']
  if not path.is_file() or _sha(path)!=spec.get('sha256'):raise ValueError('packet hash mismatch')
  packet=json.loads(path.read_text())
  if packet.get('opaque_id')!=pid:raise ValueError('packet identity mismatch')
  expected={x['field']:x['verdict'] for x in fields[pid].get('fields',[])}
  if set(expected)!=set(_fields(packet)):raise ValueError('field key coverage mismatch')
  index=path.stem.removeprefix('input-')
  reviews=[review_root/f'review-{index}-{slot}.json' for slot in ('a','b')]
  if not all(p.is_file() for p in reviews):raise ValueError(f'missing review files: {pid}')
  if len(reviews)!=2:raise ValueError(f'two reviews required: {pid}')
  rr=[]
  for rp in reviews:
   d=json.loads(rp.read_text());valid=validate_review(packet,d);vm={x['field']:x['verdict'] for x in valid['items']}
   rr.append({'path':rp.name,'sha256':_sha(rp),'reviewer_id':d.get('reviewer_id'),'matches_answer_key':vm==expected,'counts':valid['counts']})
  output.append({'opaque_id':pid,'expected_class':records[pid].get('expected_class'),'reviewers':rr,'both_match':all(x['matches_answer_key'] for x in rr)})
 return {'schema':'recordwise-calibration-v4-score.v1','packet_count':len(output),'review_count':2*len(output),'packets':output,'all_reviewers_match':all(x['both_match'] for x in output),'artifact_sha256':{'manifest.json':_sha(root/'manifest.json'),'answer-key.json':_sha(key_path),'reviewer-instructions.md':_sha(instruction)}}
