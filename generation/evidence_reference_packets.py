"""ID-only evidence packets for offline source-record review."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping
from generation.record_review_packets import build_record_packet

SCHEMA = "record-review-evidence-references.v1"
MAX_SPAN = 300
REVIEW_INSTRUCTIONS = """Return only {\"fields\":[...]} with one item per field_id.
Each item contains field_id, verdict (supported, unsupported, contradicted or
indeterminate), an item-specific reason, and evidence_ids. Never copy field
values, quotes, chunk IDs or hashes; the orchestrator resolves those IDs.
Judge scientific fields only: metadata is bookkeeping. Supported means the
cited rendered chunks state the claim or it follows without an outside
assumption. Contradicted means the cited chunks state the opposite.
Unsupported means the claim could be false while the cited chunks remained
true. Indeterminate is only for malformed evidence or unresolved referents.
Unknown fields mean information absent from the supplied cited chunks; if the
chunks answer the unknown, it is unsupported. For supported non-unknown fields,
provide one or more exact evidence_ids. For supported unknowns provide an
absence_reason and no evidence IDs. Use at most four evidence IDs per field
and per subclaim. Compound fields must include claim_assessments, each with its
own verdict, reason and evidence_ids and a field cannot pass if one subclaim
fails. For unsupported claims name the missing assumption; for contradictions
name the conflicting passage or provide a concrete counterexample consistent
with the cited text. Do not calculate provenance hashes."""

def _canon(v: Any) -> bytes:
    return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

def _sha(v: Any) -> str:
    return hashlib.sha256(_canon(v)).hexdigest()

def _windows(text: str, width: int = MAX_SPAN, overlap: int = 50) -> list[dict[str, Any]]:
    if width <= 0 or overlap < 0 or overlap >= width: raise ValueError("invalid evidence window limits")
    if not text: return [{"start": 0, "end": 0, "text": ""}]
    out=[]; start=0; step=width-overlap
    while True:
        end=min(len(text), start+width); out.append({"start":start,"end":end,"text":text[start:end]})
        if end==len(text): return out
        start += step

def _normalize_record(record: Mapping[str, Any]) -> dict[str, Any]:
    if record.get("source_id") is not None:
        return dict(record)
    meta=record.get("record_metadata", {})
    fields={x.get("field"):x.get("value") for x in record.get("scientific_fields", [])}
    if not isinstance(meta, Mapping) or not fields or not meta.get("source_id"):
        raise ValueError("record must include source_id or packet metadata")
    out={"id":record.get("opaque_id", meta.get("record_id")),"source_id":meta["source_id"],"evidence":[{"chunk_id":x} for x in meta.get("evidence_chunk_ids", [])]}
    for name,value in fields.items():
        if "[" not in name: out[name]=value; continue
        key,index=name.split("[",1); index=int(index[:-1]); out.setdefault(key,[])
        while len(out[key])<=index: out[key].append(None)
        out[key][index]=value
    for key in ("inputs", "outputs", "conditions", "unknowns"):
        out.setdefault(key, [])
    out.setdefault("evidence", [{"chunk_id": x} for x in meta.get("evidence_chunk_ids", [])])
    return out

def build_evidence_packet(kind: str, record: Mapping[str, Any], pack: Mapping[str, Any]) -> dict[str, Any]:
    base=build_record_packet(kind, _normalize_record(record), pack)
    fields=[{"field_id":f"F{i:03d}","field":f["field"],"value":f["value"],"is_unknown":bool(f.get("is_unknown"))} for i,f in enumerate(base["scientific_fields"])]
    evidence=[]
    for chunk in base["cited_chunks"]:
        for span in _windows(chunk["text"]): evidence.append({"evidence_id":f"E{len(evidence):04d}","chunk_id":chunk["chunk_id"],**span})
    body={"schema":SCHEMA,"opaque_id":base["opaque_id"],"record_metadata":base["record_metadata"],"fields":fields,"evidence":evidence}
    return {**body,"packet_sha256":_sha(body),"review_output_contract":{"fields":"field_id, verdict, reason, evidence_ids, optional claim_assessments","forbidden":["field","value","quote","chunk_id","hash","metadata"]}}

def _validate_item(item: Mapping[str, Any], fields: Mapping[str, Mapping[str, Any]], evidence_ids: set[str]) -> None:
    allowed={"field_id","verdict","reason","evidence_ids","absence_reason","claim_assessments"}
    if set(item)-allowed or item.get("field_id") not in fields: raise ValueError("unknown model field item key or id")
    if item.get("verdict") not in {"supported","unsupported","contradicted","indeterminate"}: raise ValueError("invalid verdict")
    if not isinstance(item.get("reason"),str) or not item["reason"].strip(): raise ValueError("reason required")
    refs=item.get("evidence_ids",[])
    if not isinstance(refs,list) or len(refs)>4 or len(set(refs))!=len(refs) or any(x not in evidence_ids for x in refs): raise ValueError("invalid evidence_ids")
    unknown=bool(fields[item["field_id"]].get("is_unknown"))
    if item["verdict"]=="supported" and not refs and not (unknown and str(item.get("absence_reason","")).strip()): raise ValueError("supported item needs evidence")
    if "absence_reason" in item and not unknown: raise ValueError("absence_reason only applies to unknown fields")
    if unknown and item["verdict"]=="supported" and (refs or not isinstance(item.get("absence_reason"),str) or not item["absence_reason"].strip()): raise ValueError("supported unknown requires absence reason only")
    claims=item.get("claim_assessments")
    if claims is not None:
        if not isinstance(claims,list) or not claims: raise ValueError("claim assessments malformed")
        for claim in claims:
            if not isinstance(claim,Mapping) or set(claim)-{"claim","verdict","reason","evidence_ids"} or claim.get("verdict") not in {"supported","unsupported","contradicted","indeterminate"}: raise ValueError("invalid claim assessment")
            if not isinstance(claim.get("reason"),str) or not claim["reason"].strip(): raise ValueError("claim reason required")
            ids=claim.get("evidence_ids",[])
            if not isinstance(ids,list) or len(set(ids)) != len(ids) or len(ids)>4 or any(x not in evidence_ids for x in ids): raise ValueError("claim evidence invalid")
            if claim["verdict"] == "supported" and not ids: raise ValueError("supported claim needs evidence")
        if item["verdict"] == "supported" and any(c.get("verdict") != "supported" for c in claims): raise ValueError("supported field has unsupported subclaim")

def decode_review(packet: Mapping[str, Any], model_output: Mapping[str, Any]) -> dict[str, Any]:
    if packet.get("schema")!=SCHEMA or not isinstance(model_output,Mapping) or set(model_output) != {"fields"}: raise ValueError("wrong packet/output schema")
    fields=packet.get("fields"); evidence=packet.get("evidence")
    if not isinstance(fields,list) or not isinstance(evidence,list): raise ValueError("packet fields/evidence missing")
    byfield={x["field_id"]:x for x in fields}; byeid={x["evidence_id"]:x for x in evidence}
    if len(byfield)!=len(fields) or len(byeid)!=len(evidence): raise ValueError("duplicate packet identifiers")
    body={k:packet[k] for k in ("schema","opaque_id","record_metadata","fields","evidence")}
    if packet.get("packet_sha256") != _sha(body): raise ValueError("packet hash mismatch")
    for row in evidence:
        if not isinstance(row.get("text"),str) or not row["text"] or len(row["text"])>MAX_SPAN or not isinstance(row.get("start"),int) or not isinstance(row.get("end"),int) or row["end"]-row["start"] != len(row["text"]): raise ValueError("invalid evidence catalog row")
    items=model_output.get("fields")
    if not isinstance(items,list) or len(items)!=len(fields): raise ValueError("model response must cover every field")
    seen=set(); resolved=[]
    for item in items:
        if not isinstance(item,Mapping): raise ValueError("model field item malformed")
        _validate_item(item,byfield,set(byeid)); fid=item["field_id"]
        if fid in seen: raise ValueError("duplicate model field_id")
        seen.add(fid); spans=[{"chunk_id":byeid[e]["chunk_id"],"quote":byeid[e]["text"]} for e in item.get("evidence_ids",[])]
        out={"field":byfield[fid]["field"],"value":byfield[fid]["value"],"verdict":item["verdict"],"reason":item["reason"],"evidence":spans}
        if byfield[fid].get("is_unknown") and item.get("absence_reason"): out["absence_reason"]=item["absence_reason"]
        if item.get("claim_assessments") is not None:
            assessments=[]
            for claim in item["claim_assessments"]:
                c={k:v for k,v in claim.items() if k != "evidence_ids"}
                c["evidence"]=[{"chunk_id":byeid[e]["chunk_id"],"quote":byeid[e]["text"]} for e in claim.get("evidence_ids",[])]
                assessments.append(c)
            out["claim_assessments"]=assessments
        resolved.append(out)
    if seen!=set(byfield): raise ValueError("model response omitted a field")
    return {"schema":"record-review-decoded.v1","opaque_id":packet["opaque_id"],"field_items":resolved,"packet_sha256":packet.get("packet_sha256")}

def build_probe_v5(source_dir: Path, output_dir: Path, pack_path: Path) -> dict[str, Any]:
    if output_dir.exists(): raise FileExistsError("refusing existing prototype output")
    pack=json.loads(pack_path.read_text()); output_dir.mkdir(parents=True); packets=[]
    for source in sorted(source_dir.glob("input-*.json")):
        value=json.loads(source.read_text()); record=value.get("record", value); kind="atoms" if str(value.get("representation", "atom")).startswith("atom") else "distillations"; packet=build_evidence_packet(kind,record,pack); path=output_dir/source.name; path.write_text(json.dumps(packet,ensure_ascii=False,indent=2)+"\n"); packets.append({"path":path.name,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"opaque_id":packet["opaque_id"]})
    (output_dir/"reviewer-instructions.md").write_text(REVIEW_INSTRUCTIONS + "\n")
    return {"schema":"evidence-reference-probe.v1","packets":packets,"labels_excluded":True}

