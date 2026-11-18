"""Recompute the ID-reference development calibration; no API calls."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from generation.evidence_reference_packets import decode_review
from generation.recordwise_calibration_scoring import validate_review

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def score(root: Path, source: Path):
    manifest=read(root/'manifest.json'); original=read(source/'manifest.json')
    bindings={'source_manifest_sha256':source/'manifest.json','answer_key_sha256':source/'answer-key.json','instructions_sha256':root/'reviewer-instructions.md','builder_sha256':Path(__file__).resolve().parents[1]/'generation/evidence_reference_packets.py'}
    for name,path in bindings.items():
        if sha(path)!=manifest[name]: raise ValueError(f'changed calibration dependency: {name}')
    key=read(source/'answer-key.json')
    if sha(source/'answer-key.json')!=original['answer_key_sha256']: raise ValueError('original key changed')
    expected={x['opaque_id']:{f['field']:f['verdict'] for f in x['fields']} for x in key['field_items']}
    specs=manifest['packets']; originals={s['path']:s for s in original['packets']}
    if len(specs)!=8 or len({s['path'] for s in specs})!=8 or {s['opaque_id'] for s in specs}!=set(expected): raise ValueError('packet coverage mismatch')
    rows=[]
    for spec in specs:
        path=root/spec['path']; oldpath=source/spec['path']
        if sha(path)!=spec['sha256'] or sha(oldpath)!=originals[spec['path']]['sha256']: raise ValueError('input changed')
        packet,old=read(path),read(oldpath)
        fields=[{k:v for k,v in row.items() if k!='field_id'} for row in packet['fields']]
        if fields!=old['scientific_fields'] or packet['opaque_id']!=old['opaque_id']: raise ValueError('scientific fields changed')
        chunks={x['chunk_id']:x['text'] for x in old['cited_chunks']}
        for span in packet['evidence']:
            if span['text']!=chunks[span['chunk_id']][span['start']:span['end']]: raise ValueError('evidence changed')
        index=path.stem.removeprefix('input-')
        for slot in 'ab':
            rp=root/'reviews'/f'review-{index}-{slot}.json'
            row={'packet':packet['opaque_id'],'index':int(index),'slot':slot,'agent':f'/root/id_review_{index}{slot}','input_sha256':sha(path),'review_sha256':sha(rp)}
            try:
                decoded=decode_review(packet,read(rp)); valid=validate_review(old,decoded)
                verdicts={x['field']:x['verdict'] for x in valid['items']}
                row.update(mechanically_valid=True,matches_key=verdicts==expected[packet['opaque_id']],verdicts=verdicts)
            except (ValueError,KeyError,TypeError) as exc:
                row.update(mechanically_valid=False,matches_key=False,error=str(exc))
            rows.append(row)
    return {'schema':'reference-calibration-score.v1','date':'2026-11-19','packet_count':8,'review_count':16,'mechanically_valid_count':sum(r['mechanically_valid'] for r in rows),'matches_key_count':sum(r['matches_key'] for r in rows),'calibration_passed':all(r['mechanically_valid'] and r['matches_key'] for r in rows),'production_authorized':False,'scope':'Development instrument check on the same eight frozen examples; not a population accuracy estimate or proof of reviewer independence.','manifest_sha256':sha(root/'manifest.json'),'scorer_sha256':sha(Path(__file__)),'reviews':rows}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=score(a.root,a.source)
    with a.output.open('x') as f: json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='reviews'}))
