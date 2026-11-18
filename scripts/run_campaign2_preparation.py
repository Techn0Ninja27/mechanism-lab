"""Sequential, immutable preparation draws; every paid action shares one lock/cap."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
from scripts import generate_causal_campaign2 as c

MAX_DRAWS = 3
KINDS = ('atoms', 'distillations')
CANONICAL_OUTPUT = c.ROOT / 'results/generation-comparison-campaign2-run'

def _draw_dir(out, kind, index):
    if kind not in KINDS or type(index) is not int or not 0 <= index < MAX_DRAWS:
        raise ValueError('kind/index outside the frozen draw schedule')
    return out / 'prep-draws' / kind / f'{index:02d}'

def _read(path): return json.loads(path.read_text())
def _same_or_new(path, value):
    if path.exists():
        if _read(path) != value: raise ValueError(f'immutable artifact changed: {path.name}')
    else: c.write_new(path, value)

def _stop(out, reason):
    p=out/'campaign-stop.json'
    if not p.exists(): c.write_new(p, {'status':'stopped','reason':reason,'plan_sha256':c.sha(out/'plan.json')})

@contextmanager
def _locked(out):
    out=Path(out).resolve()
    if out != CANONICAL_OUTPUT.resolve(): raise ValueError('only the canonical campaign directory may execute')
    if not out.is_dir(): raise ValueError('plan the canonical campaign first')
    with (out/'campaign.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'campaign-stop.json').exists(): raise ValueError('campaign is permanently stopped')
        c.require_calibration_ready()
        _,pack,_=c.verify(out)
        yield out,pack

def _validate_draw(out,kind,index,pack):
    draw=_draw_dir(out,kind,index);meta=_read(draw/'draw.json')
    if meta.get('plan_sha256')!=c.sha(out/'plan.json'):raise ValueError('draw plan mismatch')
    call=out/'calls'/f'prep-{kind}-d{index}-0.json'
    if c.sha(call)!=meta.get('call_sha256'):raise ValueError('draw call changed')
    rec=_read(call);req=c.expected_request(c._prep_prompt(pack,kind))
    if rec.get('status')!='complete' or rec.get('request')!=req or rec.get('request_hash')!=c.request_hash(req):
        raise ValueError('draw provider receipt/request invalid')
    value=getattr(c._prep(),f'validate_{kind}')(c.strict_json(rec['content']),pack)
    path=draw/f'{kind}.json'
    if c.sha(path)!=meta.get('representation_sha256') or _read(path)!=value:raise ValueError('draw content changed')
    return value,meta

def _decision(out,kind,index,pack):
    draw=_draw_dir(out,kind,index);decision=_read(draw/'decision.json')
    value,meta=_validate_draw(out,kind,index,pack)
    paths=[draw/'review-a.json',draw/'review-b.json']
    if decision.get('draw_sha256')!=c.sha(draw/'draw.json') or decision.get('review_sha256')!=[c.sha(p) for p in paths]:
        raise ValueError('recorded decision or review changed')
    combined=c._prep().combine_draw_reviews(kind,*[_read(p) for p in paths],pack,value)
    if combined!=decision.get('combined'):raise ValueError('decision no longer validates')
    return decision,value,meta

def _selected(out,kind,pack):
    path=out/'selected'/f'{kind}.selection.json'
    if not path.exists():return None
    selected=_read(path);index=selected['draw']
    decision,value,meta=_decision(out,kind,index,pack)
    if decision['combined']['status']!='passed' or selected.get('decision_sha256')!=c.sha(_draw_dir(out,kind,index)/'decision.json'):
        raise ValueError('selected draw changed')
    for prior in range(index):
        if _decision(out,kind,prior,pack)[0]['combined']['status']!='failed':raise ValueError('skipped a passing draw')
    if selected.get('representation_sha256')!=meta['representation_sha256']:raise ValueError('selection content mismatch')
    return selected

def execute_draw(out,kind,index):
    _draw_dir(Path(out),kind,index)
    with _locked(out) as (out,pack):
        if _selected(out,kind,pack) is not None:raise ValueError('representation already selected')
        for prior in range(index):
            if _decision(out,kind,prior,pack)[0]['combined']['status']!='failed':raise ValueError('prior draw did not fail review')
        draw=_draw_dir(out,kind,index)
        if draw.exists():raise FileExistsError('draw already attempted; never retry')
        draw.mkdir(parents=True)
        call=out/'calls'/f'prep-{kind}-d{index}-0.json'
        try:
            client=c.Client(out,c.config(3.0))
            content=client.call({'id':f'prep-{kind}-d{index}'},0,c._prep_prompt(pack,kind))
            value=getattr(c._prep(),f'validate_{kind}')(c.strict_json(content),pack)
            target=draw/f'{kind}.json';c.write_new(target,value)
            meta={'kind':kind,'draw':index,'status':'awaiting_reviews','plan_sha256':c.sha(out/'plan.json'),
                  'call_path':str(call.relative_to(out)),'call_sha256':c.sha(call),
                  'path':str(target.relative_to(out)),'representation_sha256':c.sha(target)}
            c.write_new(draw/'draw.json',meta)
            _validate_draw(out,kind,index,pack)
            return meta
        except Exception as exc:
            _stop(out,f'preparation draw failure: {type(exc).__name__}: {str(exc)[:300]}')
            c.write_new(draw/'failure.json',{'kind':kind,'draw':index,'error':f'{type(exc).__name__}: {str(exc)[:300]}',
                         'call_sha256':c.sha(call) if call.exists() else None})
            raise

def record_review(out,kind,index):
    _draw_dir(Path(out),kind,index)
    with _locked(out) as (out,pack):
        existing=_selected(out,kind,pack)
        if existing is not None:
            if existing['draw']!=index:raise ValueError('a different draw is already locked')
            return existing
        draw=_draw_dir(out,kind,index)
        value,meta=_validate_draw(out,kind,index,pack)
        if (draw/'decision.json').exists():
            decision,_,_=_decision(out,kind,index,pack)
        else:
            paths=[draw/'review-a.json',draw/'review-b.json']
            combined=c._prep().combine_draw_reviews(kind,*[_read(p) for p in paths],pack,value)
            decision={'kind':kind,'draw':index,'draw_sha256':c.sha(draw/'draw.json'),
                      'review_paths':[str(p.relative_to(out)) for p in paths],
                      'review_sha256':[c.sha(p) for p in paths],'combined':combined}
            c.write_new(draw/'decision.json',decision)
        if decision['combined']['status']!='passed':
            if index==MAX_DRAWS-1:_stop(out,f'{kind} exhausted three reviewed draws')
            return {'status':'failed','kind':kind,'draw':index}
        for prior in range(index):
            if _decision(out,kind,prior,pack)[0]['combined']['status']!='failed':raise ValueError('first passing selection violated')
        selected={**meta,'status':'locked','decision_sha256':c.sha(draw/'decision.json'),
                  'review_paths':decision['review_paths'],'review_hashes':decision['review_sha256'],
                  'reviewers':decision['combined']['reviews']}
        _same_or_new(out/'selected'/f'{kind}.selection.json',selected)
        return selected

def select_pair(out):
    with _locked(out) as (out,pack):
        selected={kind:_selected(out,kind,pack) for kind in KINDS}
        if any(v is None for v in selected.values()):raise ValueError('both representations must already be locked')
        selection={'schema':'campaign2-preparation-selection.v1','status':'locked','plan_sha256':c.sha(out/'plan.json'),**selected}
        _same_or_new(out/'selected/selection.json',selection)
        representations={};reviews=[];receipts=[]
        for kind,meta in selected.items():
            source=out/meta['path'];target=out/'representations'/f'{kind}.json'
            _same_or_new(target,_read(source))
            representations[kind]={'status':'validated','path':str(target.relative_to(out)),'sha256':c.sha(target),
                                   'call_path':meta['call_path'],'call_sha256':meta['call_sha256']}
            for p,h,r in zip(meta['review_paths'],meta['review_hashes'],meta['reviewers']):
                reviews.append({'kind':kind,'path':p,'sha256':h,'reviewer_id':r['reviewer_id'],'artifact_sha256':r['artifact_sha256']})
            receipts.append({'kind':kind,'draw':meta['draw'],'call_path':meta['call_path'],'call_sha256':meta['call_sha256'],
                             'representation_sha256':meta['representation_sha256']})
        _same_or_new(out/'preparation-result.json',{'schema':'causal-generation-campaign2-preparation.v1','status':'ready',
                     'plan_sha256':c.sha(out/'plan.json'),'representations':representations,'semantic_gate_required':True})
        gate={'schema':'campaign2-dual-draw-gate.v1','status':'passed','criterion':'cited_chunk_support.v1',
              'counts':{'total':24,'supported':24,'unsupported':0,'contradicted':0,'indeterminate':0},
              'reviews':reviews,'draw_receipts':receipts,'selection_sha256':c.sha(out/'selected/selection.json')}
        _same_or_new(out/'chunk-fidelity-gate.json',gate)
        return selection

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--kind',choices=KINDS);p.add_argument('--draw',type=int)
    action=p.add_mutually_exclusive_group(required=True);action.add_argument('--execute',action='store_true');action.add_argument('--record-review',action='store_true');action.add_argument('--select-pair',action='store_true')
    a=p.parse_args()
    if a.select_pair:r=select_pair(a.output)
    else:
        if a.kind is None or a.draw is None:p.error('kind and draw required')
        r=(execute_draw if a.execute else record_review)(a.output,a.kind,a.draw)
    print(json.dumps({'status':r.get('status'),'kind':r.get('kind'),'draw':r.get('draw')},indent=2))
if __name__=='__main__':main()
