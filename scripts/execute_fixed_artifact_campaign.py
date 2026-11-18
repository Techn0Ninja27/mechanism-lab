"""Single-attempt fixed-artifact lifecycle. Transport injection is explicit.

Base plan files never change. Preparation, review, and generation artifacts
have separate directories. This module does not launch reviewers or send HTTP
by itself; its caller must supply a transport and native-review launch records.
"""
from __future__ import annotations
import hashlib
import platform
import importlib.metadata
import fcntl
from functools import wraps
from datetime import datetime
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping

from scripts import run_fixed_artifact_campaign as planner
from scripts.generate_causal_campaign import parse_candidates
from generation.api_budget import BudgetLedger, BudgetError
from generation.fixed_artifact_client import FixedArtifactClient
from generation.fixed_artifact_fidelity import build_report
from generation import reference_draw_reviews as reviews

ROOT = planner.ROOT
CANONICAL_OUTPUT = ROOT / 'results/fixed-artifact-comparison-v1'
PHASE_LEDGER = ROOT / 'results/improvement-phase-2026-11-16/budget.json'
K = 4


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value): return planner.digest(value)
def read(path): return json.loads(Path(path).read_text())

def _atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.fixed-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def _bound(path, value):
    if Path(path).exists():
        if read(path) != value: raise ValueError(f'bound artifact changed: {path}')
    else: _atomic(path, value)


def _bound_text(path, text):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != text: raise ValueError(f'bound text changed: {path}')
    else:
        with path.open('x') as f: f.write(text)


def _canonical(out):
    if Path(CANONICAL_OUTPUT).is_symlink() or Path(out).resolve() != Path(CANONICAL_OUTPUT).resolve():
        raise ValueError('execution requires the single canonical campaign directory')


def _code_hashes():
    names = ('scripts/execute_fixed_artifact_campaign.py', 'scripts/fixed_artifact_campaign.py', 'generation/fixed_artifact_client.py',
             'generation/fixed_artifact_transport.py', 'generation/fixed_artifact_analysis.py',
             'generation/fixed_artifact_fidelity.py', 'generation/reference_draw_reviews.py',
             'generation/fixed_artifact_review_launch.py',
             'scripts/run_fixed_artifact_campaign.py', 'scripts/generate_causal_campaign.py',
             'reports/fixed-artifact-comparison-v1/requirements.txt')
    return {name: sha(ROOT/name) for name in names}


def _cases(plan): return plan['cases']
def _slot(case, n, attempt='unattempted', error=None):
    row = {'id':f"{case['id']}-cand-{n:02d}", 'call_id':case['id'], 'decoder':case['decoder'],
           'arm':case['arm'], 'status':'invalid', 'attempt_status':attempt,
           'program':None, 'program_sha256':digest(None)}
    if error: row['error'] = error
    return row

def _empty(cases): return [_slot(c,n) for c in cases for n in range(K)]
def _inventory(out, plan, rows, status, error=None):
    result = {'schema':'causal-candidate-inventory.v1', 'campaign':'fixed-artifact-comparison-v1',
              'status':status, 'plan_sha256':sha(out/'plan.json'), 'candidates':rows,
              'train_eligible':False, 'novelty_established':False}
    if error: result['error'] = error
    _atomic(out/'inventory.json', result)
    return result


def _request(decoder, prompt):
    price = planner.DECODERS[decoder]
    return {'model':price['model'], 'messages':[{'role':'system','content':planner.SYSTEM},
            {'role':'user','content':prompt}], 'max_tokens':12288, 'reasoning':{'effort':'low'},
            'provider':{'only':['anthropic'], 'order':['anthropic'], 'allow_fallbacks':False,
                        'require_parameters':True,
                        'max_price':{'prompt':price['prompt']*1e6, 'completion':price['completion']*1e6}}}


def _client(out, transport):
    return FixedArtifactClient(out/'receipts', PHASE_LEDGER, transport=transport,
                               campaign_cap_usd=planner.CAP, phase_cap_usd=planner.PHASE_CAP)


def _preflight(client, requests):
    ledger = BudgetLedger(PHASE_LEDGER, planner.PHASE_CAP)
    additional = sum(client._reserve(req) for req in requests)
    if client._charged(ledger)+additional > planner.CAP+1e-12:
        raise BudgetError('remaining complete schedule exceeds campaign cap')
    if planner._phase_charged()+additional > planner.PHASE_CAP+1e-12:
        raise BudgetError('remaining complete schedule exceeds phase cap')


def _initial_doc(out, plan):
    return {'schema':'fixed-artifact-execution-freeze.v1', 'plan_sha256':sha(out/'plan.json'),
            'code_hashes':_code_hashes(), 'schedule':_cases(plan), 'models':plan['decoders'],
            'phase_ledger':str(Path(PHASE_LEDGER).resolve()),
            'runtime':{'python':platform.python_version(),'platform':platform.platform(),
                       'packages':{name:importlib.metadata.version(name) for name in ('torch','numpy')}},
            'caps':{'campaign':planner.CAP,'phase':planner.PHASE_CAP}}


def _serialized(fn):
    @wraps(fn)
    def wrapped(out, *args, **kwargs):
        out=Path(out); _canonical(out)
        with (out/'.execution.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:return fn(out,*args,**kwargs)
            finally:fcntl.flock(lock,fcntl.LOCK_UN)
    return wrapped


@_serialized
def initial_freeze(out):
    out = Path(out); _canonical(out)
    existing = (out/'execution-freeze.json').exists()
    plan = planner.verify_plan(out, require_current_phase=not existing)
    if not existing and ((out/'prepared').exists() or (out/'receipts').exists()):
        raise ValueError('cannot create a retrospective execution freeze')
    doc = _initial_doc(out, plan)
    _bound(out/'execution-freeze.json', doc)
    return doc


def _require_initial(out):
    _canonical(out)
    plan = planner.verify_plan(out, require_current_phase=False)
    if read(out/'execution-freeze.json') != _initial_doc(out, plan):
        raise ValueError('execution freeze changed')
    return plan


def _stop(out, plan, reason, rows=None):
    _bound(out/'campaign-stop.json', {'reason':reason})
    return _inventory(out, plan, rows if rows is not None else _empty(_cases(plan)), 'incomplete', reason)


def _require_prepared(out):
    prep = read(out/'prepared/preparation.json')
    if prep.get('status') != 'validated': raise ValueError('validated prose required')
    raw = out/'prepared/raw.txt'; artifact = out/'prepared/distillations.json'
    if prep.get('raw_sha256') != sha(raw) or prep.get('artifact_sha256') != sha(artifact):
        raise ValueError('prepared artifacts changed')
    receipt = read(out/'receipts/prep-distill.json')
    if prep.get('receipt_sha256') != sha(out/'receipts/prep-distill.json') or receipt.get('content') != raw.read_text():
        raise ValueError('preparation receipt changed')
    validated = planner.c2._prep().validate_distillations(json.loads(raw.read_text()), read(out/'sourcepack.json'))
    if read(artifact) != validated: raise ValueError('prose does not match its response')
    return validated


def export_reviews(out):
    out = Path(out); _require_initial(out)
    valid = _require_prepared(out); pack = read(out/'sourcepack.json')
    for slot in ('a','b'):
        root = out/'prepared/reference-slots'/slot
        if not root.exists():
            reviews.build_review_set(root, plan_sha256=sha(out/'plan.json'), kind='distillations',
                                     draw=0, slot=slot, pack=pack, representation=valid)
        manifest = read(root/'manifest.json')
        if manifest.get('representation_sha256') != reviews._sha(valid): raise ValueError('review manifest changed')
        if (root/'reviewer-instructions.md').read_text() != reviews.REVIEW_INSTRUCTIONS+'\n':
            raise ValueError('review instruction changed')
        for i, record in enumerate(valid['distillations']):
            expected = reviews._packet_with_context('distillations',record,pack,sha(out/'plan.json'),0,slot,i)
            if read(root/f'input-{i:02d}.json') != expected: raise ValueError('review packet changed')
    return out/'prepared/reference-slots'


@_serialized
def prepare(out, transport):
    from scripts.fixed_artifact_campaign import require_execution_audit
    out = Path(out); require_execution_audit(out); plan = _require_initial(out)
    if (out/'campaign-stop.json').exists(): raise ValueError('campaign stopped')
    if (out/'generation-freeze.json').exists():
        _require_generation(out,plan)
        return read(out/'prepared/preparation.json')
    client = _client(out, transport)
    req = _request('opus-5.5',(out/'prompts/prose-preparation.txt').read_text())
    if not (out/'receipts/prep-distill.json').exists():
        # Check the complete bound before buying preparation, not merely this request.
        if planner._phase_charged()+plan['worst_case_campaign_reserve_usd'] > planner.PHASE_CAP+1e-12:
            raise BudgetError('full campaign no longer fits phase headroom')
    try:
        rec = client.call(req,'prep-distill.json')
    except BaseException as exc:
        _stop(out,plan,f'preparation transport failed: {type(exc).__name__}')
        raise
    _bound_text(out/'prepared/raw.txt', rec['content'])
    try:
        valid = planner.c2._prep().validate_distillations(json.loads(rec['content']),read(out/'sourcepack.json'))
    except Exception as exc:
        result={'status':'invalid','raw_sha256':sha(out/'prepared/raw.txt'),'error':str(exc)[:400]}
        _bound(out/'prepared/preparation.json',result)
        _stop(out,plan,'prose failed full preparation validator')
        return result
    _bound(out/'prepared/distillations.json',valid)
    result={'status':'validated','raw_sha256':sha(out/'prepared/raw.txt'),
            'artifact_sha256':sha(out/'prepared/distillations.json'),
            'receipt_sha256':sha(out/'receipts/prep-distill.json')}
    _bound(out/'prepared/preparation.json',result)
    export_reviews(out)
    return result


def _launch_bindings(out):
    bindings={}; tasks=set()
    for slot in ('a','b'):
        for i in range(12):
            base=out/'prepared/review-launches'; name=f'{slot}-{i:02d}'
            intent=base/f'{name}-intent.json'; done=base/f'{name}-completion.json'
            a=read(intent); b=read(done)
            try:
                start=datetime.fromisoformat(a['recorded_at_utc']);end=datetime.fromisoformat(b['completed_at_utc'])
                if start.tzinfo is None or end.tzinfo is None or start>end:raise ValueError('review timestamps invalid')
            except (KeyError,TypeError,ValueError) as exc:
                raise ValueError('review launch/completion times missing or invalid') from exc
            if a.get('schema')!='fixed-artifact-review-intent.v1' or b.get('schema')!='fixed-artifact-review-completion.v1':
                raise ValueError('review provenance schema invalid')
            packet=out/f'prepared/reference-slots/{slot}/input-{i:02d}.json'
            instruction=packet.parent/'reviewer-instructions.md'
            if (a.get('slot')!=slot or a.get('index')!=i or a.get('packet_sha256')!=sha(packet)
                or a.get('instruction_sha256')!=sha(instruction) or not isinstance(a.get('task_name'),str)
                or not a['task_name'] or a['task_name'] in tasks or not isinstance(a.get('prompt'),str)
                or str(packet.resolve()) not in a['prompt'] or str(instruction.resolve()) not in a['prompt']
                or b.get('intent_sha256')!=sha(intent) or b.get('agent_task')!=a['task_name']
                or b.get('status') not in {'completed','failed'}):
                raise ValueError('review launch/completion record invalid')
            tasks.add(a['task_name']); bindings[str(intent.relative_to(out))]=sha(intent)
            bindings[str(done.relative_to(out))]=sha(done)
    return bindings


def _fidelity(out):
    root=export_reviews(out)
    result=build_report(root)
    result['representation_sha256']=sha(out/'prepared/distillations.json')
    return result


@_serialized
def finalize_generation(out):
    out=Path(out); plan=_require_initial(out)
    if (out/'campaign-stop.json').exists(): raise ValueError('campaign stopped')
    if (out/'generation-freeze.json').exists():
        _require_generation(out,plan)
        return read(out/'generation-freeze.json')
    valid=_require_prepared(out); launch_hashes=_launch_bindings(out); fidelity=_fidelity(out)
    _bound(out/'prepared/prose-fidelity.json',fidelity)
    bodies=planner._prompt_bodies(out,valid); files={}; requests=[]
    for arm,body in bodies.items():
        path=out/f'generation-prompts/{arm}.txt'; _bound_text(path,body)
        files[arm]={'path':str(path.relative_to(out)),'sha256':sha(path),'bytes':len(body.encode())}
    for case in _cases(plan):
        request=_request(case['decoder'],bodies[case['arm']])
        requests.append({**case,'request_sha256':digest(request),
                         'reserve_usd':planner._reserve_bytes(len(bodies[case['arm']].encode()),planner.DECODERS[case['decoder']])})
    # This preflight makes no HTTP call and does not read a credential.
    client=_client(out,lambda _:None)
    _preflight(client,[_request(c['decoder'],bodies[c['arm']]) for c in _cases(plan)])
    doc={'schema':'fixed-artifact-generation-freeze.v1','initial_freeze_sha256':sha(out/'execution-freeze.json'),
         'plan_sha256':sha(out/'plan.json'),'code_hashes':_code_hashes(),
         'prepared_sha256':sha(out/'prepared/distillations.json'), 'preparation_sha256':sha(out/'prepared/preparation.json'),
         'atom_fidelity_sha256':sha(out/'atoms-fidelity.json'),'prose_fidelity_sha256':sha(out/'prepared/prose-fidelity.json'),
         'review_launch_hashes':launch_hashes,'prompt_files':files,'requests':requests}
    _bound(out/'generation-freeze.json',doc)
    return doc


def _require_generation(out,plan):
    frozen=read(out/'generation-freeze.json'); valid=_require_prepared(out)
    if (frozen.get('initial_freeze_sha256')!=sha(out/'execution-freeze.json')
        or frozen.get('plan_sha256')!=sha(out/'plan.json') or frozen.get('code_hashes')!=_code_hashes()
        or frozen.get('prepared_sha256')!=sha(out/'prepared/distillations.json')
        or frozen.get('preparation_sha256')!=sha(out/'prepared/preparation.json')
        or frozen.get('atom_fidelity_sha256')!=sha(out/'atoms-fidelity.json')
        or frozen.get('prose_fidelity_sha256')!=sha(out/'prepared/prose-fidelity.json')
        or frozen.get('review_launch_hashes')!=_launch_bindings(out)
        or read(out/'prepared/prose-fidelity.json')!=_fidelity(out)):
        raise ValueError('generation input/fidelity binding changed')
    bodies=planner._prompt_bodies(out,valid); requests=[]; expected=[]
    for arm,body in bodies.items():
        path=out/f'generation-prompts/{arm}.txt'
        if path.read_text()!=body or frozen['prompt_files'].get(arm)!={'path':str(path.relative_to(out)),'sha256':sha(path),'bytes':len(body.encode())}:
            raise ValueError('generation prompt changed')
    for case in _cases(plan):
        req=_request(case['decoder'],bodies[case['arm']]); requests.append(req)
        expected.append({**case,'request_sha256':digest(req),'reserve_usd':planner._reserve_bytes(len(bodies[case['arm']].encode()),planner.DECODERS[case['decoder']])})
    if frozen.get('requests')!=expected: raise ValueError('generation request order changed')
    return requests


def _restore_rows(out,plan,client):
    ledger=BudgetLedger(PHASE_LEDGER,planner.PHASE_CAP); snapshot=ledger.snapshot()['reservations']
    rows=_empty(_cases(plan)); terminal=None; unattempted_gap=False
    for i,case in enumerate(_cases(plan)):
        path=out/f"receipts/{case['id']}.json"
        reserved=any((r.get('metadata') or {}).get('receipt_path')==str(path.resolve()) for r in snapshot.values())
        if not path.exists():
            if reserved:
                terminal='reserved generation receipt is missing'
                rows[i*K:i*K+K]=[_slot(case,n,'failed',terminal) for n in range(K)]
            else:unattempted_gap=True
            continue
        try:
            rec=read(path)
            if rec.get('status')!='complete':raise ValueError('generation request failed or interrupted')
            if unattempted_gap:raise ValueError('generation receipts are out of schedule order')
            client._check_record(path.resolve(),rec,snapshot)
            parsed=parse_candidates(rec['content'],case)
        except (ValueError,RuntimeError,OSError,TypeError,AttributeError,KeyError) as exc:
            terminal=f'unverified generation receipt: {type(exc).__name__}: {str(exc)[:200]}'
            rows[i*K:i*K+K]=[_slot(case,n,'failed',terminal) for n in range(K)]
            continue
        for row in parsed:row.update(decoder=case['decoder'],attempt_status='attempted')
        rows[i*K:i*K+K]=parsed
    return rows,terminal


@_serialized
def execute_generation(out,transport):
    from scripts.fixed_artifact_campaign import require_execution_audit
    out=Path(out);require_execution_audit(out);plan=_require_initial(out);requests=_require_generation(out,plan)
    if not (out/'inventory.json').exists():_inventory(out,plan,_empty(_cases(plan)),'running')
    client=_client(out,transport);rows,terminal=_restore_rows(out,plan,client)
    if terminal or (out/'campaign-stop.json').exists() or (out/'receipts/.stop.json').exists():
        return _inventory(out,plan,rows,'incomplete',terminal or 'campaign stopped')
    cases=_cases(plan)
    for i,case in enumerate(cases):
        if rows[i*K]['attempt_status']=='attempted':continue
        _require_initial(out);_require_generation(out,plan)
        remaining=[requests[j] for j in range(i,len(cases)) if rows[j*K]['attempt_status']!='attempted']
        try:_preflight(client,remaining)
        except BudgetError as exc:return _stop(out,plan,'remaining schedule budget no longer fits',rows)
        try:rec=client.call(requests[i],f"{case['id']}.json")
        except BaseException as exc:
            reason=f'generation transport failed: {type(exc).__name__}'
            rows[i*K:i*K+K]=[_slot(case,n,'failed',reason) for n in range(K)]
            return _stop(out,plan,reason,rows)
        parsed=parse_candidates(rec['content'],case)
        for row in parsed:row.update(decoder=case['decoder'],attempt_status='attempted')
        rows[i*K:i*K+K]=parsed;_inventory(out,plan,rows,'running')
    return _inventory(out,plan,rows,'generated')
