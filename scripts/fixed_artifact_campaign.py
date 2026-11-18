"""Command entry point for the one fixed-artifact campaign.

Paid stages preview by default and require --execute plus a hash-bound audit
receipt. Credentials are read only after those checks. Local evaluation uses
run_causal_campaign on the exported per-decoder inventories.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from scripts import run_fixed_artifact_campaign as planner
from scripts import execute_fixed_artifact_campaign as lifecycle
from generation.fixed_artifact_transport import openrouter_transport
from generation.fixed_artifact_analysis import split_inventory


def require_execution_audit(out):
    lifecycle._require_initial(out)
    receipt=lifecycle.read(out/'execution-audit.json')
    if (receipt.get('schema')!='fixed-artifact-execution-audit.v1'
        or receipt.get('status')!='passed'
        or receipt.get('plan_sha256')!=lifecycle.sha(out/'plan.json')
        or receipt.get('execution_freeze_sha256')!=lifecycle.sha(out/'execution-freeze.json')):
        raise ValueError('paid execution requires a matching integrity audit')
    source=receipt.get('source_report')
    if not isinstance(source,str):raise ValueError('audit source report missing')
    path=(out/source).resolve()
    if not path.is_relative_to(out.resolve()) or not path.is_file() or lifecycle.sha(path)!=receipt.get('source_report_sha256'):
        raise ValueError('audit source report binding invalid')
    source_record=lifecycle.read(path)
    if (source_record.get('schema')!='fixed-artifact-independent-audit.v1'
        or source_record.get('status')!='passed'
        or source_record.get('plan_sha256')!=receipt['plan_sha256']
        or source_record.get('execution_freeze_sha256')!=receipt['execution_freeze_sha256']
        or not isinstance(source_record.get('reviewer'),str) or not source_record['reviewer']):
        raise ValueError('independent audit did not clear these exact artifacts')
    return receipt


def paid_stage(out, stage, *, execute=False, key_loader=None, transport_factory=openrouter_transport):
    if stage not in {'prepare','generate'}:raise ValueError('unknown paid stage')
    plan=lifecycle._require_initial(out)
    if not execute:
        return {'status':'preview','stage':stage,'paid_requests':0,
                'planned_requests':1 if stage=='prepare' else len(plan['cases']),
                'campaign_cap_usd':planner.CAP,'phase_cap_usd':planner.PHASE_CAP}
    require_execution_audit(out)
    if key_loader is None:
        from scripts.world_search import key
        key_loader=key
    # A missing credential is discovered before reserving money or recording an attempt.
    credential=key_loader()
    if not isinstance(credential,str) or not credential.strip():raise ValueError('credential unavailable')
    transport=transport_factory(lambda:credential)
    if stage=='prepare':return lifecycle.prepare(out,transport)
    return lifecycle.execute_generation(out,transport)


def export_inventories(out):
    plan=lifecycle._require_initial(out)
    inventory=lifecycle.read(out/'inventory.json')
    lifecycle._require_generation(out,plan)
    client=lifecycle._client(out,lambda _:None)
    expected,terminal=lifecycle._restore_rows(out,plan,client)
    def scientific_rows(rows):
        return [{k:v for k,v in row.items() if k!='error'} for row in rows]
    if scientific_rows(inventory.get('candidates',[]))!=scientific_rows(expected):
        raise ValueError('inventory no longer matches verified request receipts')
    if inventory.get('status')=='generated' and (terminal or any(r['attempt_status']!='attempted' for r in expected)):
        raise ValueError('generated status contradicts receipt coverage')
    if inventory.get('status') not in {'generated','incomplete'}:
        raise ValueError('generation must be finished or stopped before local evaluation')
    shared=lifecycle.read(out/'random-inventory.json')['candidates']+lifecycle.read(out/'controls.json')['linear_controls']
    split=split_inventory(inventory,shared,expected_cases=plan['cases'])
    for decoder,value in split.items():
        value['generation_inventory_sha256']=lifecycle.sha(out/'inventory.json')
        lifecycle._bound(out/f'evaluation/{decoder}/inventory.json',value)
    return {'status':'exported','decoders':{k:len(v['candidates']) for k,v in split.items()}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('plan','freeze','prepare','finalize','generate','export'))
    parser.add_argument('--execute',action='store_true',help='perform the paid stage after audit verification')
    args=parser.parse_args();out=lifecycle.CANONICAL_OUTPUT
    if args.stage=='plan':result=planner.build_plan(out)
    elif args.stage=='freeze':result=lifecycle.initial_freeze(out)
    elif args.stage in {'prepare','generate'}:result=paid_stage(out,args.stage,execute=args.execute)
    elif args.stage=='finalize':result=lifecycle.finalize_generation(out)
    else:result=export_inventories(out)
    omitted={'code_hashes','dependency_hashes','cases','schedule','candidates','requests','prompt_files','review_launch_hashes'}
    print(json.dumps({k:v for k,v in result.items() if k not in omitted},indent=2))


if __name__=='__main__':main()
