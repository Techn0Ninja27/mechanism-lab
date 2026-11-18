import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from contextlib import ExitStack

from generation.api_budget import BudgetLedger
from generation.fixed_artifact_analysis import split_inventory
from scripts import execute_fixed_artifact_campaign as execution
from scripts import run_fixed_artifact_campaign as planner

PROGRAM={'predict':{'var':'last_sync'},'update':{'var':'memory'}}

def candidate_text():
    return json.dumps({'candidates':[{'id':f'cand-{i:02d}','program':PROGRAM,
                      'hypothesis':'Fixture only','falsifier':'Fixture comparison'} for i in range(4)]})

class FixedExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.out=Path(self.temp.name)/'campaign'; self.ledger=Path(self.temp.name)/'ledger.json'
        BudgetLedger(self.ledger,30).snapshot()
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(execution,'CANONICAL_OUTPUT',self.out))
        self.stack.enter_context(patch.object(execution,'PHASE_LEDGER',self.ledger))
        def charged():
            rows=BudgetLedger(self.ledger,30).snapshot()['reservations'].values()
            return sum(r.get('actual_usd',r['reserved_usd']) if r['status'] in {'reconciled','over_budget'} else r['reserved_usd'] for r in rows)
        self.stack.enter_context(patch.object(planner,'_phase_charged',side_effect=charged))
        self.plan=planner.build_plan(self.out)
        atoms=json.loads(planner.stopped_atoms().read_text())['atoms']
        self.prose=json.dumps({'distillations':[{'id':a['id'],'source_id':a['source_id'],
              'summary':a['statement'],'unknowns':a['unknowns'],'evidence':a['evidence']} for a in atoms]})
        self.sent=[]

    def response(self,request,content):
        return {'response_id':f'fixture-{len(self.sent)}','resolved_model':request['model'],
                'finish_reason':'stop','content':content,
                'usage':{'prompt_tokens':3,'completion_tokens':2,'cost':0.001}}

    def transport(self,request):
        self.sent.append(request)
        return self.response(request,self.prose if len(self.sent)==1 else candidate_text())

    def launch_fixtures(self):
        from generation.fixed_artifact_review_launch import launch_intent, record_completion
        # Explicit simulated failed reviews: missing measurements do not become support labels.
        for slot in ('a','b'):
            for i in range(12):
                task=f'/root/fixture-{slot}-{i}'
                launch_intent(self.out,slot,i,task)
                record_completion(self.out,slot,i,task,'failed')

    def freeze(self):
        execution.initial_freeze(self.out)
        source={'schema':'fixed-artifact-independent-audit.v1','status':'passed','reviewer':'fixture-only',
                'plan_sha256':execution.sha(self.out/'plan.json'),
                'execution_freeze_sha256':execution.sha(self.out/'execution-freeze.json')}
        execution._bound(self.out/'audit-source.json',source)
        wrapper={'schema':'fixed-artifact-execution-audit.v1','status':'passed',
                 'plan_sha256':source['plan_sha256'],'execution_freeze_sha256':source['execution_freeze_sha256'],
                 'source_report':'audit-source.json','source_report_sha256':execution.sha(self.out/'audit-source.json')}
        execution._bound(self.out/'execution-audit.json',wrapper)

    def ready(self):
        self.freeze()
        execution.prepare(self.out,self.transport)
        self.launch_fixtures()
        execution.finalize_generation(self.out)

    def test_real_plan_full_mocked_lifecycle_and_resume(self):
        original=(self.out/'plan.json').read_bytes(); self.ready()
        result=execution.execute_generation(self.out,self.transport)
        self.assertEqual(result['status'],'generated');self.assertEqual(len(result['candidates']),192)
        self.assertEqual(len(self.sent),49)
        self.assertTrue(all(r['status']=='valid' for r in result['candidates']))
        controls=json.loads((self.out/'random-inventory.json').read_text())['candidates']+json.loads((self.out/'controls.json').read_text())['linear_controls']
        split=split_inventory(result,controls,expected_cases=self.plan['cases'])
        self.assertEqual([len(v['candidates']) for v in split.values()],[124,124])
        self.assertEqual(execution.execute_generation(self.out,self.transport),result)
        self.assertEqual(len(self.sent),49);self.assertEqual(original,(self.out/'plan.json').read_bytes())
        execution.finalize_generation(self.out)
        from scripts.fixed_artifact_campaign import export_inventories
        self.assertEqual(export_inventories(self.out)['decoders'],{'opus-5.5':124,'fable-5.2':124})
        self.assertEqual(len(self.sent),49)
        self.assertEqual([r['model'] for r in self.sent[1:]], [self.plan['decoders'][c['decoder']]['model'] for c in self.plan['cases']])

    def test_missing_initial_freeze_cannot_construct_paid_request(self):
        with self.assertRaises(FileNotFoundError):execution.prepare(self.out,self.transport)
        self.assertEqual(self.sent,[])

    def test_invalid_prose_stops_all_generation_and_preserves_slots(self):
        self.freeze()
        def bad(request): self.sent.append(request);return self.response(request,'{invalid')
        result=execution.prepare(self.out,bad)
        self.assertEqual(result['status'],'invalid');self.assertEqual(len(self.sent),1)
        inventory=json.loads((self.out/'inventory.json').read_text())
        self.assertEqual(len(inventory['candidates']),192)
        self.assertEqual({r['attempt_status'] for r in inventory['candidates']},{'unattempted'})
        with self.assertRaises(ValueError):execution.prepare(self.out,bad)
        self.assertEqual(len(self.sent),1)

    def test_provider_stop_preserves_prefix_and_never_retries(self):
        self.ready()
        def fail_third(request):
            self.sent.append(request)
            if len(self.sent)==4: raise KeyboardInterrupt()
            return self.response(request,candidate_text())
        result=execution.execute_generation(self.out,fail_third)
        self.assertEqual(result['status'],'incomplete');self.assertEqual(len(self.sent),4)
        self.assertEqual(sum(r['attempt_status']=='attempted' for r in result['candidates']),8)
        replay=execution.execute_generation(self.out,fail_third)
        self.assertEqual(len(self.sent),4)
        self.assertEqual(sum(r['attempt_status']=='attempted' for r in replay['candidates']),8)

    def test_bad_candidate_json_is_invalid_but_schedule_continues(self):
        self.ready()
        def bad_first(request):
            self.sent.append(request)
            return self.response(request,'bad JSON' if len(self.sent)==2 else candidate_text())
        result=execution.execute_generation(self.out,bad_first)
        self.assertEqual(len(self.sent),49)
        self.assertEqual(sum(r['status']=='invalid' for r in result['candidates']),4)

    def test_prompt_or_review_drift_blocks_before_generation(self):
        self.ready(); path=self.out/'generation-prompts/atoms.txt';path.write_text(path.read_text()+' changed')
        with self.assertRaises(ValueError):execution.execute_generation(self.out,self.transport)
        self.assertEqual(len(self.sent),1)

    def test_missing_review_launches_block_generation_freeze(self):
        self.freeze();execution.prepare(self.out,self.transport)
        with self.assertRaises(FileNotFoundError):execution.finalize_generation(self.out)
        self.assertEqual(len(self.sent),1)

    def test_missing_checkpoint_reconstructed_from_receipts(self):
        self.ready(); original=execution._inventory
        def crash_after_first(out,plan,rows,status,error=None):
            if status=='running' and any(r['attempt_status']=='attempted' for r in rows): raise RuntimeError('local checkpoint crash')
            return original(out,plan,rows,status,error)
        with patch.object(execution,'_inventory',side_effect=crash_after_first):
            with self.assertRaises(RuntimeError):execution.execute_generation(self.out,self.transport)
        self.assertEqual(len(self.sent),2)
        result=execution.execute_generation(self.out,self.transport)
        self.assertEqual(result['status'],'generated');self.assertEqual(len(self.sent),49)

    def test_replaced_receipt_returns_incomplete_without_more_transport(self):
        self.ready(); original=execution._inventory
        def crash(out,plan,rows,status,error=None):
            if status=='running' and any(r['attempt_status']=='attempted' for r in rows):
                raise RuntimeError('checkpoint crash')
            return original(out,plan,rows,status,error)
        with patch.object(execution,'_inventory',side_effect=crash):
            with self.assertRaises(RuntimeError):execution.execute_generation(self.out,self.transport)
        first=self.plan['cases'][0]['id'];path=self.out/f'receipts/{first}.json'
        receipt=json.loads(path.read_text());receipt['content']='another response';path.write_text(json.dumps(receipt))
        result=execution.execute_generation(self.out,self.transport)
        self.assertEqual(result['status'],'incomplete');self.assertEqual(len(result['candidates']),192)
        self.assertEqual(len(self.sent),2)

    def test_direct_paid_stages_require_audit_with_valid_freeze(self):
        self.freeze();audit=self.out/'execution-audit.json';audit.unlink()
        with self.assertRaises(FileNotFoundError):execution.prepare(self.out,self.transport)
        self.assertEqual(len(self.sent),0)
        self.ready();audit.unlink()
        with self.assertRaises(FileNotFoundError):execution.execute_generation(self.out,self.transport)
        self.assertEqual(len(self.sent),1)

    def test_canonical_copy_cannot_start_another_attempt(self):
        import shutil
        self.freeze();other=Path(self.temp.name)/'other';shutil.copytree(self.out,other)
        with self.assertRaises(ValueError):execution.prepare(other,self.transport)
        self.assertEqual(len(self.sent),0)

    def test_phase_headroom_for_remaining_schedule_checked_before_call(self):
        self.ready()
        BudgetLedger(self.ledger,30).reserve('outside-campaign',10,{'scope':'unit fixture'})
        result=execution.execute_generation(self.out,self.transport)
        self.assertEqual(result['status'],'incomplete');self.assertEqual(len(self.sent),1)
        self.assertEqual({x['attempt_status'] for x in result['candidates']},{'unattempted'})

if __name__=='__main__': unittest.main()
