import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts import generate_causal_campaign2 as c
from scripts import run_campaign2_preparation as life
from scripts import prepare_chunk_representations as prep
from scripts.run_causal_campaign import load_inventory
PACK=c.ROOT/'results/gnn-useful-proposal-v1/pack.json'

class FakeClient:
    calls=[]
    fail=False
    def __init__(self,out,config):self.out=out;self.config=config
    def call(self,case,stage,prompt):
        self.calls.append(case['id'])
        if self.fail:raise RuntimeError('provider unavailable')
        if case['id'].startswith('prep-'):
            kind='atoms' if case['id'].startswith('prep-atoms') else 'distillations'
            pack=prep._pack(self.out/'sourcepack.json');chunks=prep.render_source_context(pack)['chunks'];rows=[]
            for sid,n in prep.source_quotas(pack).items():
                chunk=next(x for x in chunks if x['source_id']==sid)
                for _ in range(n):
                    row={'id':f'r{len(rows)}','source_id':sid,'unknowns':['Unspecified'],
                         'evidence':[{'chunk_id':chunk['id'],'quote':chunk['text'][:40]}]}
                    if kind=='atoms':row.update(statement=chunk['text'][:40],inputs=[],outputs=[],conditions=[])
                    else:row['summary']=chunk['text'][:40]
                    rows.append(row)
            content=json.dumps({kind:rows})
        else:content=json.dumps({'candidates':[{'id':f'cand-{i:02d}','program':{'predict':{'var':'last_sync'},'update':{'var':'memory'}},'hypothesis':'h','falsifier':'f'} for i in range(4)]})
        req=c.expected_request(prompt)
        c.write_new(self.out/'calls'/f"{case['id']}-{stage}.json",{'status':'complete','request':req,'request_hash':c.request_hash(req),'content':content,'reserved_usd':c.reserve(prompt)})
        return content

def reviews(out,kind,index,failed=False):
    draw=life._draw_dir(out,kind,index);value=json.loads((draw/f'{kind}.json').read_text());pack=prep._pack(out/'sourcepack.json')
    for reviewer in ('a','b'):
        rows=[]
        for record in value[kind]:
            fields=[]
            for field,text,unknown in prep._draw_fields(kind,record):
                item={'field':field,'value':text,'verdict':'supported','reason':'fixture'}
                if unknown:item['absence_reason']='fixture absent'
                else:item.update(chunk_id=record['evidence'][0]['chunk_id'],support_quote=record['evidence'][0]['quote'])
                fields.append(item)
            rows.append({'record_id':record['id'],'criterion':'cited_chunk_support.v1','field_items':fields})
        if failed and reviewer=='b':
            rows[0]['field_items'][0].update(verdict='unsupported',disputed_field=rows[0]['field_items'][0]['field'],reason='fixture unsupported')
        c.write_new(draw/f'review-{reviewer}.json',{'schema':prep.DRAW_REVIEW_SCHEMA,'reviewer_id':f'{kind}-{index}-{reviewer}',
                    'context':{'plan_sha256':c.sha(out/'plan.json'),'kind':kind,'draw':index,'slot':reviewer},'input_sha256':{'source_pack':pack['digest'],kind:c.digest(value)},'records':rows})

class Campaign2HarnessTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.out=Path(self.tmp.name)/'campaign';c.plan(PACK,self.out)
        p=patch.object(life,'CANONICAL_OUTPUT',self.out);p.start();self.addCleanup(p.stop)
        from generation.api_budget import BudgetLedger
        anchor=BudgetLedger(Path(self.tmp.name)/'phase-ledger.json',30)
        p=patch.object(life,'_anchor',return_value=anchor);p.start();self.addCleanup(p.stop)
        self.actual_calibration_check=c.require_calibration_ready
        p=patch.object(c,'require_calibration_ready',return_value=None);p.start();self.addCleanup(p.stop)
        FakeClient.calls=[];FakeClient.fail=False
        p=patch.object(c,'Client',FakeClient);p.start();self.addCleanup(p.stop)
    def test_current_failed_calibration_blocks_client(self):
        with patch.object(c,'require_calibration_ready',self.actual_calibration_check):
            with self.assertRaisesRegex(ValueError,'calibration failed'):
                life.execute_draw(self.out,'atoms',0)
        self.assertEqual(FakeClient.calls,[])

    def accept(self,kind,index=0):
        life.execute_draw(self.out,kind,index);reviews(self.out,kind,index);return life.record_review(self.out,kind,index)
    def test_blocked_random_and_inventory_counts(self):
        doc=json.loads((self.out/'plan.json').read_text())
        for i in range(6):self.assertEqual({r['arm'] for r in doc['cases'][i*4:i*4+4]},set(c.MODEL_ARMS))
        self.assertEqual(len(json.loads((self.out/'random-inventory.json').read_text())['candidates']),24)
    def test_invalid_order_path_and_gate_never_spend(self):
        for index in (1,3,-1,True):
            with self.assertRaises((ValueError,FileNotFoundError)):life.execute_draw(self.out,'atoms',index)
        with self.assertRaises(ValueError):life.execute_draw(self.out.parent/'other','atoms',0)
        with self.assertRaises(ValueError):c.generate(self.out,True)
        self.assertEqual(FakeClient.calls,[])
    def test_missing_review_cannot_advance(self):
        life.execute_draw(self.out,'atoms',0)
        with self.assertRaises(FileNotFoundError):life.record_review(self.out,'atoms',0)
        with self.assertRaises(FileNotFoundError):life.execute_draw(self.out,'atoms',1)
        self.assertEqual(len(FakeClient.calls),1)
    def test_full_first_pass_lifecycle_and_generation(self):
        life.execute_draw(self.out,'atoms',0);reviews(self.out,'atoms',0,failed=True)
        self.assertEqual(life.record_review(self.out,'atoms',0)['status'],'failed')
        locked=self.accept('atoms',1)
        self.assertEqual(life.record_review(self.out,'atoms',1),locked)
        with self.assertRaises(ValueError):life.execute_draw(self.out,'atoms',2)
        self.accept('distillations',0)
        selected=life.select_pair(self.out);self.assertEqual(selected['atoms']['draw'],1);self.assertEqual(selected['distillations']['draw'],0)
        self.assertEqual(life.select_pair(self.out),selected)
        plan=c.generation_plan(self.out);self.assertEqual(c.generation_plan(self.out),plan)
        result=c.generate(self.out,True)
        self.assertEqual(result['candidates'],124)
        inv,_=load_inventory(self.out/'inventory.json');self.assertEqual(len(inv['candidates']),124)
        self.assertEqual(len(FakeClient.calls),27)
        with self.assertRaises(FileExistsError):c.generate(self.out,True)
    def test_provider_failure_stops_other_representation(self):
        FakeClient.fail=True
        with self.assertRaises(RuntimeError):life.execute_draw(self.out,'atoms',0)
        FakeClient.fail=False
        with self.assertRaisesRegex(ValueError,'permanently stopped'):life.execute_draw(self.out,'distillations',0)
        self.assertEqual(len(FakeClient.calls),1)
    def test_keyboard_interrupt_stops_even_if_stop_marker_removed(self):
        with patch.object(FakeClient,'call',side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):life.execute_draw(self.out,'atoms',0)
        (self.out/'campaign-stop.json').unlink()
        with self.assertRaisesRegex(ValueError,'permanently stopped'):
            life.execute_draw(self.out,'distillations',0)
        self.assertEqual(FakeClient.calls,[])

    def test_system_exit_and_unfinished_draw_block_other_kind(self):
        with patch.object(c,'Client',side_effect=SystemExit('missing key')):
            with self.assertRaises(SystemExit):life.execute_draw(self.out,'atoms',0)
        with self.assertRaisesRegex(ValueError,'permanently stopped'):life.execute_draw(self.out,'distillations',0)

    def test_review_from_wrong_draw_is_rejected(self):
        life.execute_draw(self.out,'atoms',0);reviews(self.out,'atoms',0)
        path=life._draw_dir(self.out,'atoms',0)/'review-b.json';data=json.loads(path.read_text());data['context']['draw']=1;path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,'bound to this draw'):life.record_review(self.out,'atoms',0)
        self.assertFalse((self.out/'selected/atoms.selection.json').exists())

    def test_exhaustion_stops_other_representation(self):
        for i in range(3):
            life.execute_draw(self.out,'atoms',i);reviews(self.out,'atoms',i,failed=True);life.record_review(self.out,'atoms',i)
        with self.assertRaisesRegex(ValueError,'permanently stopped'):life.execute_draw(self.out,'distillations',0)
        self.assertEqual(len(FakeClient.calls),3)
    def test_recover_review_event_after_external_anchor_before_append(self):
        life.execute_draw(self.out,'atoms',0);reviews(self.out,'atoms',0)
        original=life.journal._append_row
        def interrupt(out,row):
            if row['event']=='review_decided':raise KeyboardInterrupt()
            return original(out,row)
        with patch.object(life.journal,'_append_row',side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):life.record_review(self.out,'atoms',0)
        locked=life.record_review(self.out,'atoms',0)
        self.assertEqual(locked['status'],'locked');self.assertEqual(len(FakeClient.calls),1)
        self.assertEqual(life._anchor(self.out).read_event_anchor(life._chain(self.out))['head'],life.journal.audit(self.out)['last_hash'])

    def test_recover_locked_artifact_before_semantic_commit(self):
        life.execute_draw(self.out,'atoms',0);reviews(self.out,'atoms',0)
        original=life.journal.atomic_write
        def interrupt(out,relative,data):
            result=original(out,relative,data)
            if relative=='selected/atoms.selection.json':raise KeyboardInterrupt()
            return result
        with patch.object(life.journal,'atomic_write',side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):life.record_review(self.out,'atoms',0)
        locked=life.record_review(self.out,'atoms',0)
        self.assertEqual(locked['draw'],0)
        with self.assertRaises(ValueError):life.execute_draw(self.out,'atoms',1)
        self.assertEqual(len(FakeClient.calls),1)

    def test_partial_anchored_tail_recovers_without_new_call(self):
        expected=self.accept('atoms',0)
        log=self.out/life.journal.JOURNAL_NAME;lines=log.read_bytes().splitlines(keepends=True)
        log.write_bytes(b''.join(lines[:-1])+lines[-1][:len(lines[-1])//2])
        self.assertEqual(life.record_review(self.out,'atoms',0),expected)
        self.assertEqual(len(FakeClient.calls),1)

    def test_trimmed_journal_and_lock_cannot_move_selection(self):
        self.accept('atoms',0)
        log=self.out/life.journal.JOURNAL_NAME;lines=log.read_bytes().splitlines(keepends=True)
        log.write_bytes(b''.join(lines[:-2]));(self.out/'selected/atoms.selection.json').unlink()
        with self.assertRaises(ValueError):life.execute_draw(self.out,'atoms',1)
        self.assertEqual(len(FakeClient.calls),1)

    def test_deleted_stop_after_exhaustion_does_not_reopen(self):
        for i in range(3):
            life.execute_draw(self.out,'atoms',i);reviews(self.out,'atoms',i,failed=True);life.record_review(self.out,'atoms',i)
        (self.out/'campaign-stop.json').unlink()
        with self.assertRaisesRegex(ValueError,'permanently stopped'):life.execute_draw(self.out,'distillations',0)
        self.assertEqual(len(FakeClient.calls),3)

    def test_deleted_failed_decision_cannot_be_rereviewed(self):
        life.execute_draw(self.out,'atoms',0);reviews(self.out,'atoms',0,failed=True);life.record_review(self.out,'atoms',0)
        (life._draw_dir(self.out,'atoms',0)/'decision.json').unlink()
        with self.assertRaises(ValueError):life.record_review(self.out,'atoms',0)
        with self.assertRaises(ValueError):life.execute_draw(self.out,'distillations',0)
        self.assertEqual(len(FakeClient.calls),1)

    def test_review_tamper_after_fail_cannot_advance(self):
        life.execute_draw(self.out,'atoms',0);reviews(self.out,'atoms',0,failed=True);life.record_review(self.out,'atoms',0)
        path=life._draw_dir(self.out,'atoms',0)/'review-b.json';d=json.loads(path.read_text());d['reviewer_id']='changed';path.write_text(json.dumps(d))
        with self.assertRaises(ValueError):life.execute_draw(self.out,'atoms',1)
        self.assertEqual(len(FakeClient.calls),1)
if __name__=='__main__':unittest.main()
