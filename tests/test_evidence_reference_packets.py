import json
import tempfile
import unittest
from pathlib import Path
from generation.evidence_reference_packets import build_evidence_packet, decode_review, build_probe_v5
from scripts import prepare_chunk_representations as prep

ROOT=Path(__file__).resolve().parents[1]
PACK=ROOT/'results/gnn-useful-proposal-v1/pack.json'
V4=ROOT/'results/recordwise-calibration-v4'

class EvidenceReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pack=prep._pack(PACK)
        cls.record=json.loads((V4/'input-0.json').read_text())

    def test_packet_has_stable_ids_and_windows(self):
        p=build_evidence_packet('atoms',self.record,self.pack)
        self.assertEqual([x['field_id'] for x in p['fields']], [f'F{i:03d}' for i in range(len(p['fields']))])
        self.assertEqual([x['evidence_id'] for x in p['evidence']], [f'E{i:04d}' for i in range(len(p['evidence']))])
        self.assertTrue(all(len(x['text'])<=300 for x in p['evidence']))
        self.assertNotIn('Selected Section IV only', ''.join(x['text'] for x in p['evidence']))

    def test_decoder_resolves_values_and_rejects_copied_text(self):
        p=build_evidence_packet('atoms',self.record,self.pack)
        out={'fields':[{'field_id':x['field_id'],'verdict':'supported','reason':'field-specific review','evidence_ids':[p['evidence'][0]['evidence_id']]} for x in p['fields']]}
        decoded=decode_review(p,out)
        self.assertEqual([x['value'] for x in decoded['field_items']], [x['value'] for x in p['fields']])
        bad={'fields':[dict(out['fields'][0],value='copied')]+out['fields'][1:]}
        with self.assertRaises(ValueError): decode_review(p,bad)

    def test_unknown_ids_and_duplicate_fields_rejected(self):
        p=build_evidence_packet('atoms',self.record,self.pack)
        item={'field_id':'F000','verdict':'unsupported','reason':'missing','evidence_ids':[]}
        items=[item]+[{'field_id':x['field_id'],'verdict':'supported','reason':'ok','evidence_ids':[p['evidence'][0]['evidence_id']]} for x in p['fields'][1:]]
        with self.assertRaises(ValueError): decode_review(p,{'fields':items+[dict(item)]})
        with self.assertRaises(ValueError): decode_review(p,{'fields':[dict(x,field_id='F999') if i==0 else x for i,x in enumerate(items)]})

    def test_claim_assessments_and_unsupported_preserved(self):
        p=build_evidence_packet('atoms',self.record,self.pack)
        items=[]
        for i,x in enumerate(p['fields']):
            items.append({'field_id':x['field_id'],'verdict':'unsupported' if i==0 else 'supported','reason':'specific reason','evidence_ids':[] if i==0 else [p['evidence'][0]['evidence_id']]})
        items[0]['claim_assessments']=[{'verdict':'unsupported','reason':'missing condition','evidence_ids':[]}]
        d=decode_review(p,{'fields':items})
        self.assertEqual(d['field_items'][0]['verdict'],'unsupported')

    def test_packet_hash_top_level_and_unknown_bypass_are_rejected(self):
        p=build_evidence_packet('atoms',self.record,self.pack)
        out={'fields':[{'field_id':x['field_id'],'verdict':'supported','reason':'specific','evidence_ids':[p['evidence'][0]['evidence_id']]} for x in p['fields']]}
        tampered=json.loads(json.dumps(p));tampered['fields'][0]['value']='changed'
        with self.assertRaises(ValueError): decode_review(tampered,out)
        with self.assertRaises(ValueError): decode_review(p,{**out,'extra':1})
        unknown=next((x for x in p['fields'] if x['is_unknown']),None)
        if unknown:
            items=json.loads(json.dumps(out['fields'])); item=next(x for x in items if x['field_id']==unknown['field_id']); item['evidence_ids']=[]; item['absence_reason']='not stated'; item['is_unknown']=False
            with self.assertRaises(ValueError): decode_review(p,{'fields':items})

    def test_nested_claims_are_resolved_and_supported_claims_require_evidence(self):
        p=build_evidence_packet('atoms',self.record,self.pack); items=[]
        for x in p['fields']:
            items.append({'field_id':x['field_id'],'verdict':'supported','reason':'specific','evidence_ids':[p['evidence'][0]['evidence_id']]})
        items[0]['claim_assessments']=[{'verdict':'supported','reason':'sub','evidence_ids':[p['evidence'][0]['evidence_id']]}]
        d=decode_review(p,{'fields':items}); self.assertIn('evidence',d['field_items'][0]['claim_assessments'][0]); self.assertNotIn('evidence_ids',d['field_items'][0]['claim_assessments'][0])
        items[0]['claim_assessments'][0]['verdict']='unsupported'
        with self.assertRaises(ValueError): decode_review(p,{'fields':items})

    def test_field_reference_cap_and_subclaim_absence_bypass(self):
        p=build_evidence_packet('atoms',self.record,self.pack)
        items=[{'field_id':x['field_id'],'verdict':'supported','reason':'specific','evidence_ids':[p['evidence'][0]['evidence_id']]} for x in p['fields']]
        items[0]['evidence_ids']=[x['evidence_id'] for x in p['evidence'][:5]]
        self.assertEqual(len(items[0]['evidence_ids']),5)
        with self.assertRaises(ValueError): decode_review(p,{'fields':items})
        items[0]['evidence_ids']=[p['evidence'][0]['evidence_id']]
        items[0]['claim_assessments']=[{'verdict':'supported','reason':'missing evidence','evidence_ids':[], 'absence_reason':'invented unknown'}]
        with self.assertRaises(ValueError): decode_review(p,{'fields':items})

    def test_probe_builder_omits_answer_key(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'probe'; m=build_probe_v5(V4,out,PACK)
            self.assertEqual(len(m['packets']),8); self.assertFalse((out/'answer-key.json').exists())

if __name__=='__main__': unittest.main()

