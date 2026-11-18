import json,tempfile,unittest,hashlib
from pathlib import Path
from generation.recordwise_calibration_scoring import validate_review,score_directory,score_v4
class T(unittest.TestCase):
 def p(self):return {"scientific_fields":[{"field":"statement","value":"alpha"}],"cited_chunks":[{"chunk_id":"c0","text":"alpha is present"}]}
 def r(self,v="supported",e=None):return {"field_items":[{"field":"statement","value":"alpha","verdict":v,"reason":"specific","evidence":e if e is not None else [{"chunk_id":"c0","quote":"alpha is present"}]}]}
 def test_mechanical(self):self.assertTrue(validate_review(self.p(),self.r())["all_supported"])
 def test_bad_span(self):
  with self.assertRaises(ValueError):validate_review(self.p(),self.r(e=[{"chunk_id":"c1","quote":"alpha"}]))
 def test_directory(self):
  with tempfile.TemporaryDirectory() as t:
   q=Path(t);(q/'input-p0.json').write_text(json.dumps({**self.p(),"opaque_id":"p0"}));(q/'answer-key.json').write_text(json.dumps({"records":[{"packet_id":"p0","fields":{"statement":"supported"}}]}))
   for n in ('a','b'):(q/f'review-p0-{n}.json').write_text(json.dumps(self.r()))
   self.assertTrue(score_directory(q)['all_reviewers_match'])
 def test_unknown_flag_cannot_remove_evidence_requirement(self):
  review=self.r(e=[]);review['field_items'][0].update(is_unknown=True,absence_reason='not stated')
  with self.assertRaises(ValueError):validate_review(self.p(),review)
 def test_subclaim_evidence_must_be_bound(self):
  review=self.r();review['field_items'][0]['claim_assessments']=[{'claim':'alpha','verdict':'supported','evidence':[{'chunk_id':'c0','quote':'not in source'}]}]
  with self.assertRaises(ValueError):validate_review(self.p(),review)
 def test_v4_full_frozen_shape(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);reviewroot=root/'direct';reviewroot.mkdir();specs=[];records=[];fields=[]
   def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
   for i in range(8):
    pid=f'p{i}';packet={**self.p(),'opaque_id':pid};path=root/f'input-{i}.json';path.write_text(json.dumps(packet));specs.append({'path':path.name,'opaque_id':pid,'sha256':sha(path)});records.append({'opaque_id':pid});fields.append({'opaque_id':pid,'fields':[{'field':'statement','verdict':'supported'}]})
    for slot in ('a','b'):(reviewroot/f'review-{i}-{slot}.json').write_text(json.dumps(self.r()))
   key=root/'answer-key.json';key.write_text(json.dumps({'records':records,'field_items':fields}));inst=root/'reviewer-instructions.md';inst.write_text('fixed instructions')
   manifest={'schema':'recordwise-calibration-v4-manifest.v1','packet_count':8,'packets':specs,'answer_key_sha256':sha(key),'instructions_sha256':sha(inst)};(root/'manifest.json').write_text(json.dumps(manifest))
   result=score_v4(root,reviewroot);self.assertEqual(result['review_count'],16);self.assertTrue(result['all_reviewers_match'])
   (reviewroot/'review-0-a.json').unlink()
   with self.assertRaises(ValueError):score_v4(root,reviewroot)
if __name__=='__main__':unittest.main()
