import hashlib, json, tempfile, unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from generation.interface_graph import load_atoms, build

class InterfaceGraphTest(unittest.TestCase):
 def setUp(self):
  self.d=Path(tempfile.mkdtemp()); s='Producer output statement.'
  self.atom={"source_id":"p","source_hash":"ph","statement":s,"kind":"method","inputs":["x"],"outputs":["shared output"],"assumptions":[{"text":"A"}]}
  t={"source_id":"t","source_hash":"th","statement":"Target method statement.","kind":"method","inputs":["shared output"],"outputs":["y"],"assumptions":[{"text":"B"}]}
  self.atoms=self.d/'a.jsonl'; self.atoms.write_text(json.dumps(self.atom)+'\n'+json.dumps(t)+'\n')
  self.papers=self.d/'p.json'; self.papers.write_text(json.dumps([{"source_id":"p","corpus_group":"ml","url":"u"},{"source_id":"t","corpus_group":"systems","url":"v"}]))
  self.review=self.d/'r.json'; self.review.write_text(json.dumps({"atoms":[{"source_id":"p","source_hash":"ph","statement_sha256":hashlib.sha256(s.encode()).hexdigest(),"reviewed":True,"flag_type":"none"},{"source_id":"t","source_hash":"th","statement_sha256":hashlib.sha256(t['statement'].encode()).hexdigest(),"reviewed":True,"flag_type":"none"}],"recovered_batch2_atoms":[]}))
 def test_assumptions_and_recovered_review(self):
  a,r,c=load_atoms(self.atoms,self.review,self.papers); o=build(a,r,c,limit=1); self.assertTrue(o['candidates'][0]['producer_assumptions'] in ([{"text":"A"}],[{"text":"B"}]))
 def test_bounds(self):
  a,r,c=load_atoms(self.atoms,self.review,self.papers)
  for n in (True,0,21,1.0):
   with self.assertRaises(ValueError): build(a,r,c,limit=n)
 def test_bad_review_or_atom_rejected(self):
  d=json.loads(self.review.read_text()); d['atoms'][0]['reviewed']=False; self.review.write_text(json.dumps(d)); self.assertEqual(len(load_atoms(self.atoms,self.review,self.papers)[0]),1)
  self.atom['statement']=3; self.atoms.write_text(json.dumps(self.atom)+'\n'); self.assertEqual(len(load_atoms(self.atoms,self.review,self.papers)[0]),0)

 def test_recovered_array_and_exact_provenance(self):
  original=json.loads(self.review.read_text())
  d=json.loads(json.dumps(original)); d['recovered_batch2_atoms']=[d['atoms'].pop(0)]
  self.review.write_text(json.dumps(d))
  self.assertEqual(len(load_atoms(self.atoms,self.review,self.papers)[0]),2)
  for field,value in [('source_hash','forged'),('source_hash',None),('statement_sha256','forged'),('flag_type','unknown')]:
   d=json.loads(json.dumps(original)); d['atoms'][0][field]=value
   self.review.write_text(json.dumps(d))
   self.assertEqual(len(load_atoms(self.atoms,self.review,self.papers)[0]),1,field)

 def test_conflicting_review_and_semantic_exclusion(self):
  d=json.loads(self.review.read_text()); other=dict(d['atoms'][0],flag_type='semantic')
  d['recovered_batch2_atoms']=[other]; self.review.write_text(json.dumps(d))
  self.assertEqual(len(load_atoms(self.atoms,self.review,self.papers)[0]),1)
  d['atoms'][0]['flag_type']='semantic'; self.review.write_text(json.dumps(d))
  self.assertEqual(len(load_atoms(self.atoms,self.review,self.papers)[0]),1)

 def test_invalid_interfaces(self):
  for field,value in [('inputs','not a list'),('outputs',[1]),('inputs',['']),('source_id',1),('source_hash',3),('kind','unknown')]:
   atom=dict(self.atom); atom[field]=value
   self.atoms.write_text(json.dumps(atom)+'\n')
   self.assertEqual(load_atoms(self.atoms,self.review,self.papers)[0],[],field)

if __name__=='__main__': unittest.main()
