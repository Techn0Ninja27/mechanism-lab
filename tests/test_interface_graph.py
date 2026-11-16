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

 def test_mechanism_diverse_is_method_only_and_deterministic(self):
  a,r,c=load_atoms(self.atoms,self.review,self.papers)
  lexical=build(a,r,c,limit=1)
  diverse=build(a,r,c,limit=1,selection_mode='mechanism_diverse',max_paper_uses=1)
  self.assertEqual(lexical['candidates'][0]['producer_atom_id'],diverse['candidates'][0]['producer_atom_id'])
  self.assertEqual(diverse['selection']['selection_mode'],'mechanism_diverse')
  self.assertTrue(all(x['producer_kind']=='method' and x['target_kind']=='method' for x in diverse['candidates']))
  self.assertEqual(diverse,build(a,r,c,limit=1,selection_mode='mechanism_diverse',max_paper_uses=1))

 def test_mechanism_diverse_rejects_finding_and_reuses_no_unordered_pair(self):
  a,r,c=load_atoms(self.atoms,self.review,self.papers)
  finding=dict(a[0],source_id='f',source_hash='fh',kind='finding',statement='Finding statement.',inputs=['x'],outputs=['shared output'])
  fhash=hashlib.sha256(finding['statement'].encode()).hexdigest()
  papers=json.loads(self.papers.read_text());papers.append({'source_id':'f','corpus_group':'ml','url':'f'})
  review=json.loads(self.review.read_text());review['atoms'].append({'source_id':'f','source_hash':'fh','statement_sha256':fhash,'reviewed':True,'flag_type':'none'})
  self.atoms.write_text('\n'.join(json.dumps(x) for x in [self.atom,a[1],finding])+'\n')
  self.papers.write_text(json.dumps(papers));self.review.write_text(json.dumps(review))
  aa,rr,cc=load_atoms(self.atoms,self.review,self.papers)
  out=build(aa,rr,cc,limit=20,selection_mode='mechanism_diverse',max_paper_uses=2)
  pairs=[frozenset((x['producer_source'],x['target_source'])) for x in out['candidates']]
  self.assertEqual(len(pairs),len(set(pairs)))
  self.assertTrue(all(x['producer_kind']=='method' for x in out['candidates']))

 def test_selection_bounds(self):
  a,r,c=load_atoms(self.atoms,self.review,self.papers)
  for mode in ('bad',):
   with self.assertRaises(ValueError):build(a,r,c,selection_mode=mode)
  for n in (True,0,21,1.0):
   with self.assertRaises(ValueError):build(a,r,c,selection_mode='mechanism_diverse',max_paper_uses=n)

 def test_degree_cap_applies_to_ranked_and_random_with_real_competition(self):
  atoms=[]
  for group in ['ml','systems']:
   for i in range(3):
    atoms.append({'source_id':f'{group}-{i}','stable_atom_id':f'{group}-{i}',
                  'kind':'method','corpus_group':group,'inputs':['tensor'],
                  'outputs':['tensor'],'statement':f'tensor method {i}',
                  'source_hash':f'hash-{group}-{i}'})
  result=build(atoms,{}, {},selection_mode='mechanism_diverse',max_paper_uses=1)
  for key in ['candidates','seeded_random_shortlist']:
   self.assertEqual(len(result[key]),3)
   used=[r[side] for r in result[key] for side in ['producer_source','target_source']]
   self.assertEqual(len(used),len(set(used)))
  self.assertEqual(result,build(atoms,{}, {},selection_mode='mechanism_diverse',max_paper_uses=1))
  lexical=build(atoms,{}, {},limit=8)
  self.assertEqual(lexical['candidate_pool_size'],18)
  self.assertEqual(len(lexical['candidates']),8)

if __name__=='__main__': unittest.main()
