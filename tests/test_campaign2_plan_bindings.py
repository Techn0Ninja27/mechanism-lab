import json,tempfile,unittest
from pathlib import Path
from scripts import generate_causal_campaign2 as c
class PlanBindingsTests(unittest.TestCase):
 def test_metadata_and_control_mutations_rejected(self):
  for target in ('task','grammar','controls','manifest'):
   with self.subTest(target=target), tempfile.TemporaryDirectory() as td:
    out=Path(td)/'plan';c.plan(c.ROOT/'results/gnn-useful-proposal-v1/pack.json',out)
    if target in ('task','grammar'):
     p=out/'plan.json';d=json.loads(p.read_text());d[target]+=' changed';p.write_text(json.dumps(d))
    elif target=='controls':
     p=out/'controls.json';d=json.loads(p.read_text());d['linear_controls'][0]['program']['predict']={'var':'last_sync'};p.write_text(json.dumps(d))
    else:
     p=out/'preparation/manifest.json';p.write_text(p.read_text()+' ')
    with self.assertRaises(ValueError):c.verify(out)
if __name__=='__main__':unittest.main()
