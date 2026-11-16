import hashlib,json,tempfile,unittest
from pathlib import Path
from scripts.recover_atomization_atoms import recover,ROOT
class AtomRecoveryTests(unittest.TestCase):
    def test_valid_neighbor_survives_bad_quote_without_changing_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);papers=root/'papers.json';papers.write_text(json.dumps([{'source_id':'p','title':'T','url':'https://example.org/p','abstract':'A measured result.'}]))
            run=root/'run';(run/'calls').mkdir(parents=True)
            (run/'plan.json').write_text(json.dumps({'input_sha256':hashlib.sha256(papers.read_bytes()).hexdigest(),'source_hashes':{'scripts/literature_atomization.py':hashlib.sha256((ROOT/'scripts/literature_atomization.py').read_bytes()).hexdigest()}}))
            atom={'source_id':'p','source_quote':'A measured result.','statement':'The source reports a measurement.','kind':'finding','inputs':[],'outputs':[],'assumptions':[]}
            bad=dict(atom,source_quote='Invented quote')
            receipt=run/'calls/atom-batch-000-0.json';receipt.write_text(json.dumps({'status':'complete','content':json.dumps({'atoms':[atom,bad]})}));before=receipt.read_bytes()
            result=recover(papers,run,root/'recovery')
            self.assertEqual((result['atoms'],result['covered_papers'],len(result['rejected'])),(1,1,1))
            self.assertEqual(receipt.read_bytes(),before)
            self.assertEqual(result['new_api_calls'],0)
            with self.assertRaises(FileExistsError):recover(papers,run,root/'recovery')
