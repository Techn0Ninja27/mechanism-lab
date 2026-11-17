import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from scripts.atomize_budgeted import execute,plan
class BudgetedAtomizationTests(unittest.TestCase):
    def papers(self,path,n=1):
        path.write_text(json.dumps([{'source_id':str(i),'title':'T','url':'https://example.org/p','abstract':'Evidence.'} for i in range(n)]))
    def test_offline_plan_never_constructs_paid_client(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'papers.json';self.papers(p)
            with patch('scripts.atomize_budgeted.Client') as c:
                result=execute(p,root/'run');c.assert_not_called()
            self.assertEqual(result['shared_budget_usd'],20)
            self.assertEqual(result['budget_usd'],.5)
    def test_overlarge_packet_rejected_before_client(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'papers.json';self.papers(p,33)
            with patch('scripts.atomize_budgeted.Client') as c:
                with self.assertRaises(ValueError):execute(p,root/'run',True)
                c.assert_not_called()
    def test_finished_run_is_not_reexecuted(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'papers.json';self.papers(p);out=root/'run';execute(p,out)
            (out/'summary.json').write_text('{}')
            with patch('scripts.atomize_budgeted.Client') as c:
                with self.assertRaises(FileExistsError):execute(p,out,True)
                c.assert_not_called()

    def test_historical_ten_dollar_plan_is_not_rewritten(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'papers.json';self.papers(p);out=root/'run'
            out.mkdir(); old={'shared_budget_usd':10.0,'input_sha256':'historical'}
            (out/'plan.json').write_text(json.dumps(old))
            with patch('scripts.atomize_budgeted.Client') as c:
                with self.assertRaises(ValueError): execute(p,out)
                c.assert_not_called()
            self.assertEqual(json.loads((out/'plan.json').read_text()),old)
