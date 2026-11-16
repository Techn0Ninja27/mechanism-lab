import unittest
from scripts.select_arxiv_metadata import group
class ArxivSelectionTests(unittest.TestCase):
    def test_crosslisted_learning_retains_ml_priority(self):
        self.assertEqual(group(['math.PR','stat.ML']),'ml')
        self.assertEqual(group(['cs.DC','cs.LG']),'ml')
    def test_requested_systems_math_and_unrelated_boundaries(self):
        self.assertEqual(group('cs.OS cs.DC'),'systems')
        self.assertEqual(group('math.NA math.OC'),'math')
        self.assertIsNone(group('astro-ph.CO hep-th'))

    def test_all_ml_removes_only_ml_cap_and_preserves_supporting_selection(self):
        import contextlib,hashlib,io,json,tempfile
        from pathlib import Path
        from unittest.mock import patch
        from scripts import select_arxiv_metadata as selector
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest('optional local corpus dependency pyarrow unavailable')
        rows=[]
        for i,category in enumerate(['cs.LG','cs.AI','stat.ML','cs.DC','cs.OS','math.NA','math.OC','cs.LG']):
            rows.append(dict(paper_id=f'2601.{i:05d}',title=f'Paper {i}',abstract='Primary text',authors=['A'],
                             categories=category,primary_category=category,first_version_date='2026-01-01',
                             latest_version_date='2026-11-17' if i==7 else '2026-01-01',doi=None,license=None,
                             arxiv_abs_url=f'https://arxiv.org/abs/2601.{i:05d}'))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'input.parquet';pq.write_table(pa.Table.from_pylist(rows),source)
            digest=hashlib.sha256(source.read_bytes()).hexdigest()
            for mode in ['sample','all']:
                args=['selector','--input',str(source),'--output',str(root/mode)] + (['--all-ml'] if mode=='all' else [])
                with patch.object(selector,'EXPECTED_SHA',digest),patch.object(selector,'TARGETS',{'ml':1,'systems':1,'math':1}),patch('sys.argv',args),contextlib.redirect_stdout(io.StringIO()):
                    selector.main()
            sample=[json.loads(x) for x in (root/'sample/records.jsonl').read_text().splitlines()]
            all_rows=[json.loads(x) for x in (root/'all/records.jsonl').read_text().splitlines()]
            self.assertEqual(len(sample),3);self.assertEqual(len(all_rows),5)
            self.assertEqual([r for r in sample if r['corpus_group']!='ml'],[r for r in all_rows if r['corpus_group']!='ml'])
            self.assertEqual({r['id'] for r in all_rows if r['corpus_group']=='ml'},{'2601.00000','2601.00001','2601.00002'})
            manifest=json.loads((root/'all/selection-manifest.json').read_text())
            self.assertEqual(manifest['selected']['ml'],manifest['eligible']['ml'])
            self.assertIsNone(manifest['targets']['ml'])
