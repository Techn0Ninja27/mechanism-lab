import json,tempfile,unittest
from pathlib import Path
from generation.corpus_registry import active_corpus
class CorpusRegistryTests(unittest.TestCase):
    def test_all_ml_snapshot_requires_completion_marker(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name in ('corpus-ml-complete-mirror-v6','corpus-ml-grounded-v4'):
                p=root/'results'/name;p.mkdir(parents=True);(p/'corpus.sqlite').touch()
            old=root/'results/corpus-ml-grounded-v4';(old/'bulk-manifest.json').write_text('{"records":10}')
            self.assertEqual(active_corpus(root)['snapshot'],'corpus-ml-grounded-v4')
            new=root/'results/corpus-ml-complete-mirror-v6';(new/'bulk-manifest.json').write_text('{"records":20}')
            self.assertEqual(active_corpus(root)['snapshot'],'corpus-ml-complete-mirror-v6')

    def test_incomplete_snapshot_cannot_replace_completed_corpus(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name in ('corpus-arxiv-100k','corpus-integrated-v3'):
                p=root/'results'/name;p.mkdir(parents=True);(p/'corpus.sqlite').touch()
            old=root/'results/corpus-integrated-v3';(old/'manifest.json').write_text('{"records":16357}')
            self.assertEqual(active_corpus(root)['records'],16357)
            new=root/'results/corpus-arxiv-100k';(new/'bulk-manifest.json').write_text('{"records":115077}')
            self.assertEqual(active_corpus(root)['records'],115077)
    def test_influence_audit_must_match_snapshot(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'results/corpus-openalex-enriched-v3';p.mkdir(parents=True);(p/'corpus.sqlite').touch()
            (p/'bulk-manifest.json').write_text(json.dumps({'records':10,'corpus_sha256':'right'}))
            audit={'records':10,'source_sha256':'wrong','influence_fields':{'citation_count':{'known':3},'citation_normalized_percentile':{'known':2}}}
            (p/'metadata-audit.json').write_text(json.dumps(audit));self.assertIsNone(active_corpus(root)['influence_coverage'])
            audit['source_sha256']='right';(p/'metadata-audit.json').write_text(json.dumps(audit))
            self.assertEqual(active_corpus(root)['influence_coverage']['citation_count'],3)

    def test_missing_corpus_stays_unavailable(self):
        with tempfile.TemporaryDirectory() as d:self.assertIsNone(active_corpus(d))

    def test_grounded_snapshot_only_replaces_completed_predecessor_after_marker(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name in ('corpus-ml-grounded-v4','corpus-openalex-enriched-v3'):
                p=root/'results'/name;p.mkdir(parents=True);(p/'corpus.sqlite').touch()
            old=root/'results/corpus-openalex-enriched-v3'
            (old/'bulk-manifest.json').write_text('{"records":206917}')
            self.assertEqual(active_corpus(root)['snapshot'],'corpus-openalex-enriched-v3')
            new=root/'results/corpus-ml-grounded-v4'
            (new/'bulk-manifest.json').write_text('{"records":206920}')
            self.assertEqual(active_corpus(root)['snapshot'],'corpus-ml-grounded-v4')
