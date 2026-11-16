import json,tempfile,unittest
from pathlib import Path
from generation.corpus_registry import active_corpus
class CorpusRegistryTests(unittest.TestCase):
    def test_incomplete_snapshot_cannot_replace_completed_corpus(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name in ('corpus-arxiv-100k','corpus-integrated-v3'):
                p=root/'results'/name;p.mkdir(parents=True);(p/'corpus.sqlite').touch()
            old=root/'results/corpus-integrated-v3';(old/'manifest.json').write_text('{"records":16357}')
            self.assertEqual(active_corpus(root)['records'],16357)
            new=root/'results/corpus-arxiv-100k';(new/'bulk-manifest.json').write_text('{"records":115077}')
            self.assertEqual(active_corpus(root)['records'],115077)
    def test_missing_corpus_stays_unavailable(self):
        with tempfile.TemporaryDirectory() as d:self.assertIsNone(active_corpus(d))
