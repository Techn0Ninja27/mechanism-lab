import gzip,hashlib,json,tempfile,unittest
from pathlib import Path
from scripts.verify_openalex_archive import verify
from scripts.fetch_openalex_corpus import normalize
class OpenAlexArchiveTests(unittest.TestCase):
    def test_normalized_content_is_verified_not_just_source_hash(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'responses').mkdir()
            row={'id':'https://openalex.org/W1','title':'Original','abstract_inverted_index':{'Evidence':[0]}}
            body=json.dumps({'results':[row]}).encode()
            with gzip.open(root/'responses/000001.json.gz','wb') as f:f.write(body)
            receipt={'request':1,'status':200,'error':None,'slot':'machine_learning:T1:0','timestamp':'2026-11-16T00:00:00Z','bytes':len(body),'response_sha256':hashlib.sha256(body).hexdigest()}
            (root/'receipts.jsonl').write_text(json.dumps(receipt)+'\n')
            record=normalize(row,{'id':receipt['slot']},'2026-11-16')
            (root/'records.jsonl').write_text(json.dumps(record)+'\n')
            self.assertEqual(verify(root)['collected_records_verified'],1)
            record['title']='Changed while preserving raw hash';(root/'records.jsonl').write_text(json.dumps(record)+'\n')
            with self.assertRaises(ValueError):verify(root)
