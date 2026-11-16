import json,sqlite3,tempfile,unittest
from pathlib import Path
from scripts.audit_corpus_preservation import audit,CORE
class PreservationTests(unittest.TestCase):
    def build(self,root):
        root.mkdir();p=root/'corpus.sqlite'
        with sqlite3.connect(p) as db:
            db.execute('CREATE TABLE works (source_id TEXT PRIMARY KEY,'+','.join(k+' TEXT' for k in CORE)+',bulk_metadata_json TEXT)')
            db.execute('INSERT INTO works VALUES ('+','.join('?' for _ in range(len(CORE)+2))+')',['id']+['value']*len(CORE)+[json.dumps({'citation_count':4})])
            db.executescript("CREATE TABLE aliases(alias TEXT,canonical_id TEXT); INSERT INTO aliases VALUES ('alias','value'); CREATE TABLE works_fts(source_id TEXT,title TEXT,abstract TEXT); INSERT INTO works_fts VALUES ('id','value','value');")
        (root/'openai_sources.jsonl').write_text('{"separate":true}\n');return p
    def test_detects_changed_content_and_index_even_when_counts_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=self.build(root/'old');new=self.build(root/'new')
            self.assertTrue(audit(old,new)['passed'])
            with sqlite3.connect(new) as db:db.execute("UPDATE works_fts SET abstract='wrong'")
            self.assertFalse(audit(old,new)['passed'])
            with sqlite3.connect(new) as db:
                db.execute("UPDATE works_fts SET abstract='value'")
                db.execute("UPDATE works SET bulk_metadata_json='{}'")
            self.assertFalse(audit(old,new)['passed'])
