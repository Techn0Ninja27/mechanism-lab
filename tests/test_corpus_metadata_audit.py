import json,tempfile,unittest
from pathlib import Path
from scripts.audit_corpus_metadata import audit
class CorpusMetadataAuditTests(unittest.TestCase):
    def test_missing_zero_and_invalid_are_separate(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'rows.jsonl';p.write_text('\n'.join(json.dumps(x) for x in [
                {'citation_count':None},{'citation_count':0,'citation_normalized_percentile':{'value':.5}},
                {'citation_count':True,'field_weighted_citation_impact':-1,'citation_normalized_percentile':{'value':2}}])+'\n')
            r=audit(p);c=r['influence_fields']['citation_count']
            self.assertEqual((c['known'],c['missing'],c['invalid'],c['zero']),(1,1,1,1))
            self.assertEqual(r['influence_fields']['citation_normalized_percentile']['known'],1)
            self.assertEqual(r['influence_rows_without_explicit_provenance'],1)
    def test_explicit_provenance_is_recorded_without_inventing_scores(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'rows.jsonl';p.write_text(json.dumps({'citation_count':7,'influence_provenance':[{'provider':'openalex','as_of':'2026-11-16'}]})+'\n')
            r=audit(p);self.assertEqual(r['influence_rows_without_explicit_provenance'],0)
            self.assertNotIn('usefulness_score',r)
