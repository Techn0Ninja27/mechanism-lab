import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import generate_causal_campaign2 as c
from scripts import run_campaign2_preparation as life
from tests.test_campaign2_generation import FakeClient, PACK
from generation.api_budget import BudgetLedger


class ReferenceLifecycleTests(unittest.TestCase):
    def setUp(self):
        live=Path(c.config()['shared_budget_path']);before=live.read_bytes() if live.exists() else None
        self.addCleanup(lambda:self.assertEqual(live.read_bytes() if live.exists() else None,before))
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name) / 'campaign'
        c.plan(PACK, self.out)
        self.patches = [
            patch.object(life, 'CANONICAL_OUTPUT', self.out),
            patch.object(life, '_anchor', return_value=BudgetLedger(Path(self.tmp.name) / 'ledger.json', 30)),
            patch.object(c, 'require_calibration_ready', return_value=None),
            patch.object(c, 'Client', FakeClient),
        ]
        for p in self.patches:
            p.start(); self.addCleanup(p.stop)
        FakeClient.calls = []; FakeClient.fail = False

    def _raw_reviews(self, kind='atoms'):
        for slot in ('a', 'b'):
            root = self.out / 'prep-draws' / kind / '00' / 'reference-slots' / slot
            for packet_path in sorted(root.glob('input-*.json')):
                packet = json.loads(packet_path.read_text())
                first = packet['evidence'][0]['evidence_id']
                items = []
                for field in packet['fields']:
                    item = {'field_id': field['field_id'], 'verdict': 'supported',
                            'reason': f'source-specific reason slot {slot}', 'evidence_ids': []}
                    if field['is_unknown']:
                        item['absence_reason'] = 'the cited source does not specify this'
                    else:
                        item['evidence_ids'] = [first]
                    items.append(item)
                (root / f'review-{packet_path.stem.removeprefix("input-")}.json').write_text(
                    json.dumps({'fields': items}) + '\n')

    def test_reference_packets_and_dual_review_lock(self):
        life.execute_draw(self.out, 'atoms', 0)
        life.export_reference_review_sets(self.out, 'atoms', 0)
        self._raw_reviews('atoms')
        life.execute_draw(self.out, 'distillations', 0)
        life.export_reference_review_sets(self.out, 'distillations', 0)
        self._raw_reviews('distillations')
        selected = life.record_review(self.out, 'atoms', 0)
        self.assertEqual(selected['status'], 'locked')
        self.assertEqual(selected['draw'], 0)
        self.assertTrue((self.out / 'prep-draws/atoms/00/decision.json').is_file())
        self.assertEqual(json.loads((self.out / 'prep-draws/atoms/00/decision.json').read_text())['review_schema'],
                         'reference-dual-draw-review.v1')
        self.assertEqual(life.record_review(self.out, 'distillations', 0)['status'], 'locked')
        self.assertEqual(life.select_pair(self.out)['status'], 'locked')
        c._require_chunk_gate(self.out)
        generated=c.generate(self.out,True)
        self.assertEqual(generated['candidates'],124)
        self.assertEqual(json.loads((self.out/'inventory.json').read_text())['status'],'generated')
        self.assertEqual(len(FakeClient.calls),26)
        path=self.out/'prep-draws/atoms/00/reference-slots/a/review-00.json'
        path.write_text(path.read_text()+' ')
        with self.assertRaises(ValueError): c._require_chunk_gate(self.out)

    def test_copied_raw_review_set_cannot_lock(self):
        life.execute_draw(self.out,'atoms',0)
        life.export_reference_review_sets(self.out,'atoms',0)
        self._raw_reviews()
        root=self.out/'prep-draws/atoms/00/reference-slots'
        for path in (root/'a').glob('review-*.json'):
            (root/'b'/path.name).write_bytes(path.read_bytes())
        with self.assertRaisesRegex(ValueError,'duplicate raw review sets'):
            life.record_review(self.out,'atoms',0)
        self.assertFalse((self.out/'selected/atoms.selection.json').exists())

    def test_legacy_review_cannot_lock(self):
        life.execute_draw(self.out,'atoms',0)
        # Old artifacts must not be sufficient to select a representation.
        draw=self.out/'prep-draws/atoms/00'
        for slot in 'ab':
            (draw/f'review-{slot}.json').write_text(json.dumps({'schema':'chunk-grounded-draw-review.v1','records':[]}))
        with self.assertRaisesRegex(ValueError,'legacy review format is disabled'):
            life.record_review(self.out,'atoms',0)
        self.assertFalse((self.out/'selected/atoms.selection.json').exists())

    def test_missing_raw_review_fails_closed(self):
        life.execute_draw(self.out, 'atoms', 0)
        life.export_reference_review_sets(self.out, 'atoms', 0)
        self._raw_reviews()
        path = self.out / 'prep-draws/atoms/00/reference-slots/b/review-03.json'
        path.unlink()
        with self.assertRaises(ValueError):
            life.record_review(self.out, 'atoms', 0)

    def test_failed_supported_subclaim_fails_closed(self):
        life.execute_draw(self.out, 'atoms', 0)
        life.export_reference_review_sets(self.out, 'atoms', 0)
        self._raw_reviews('atoms')
        path = self.out / 'prep-draws/atoms/00/reference-slots/a/review-00.json'
        data = json.loads(path.read_text())
        field = next(x for x in data['fields'] if x['evidence_ids'])
        field['claim_assessments'] = [{'verdict': 'unsupported', 'reason': 'missing condition', 'evidence_ids': []}]
        path.write_text(json.dumps(data) + '\n')
        with self.assertRaises(ValueError):
            life.record_review(self.out, 'atoms', 0)


if __name__ == '__main__':
    unittest.main()
