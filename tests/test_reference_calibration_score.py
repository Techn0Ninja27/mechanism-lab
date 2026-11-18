import json
import shutil
import tempfile
import unittest
from pathlib import Path
from scripts.score_reference_calibration import score

ROOT = Path(__file__).resolve().parents[1]

class ReferenceCalibrationScoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'v5'
        self.source = Path(self.tmp.name) / 'v4'
        shutil.copytree(ROOT/'results/recordwise-calibration-v5', self.root)
        self.source.mkdir()
        for name in ['manifest.json','answer-key.json'] + [f'input-{i}.json' for i in range(8)]:
            shutil.copyfile(ROOT/'results/recordwise-calibration-v4'/name,self.source/name)

    def test_actual_review_set_matches(self):
        result = score(self.root, self.source)
        self.assertTrue(result['calibration_passed'])
        self.assertEqual(result['matches_key_count'],16)
        self.assertFalse(result['production_authorized'])

    def test_blanket_supported_does_not_pass(self):
        for path in (self.root/'reviews').glob('*.json'):
            raw=json.loads(path.read_text())
            for field in raw['fields']:
                field['verdict']='supported'
                field['evidence_ids']=['E0000']
                field.pop('claim_assessments',None)
            path.write_text(json.dumps(raw))
        self.assertFalse(score(self.root,self.source)['calibration_passed'])

    def test_changed_source_or_missing_review_fails(self):
        path=self.source/'input-0.json'
        path.write_text(path.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'input changed'):score(self.root,self.source)
        shutil.copyfile(ROOT/'results/recordwise-calibration-v4/input-0.json',path)
        (self.root/'reviews/review-0-a.json').unlink()
        with self.assertRaises(FileNotFoundError):score(self.root,self.source)

if __name__=='__main__': unittest.main()
