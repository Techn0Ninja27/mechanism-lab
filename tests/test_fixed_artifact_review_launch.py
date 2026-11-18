import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from generation.fixed_artifact_review_launch import launch_intent, record_completion

class ReviewLaunchTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.out=Path(self.tmp.name);self.slots=self.out/'prepared/reference-slots'
        for slot in ('a','b'):
            (self.slots/slot).mkdir(parents=True)
            (self.slots/slot/'input-00.json').write_text('{}')
            (self.slots/slot/'reviewer-instructions.md').write_text('fixture')
        self.patch=patch('generation.fixed_artifact_review_launch.campaign.export_reviews',return_value=self.slots)
        self.patch.start();self.addCleanup(self.patch.stop)

    def test_intent_precedes_output_and_completion_separate(self):
        intent=launch_intent(self.out,'a',0,'/root/testa')
        self.assertFalse((self.slots/'a/review-00.json').exists())
        (self.slots/'a/review-00.json').write_text('{}')
        done=record_completion(self.out,'a',0,'/root/testa','completed')
        self.assertLessEqual(intent['recorded_at_utc'],done['completed_at_utc'])
        self.assertEqual(intent,launch_intent(self.out,'a',0,'/root/testa'))
        self.assertEqual(done,record_completion(self.out,'a',0,'/root/testa','completed'))

    def test_existing_output_or_reused_handle_rejected(self):
        (self.slots/'a/review-00.json').write_text('{}')
        with self.assertRaises(ValueError):launch_intent(self.out,'a',0,'/root/testa')
        launch_intent(self.out,'b',0,'/root/testb')
        (self.slots/'a/review-00.json').unlink()
        with self.assertRaises(ValueError):launch_intent(self.out,'a',0,'/root/testb')

    def test_wrong_task_or_changed_status_rejected(self):
        launch_intent(self.out,'a',0,'/root/testa')
        with self.assertRaises(ValueError):record_completion(self.out,'a',0,'/root/wrong','completed')
        record_completion(self.out,'a',0,'/root/testa','failed')
        with self.assertRaises(ValueError):record_completion(self.out,'a',0,'/root/testa','completed')

if __name__=='__main__':unittest.main()
