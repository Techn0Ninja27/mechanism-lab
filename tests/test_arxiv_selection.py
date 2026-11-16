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
