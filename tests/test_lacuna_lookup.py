import unittest
from scripts.lacuna_lookup import search_url,context_url
class LacunaTests(unittest.TestCase):
 def test_bounded_search(self):
  with self.assertRaises(ValueError):search_url('x',limit=11)
 def test_query_is_encoded(self):self.assertIn('q=a%26type%3Dhypothesis',search_url('a&type=hypothesis'))
 def test_context_rejects_path(self):
  with self.assertRaises(ValueError):context_url('paper','../secrets')
 def test_context_known_type(self):self.assertEqual(context_url('paper','art_a'),'https://lacuna.tiptreesystems.com/api/v1/context/paper/art_a?view=compact')
