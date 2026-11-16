import unittest
from scripts.fetch_known_arxiv import AbstractPage
class AbstractParserTests(unittest.TestCase):
    def test_primary_abstract_excludes_surrounding_page(self):
        p=AbstractPage();p.feed('<meta name="citation_title" content="A &amp; B"><meta name="citation_author" content="Author"><p>Outside</p><blockquote class="abstract mathjax"><span>Abstract:</span> Mechanism <em>inside</em>.</blockquote><p>Not abstract</p>')
        self.assertEqual(p.meta['citation_title'],['A & B'])
        self.assertIn('inside',''.join(p.abstract))
        self.assertNotIn('Outside',''.join(p.abstract))
        self.assertNotIn('Not abstract',''.join(p.abstract))
