import unittest
from scripts.build_arxiv_primary_packet import parse

class PrimaryPacketTests(unittest.TestCase):
    def body(self, extra=''):
        return ('''<meta name="citation_arxiv_id" content="2512.19206">
<meta name="citation_title" content="A &amp; B">
<meta name="citation_abstract" content="Exact primary\n abstract.">
<meta name="citation_date" content="2025/12/22">
<meta name="citation_author" content="One, Author">
''' + extra).encode()

    def test_primary_fields_and_normalization(self):
        row = parse(self.body(), '2512.19206v1', 'https://arxiv.org/abs/2512.19206v1')
        self.assertEqual(row['title'], 'A & B')
        self.assertEqual(row['abstract'], 'Exact primary abstract.')
        self.assertEqual(row['updated'], '2025-12-22')
        self.assertEqual(row['authors'], ['One, Author'])
        self.assertEqual(len(row['source_hash']), 64)

    def test_wrong_paper_revision_or_ambiguous_metadata_rejected(self):
        for body, pin in [(self.body(), '2501.00001v1'), (self.body(), '2512.19206v2'),
                          (self.body('<meta name="citation_abstract" content="Altered">'), '2512.19206v1'),
                          (b'<html>rate limited</html>', '2512.19206v1')]:
            with self.subTest(pin=pin, body=body), self.assertRaises(ValueError):
                parse(body, pin, 'https://arxiv.org/abs/'+pin)

if __name__ == '__main__': unittest.main()
