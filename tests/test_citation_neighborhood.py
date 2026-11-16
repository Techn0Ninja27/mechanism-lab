import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from generation.citation_neighborhood import abstract, snapshot, work_id


class Response:
    def __init__(self, value): self.body = json.dumps(value).encode()
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def read(self, size): return self.body[:size]


def row(i, refs=(), related=()):
    return {'id': f'https://openalex.org/W{i}', 'title': str(i),
            'referenced_works': [f'https://openalex.org/W{x}' for x in refs],
            'related_works': [f'https://openalex.org/W{x}' for x in related]}


class CitationTests(unittest.TestCase):
    def test_edges_missing_metadata_and_offline_replay(self):
        opener = Mock(side_effect=[Response(row(1, [2, 4], [3])),
                    Response({'results': [row(2), row(3)]}),
                    Response({'results': [row(5, [1])], 'meta': {'count': 8}})])
        with tempfile.TemporaryDirectory() as directory:
            result = snapshot('W1', directory, allow_network=True, opener=opener)
            self.assertEqual(opener.call_count, 3)
            self.assertEqual(result['coverage']['unresolved_selected_ids'], ['W4'])
            self.assertIn({'source': 'W5', 'target': 'W1', 'type': 'cites'}, result['edges'])
            self.assertIn({'source': 'W1', 'target': 'W3', 'type': 'provider_related'}, result['edges'])
            self.assertFalse(result['novelty_established'])
            self.assertEqual(snapshot('W1', directory), result)
            path = next(p for p in Path(directory).glob('*.json') if '.receipt.' not in p.name)
            path.write_text('{}')
            with self.assertRaises(ValueError): snapshot('W1', directory)

    def test_network_opt_in_and_identity_validation(self):
        for value in ('https://evil.test/W1', 'W1?x', None):
            with self.assertRaises(ValueError): work_id(value)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError): snapshot('W1', directory)
            with self.assertRaises(ValueError): snapshot('W1', directory, limit=True)

    def test_wrong_incoming_edge_rejected(self):
        opener = Mock(side_effect=[Response(row(1)), Response({'results': [row(2)], 'meta': {'count': 1}})])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'lacks reported citation'):
                snapshot('W1', directory, allow_network=True, opener=opener)

    def test_abstract_restores_order_and_rejects_collisions(self):
        self.assertEqual(abstract({'b': [1], 'a': [0, 2]}), 'a b a')
        with self.assertRaises(ValueError): abstract({'a': [0], 'b': [0]})


if __name__ == '__main__': unittest.main()
