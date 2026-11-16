import hashlib
import pathlib
import tempfile
import unittest
from unittest.mock import patch
from scripts.fetch_method_sources import fetch

class MethodSourceTests(unittest.TestCase):
    def manifest(self, local='results/sources'):
        return {'snapshots':[{'repository':'https://github.com/test/repo','commit':'a'*40,
                'local_directory':local,'files':{'method.py':hashlib.sha256(b'example').hexdigest()}}]}
    def test_existing_hash_verified_without_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp);p=root/'results/sources/method.py';p.parent.mkdir(parents=True);p.write_bytes(b'example')
            with patch('scripts.fetch_method_sources.urllib.request.urlopen') as network:
                self.assertEqual(fetch(self.manifest(),root)[0]['status'],'verified_cache');network.assert_not_called()
                p.write_bytes(b'changed')
                with self.assertRaises(ValueError):fetch(self.manifest(),root)
    def test_path_escape_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch('scripts.fetch_method_sources.urllib.request.urlopen') as network:
                with self.assertRaises(ValueError):fetch(self.manifest('../outside'),pathlib.Path(tmp))
                network.assert_not_called()
