"""HTTP contract checks with isolated writable state; no model/API calls."""
import json
import base64
import zipfile
import io
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
import server


class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.original = (server.ROOT, server.RESULTS, server.JOBS_DIR, server.ADJUDICATIONS)
        server.ROOT = Path(cls.tmp.name)
        server.RESULTS = server.ROOT / 'results'
        fixture = server.RESULTS / 'proposals-v2'
        fixture.mkdir(parents=True)
        (fixture / 'selection.json').write_text(json.dumps({'candidates': [{'id': 'test-triple', 'cluster_ids': [1, 2, 3]}]}))
        server.get_workspace.cache_clear()
        server.JOBS_DIR = Path(cls.tmp.name)
        server.ADJUDICATIONS = Path(cls.tmp.name) / 'adjudications.json'
        server.JOBS.clear()
        cls.http = server.http.server.ThreadingHTTPServer(('127.0.0.1', 0), server.Handler)
        server.PORT = cls.http.server_port
        cls.base = f'http://127.0.0.1:{server.PORT}'
        threading.Thread(target=cls.http.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.http.shutdown(); cls.http.server_close()
        if server.get_workspace.cache_info().currsize: server.get_workspace().close()
        server.get_workspace.cache_clear()
        server.ROOT, server.RESULTS, server.JOBS_DIR, server.ADJUDICATIONS = cls.original
        cls.tmp.cleanup()

    def request(self, path, body=None, headers=None):
        h = {'Content-Type': 'application/json', **(headers or {})}
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode() if body is not None else None, headers=h)
        try:
            with urllib.request.urlopen(req) as response: return response.status, json.load(response)
        except urllib.error.HTTPError as response:
            with response: return response.code, json.load(response)

    def test_inventory_job_persisted(self):
        status, data = self.request('/api/jobs', {'action': 'inventory'})
        self.assertEqual(status, 202)
        self.assertEqual(data['status'], 'completed')
        self.assertEqual(json.loads((server.JOBS_DIR / (data['id'] + '.json')).read_text())['status'], 'completed')
        self.assertEqual(self.request('/api/jobs/' + data['id'])[1]['status'], 'completed')

    def test_cross_origin_cannot_start_job(self):
        self.assertEqual(self.request('/api/jobs', {'action': 'inventory'}, {'Origin': 'https://elsewhere.example'})[0], 403)

    def test_arbitrary_commands_rejected(self):
        self.assertEqual(self.request('/api/jobs', {'action': 'shell', 'command': 'touch should-not-exist'})[0], 400)

    def test_bad_pagination_returns_client_error(self):
        self.assertEqual(self.request('/api/candidates?limit=bad')[0], 400)

    def test_run_traversal_rejected(self):
        self.assertEqual(self.request('/api/candidates?run=../../..')[0], 400)

    def test_paid_decode_blocked_without_process(self):
        status, data = self.request('/api/jobs', {'action': 'decode_paid'})
        self.assertEqual(status, 409)
        self.assertNotIn(data['id'], server.PROCESSES)

    def test_adjudication_persists_without_changing_candidate(self):
        before = self.request('/api/candidates')[1]['items'][0]
        status, record = self.request('/api/adjudications', {'candidate_id': before['id'], 'decision': 'hold', 'notes': 'HTTP test'})
        self.assertEqual(status, 201)
        self.assertEqual(self.request('/api/adjudications')[1]['items'][-1]['id'], record['id'])
        self.assertEqual(self.request('/api/candidates')[1]['items'][0], before)

    def test_invalid_job_options_fail_visibly(self):
        status, data = self.request('/api/jobs', {'action': 'sample_offline', 'options': {'n_gen': 'bad'}})
        self.assertEqual(status, 400)
        self.assertEqual(data['status'], 'failed')

    def test_project_import_read_and_export_roundtrip(self):
        status, p = self.request('/api/projects', {'name': 'HTTP document project'})
        self.assertEqual(status, 201)
        route = '/api/projects/' + p['id']
        status, imported = self.request(route + '/documents', {'files': [{'name': 'papers/a.md', 'content_base64': base64.b64encode(b'# Source\nactual bytes').decode()}]})
        self.assertEqual(status, 201)
        doc = imported['items'][0]
        self.assertEqual(self.request(route + '/documents/' + doc['id'])[1]['content'], '# Source\nactual bytes')
        status, exported = self.request(route + '/export', {})
        self.assertEqual(status, 201)
        with urllib.request.urlopen(self.base + exported['download_url']) as response:
            with zipfile.ZipFile(io.BytesIO(response.read())) as z:
                self.assertEqual(z.read('sources/papers/a.md'), b'# Source\nactual bytes')

if __name__ == '__main__': unittest.main()
