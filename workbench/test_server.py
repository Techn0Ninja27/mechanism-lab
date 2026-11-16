"""HTTP contract checks with isolated writable state; no model/API calls."""
import json
import base64
import zipfile
import io
import subprocess
from unittest import mock
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

    def test_study_registry_reports_verdict_and_qualification(self):
        status, data = self.request('/api/studies')
        self.assertEqual(status, 200)
        self.assertGreaterEqual(data['total'], 27)
        no_gain = next(item for item in data['items'] if item['raw_verdict'] == 'no_gain')
        self.assertIn(no_gain['audit_qualification'], {'clean_negative', 'pre_final_audit_reviewed', 'unknown'})
        self.assertFalse(no_gain['novelty_established'])

    def test_study_artifact_and_traversal_are_bounded(self):
        status, data = self.request('/api/studies/abstracts-0')
        self.assertEqual(status, 200)
        self.assertIn('verdict.json', data['study']['artifacts'])
        self.assertEqual(self.request('/api/studies/abstracts-0/verdict.json')[0], 200)
        self.assertEqual(self.request('/api/studies/abstracts-0/../../README.md')[0], 400)
        self.assertEqual(self.request('/api/studies/does-not-exist')[0], 404)

    def test_generated_research_is_machine_review_only_and_bounded(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(server, 'ENGINE', Path(tmp)):
            run_dir=Path(tmp)/'results/improvement-phase-2026-11-16/grounded-01'
            run_dir.mkdir(parents=True)
            (run_dir/'result.json').write_text(json.dumps({'status':'reviewed','proposal':{'title':'Test','evidence':[]},'review':{'decision':'reject'}}))
            status, data = self.request('/api/generated-research')
            self.assertEqual(status, 200)
            run = next(r for r in data['items'] if r['id']=='grounded-01')
            self.assertTrue(run['machine_review_only'])
            self.assertFalse(run['empirical_result'])
            self.assertNotIn('api_key', json.dumps(run).lower())
            self.assertIn(run['operator_evaluation']['status'], {'unavailable', 'unverified'})

    def test_project_grounded_result_is_discoverable(self):
        status, project = self.request('/api/projects', {'name': 'Completed grounded run'})
        self.assertEqual(status, 201)
        output = Path(project['path']) / 'runs' / 'sourcepack-0123456789ab' / 'execution'
        output.mkdir(parents=True)
        (output / 'result.json').write_text(json.dumps({'status': 'reviewed', 'pack_hash': 'abc', 'novelty_established': False,
            'proposal': {'title': 'Test', 'hypothesis': 'H', 'evidence': []}, 'review': {'decision': 'reject', 'failure_type': 'insufficient_evidence'}}))
        status, data = self.request('/api/generated-research')
        self.assertEqual(status, 200)
        self.assertTrue(any(x.get('project_id') == project['id'] for x in data['items']))

    def test_grounded_launch_reaches_fixed_subprocess_and_duplicate_is_refused(self):
        status, project = self.request('/api/projects', {'name': 'Launch pack'})
        self.assertEqual(status, 201)
        route = '/api/projects/' + project['id']
        status, imported = self.request(route + '/documents', {'files': [{'name': 'a.md', 'content_base64': base64.b64encode(b'# Evidence').decode()}]})
        self.assertEqual(status, 201)
        status, pack = self.request(route + '/source-pack', {'document_ids': [imported['items'][0]['id']], 'question': 'Test?'})
        self.assertEqual(status, 201)
        prior = {'id': 'prior', 'action': 'grounded_execute', 'status': 'completed', 'options': {'project_id': project['id'], 'sourcepack_run': pack['run_id']}}
        current = {'id': 'current', 'action': 'grounded_execute', 'status': 'queued', 'options': {'project_id': project['id'], 'sourcepack_run': pack['run_id']}}
        class FakeProcess:
            pid = 12345
            def poll(self): return 0
            def wait(self): return 0
        server.JOBS['prior'] = prior; server.JOBS['current'] = current
        with self.assertRaises(ValueError): server.launch_job('current', 'grounded_execute', current['options'])
        del server.JOBS['prior']
        with mock.patch.object(server.subprocess, 'Popen', return_value=FakeProcess()) as popen:
            server.launch_job('current', 'grounded_execute', current['options'])
            command = popen.call_args.args[0]
            self.assertEqual(command[-1], '--execute')
            self.assertIn(str(Path(project['path']) / 'runs' / pack['run_id'] / 'source-pack.json'), command)

    def test_run_traversal_rejected(self):
        self.assertEqual(self.request('/api/candidates?run=../../..')[0], 400)

    def test_paid_decode_blocked_without_process(self):
        status, data = self.request('/api/jobs', {'action': 'decode_paid'})
        self.assertEqual(status, 409)
        self.assertNotIn(data['id'], server.PROCESSES)

    def test_grounded_execute_rejects_untrusted_pack_reference_without_launching(self):
        status, data = self.request('/api/jobs', {'action': 'grounded_execute', 'options': {'project_id': 'missing', 'sourcepack_run': '../../etc'}})
        self.assertEqual(status, 400)
        self.assertEqual(data['status'], 'failed')
        self.assertNotIn('subprocess', json.dumps(data).lower())

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

    def test_operator_links_require_matching_artifact_hash(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(server, 'ENGINE', Path(tmp)):
            root=Path(tmp);run=root/'results/improvement-phase-2026-11-16/grounded-01'
            run.mkdir(parents=True);(run/'result.json').write_text('{"status":"reviewed","proposal":{},"review":{}}')
            op=root/'operator-studies/test';op.mkdir(parents=True)
            (op/'results.json').write_text('{}');(op/'README.md').write_text('Operator only')
            report=root/'reports/improvement-2026-11-16';report.mkdir(parents=True)
            links={'grounded-01':{'status':'rejected_as_written','summary':'Tested','result_path':'operator-studies/test/results.json','report_path':'operator-studies/test/README.md','result_sha256':hashlib.sha256(b'{}').hexdigest()}}
            (report/'evaluation-links.json').write_text(json.dumps(links))
            result=next(x for x in server.grounded_rows() if x['id']=='grounded-01')['operator_evaluation']
            self.assertTrue(result['verified'])
            (op/'results.json').write_text('{"changed":true}')
            result=next(x for x in server.grounded_rows() if x['id']=='grounded-01')['operator_evaluation']
            self.assertFalse(result['verified']);self.assertEqual(result['status'],'unverified')

    def test_source_pack_requires_explicit_documents_and_is_plan_only(self):
        status, project = self.request('/api/projects', {'name': 'Source pack project'})
        self.assertEqual(status, 201)
        route = '/api/projects/' + project['id']
        status, imported = self.request(route + '/documents', {'files': [{'name': 'papers/a.md', 'content_base64': base64.b64encode(b'# Source\nactual bytes').decode()}]})
        self.assertEqual(status, 201)
        doc_id = imported['items'][0]['id']
        status, plan = self.request(route + '/source-pack', {'document_ids': [doc_id], 'question': 'What is testable here?'})
        self.assertEqual(status, 201)
        self.assertFalse(plan['execution_enabled'])
        self.assertTrue((Path(server.ROOT) / plan['source_pack']).is_file())
        plan_file = Path(project['path']) / plan['plan'].split('/runs/', 1)[-1] if '/runs/' in plan['plan'] else Path(server.ROOT) / plan['plan']
        if not plan_file.is_file():
            plan_file = Path(server.ROOT) / plan['plan']
        command = json.loads(plan_file.read_text())['execution']['command']
        completed = subprocess.run(command, cwd=server.ENGINE, capture_output=True, text=True, check=True)
        self.assertIn('planned', completed.stdout)
        self.assertEqual(self.request(route + '/source-pack', {'document_ids': [], 'question': 'x'})[0], 400)

if __name__ == '__main__': unittest.main()
