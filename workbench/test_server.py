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

    def test_new_study_reports_are_accessible_in_their_own_directories(self):
        for sid, artifact in (("utility-ranker", "final.json"), ("integer-composition", "results.json")):
            status, data = self.request('/api/studies/' + sid + '/' + artifact)
            self.assertEqual(status, 200)
            self.assertIn('summary', json.loads(data['content']))
            self.assertEqual(self.request('/api/studies/' + sid + '/../../README.md')[0], 400)
        status, data = self.request('/api/studies/utility-ranker')
        self.assertEqual(data['study']['raw_verdict'], 'not_supported')
        self.assertEqual(data['study']['audit_qualification'], 'frozen_sources_and_validation_match')

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
            self.assertEqual(run['proposal']['mechanism_check']['status'], 'not_recorded')
            self.assertEqual(run['review']['mechanism_check']['status'], 'not_recorded')
            self.assertNotIn('api_key', json.dumps(run).lower())
            self.assertIn(run['operator_evaluation']['status'], {'unavailable', 'unverified'})
            self.assertEqual(run['claim_assessment'], {'status': 'not_recorded'})

    def test_claim_assessment_is_validated_and_never_promotes_novelty(self):
        valid = {
            'status': 'no_match_in_examined_evidence',
            'claims': [{
                'claim_id': 'c1', 'coverage': 'not_found', 'checked_chunk_ids': ['chunk-1'],
                'evidence': [{'chunk_id': 'chunk-1', 'quote': 'No matching statement.', 'supports': 'does not establish claim'}],
                'reason': 'The examined chunks do not state this claim.',
                'claim': {'id': 'c1', 'text': 'A bounded claim', 'role': 'central', 'kind': 'problem'},
            }],
            'novelty_estimate': None, 'novelty_established': False, 'train_eligible': False,
            'scope': 'Only the supplied evidence pack was examined.',
        }
        self.assertEqual(server.claim_assessment(None), {'status': 'not_recorded'})
        self.assertEqual(server.claim_assessment(valid)['novelty_established'], False)
        self.assertEqual(server.claim_assessment(valid)['train_eligible'], False)
        for malformed in [valid | {'scope': 4}, valid | {'novelty_estimate': 'low'}, valid | {'status': []}, valid | {'claims': []},
                          valid | {'claims': [valid['claims'][0]] * 2},
                          valid | {'claims': [{**valid['claims'][0], 'coverage': []}]},
                          valid | {'claims': [valid['claims'][0]] * 7},
                          {**valid, 'claims': [{**valid['claims'][0], 'claim': {'id': 'c1', 'text': 'x', 'role': 'central', 'kind': 'bad'}}]}]:
            self.assertEqual(server.claim_assessment(malformed), {'status': 'invalid'})
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(server, 'ENGINE', Path(tmp)):
            run_dir = Path(tmp) / 'results/improvement-phase-2026-11-16/grounded-claim'
            run_dir.mkdir(parents=True)
            (run_dir / 'result.json').write_text(json.dumps({'status': 'reviewed', 'novelty_established': True,
                'claim_assessment': valid, 'proposal': {}, 'review': {}}))
            row = next(x for x in server.grounded_rows() if x['id'] == 'grounded-claim')
            self.assertEqual(row['claim_assessment']['status'], valid['status'])
            self.assertFalse(row['claim_assessment']['novelty_established'])
            self.assertTrue(row['novelty_established'])

    def test_citation_evidence_is_bounded_and_not_a_novelty_signal(self):
        valid = {
            'status': 'included', 'snapshot_sha256': 'a' * 64, 'seed': 'W4409983225',
            'selected_ids': ['W4409983225', 'W7160826828'],
            'included_ids': ['W7160826828'], 'omitted_ids': ['W4409983225'],
            'included_chars': 1200, 'scope': 'Provider abstracts selected from a bounded current snapshot.'}
        self.assertEqual(server.citation_evidence(None), {'status': 'not_recorded'})
        self.assertEqual(server.citation_evidence(valid), valid)
        for malformed in [
            valid | {'snapshot_sha256': 'x'},
            valid | {'seed': 'arxiv:2608.06441'},
            valid | {'selected_ids': ['W1'] * 2},
            valid | {'included_ids': ['W999']},
            valid | {'included_chars': True},
            valid | {'included_chars': 16001},
            valid | {'scope': ''},
            valid | {'status': []},
            valid | {'selected_ids': ['W1'] * 9},
            {k: v for k, v in valid.items() if k != 'scope'},
        ]:
            self.assertEqual(server.citation_evidence(malformed), {'status': 'invalid'})
        self.assertEqual(server.citation_evidence({'status': 'not_requested'}), {'status': 'not_requested'})
        empty = valid | {'included_ids': [], 'omitted_ids': valid['selected_ids'], 'included_chars': 0}
        self.assertEqual(server.citation_evidence(empty), empty)
        self.assertEqual(server.citation_evidence(valid | {'omitted_ids': ['W999']}), {'status': 'invalid'})
        self.assertEqual(server.citation_evidence(valid | {'omitted_ids': valid['included_ids']}), {'status': 'invalid'})

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(server, 'ENGINE', Path(tmp)):
            run_dir = Path(tmp) / 'results/improvement-phase-2026-11-16/grounded-citation'
            run_dir.mkdir(parents=True)
            (run_dir / 'result.json').write_text(json.dumps({'status': 'reviewed', 'proposal': {}, 'review': {},
                'citation_evidence': valid, 'novelty_established': True}))
            row = next(x for x in server.grounded_rows() if x['id'] == 'grounded-citation')
            self.assertEqual(row['citation_evidence'], valid)
            self.assertTrue(row['novelty_established'])

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
            self.assertIn('--execute', command)
            self.assertIn(str(Path(project['path']) / 'runs' / pack['run_id'] / 'source-pack.json'), command)

    def test_run_traversal_rejected(self):
        self.assertEqual(self.request('/api/candidates?run=../../..')[0], 400)

    def test_abstract_notes_remain_separate_and_unpromoted(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(server,'ENGINE',Path(tmp)):
            path=Path(tmp)/'reports/operational-notes-v2/notes.json';path.parent.mkdir(parents=True)
            note={'source_id':'arxiv:test','operation':'Estimate an object','source_url':'https://example.test',
                  'stated_problems':[{'name':'Source-scoped cost'}],'inputs':[],'outputs':[],
                  'unknowns':['Missing interface'],'problem_review':{'status':'flag','reason':'Scope uncertain'},
                  'ready_for_composition':False,'train_eligible':False,'novelty_established':False}
            path.write_text(json.dumps({'notes':[note,{**note,'train_eligible':True}]}))
            status,data=self.request('/api/operator-library')
            self.assertEqual(status,200);self.assertEqual(data['operators'],[])
            self.assertEqual(len(data['operational_notes']),1)
            self.assertEqual(data['operational_notes'][0]['problem_review_status'],'flag')
            self.assertFalse(data['operational_notes'][0]['ready_for_composition'])
            path.write_text('x'*200001);self.assertEqual(server.operational_notes(),[])

    def test_operator_library_missing_catalog_is_empty(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(server, 'ENGINE', Path(tmp)):
            status, data = self.request('/api/operator-library')
            self.assertEqual(status, 200)
            self.assertEqual(data['operators'], [])
            self.assertEqual(data['chains'], [])

    def test_operator_library_is_data_only_and_bounds_report_path(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(server, 'ENGINE', Path(tmp)):
            catalog = Path(tmp) / 'reports/operator-composition/catalog.json'
            catalog.parent.mkdir(parents=True)
            catalog.write_text(json.dumps({
                'schema_version': 1, 'scope': 'test',
                'operators': [{'id': 'safe-op', 'content_id': 'c1', 'input_type': 'x', 'output_type': 'y',
                               'source_url': 'javascript:alert(1)', 'assumptions': ['none'], 'contract_status': 'hand-curated'},
                              {'id': 'evil-op', 'source_url': 'https://example.test/source'}],
                'chains': [{'id': 'chain-1', 'operator_ids': ['safe-op'], 'input_type': 'x', 'output_type': 'y', 'molecule_id': 'm1'}],
                'study': {'report_path': '../escape.md', 'status': 'draft'},
            }))
            status, data = self.request('/api/operator-library')
            self.assertEqual(status, 200)
            self.assertEqual(data['operators'][0]['source_url'], 'javascript:alert(1)')
            self.assertEqual(data['chains'][0]['operator_ids'], ['safe-op'])
            self.assertIsNone(data['study']['report_path'])
            self.assertNotIn('import', json.dumps(data).lower())

    def test_operator_check_roundtrip_persists_minimal_artifact(self):
        status, data = self.request('/api/operator-check', {'operator_ids': ['for', 'lzma'], 'values': [1, 2, 3]})
        self.assertEqual(status, 200)
        self.assertTrue(data['result']['round_trip'])
        self.assertEqual(data['result']['input_count'], 3)
        self.assertEqual(data['result']['scope'].split('.')[0], 'Single supplied-input roundtrip check only')
        artifact = server.ROOT / 'results/operator-checks' / f"{data['id']}.json"
        self.assertTrue(artifact.is_file())
        saved = json.loads(artifact.read_text())
        self.assertNotIn('values', saved)
        self.assertEqual(saved['input_sha256'], data['input_sha256'])

    def test_operator_check_rejects_bad_types_and_unknown_ids(self):
        self.assertEqual(self.request('/api/operator-check', {'operator_ids': ['for'], 'values': [True]})[0], 400)
        self.assertEqual(self.request('/api/operator-check', {'operator_ids': ['no-such-op'], 'values': [1]})[0], 400)
        self.assertEqual(self.request('/api/operator-check', {'operator_ids': ['for'], 'values': [1.5]})[0], 400)
        self.assertEqual(self.request('/api/operator-check', {'operator_ids': ['for'], 'values': [1] * 4097})[0], 400)

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

    def test_mechanism_summary_preserves_unknown_and_rejects_invalid_claims(self):
        self.assertEqual(server.mechanism_summary(None), {'status':'not_recorded'})
        check={k:'Explicitly unresolved' for k in ('assumption_scope','specialized_operation','remaining_effect','simplest_baseline','distinguishing_test')}
        check['status']='unresolved'
        self.assertEqual(server.mechanism_summary(check),check)
        for bad in ({**check,'status':'verified'}, {**check,'status':[]}, {**check,'remaining_effect':[]}, {**check,'simplest_baseline':'x'*3001}):
            self.assertEqual(server.mechanism_summary(bad), {'status':'invalid'})

    def test_factual_audit_attaches_only_with_matching_result_hash(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(server, 'ENGINE', Path(tmp)):
            run=Path(tmp)/'results/improvement-phase-2026-11-16/grounded-01';run.mkdir(parents=True)
            result=run/'result.json';result.write_text('{"status":"reviewed","proposal":{},"review":{}}')
            self.assertEqual(next(x for x in server.grounded_rows() if x['id']=='grounded-01')['factual_audit']['status'],'unavailable')
            audit={'source_result_sha256':hashlib.sha256(result.read_bytes()).hexdigest(),'scope':'Claims checked against saved evidence.',
                   'findings':[{'claim':'A claim','assessment':'Unsupported by the cited span.','evidence':'The method text does not state it.'}]}
            (run/'factual-audit.json').write_text(json.dumps(audit))
            row=next(x for x in server.grounded_rows() if x['id']=='grounded-01')
            self.assertEqual(row['factual_audit']['status'],'attached');self.assertTrue(row['factual_audit']['verified'])
            self.assertEqual(row['factual_audit']['findings'][0]['claim'],'A claim')
            result.write_text('{"changed":true}')
            row=next(x for x in server.grounded_rows() if x['id']=='grounded-01')
            self.assertEqual(row['factual_audit']['status'],'mismatch');self.assertEqual(row['factual_audit']['findings'],[])

    def test_factual_audit_invalid_and_oversize_are_quarantined(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(server, 'ENGINE', Path(tmp)):
            run=Path(tmp)/'results/improvement-phase-2026-11-16/grounded-01';run.mkdir(parents=True)
            result=run/'result.json';result.write_text('{}')
            for payload in [{'scope':'x','findings':[]}, {'source_result_sha256':'0'*64,'scope':'x','findings':[{'claim':'x','assessment':'y','evidence':'z'}]}]:
                (run/'factual-audit.json').write_text(json.dumps(payload))
                row=next(x for x in server.grounded_rows() if x['id']=='grounded-01')
                self.assertIn(row['factual_audit']['status'], {'invalid','mismatch'});self.assertEqual(row['factual_audit']['findings'],[])
            (run/'factual-audit.json').write_text('x'*100001)
            row=next(x for x in server.grounded_rows() if x['id']=='grounded-01')
            self.assertEqual(row['factual_audit']['status'],'invalid');self.assertEqual(row['factual_audit']['findings'],[])

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
        self.assertIn('--search-config', command)
        self.assertIn(str(Path(project['path']) / 'runs' / plan['run_id'] / 'search-config.json'), command)
        self.assertEqual(self.request(route + '/source-pack', {'document_ids': [], 'question': 'x'})[0], 400)

    def test_source_pack_search_config_is_bounded_and_opt_in(self):
        status, project = self.request('/api/projects', {'name': 'Prior art config'})
        self.assertEqual(status, 201)
        route = '/api/projects/' + project['id']
        status, imported = self.request(route + '/documents', {'files': [{'name': 'a.md', 'content_base64': base64.b64encode(b'# Source').decode()}]})
        self.assertEqual(status, 201)
        doc_id = imported['items'][0]['id']
        status, plan = self.request(route + '/source-pack', {'document_ids': [doc_id], 'question': 'Check?', 'prior_art_queries': ['memory-centric agents', 'continual learning'], 'allow_arxiv': True})
        self.assertEqual(status, 201)
        run_root = Path(project['path']) / 'runs' / plan['run_id']
        self.assertEqual(json.loads((run_root / 'search-config.json').read_text()), {'queries': ['memory-centric agents', 'continual learning'], 'allow_arxiv': True})
        command = json.loads((run_root / 'grounded-plan.json').read_text())['execution']['command']
        self.assertIn('--search-config', command)
        self.assertIn(str(run_root / 'search-config.json'), command)
        self.assertEqual(self.request(route + '/source-pack', {'document_ids': [doc_id], 'question': 'x', 'allow_arxiv': True})[0], 400)
        self.assertEqual(self.request(route + '/source-pack', {'document_ids': [doc_id], 'question': 'x', 'prior_art_queries': ['x'] * 4})[0], 400)
        self.assertEqual(self.request(route + '/source-pack', {'document_ids': [doc_id], 'question': 'x', 'prior_art_queries': ['x' * 201]})[0], 400)

if __name__ == '__main__': unittest.main()
