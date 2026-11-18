import tempfile
import json
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts import fixed_artifact_campaign as command

class FixedCommandTests(unittest.TestCase):
    def test_preview_never_reads_key_or_builds_transport(self):
        with patch.object(command.lifecycle,'_require_initial',return_value={'cases':[1]*48}):
            def forbidden(*args):raise AssertionError('unexpected paid dependency')
            result=command.paid_stage(Path('/fixture'),'generate',key_loader=forbidden,transport_factory=forbidden)
            self.assertEqual(result['paid_requests'],0);self.assertEqual(result['planned_requests'],48)

    def test_audit_before_credentials(self):
        with patch.object(command.lifecycle,'_require_initial',return_value={}),patch.object(command,'require_execution_audit',side_effect=ValueError('no audit')):
            def forbidden():raise AssertionError('credential read before audit')
            with self.assertRaisesRegex(ValueError,'no audit'):
                command.paid_stage(Path('/fixture'),'prepare',execute=True,key_loader=forbidden)

    def test_empty_key_before_lifecycle_attempt(self):
        with patch.object(command.lifecycle,'_require_initial',return_value={}),patch.object(command,'require_execution_audit',return_value={}),patch.object(command.lifecycle,'prepare',side_effect=AssertionError('attempt')):
            with self.assertRaisesRegex(ValueError,'credential unavailable'):
                command.paid_stage(Path('/fixture'),'prepare',execute=True,key_loader=lambda:'')

    def test_key_stays_only_in_transport_closure(self):
        seen=[]
        def factory(loader):seen.append(loader());return lambda req:None
        with patch.object(command.lifecycle,'_require_initial',return_value={}),patch.object(command,'require_execution_audit',return_value={}),patch.object(command.lifecycle,'prepare',return_value={'status':'fixture'}):
            result=command.paid_stage(Path('/fixture'),'prepare',execute=True,key_loader=lambda:'fake-secret',transport_factory=factory)
            self.assertEqual(result,{'status':'fixture'});self.assertEqual(seen,['fake-secret'])

    def test_bound_source_must_itself_clear_exact_plan(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);out=root/"campaign";out.mkdir()
            (out/"plan.json").write_text('{}');(out/"execution-freeze.json").write_text('{}')
            source=out/"source.json"
            raw={"schema":"fixed-artifact-independent-audit.v1","status":"passed","reviewer":"fixture",
                 "plan_sha256":command.lifecycle.sha(out/"plan.json"),"execution_freeze_sha256":command.lifecycle.sha(out/"execution-freeze.json")}
            source.write_text(json.dumps(raw))
            receipt={"schema":"fixed-artifact-execution-audit.v1","status":"passed",
                     "plan_sha256":raw["plan_sha256"],"execution_freeze_sha256":raw["execution_freeze_sha256"],
                     "source_report":"source.json","source_report_sha256":command.lifecycle.sha(source)}
            (out/"execution-audit.json").write_text(json.dumps(receipt))
            with patch.object(command.planner,"ROOT",root),patch.object(command.lifecycle,"_require_initial",return_value={}):
                command.require_execution_audit(out)
                raw["status"]="blocked";source.write_text(json.dumps(raw))
                receipt["source_report_sha256"]=command.lifecycle.sha(source)
                (out/"execution-audit.json").write_text(json.dumps(receipt))
                with self.assertRaisesRegex(ValueError,"did not clear"):
                    command.require_execution_audit(out)

if __name__=='__main__':unittest.main()
