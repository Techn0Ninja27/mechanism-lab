import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts import run_causal_campaign as c
from generation.campaign_selection import select_development


def program():
    return {"predict": {"var": "last_sync"}, "update": {"var": "memory"}}


def inventory(path: Path, *, invalid=False):
    p = program(); digest = hashlib.sha256(json.dumps(p, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    rows = [{"id": "cand-00", "call_id": "call-00", "arm": "task", "status": "valid", "program": p, "program_sha256": digest}]
    if invalid: rows.append({"id": "cand-01", "call_id": "call-01", "arm": "raw", "status": "invalid", "error": "malformed AST"})
    path.write_text(json.dumps({"schema": c.SCHEMA, "candidates": rows}) + "\n")
    return rows


class CampaignRunTests(unittest.TestCase):
    def test_default_is_plan_only_and_binds_inventory(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); inv = root / "inventory.json"; inventory(inv); out = root / "plan"
            result = c.plan(inv, out, "dev")
            self.assertEqual(result["schema"], "causal-campaign-plan.v1")
            self.assertTrue((out / "plan.json").is_file())
            self.assertFalse(result["auto_execute"])

    def test_stage0_plan_opens_no_data_or_training(self):
        with tempfile.TemporaryDirectory() as t:
            result = c.stage0_plan(Path(t) / "stage0")
            self.assertEqual(result["schema"], "causal-campaign-stage0.v1")
            self.assertFalse(result["provenance"]["training_opened"])
            self.assertEqual(result["development_seeds"][0], 100)
            self.assertEqual(result["final_seeds"][0], 200)
            self.assertEqual(result["task"]["primary_window"], list(range(88, 120)))

    def test_dev_plan_retains_invalid_candidates(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); inv = root / "inventory.json"; rows = inventory(inv, invalid=True)
            result = c.plan(inv, root / "plan", "dev")
            self.assertEqual(result["candidate_count"], len(rows))

    def test_final_requires_selection_before_data_access(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); inv = root / "inventory.json"; inventory(inv)
            with self.assertRaises(ValueError): c.plan(inv, root / "plan", "final")
            # A malformed selection is rejected while still in the guard.
            sel = root / "selection.json"; sel.write_text(json.dumps({"schema": "wrong"}))
            with self.assertRaises(ValueError): c.plan(inv, root / "plan2", "final", sel)

    def test_final_selection_binds_inventory_hash_and_call(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); inv = root / "inventory.json"; rows = inventory(inv)
            _, digest = c.load_inventory(inv)
            runs = []
            for seed in c.DEV:
                runs.append({"candidate_id":"cand-00","call_id":"call-00","arm":"task","seed":seed,"status":"complete","window":[{"t":t,"exact_eval":.8} for t in range(88,120)]})
            dev = root / "dev.json"; dev.write_text(json.dumps({"schema":"causal-campaign-result.v1","stage":"dev","inventory_sha256":digest,"dependency_hashes":c._dependency_hashes(),"runs":runs}))
            derived = select_development(rows, runs, inventory_sha256=digest)
            sel = root / "selection.json"; sel.write_text(json.dumps(derived))
            chosen = c.validate_selection(sel, digest, json.loads(inv.read_text()), "final", dev_result_path=dev)
            self.assertEqual(chosen["candidates"][0]["id"], "cand-00")

    def test_cached_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); inv = root / "inventory.json"; inventory(inv); out = root / "plan"
            c.plan(inv, out, "dev")
            with self.assertRaises(FileExistsError): c.plan(inv, out, "dev")

    def test_program_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); inv = root / "inventory.json"; inventory(inv)
            data = json.loads(inv.read_text()); data["candidates"][0]["program_sha256"] = "0" * 64; inv.write_text(json.dumps(data))
            with self.assertRaises(ValueError): c.load_inventory(inv)

    def test_all_failed_call_is_preserved_and_final_guard_allows_hold_only(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); inv = root / "inventory.json"; rows = inventory(inv)
            _, digest = c.load_inventory(inv)
            runs = [{"candidate_id":"cand-00","call_id":"call-00","arm":"task","seed":s,
                     "status":"failed","error":"runtime failure"} for s in c.DEV]
            dev = root / "dev.json"; dev.write_text(json.dumps({"schema":"causal-campaign-result.v1","stage":"dev","inventory_sha256":digest,"dependency_hashes":c._dependency_hashes(),"runs":runs}))
            selection = select_development(rows, runs, inventory_sha256=digest)
            self.assertEqual(selection["selected"][0]["status"], "no_eligible_candidate")
            sel = root / "selection.json"; sel.write_text(json.dumps(selection))
            chosen = c.validate_selection(sel, digest, json.loads(inv.read_text()), "final", dev_result_path=dev)
            self.assertEqual(chosen["candidates"], [])

    def test_real_selection_to_mocked_final_execution_and_report(self):
        from unittest.mock import patch
        from generation.campaign_selection import final_report
        from scripts import evaluate_causal_candidates as evaluator
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); inv=root/'inventory.json'; rows=inventory(inv)
            digest=c.sha(inv)
            def fake_run(x,y,adj,sp,candidate,seed,**kwargs):
                return {'id':candidate['id'],'call_id':candidate['call_id'],'arm':candidate['arm'],
                        'seed':seed,'status':'complete','window':[{'t':i,'exact_eval':.9 if candidate['arm']=='linear-control' else (.8 if candidate['arm']!='hold' else .7)} for i in c.WINDOW]}
            devruns=[fake_run(None,None,None,None,rows[0],seed) for seed in c.DEV]
            selection=select_development(rows,devruns,inventory_sha256=digest)
            dev=root/'dev.json'; dev.write_text(json.dumps({'schema':'causal-campaign-result.v1','stage':'dev','inventory_sha256':digest,'dependency_hashes':c._dependency_hashes(),'runs':devruns}))
            sel=root/'selection.json';sel.write_text(json.dumps(selection))
            with patch.object(evaluator.gap.screen,'load_cora',return_value=(None,None,None)), patch.object(evaluator.gap,'splits',return_value=None), patch.object(evaluator,'run_one',side_effect=fake_run):
                result=c.execute(inv,root/'final','final',sel,dev_result_path=dev,control_selection_path=c.ROOT/'reports/generation-comparison-v2/control-selection.json')
            hold=[r for r in result['runs'] if r['arm']=='hold']
            candidates=[r for r in result['runs'] if r['arm'] not in ('hold','linear-control')]
            report=final_report(rows,selection,candidates,hold,inventory_sha256=digest)
            self.assertEqual(len(hold),32)
            self.assertTrue(report['reports'][0]['better_than_hold'])
            strong=c.report_final(inv,sel,root/'final/result.json',None,root/'report',dev_result_path=dev,control_selection_path=c.ROOT/'reports/generation-comparison-v2/control-selection.json')
            self.assertFalse(strong['reports'][0]['better_than_control'])


if __name__ == "__main__":
    unittest.main()
