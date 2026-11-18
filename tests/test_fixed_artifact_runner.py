import json, tempfile, unittest
from unittest.mock import patch
from pathlib import Path
from scripts.run_fixed_artifact_campaign import ARMS, RAW_RENDER_UPPER_BYTES, PHASE_CAP, build_plan, verify_plan

ROOT=Path(__file__).resolve().parents[1]

class FixedArtifactRunnerTests(unittest.TestCase):
    def test_offline_plan_preserves_failed_atom_and_48_slots(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/"plan"
            plan=build_plan(out)
            self.assertEqual(plan["execution_authority"],"none_requires_separate_audited_freeze")
            self.assertEqual(len(plan["cases"]),48)
            self.assertEqual(plan["prose_status"],"unavailable")
            self.assertEqual(json.loads((out/"fidelity.json").read_text())["atoms"]["status"],"gate_failed_reused_unselected")
            self.assertEqual(verify_plan(out)["paid_calls"],0)

    def test_all_arms_have_same_call_denominator_per_decoder(self):
        with tempfile.TemporaryDirectory() as td:
            d=build_plan(Path(td)/"plan")
            for decoder in d["decoders"]:
                for arm in ARMS:
                    self.assertEqual(sum(x["decoder"]==decoder and x["arm"]==arm for x in d["cases"]),6)

    def test_inputs_are_bound_and_plan_is_not_resumable(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/"plan";build_plan(out)
            with self.assertRaises(FileExistsError): build_plan(out)
            (out/"representations/atoms.json").write_text((out/"representations/atoms.json").read_text()+"\n")
            with self.assertRaises(ValueError): verify_plan(out)

    def test_same_header_is_present_in_raw_and_atoms_prompts(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/"plan";build_plan(out)
            hashes=json.loads((out/"prompt-hashes.json").read_text())
            self.assertEqual(set(hashes),{"task-only","raw","atoms","distill"})
            self.assertNotEqual(hashes["raw"],hashes["atoms"])

    def test_control_model_case_task_code_and_prompt_mutations_fail(self):
        import scripts.run_fixed_artifact_campaign as m
        for field in ("decoders", "cases", "task", "controls", "prompt"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as td:
                out=Path(td)/"plan"; build_plan(out)
                p=out/"plan.json"; d=json.loads(p.read_text())
                if field=="decoders": d["decoders"]["opus-5.5"]["model"]="tampered"
                elif field=="cases": d["cases"]=d["cases"][:-1]
                elif field=="task": d["task_sha256"]="0"*64
                elif field=="controls": d["controls_file_sha256"]="0"*64
                else: (out/"representations/atoms.json").write_text((out/"representations/atoms.json").read_text()+" ")
                p.write_text(json.dumps(d))
                with self.assertRaises(ValueError): verify_plan(out)

    def test_cap_failure_is_preflight_only(self):
        import scripts.run_fixed_artifact_campaign as m
        with tempfile.TemporaryDirectory() as td:
            with patch.object(m,"CAP",0.01):
                with self.assertRaises(ValueError): build_plan(Path(td)/"plan")

    def test_absent_prose_uses_render_upper_and_phase_headroom(self):
        with tempfile.TemporaryDirectory() as td:
            plan=build_plan(Path(td)/"plan")
            self.assertGreaterEqual(plan["generation_reserve_upper_usd"], 23.0)
            self.assertEqual(plan["raw_render_upper_bytes"], RAW_RENDER_UPPER_BYTES)
            self.assertGreater(plan["phase_cap_usd"], plan["phase_worst_case_charged_usd"])
            self.assertEqual(plan["phase_cap_usd"], PHASE_CAP)

    def test_copied_control_content_and_prompt_files_are_bound(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/"plan"; build_plan(out)
            (out/"controls.json").write_text((out/"controls.json").read_text()+"\n")
            with self.assertRaises(ValueError): verify_plan(out)

    def test_dependency_and_reviewer_instruction_bindings_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/"plan"; build_plan(out)
            d=json.loads((out/"plan.json").read_text())
            d["dependency_hashes"]["generation/campaign_selection.py"] = "0" * 64
            (out/"plan.json").write_text(json.dumps(d))
            with self.assertRaises(ValueError): verify_plan(out)
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/"plan"; build_plan(out)
            d=json.loads((out/"plan.json").read_text())
            d["reviewer_metadata"]["applies_to"] = ["summary"]
            (out/"plan.json").write_text(json.dumps(d))
            with self.assertRaises(ValueError): verify_plan(out)
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/"plan"; build_plan(out)
            (out/"prompts/raw.txt").write_text((out/"prompts/raw.txt").read_text()+"tamper")
            with self.assertRaises(ValueError): verify_plan(out)

    def test_seeded_case_order_is_exact_and_decoder_unique(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/"plan"; plan=build_plan(out)
            self.assertEqual(plan["cases"], verify_plan(out)["cases"])
            for case in plan["cases"]:
                self.assertEqual(case["id"], f"{case['decoder']}-{case['arm']}-{case['replicate']:02d}")

    def test_validated_prose_roundtrip_verifies(self):
        import scripts.run_fixed_artifact_campaign as m
        pack=json.loads(m.source_pack().read_text())
        atoms=json.loads(m.stopped_atoms().read_text())
        prose={"distillations":[{"id":a["id"],"source_id":a["source_id"],"summary":a["statement"],"unknowns":a["unknowns"],"evidence":a["evidence"]} for a in atoms["atoms"]]}
        with tempfile.TemporaryDirectory() as td:
            src=Path(td)/"prose.json"; src.write_text(json.dumps(prose))
            out=Path(td)/"plan"; plan=build_plan(out, prose_path=src)
            self.assertEqual(plan["prose_status"], "validated")
            self.assertEqual(verify_plan(out)["prose_status"], "validated")

    def test_invalid_prose_is_preserved_unavailable(self):
        with tempfile.TemporaryDirectory() as td:
            raw=Path(td)/"bad.json"; raw.write_text("{bad")
            out=Path(td)/"plan"; d=build_plan(out, prose_path=raw)
            self.assertEqual(d["prose_status"], "unavailable")
            self.assertEqual((out/"representations/distillations-raw.json").read_bytes(), raw.read_bytes())
            self.assertEqual(verify_plan(out)["execution_authority"],"none_requires_separate_audited_freeze")

    def test_fee_contingency_bound_and_tamper(self):
        from scripts.run_fixed_artifact_campaign import _reserve_bytes, SYSTEM
        price={"prompt":4e-6,"completion":20e-6}
        inference=(len(SYSTEM.encode())+100+512)*price["prompt"]+12288*price["completion"]
        self.assertAlmostEqual(_reserve_bytes(100,price),inference*1.05)
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/"plan"; plan=build_plan(out)
            self.assertAlmostEqual(plan["worst_case_campaign_reserve_usd"],24.8005002)
            self.assertLess(plan["phase_worst_case_charged_usd"],30)
            plan["byok_fee_rate"]=0
            (out/"plan.json").write_text(json.dumps(plan))
            with self.assertRaises(ValueError): verify_plan(out)

    def test_historical_verification_allows_increase_never_reset(self):
        import scripts.run_fixed_artifact_campaign as m
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/"plan"; plan=build_plan(out)
            initial=plan["phase_charged_before_usd"]
            with patch.object(m,"_phase_charged",return_value=initial+0.5):
                with self.assertRaises(ValueError):verify_plan(out)
                verify_plan(out,require_current_phase=False)
            with patch.object(m,"_phase_charged",return_value=initial-0.5):
                with self.assertRaises(ValueError):verify_plan(out,require_current_phase=False)

if __name__=='__main__': unittest.main()
