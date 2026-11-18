import hashlib, json, shutil, tempfile, unittest
from pathlib import Path
from generation.field_calibration_scoring import score_calibration
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"reports/chunk-field-calibration-v1"
INSTRUCTIONS=ROOT/"reports/generation-comparison-campaign2/reviewer-instructions.md"
class Tests(unittest.TestCase):
    def copy(self):
        td=tempfile.TemporaryDirectory(); out=Path(td.name)/"cal"; shutil.copytree(SOURCE,out); return td,out
    def test_failed_readiness_ignores_old_score(self):
        td,out=self.copy(); self.addCleanup(td.cleanup); old=json.loads((out/"score.json").read_text()); old["ready"]=True; (out/"score.json").write_text(json.dumps(old)); self.assertFalse(score_calibration(out,INSTRUCTIONS)["ready"])
    def test_known_variant_can_pass(self):
        td,out=self.copy(); self.addCleanup(td.cleanup); kp=out/"scoring-key.json"; key=json.loads(kp.read_text()); key["required_additional_negative"]={"record_id":key["reviewer_aliases"]["cal-unsupported-atom-speed"],"field":"outputs[0]","verdict":"unsupported"}; kp.write_text(json.dumps(key,indent=2)+"\n"); fp=out/"calibration-freeze.json"; freeze=json.loads(fp.read_text()); freeze["scoring_key_sha256"]=hashlib.sha256(kp.read_bytes()).hexdigest(); fp.write_text(json.dumps(freeze,indent=2)+"\n"); self.assertTrue(score_calibration(out,INSTRUCTIONS)["ready"])
    def test_mutated_key_hash_rejected(self):
        td,out=self.copy(); self.addCleanup(td.cleanup); kp=out/"scoring-key.json"; key=json.loads(kp.read_text()); key["required_positive_checks"]["fresh-2"]="unsupported"; kp.write_text(json.dumps(key)); report=score_calibration(out,INSTRUCTIONS); self.assertFalse(report["ready"]); self.assertIn("scoring key hash mismatch",report["errors"])
    def test_changed_instructions_rejected(self):
        td,out=self.copy(); self.addCleanup(td.cleanup); ip=Path(td.name)/"i.md"; ip.write_text(INSTRUCTIONS.read_text()+"changed"); report=score_calibration(out,ip); self.assertFalse(report["ready"]); self.assertIn("instructions hash mismatch",report["errors"])
