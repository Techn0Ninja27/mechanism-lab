import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import replay_proposal_readiness as replay


class ReadinessReplayTests(unittest.TestCase):
    def fixture(self, root: Path, digest_override=None):
        (root / "generation").mkdir(parents=True)
        (root / "generation/proposal_readiness.py").write_text("# frozen classifier fixture\n")
        result = root / "results" / "sample" / "result.json"
        result.parent.mkdir(parents=True)
        result.write_text(json.dumps({"status": "failed", "error": "preserved"}) + "\n")
        digest = digest_override or hashlib.sha256(result.read_bytes()).hexdigest()
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({
            "schema": "proposal-readiness-manifest.v1",
            "records": [{"id": "sample", "path": "results/sample/result.json",
                         "sha256": digest, "expected_status": "incomplete"}],
        }))
        return manifest

    def test_success_records_classifier_hash_and_does_not_mutate_input(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self.fixture(root)
            before = (root / "results/sample/result.json").read_bytes()
            output = root / "out.json"
            with patch.object(replay, "ROOT", root.resolve()):
                report = replay.replay(manifest, output)
            self.assertEqual(report["records"][0]["readiness"]["status"], "incomplete")
            self.assertRegex(report["classifier_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual((root / "results/sample/result.json").read_bytes(), before)

    def test_hash_mismatch_refuses_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self.fixture(root, "0" * 64)
            output = root / "out.json"
            with patch.object(replay, "ROOT", root.resolve()):
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    replay.replay(manifest, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
