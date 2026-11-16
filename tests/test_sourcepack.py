import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from generation.sourcepack import SourcePackError, build_source_pack, validate_source_pack


class SourcePackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "paper.md").write_text("alpha\nbeta\ngamma\ndelta\n", encoding="utf-8")
        (self.root / "code.py").write_text("print('never run')\n", encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def manifest(self, *sources):
        return {"sources": list(sources)}

    def test_hashes_and_line_chunks_are_deterministic(self):
        m = self.manifest({"id": "p", "title": "Paper", "url": "https://example/p",
                           "path": "paper.md", "evidence_level": "full_text"})
        one = build_source_pack(m, self.root, max_total_chars=100, chunk_chars=8)
        two = build_source_pack(m, self.root, max_total_chars=100, chunk_chars=8)
        self.assertEqual(one, two)
        self.assertEqual(one["digest"], build_source_pack(m, self.root, max_total_chars=100, chunk_chars=8)["digest"])
        self.assertEqual(one["chunks"][0]["start_line"], 1)
        self.assertEqual(one["chunks"][0]["source_sha256"], one["sources"][0]["content_sha256"])

    def test_total_limit_records_omissions(self):
        m = self.manifest({"id": "p", "title": "Paper", "url": "u", "path": "paper.md",
                           "evidence_level": "abstract"})
        pack = build_source_pack(m, self.root, max_total_chars=6, chunk_chars=5)
        self.assertTrue(pack["sources"][0]["truncated"])
        self.assertGreater(pack["sources"][0]["omitted_chunks"], 0)
        self.assertLessEqual(pack["limits"]["included_chars"], 6)

    def test_rejects_traversal_and_symlink_escape(self):
        outside = self.root.parent / "outside.txt"
        outside.write_text("secret", encoding="utf-8")
        m = self.manifest({"id": "x", "title": "x", "url": "u", "path": "../outside.txt",
                           "evidence_level": "user_document"})
        with self.assertRaises(SourcePackError):
            build_source_pack(m, self.root)
        link = self.root / "link.txt"
        try:
            link.symlink_to(outside)
        except OSError:
            self.skipTest("symlinks unavailable")
        m["sources"][0]["path"] = "link.txt"
        with self.assertRaises(SourcePackError):
            build_source_pack(m, self.root)

    def test_rejects_bad_encoding_extension_and_duplicate_ids(self):
        (self.root / "bad.bin").write_bytes(b"x")
        base = {"id": "x", "title": "x", "url": "u", "path": "bad.bin", "evidence_level": "abstract"}
        with self.assertRaises(SourcePackError):
            build_source_pack(self.manifest(base), self.root)
        dup = {"id": "x", "title": "x", "url": "u", "path": "paper.md", "evidence_level": "abstract"}
        with self.assertRaises(SourcePackError):
            build_source_pack(self.manifest(dup, dup), self.root)

    def test_rejects_oversized_and_empty_sources(self):
        big = self.root / "big.txt"
        big.write_text("x" * 20, encoding="utf-8")
        m = self.manifest({"id": "b", "title": "b", "url": "u", "path": "big.txt", "evidence_level": "user_document"})
        with self.assertRaises(SourcePackError):
            build_source_pack(m, self.root, max_source_bytes=10)
        empty = self.root / "empty.txt"
        empty.write_bytes(b"")
        m["sources"][0].update(id="e", path="empty.txt")
        with self.assertRaises(SourcePackError):
            build_source_pack(m, self.root)

    def test_cli_runs_as_a_standalone_script(self):
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps(self.manifest({"id": "p", "title": "Paper", "url": "u",
                                                       "path": "paper.md", "evidence_level": "abstract"})), encoding="utf-8")
        output = self.root / "pack.json"
        script = Path(__file__).resolve().parents[1] / "scripts" / "build_sourcepack.py"
        completed = subprocess.run([sys.executable, str(script), "--manifest", str(manifest), "--root",
                                    str(self.root), "--output", str(output)], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue(output.exists())
        self.assertIn("digest", json.loads(output.read_text(encoding="utf-8")))

    def test_validate_snapshot_integrity_and_recomputed_digest(self):
        m = self.manifest({"id": "p", "title": "Paper", "url": "u", "path": "paper.md",
                           "evidence_level": "full_text"})
        pack = build_source_pack(m, self.root, max_total_chars=100, chunk_chars=8)
        self.assertEqual(validate_source_pack(pack)["digest"], pack["digest"])
        tampered = json.loads(json.dumps(pack))
        tampered["chunks"][0]["text"] += "!"
        with self.assertRaises(SourcePackError):
            validate_source_pack(tampered)
        recomputed = json.loads(json.dumps(tampered))
        body = {key: value for key, value in recomputed.items() if key != "digest"}
        import hashlib
        recomputed["digest"] = hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True,
                                                         separators=(",", ":")).encode()).hexdigest()
        with self.assertRaises(SourcePackError):
            validate_source_pack(recomputed)


if __name__ == "__main__":
    unittest.main()
