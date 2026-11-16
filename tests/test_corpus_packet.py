import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ingest_research_corpus.py"
spec = importlib.util.spec_from_file_location("ingest_research_corpus", SCRIPT)
ingest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ingest)


class CorpusPacketTests(unittest.TestCase):
    def packet(self, records):
        root = Path(self.tmp.name)
        path = root / "packet.json"
        path.write_text(json.dumps(records), encoding="utf-8")
        return path

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_maps_version_dates_and_file_hash_without_summary_as_abstract(self):
        path = self.packet([{"id": "2106.00311v2", "title": "NeuMiss", "url": "https://arxiv.org/abs/2106.00311v2", "abstract": "real abstract", "mechanism_summary": "generated summary", "published": "2021-06-01T00:00:00Z", "updated": "2026-01-02T00:00:00Z", "evidence_level": "primary_abstract"}])
        rows, manifest = ingest._load_packet(path, "2026-11-16")
        self.assertEqual(rows[0]["source_id"], "arxiv:2106.00311v2")
        self.assertEqual(rows[0]["canonical_id"], "arxiv:2106.00311")
        self.assertEqual(rows[0]["version"], "v2")
        self.assertEqual(rows[0]["published_date"], "2021-06-01")
        self.assertEqual(rows[0]["updated_date"], "2026-01-02")
        self.assertEqual(rows[0]["abstract"], "real abstract")
        self.assertEqual(manifest["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_future_filter_checks_both_published_and_updated(self):
        path = self.packet([
            {"id": "2401.00001v1", "title": "published future", "abstract": "a", "published": "2026-11-17"},
            {"id": "2401.00002v1", "title": "updated future", "abstract": "b", "published": "2020-01-01", "updated": "2026-11-17"},
            {"id": "2401.00003v1", "title": "kept", "abstract": "c", "published": "2020-01-01", "updated": "2026-11-16"},
        ])
        rows, manifest = ingest._load_packet(path, "2026-11-16")
        self.assertEqual([r["source_id"] for r in rows], ["arxiv:2401.00003v1"])
        self.assertEqual(manifest["future_dropped"], 2)

    def test_alias_and_content_dedup_preserve_newest_version(self):
        rows = [
            {"source_id": "arxiv:1v1", "canonical_id": "arxiv:1", "title": "T", "abstract": "A", "updated_date": "2024-01-01", "content_sha256": ingest.content_sha("T", "A")},
            {"source_id": "arxiv:1v2", "canonical_id": "arxiv:1", "title": "T", "abstract": "B", "updated_date": "2025-01-01", "content_sha256": ingest.content_sha("T", "B")},
            {"source_id": "packet:x", "canonical_id": "packet:x", "title": "T", "abstract": "B", "updated_date": "2025-01-01", "content_sha256": ingest.content_sha("T", "B")},
        ]
        kept, aliases = ingest.deduplicate(rows)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["source_id"], "arxiv:1v2")
        self.assertEqual(aliases["arxiv:1v1"], "arxiv:1")
        self.assertEqual(aliases["packet:x"], "arxiv:1")

    def test_existing_snapshot_rows_are_retained_and_future_rows_filtered(self):
        path = Path(self.tmp.name) / "works.jsonl"
        rows = [
            {"source_id": "legacy:keep", "canonical_id": "legacy:keep", "title": "Keep", "abstract": "a", "published_date": "2020-01-01", "updated_date": "", "date_precision": "year"},
            {"source_id": "legacy:future", "canonical_id": "legacy:future", "title": "Future", "abstract": "b", "published_date": "2026-12-01", "updated_date": "", "date_precision": "day"},
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        loaded = ingest._existing_corpus_rows(path, "2026-11-16")
        self.assertEqual([row["source_id"] for row in loaded], ["legacy:keep"])
        self.assertEqual(loaded[0]["date_precision"], "year")


if __name__ == "__main__":
    unittest.main()
