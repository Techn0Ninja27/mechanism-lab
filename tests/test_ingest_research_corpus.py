import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.ingest_research_corpus import arxiv_canonical_id, content_sha, deduplicate, future_filtered, parse_arxiv, write_corpus


XML = b'''<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom"><entry><id>http://arxiv.org/abs/2401.00001v2</id><title> Test   Paper </title><summary>One abstract.</summary><published>2024-01-02T00:00:00Z</published><updated>2024-02-01T00:00:00Z</updated><author><name>A. Author</name></author><link title="pdf" href="http://arxiv.org/pdf/2401.00001v2"/></entry></feed>'''


class IngestTests(unittest.TestCase):
    def test_arxiv_id_and_parse(self):
        self.assertEqual(arxiv_canonical_id("http://arxiv.org/abs/2401.00001v2"), "2401.00001")
        rows = parse_arxiv(XML, query="q", retrieved_at="2026-11-16")
        self.assertEqual(rows[0]["source_id"], "arxiv:2401.00001")
        self.assertEqual(rows[0]["authors"], ["A. Author"])

    def test_future_filter(self):
        rows = [{"source_id": "a", "published_date": "2026-11-17"}, {"source_id": "b", "published_date": "2026-11-16"}, {"source_id": "c", "published_date": "2024-01-01", "updated_date": "2026-11-17"}]
        kept, dropped = future_filtered(rows, "2026-11-16")
        self.assertEqual([x["source_id"] for x in kept], ["b"])
        self.assertEqual(dropped, 2)

    def test_old_style_arxiv_category_is_preserved(self):
        self.assertEqual(arxiv_canonical_id("http://arxiv.org/abs/quant-ph/0001001v3"), "quant-ph/0001001")

    def test_version_and_content_dedup_are_stable(self):
        base = {"source_id": "arxiv:1v1", "canonical_id": "arxiv:1", "updated_date": "2024-01-01", "title": "A", "abstract": "x", "content_sha256": content_sha("A", "x")}
        newer = {**base, "source_id": "arxiv:1v2", "updated_date": "2024-02-01", "abstract": "y", "content_sha256": content_sha("A", "y")}
        duplicate = {"source_id": "doi:x", "canonical_id": "doi:x", "updated_date": "", "title": "A", "abstract": "y", "content_sha256": content_sha("A", "y")}
        rows, aliases = deduplicate([duplicate, newer, base])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source_id"], "arxiv:1v2")
        self.assertEqual(aliases["arxiv:1"], "arxiv:1")

    def test_sqlite_snapshot_is_idempotent(self):
        row = {"source_id": "local:a", "canonical_id": "local:a", "title": "A", "abstract": "B", "authors": [], "published_date": "2020-01-01", "updated_date": "", "landing_url": "x", "pdf_url": "x", "retrieval_method": "test", "retrieved_at": "2026-11-16", "content_sha256": content_sha("A", "B"), "evidence_level": "abstract_only", "full_text_available": False}
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            write_corpus([row], {"local:a": "local:a"}, [], out, cutoff="2026-11-16")
            write_corpus([row], {"local:a": "local:a"}, [], out, cutoff="2026-11-16")
            import sqlite3
            with sqlite3.connect(out / "corpus.sqlite") as db:
                self.assertEqual(db.execute("select count(*) from works").fetchone()[0], 1)
                self.assertEqual(db.execute("select count(*) from works_fts").fetchone()[0], 1)

    def test_content_hash_is_sha256(self):
        self.assertEqual(len(content_sha("A", "B")), 64)
        self.assertEqual(content_sha("A", "B"), hashlib.sha256(b"a\nb").hexdigest())


if __name__ == "__main__":
    unittest.main()
