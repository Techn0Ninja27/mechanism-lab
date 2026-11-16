import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from generation.prior_art_retrieval import CacheMismatchError, _parse_arxiv, retrieve_prior_art


XML = b'''<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>https://arxiv.org/abs/2401.00001v2</id><title>Memory and Agents</title><summary>Memory abstract.</summary><published>2024-01-01T00:00:00Z</published><updated>2024-01-02T00:00:00Z</updated></entry></feed>'''


class Response:
    def __init__(self, body): self.body = body
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def read(self, n=-1): return self.body if n < 0 else self.body[:n]


class PriorArtTests(unittest.TestCase):
    def db(self, path):
        with sqlite3.connect(path) as db:
            db.executescript("CREATE TABLE works (source_id TEXT PRIMARY KEY, canonical_id TEXT, title TEXT, abstract TEXT, published_date TEXT, updated_date TEXT, landing_url TEXT, pdf_url TEXT, content_sha256 TEXT, evidence_level TEXT); CREATE VIRTUAL TABLE works_fts USING fts5(source_id UNINDEXED,title,abstract);")
            db.execute("INSERT INTO works VALUES (?,?,?,?,?,?,?,?,?,?)", ("local:1", "local:1", "Memory agents", "Memory retrieval", "2024-01-01", "", "https://example.test/1", "", "hash", "abstract_only"))
            db.execute("INSERT INTO works_fts VALUES (?,?,?)", ("local:1", "Memory agents", "Memory retrieval"))

    def test_local_offline_and_escaped_fts(self):
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "c.sqlite"; self.db(db)
            out = retrieve_prior_art(['memory agents'], db, cutoff="2024-12-31")
            self.assertEqual(out["records"][0]["source_id"], "local:1")
            self.assertIsNone(out["novelty_label"])

    def test_input_and_dates(self):
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "c.sqlite"; self.db(db)
            with self.assertRaises(ValueError): retrieve_prior_art("whole document", db)
            self.assertEqual(retrieve_prior_art(["memory"], db, cutoff="2020-01-01")["records"], [])

    def test_bounds_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "c.sqlite"; self.db(db)
            with self.assertRaises(ValueError): retrieve_prior_art(["x" * 201], db)
            with self.assertRaises(ValueError): retrieve_prior_art(["x"], db, limit=21)
            with self.assertRaises(ValueError): retrieve_prior_art(["x"], db, min_interval=2.99)

    def test_cutoff_applies_before_limit(self):
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "c.sqlite"; self.db(db)
            with sqlite3.connect(db) as con:
                con.execute("INSERT INTO works VALUES (?,?,?,?,?,?,?,?,?,?)", ("local:future", "local:future", "Memory future", "Memory retrieval", "2099-01-01", "", "", "", "future", "abstract_only"))
                con.execute("INSERT INTO works_fts VALUES (?,?,?)", ("local:future", "Memory future", "Memory retrieval"))
            result = retrieve_prior_art(["memory"], db, cutoff="2024-12-31", limit=1)
            self.assertEqual(result["records"][0]["source_id"], "local:1")

    def test_later_specific_query_retains_a_slot(self):
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "c.sqlite"; self.db(db)
            with sqlite3.connect(db) as c:
                for i in range(6):
                    sid = f"broad:{i}"
                    c.execute("INSERT INTO works VALUES (?,?,?,?,?,?,?,?,?,?)", (sid,sid,"Learning",f"Learning method {i}","2024-01-01","","https://example.test/"+sid,"","hash","abstract_only"))
                    c.execute("INSERT INTO works_fts VALUES (?,?,?)", (sid,"Learning",f"Learning method {i}"))
            result = retrieve_prior_art(["learning", "memory"], db, limit=2)
            self.assertEqual(len(result["records"]), 2)
            self.assertIn("local:1", [r["source_id"] for r in result["records"]])
            self.assertEqual([r["retained_records"] for r in result["coverage"]], [1,1])

    def test_legacy_arxiv_category_and_alias_dedup(self):
        legacy = b'''<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>https://arxiv.org/abs/quant-ph/0001001v3</id><title>Memory agents</title><summary>Memory retrieval</summary><published>2020-01-01</published><updated>2020-01-01</updated></entry></feed>'''
        row = _parse_arxiv(legacy, "memory")[0]
        self.assertEqual(row["source_id"], "arxiv:quant-ph/0001001")
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "c.sqlite"; self.db(db); cache = Path(d) / "cache"
            with patch("generation.prior_art_retrieval.urllib.request.urlopen", return_value=Response(legacy)):
                result = retrieve_prior_art(["memory"], db, allow_arxiv=True, cache_dir=cache)
            self.assertEqual(len(result["records"]), 1)

    def test_arxiv_is_opt_in_cached_and_failures_recorded(self):
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "c.sqlite"; self.db(db); cache = Path(d) / "cache"
            with patch("generation.prior_art_retrieval.urllib.request.urlopen", return_value=Response(XML)) as call:
                first = retrieve_prior_art(["memory"], db, allow_arxiv=True, cache_dir=cache, cutoff="2026-11-16")
                second = retrieve_prior_art(["memory"], db, allow_arxiv=True, cache_dir=cache, cutoff="2026-11-16")
            self.assertEqual(len(first["records"]), len(second["records"])); self.assertEqual(call.call_count, 1)
            self.assertIn("all%3A%22memory%22", call.call_args[0][0].full_url)

    def test_cache_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "c.sqlite"; self.db(db); cache = Path(d) / "cache"
            with patch("generation.prior_art_retrieval.urllib.request.urlopen", return_value=Response(XML)):
                retrieve_prior_art(["memory"], db, allow_arxiv=True, cache_dir=cache)
            xml_path = next(cache.glob("*.xml")); xml_path.write_bytes(b"tampered")
            with self.assertRaises(CacheMismatchError): retrieve_prior_art(["memory"], db, allow_arxiv=True, cache_dir=cache)

    def test_artifact_is_immutable_and_failures_visible(self):
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "c.sqlite"; self.db(db); artifact = Path(d) / "run.json"
            with patch("generation.prior_art_retrieval.urllib.request.urlopen", side_effect=OSError("offline")):
                result = retrieve_prior_art(["memory"], db, allow_arxiv=True, cache_dir=Path(d)/"cache", artifact_path=artifact)
            self.assertEqual(result["failures"][0]["error_type"], "OSError")
            with self.assertRaises(FileExistsError): retrieve_prior_art(["memory"], db, artifact_path=artifact)

    def test_existing_artifact_rejected_before_network(self):
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "c.sqlite"; self.db(db); artifact = Path(d) / "run.json"; artifact.write_text("old")
            with patch("generation.prior_art_retrieval.urllib.request.urlopen") as call:
                with self.assertRaises(FileExistsError): retrieve_prior_art(["memory"], db, allow_arxiv=True, artifact_path=artifact)
            call.assert_not_called()


if __name__ == "__main__": unittest.main()
