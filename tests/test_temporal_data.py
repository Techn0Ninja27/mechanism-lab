import tempfile
import unittest
from pathlib import Path

from generation.temporal_data import (
    mark_future_triples, triples_from_papers, uniform_candidate_triples,
)


class TemporalDataTests(unittest.TestCase):
    def test_triples_and_strict_new(self):
        train = {"a": (1, 2, 3), "b": (3, 4, 5)}
        known = triples_from_papers(train)
        rows = mark_future_triples({"future": (1, 2, 3, 6)}, known)
        self.assertEqual([r["strictly_new"] for r in rows], [False, True, True, True])

    def test_uniform_reproducible_and_unique(self):
        a = uniform_candidate_triples(range(8), 100, seed=7)
        b = uniform_candidate_triples(range(8), 100, seed=7)
        self.assertEqual(a, b)
        self.assertEqual(len({tuple(r["triple"]) for r in a}), len(a))
        self.assertTrue(all(r["source"] == "uniform" for r in a))

    def test_empty_and_invalid(self):
        self.assertEqual(triples_from_papers({"x": (1, 2)}), set())
        with self.assertRaises(ValueError):
            uniform_candidate_triples([1, 2], 3)

    def test_temporal_order_validation(self):
        from generation.temporal_data import load_temporal_split
        with self.assertRaises(ValueError):
            load_temporal_split("missing.json", "missing.db", train_end=2024,
                                validation_year=2024, test_year=2025)

    def test_db_id_alias(self):
        import sqlite3
        from generation.temporal_data import _years_from_sqlite
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.db"
            with sqlite3.connect(path) as db:
                db.execute("create table papers (paper_id text, venue_year integer)")
                db.execute("insert into papers values ('conf/naacl/X', 2024)")
                db.commit()
            years = _years_from_sqlite(path)
            self.assertEqual(years["conf/naacl/X"], 2024)
            self.assertEqual(years["conf__naacl__X"], 2024)


if __name__ == "__main__":
    unittest.main()
