import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import importlib.util
HAS_SKLEARN = importlib.util.find_spec('sklearn') is not None
if HAS_SKLEARN:
    from scripts.world_search_benchmark import _digits, _tabular, run


@unittest.skipUnless(HAS_SKLEARN, "Install requirements-benchmark.lock in benchmark environment")
class WorldSearchBenchmarkTests(unittest.TestCase):
    def test_splits_are_stable_and_disjoint(self):
        x1, y1, s1 = _tabular()
        x2, y2, s2 = _tabular()
        self.assertEqual(x1.shape, (5000, 20))
        self.assertEqual(s1["train"].tolist(), s2["train"].tolist())
        self.assertEqual(set(s1["train"]) & set(s1["test"]), set())
        self.assertTrue((y1 == y2).all())
        self.assertEqual(_digits()[0].shape[1], 64)

    def test_validation_does_not_evaluate_test(self):
        import numpy as np
        from scripts import world_search_benchmark as bench
        x = np.repeat(np.arange(20)[:, None], 20, axis=1).astype(float)
        y = np.arange(20) % 2
        split = {"train": np.arange(12), "validation": np.arange(12,16), "test": np.arange(16,20)}
        seen = []
        class Spy:
            def check(self, data):
                seen.append(data[:,0].tolist())
                if set(data[:,0]) != set(split["validation"]):
                    raise AssertionError("Prediction accessed non-validation rows")
            def predict(self,data):
                self.check(data); return np.zeros(len(data), dtype=int)
            def predict_proba(self,data):
                self.check(data); return np.full((len(data),2), .5)
        with tempfile.TemporaryDirectory() as d, \
             patch.object(bench, '_tabular', return_value=(x,y,split)), \
             patch.object(bench, '_digits', return_value=(x,y,split)), \
             patch.object(bench, '_fit_tabular', return_value=Spy()), \
             patch.object(bench, '_fit_digits', return_value=Spy()), \
             patch.object(bench, '_mask', side_effect=lambda data,*args:data), \
             patch.object(bench, '_digit_corrupt', side_effect=lambda data,*args:data):
            result = run("validation", Path(d) / "result.json")
        self.assertTrue(seen)
        self.assertFalse(result["config"]["test_evaluated"])


if __name__ == "__main__":
    unittest.main()
