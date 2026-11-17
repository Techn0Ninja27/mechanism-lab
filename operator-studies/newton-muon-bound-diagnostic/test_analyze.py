import unittest
import numpy as np
from analyze import choose


class BoundTests(unittest.TestCase):
    def test_psd_bounds_and_selection(self):
        k = np.diag([1.0, 2.0, 3.0])
        fro = np.linalg.norm(k, "fro") * 1.001
        row = np.max(np.sum(np.abs(k), axis=1)) * 1.001
        trace = np.trace(k) * 1.001
        self.assertGreaterEqual(fro, np.linalg.eigvalsh(k)[-1])
        self.assertGreaterEqual(row, np.linalg.eigvalsh(k)[-1])
        self.assertGreaterEqual(trace, np.linalg.eigvalsh(k)[-1])
        for upper in (fro, row, trace):
            self.assertIn("plan_index", choose(k, upper))

    def test_uncovered_is_explicit(self):
        k = np.eye(4)
        result = choose(k, 1e9)
        self.assertIsNone(result["plan_index"])


if __name__ == "__main__":
    unittest.main()
