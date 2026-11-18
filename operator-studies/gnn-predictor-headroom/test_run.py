from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("headroom", STUDY / "run.py")
headroom = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(headroom)


class HeadroomTests(unittest.TestCase):
    def setUp(self):
        self.parts = np.asarray([0, 1, 0, 1], dtype=np.int64)
        self.pairs = [(0, 1), (1, 0), (2, 1), (3, 0)]

    def test_constant_process_has_zero_stale_error(self):
        trajectory = np.ones((32, 4, 3), dtype=np.float32)
        row = headroom._result_for_trajectory(trajectory, self.parts, self.pairs)
        self.assertLess(row["damped"]["0.2"]["0.2"]["predictor"], 1e-7)
        self.assertLess(min(row["continuous_mix_curve"]["metrics"]), 1e-7)
        self.assertTrue(all(np.isfinite(v["metric"]) for v in row["ridge"].values()))

    def test_linear_process_is_finite(self):
        t = np.arange(40, dtype=np.float32)[:, None, None]
        d = np.arange(3, dtype=np.float32)[None, None, :]
        trajectory = t + d + np.zeros((40, 4, 1), dtype=np.float32)
        row = headroom._result_for_trajectory(trajectory, self.parts, self.pairs)
        self.assertTrue(np.isfinite(row["damped"]["0.2"]["0.2"]["predictor"]))
        self.assertTrue(np.isfinite(row["continuous_mix_curve"]["metrics"]).all())

    def test_hidden_intermediate_truth_does_not_change_predictions(self):
        rng = np.random.default_rng(7)
        trajectory = rng.normal(size=(40, 4, 3)).astype(np.float32)
        altered = trajectory.copy()
        for t in range(headroom.WARMUP + 1, len(altered)):
            if (t - headroom.WARMUP) % headroom.SYNC:
                altered[t] += 1000.0
        a = headroom._damped_arrays(trajectory, self.parts, self.pairs, beta=.2)
        b = headroom._damped_arrays(altered, self.parts, self.pairs, beta=.2)
        for key in ("preds", "holds"):
            self.assertEqual(len(a[key]), len(b[key]))
            for x, y in zip(a[key], b[key]):
                np.testing.assert_array_equal(x, y)

    def test_future_prefix_does_not_change_prior_predictions(self):
        rng = np.random.default_rng(8)
        trajectory = rng.normal(size=(40, 4, 3)).astype(np.float32)
        altered = trajectory.copy(); altered[headroom.WARMUP + headroom.SYNC + 1:] += 500.0
        a = headroom._damped_arrays(trajectory, self.parts, self.pairs, beta=.2)
        b = headroom._damped_arrays(altered, self.parts, self.pairs, beta=.2)
        for x, y in zip(a["preds"][:headroom.SYNC - 1], b["preds"][:headroom.SYNC - 1]):
            np.testing.assert_array_equal(x, y)

    def test_imported_replay_baseline_is_exact(self):
        rng = np.random.default_rng(9)
        trajectory = rng.normal(size=(40, 4, 3)).astype(np.float32)
        ours = headroom._damped_arrays(trajectory, self.parts, self.pairs, beta=.2)
        imported = headroom.screen.replay_gate(trajectory, self.parts, boundary_pairs=self.pairs,
                                               sync=headroom.SYNC, warmup=headroom.WARMUP, eta=1.0)
        self.assertAlmostEqual(ours["predictor"], imported["aggregate"]["predictor"], places=6)
        self.assertAlmostEqual(ours["hold"], imported["aggregate"]["hold"], places=6)

    def test_ridge_uses_sync_history_not_hidden_truth(self):
        rng = np.random.default_rng(11)
        trajectory = rng.normal(size=(40, 4, 3)).astype(np.float32)
        altered = trajectory.copy()
        for t in range(headroom.WARMUP + 1, len(altered)):
            if (t - headroom.WARMUP) % headroom.SYNC:
                altered[t] += 900.0
        a = headroom._ridge_arrays(trajectory, self.parts, self.pairs, lag=1, ridge=.1)
        b = headroom._ridge_arrays(altered, self.parts, self.pairs, lag=1, ridge=.1)
        for x, y in zip(a["preds"], b["preds"]):
            np.testing.assert_array_equal(x, y)


if __name__ == "__main__":
    unittest.main()
