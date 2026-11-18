"""Regression for forecast time units, independently checked by Codex."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

spec = importlib.util.spec_from_file_location('headroom_horizon', Path(__file__).with_name('run.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class HorizonTest(unittest.TestCase):
    def test_next_sync_forecast_is_interpolated_not_recurred_per_epoch(self):
        trajectory = np.repeat(np.arange(29, dtype=np.float32)[:, None, None], 2, axis=1)
        # Features are [intercept, last observation]; next sync is four epochs away.
        coef = np.array([[4.0], [1.0]])
        with patch.object(m, '_fit_global_ar', return_value=coef):
            result = m._ridge_arrays(trajectory, np.array([0, 1]), [(0, 1), (1, 0)], lag=1, ridge=.1)
        expected = [21, 22, 23, 25, 26, 27]
        self.assertEqual(len(result['preds']), len(expected))
        for prediction, value in zip(result['preds'], expected):
            np.testing.assert_array_equal(prediction, np.full((2, 1), value, dtype=np.float32))
        self.assertEqual(result['metric'], 0.0)


if __name__ == '__main__':
    unittest.main()
