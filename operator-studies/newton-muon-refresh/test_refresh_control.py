import unittest
from refresh_control import expected_refreshes


class RefreshProtocolTests(unittest.TestCase):
    def test_full_horizon_counts(self):
        self.assertEqual([expected_refreshes(x) for x in (16, 32, 64)], [8, 4, 2])

    def test_windows_sum_to_full_horizon(self):
        for refresh in (16, 32, 64):
            self.assertEqual(sum(expected_refreshes(refresh, 8 + i * 32, 32) for i in range(4)), expected_refreshes(refresh))

    def test_invalid(self):
        with self.assertRaises(ValueError):
            expected_refreshes(0)
