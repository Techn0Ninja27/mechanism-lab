import unittest

import numpy as np

import importlib.util
HAS_SKLEARN=importlib.util.find_spec('sklearn') is not None
if HAS_SKLEARN:
    from scripts.missingness_shift_fixture import SEEDS, evaluate, load_fixture


@unittest.skipUnless(HAS_SKLEARN, "Requires benchmark environment")
class MissingnessShiftFixtureTests(unittest.TestCase):
    def test_validation_hides_test_arrays_and_labels(self):
        fixture = load_fixture(SEEDS[0], "validation")
        self.assertIsNone(fixture.test_x)
        self.assertIsNone(fixture.test_y)
        self.assertEqual(fixture.train_x.shape, (3000, 20))
        self.assertEqual(fixture.train_y.shape, (3000,))

    def test_final_is_explicit_and_has_hidden_conditions(self):
        fixture = load_fixture(SEEDS[0], "final")
        self.assertIsNone(fixture.test_x)
        self.assertEqual(fixture.test_y.shape, (1000,))
        self.assertEqual(set(fixture.final_cases), {"clean", "mcar_0.2", "mnar_own_feature", "mnar_reversed_sign", "block_mcar_2_8", "block_mcar_8_14"})
        self.assertIn("mnar_own_feature", fixture.metadata["final_conditions"])
        self.assertIn("mnar_reversed_sign", fixture.metadata["final_conditions"])
        self.assertIn("block_mcar_2_8", fixture.metadata["final_conditions"])
        self.assertIn("block_mcar_8_14", fixture.metadata["final_conditions"])

    def test_same_seed_is_reproducible_and_anchors_are_observed(self):
        first = load_fixture(SEEDS[1], "validation")
        second = load_fixture(SEEDS[1], "validation")
        self.assertTrue(np.array_equal(np.isnan(first.train_x), np.isnan(second.train_x)))
        np.testing.assert_array_equal(np.nan_to_num(first.train_x, nan=0.0), np.nan_to_num(second.train_x, nan=0.0))
        for x, _ in first.validation_cases.values():
            self.assertFalse(np.isnan(x[:, :2]).any())
        self.assertFalse(np.isnan(first.train_x[:, :2]).any())

    def test_validation_cases_have_expected_names_and_labels(self):
        fixture = load_fixture(SEEDS[2], "validation")
        self.assertEqual(set(fixture.validation_cases), {"clean", "mcar_0.2", "mcar_0.4", "mar_anchor_x0"})
        for x, y in fixture.validation_cases.values():
            self.assertEqual(x.shape, (1000, 20))
            self.assertEqual(y.shape, (1000,))

    def test_metrics(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.2, 0.8, 0.9])
        self.assertGreater(evaluate(y, p, "average_precision"), 0.9)
        self.assertLess(evaluate(y, p, "brier"), 0.1)
        with self.assertRaises(ValueError):
            evaluate(y, p, "unknown")


if __name__ == "__main__":
    unittest.main()
