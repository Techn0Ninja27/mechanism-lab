import unittest
import numpy as np

from generation.mechanism_preflight import (
    check_anchor_volatile_sets, check_feature_scaling_then_standardization,
    check_mask_cluster_capacity, check_mask_information_claim,
)


class MechanismPreflightTests(unittest.TestCase):
    def test_positive_scale_cancels_and_zero_masks(self):
        report = check_feature_scaling_then_standardization([2.0, 7.0, 0.0])
        self.assertEqual(report["positive_scale_indices"], [0, 1])
        self.assertEqual(report["zero_scale_indices"], [2])
        self.assertTrue(report["unsupported_mechanism"])
        self.assertTrue(any(f["code"] == "zero_scale_masks_feature" for f in report["findings"]))

    def test_numeric_standardization_cancellation(self):
        x=np.array([[1.,4.],[2.,8.],[4.,3.],[6.,9.]])
        weights=np.array([.25,7.])
        normalize=lambda z:(z-z.mean(axis=0))/z.std(axis=0)
        np.testing.assert_allclose(normalize(x),normalize(x*weights),rtol=1e-12,atol=1e-12)
        self.assertTrue(check_feature_scaling_then_standardization(weights)["unsupported_mechanism"])

    def test_disjoint_features_can_reconstruct(self):
        anchor=np.arange(10,dtype=float);target=2*anchor+3
        coef=np.linalg.lstsq(np.c_[anchor[:6],np.ones(6)],target[:6],rcond=None)[0]
        np.testing.assert_allclose(np.c_[anchor[6:],np.ones(4)]@coef,target[6:])
        self.assertTrue(check_anchor_volatile_sets(["anchor"],["target"])["reconstruction_possible"])
        self.assertFalse(check_anchor_volatile_sets(["target"],["target"])["reconstruction_possible"])

    def test_mask_capacity(self):
        report = check_mask_cluster_capacity(np.array([[0, 0], [0, 0], [1, 0]]), 3)
        self.assertEqual(report["unique_mask_rows"], 2)
        self.assertTrue(report["nonidentifiable"])

    def test_anchor_volatile_overlap(self):
        self.assertFalse(check_anchor_volatile_sets([], ["x"])["reconstruction_possible"])
        self.assertTrue(check_anchor_volatile_sets(["a"], ["b"])["reconstruction_possible"])
        self.assertTrue(check_anchor_volatile_sets(["a", "b"], ["b"])["reconstruction_possible"])

    def test_mask_claim_flags_native_route(self):
        report = check_mask_information_claim(explicitmaskremoved=True, native_nan_routing=True, mask_gated_selection=False)
        self.assertTrue(report["mask_information_preserved"])
        self.assertTrue(report["unsupported_mechanism"])
        clean = check_mask_information_claim(explicitmaskremoved=True, native_nan_routing=False, mask_gated_selection=False)
        self.assertFalse(clean["unsupported_mechanism"])


if __name__ == "__main__":
    unittest.main()
