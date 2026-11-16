import unittest
import numpy as np
from generation.mask_embedding import embed_mask


class MaskEmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.mask = np.array([[0, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=float)

    def test_finite_and_deterministic(self):
        a = embed_mask(self.mask, seed=16021)
        b = embed_mask(self.mask, seed=16021)
        self.assertTrue(np.isfinite(a).all())
        np.testing.assert_array_equal(a, b)

    def test_row_and_column_permutation_equivariance(self):
        rows = [2, 0, 1]
        cols = [2, 0, 1]
        np.testing.assert_allclose(embed_mask(self.mask[rows], seed=16023), embed_mask(self.mask, seed=16023)[rows])
        np.testing.assert_allclose(embed_mask(self.mask[:, cols], seed=16023), embed_mask(self.mask, seed=16023)[:, cols])

    def test_batch_composition_invariant(self):
        full = embed_mask(np.vstack([self.mask, [[1, 0, 1]]]), seed=16027)
        part = embed_mask(self.mask, seed=16027)
        np.testing.assert_array_equal(part, full[:3])

    def test_constant_columns_and_binary_distinction(self):
        m = np.array([[0, 0], [0, 1], [0, 1]], dtype=float)
        e = embed_mask(m, seed=16021)
        self.assertTrue(np.isfinite(e).all())
        self.assertGreater(np.linalg.norm(e[0, 1] - e[1, 1]), 0.0)
