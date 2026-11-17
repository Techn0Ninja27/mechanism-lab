import unittest

import torch

from pera_algebra import (
    delta_expanded,
    delta_sum,
    expanded_width,
    nonlinear_component,
    parameter_count,
    rank_bound,
)


class PeraAlgebraTests(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(17)

    def test_factor_and_sum_match_with_gradients(self):
        B = torch.randn(7, 3, requires_grad=True)
        A = torch.randn(3, 5, requires_grad=True)
        h = torch.randn(6, requires_grad=True)
        left = delta_expanded(B, A, h)
        right = delta_sum(B, A, h)
        self.assertTrue(torch.allclose(left, right, atol=1e-12, rtol=1e-12))
        loss_left = (left.square()).sum()
        loss_right = (right.square()).sum()
        grads_left = torch.autograd.grad(loss_left, (B, A, h), retain_graph=True)
        grads_right = torch.autograd.grad(loss_right, (B, A, h))
        for x, y in zip(grads_left, grads_right):
            self.assertTrue(torch.allclose(x, y, atol=1e-11, rtol=1e-11))

    def test_zero_h_is_lora_and_special_coefficients_recover_square(self):
        B = torch.randn(8, 3)
        A = torch.randn(3, 6)
        h0 = torch.zeros(6)
        self.assertTrue(torch.allclose(delta_expanded(B, A, h0), B @ A))
        lam = 0.37
        # Pair order is (0,0),(0,1),(0,2),(1,1),(1,2),(2,2).
        h = torch.tensor([lam, 2 * lam, 2 * lam, lam, 2 * lam, lam])
        expected = B @ A + lam * ((B @ A) * (B @ A))
        self.assertTrue(torch.allclose(delta_expanded(B, A, h), expected, atol=1e-12, rtol=1e-12))

    def test_shapes_counts_and_rank_bound(self):
        for rank in (1, 2, 4):
            m, n = 16, 12
            B = torch.randn(m, rank)
            A = torch.randn(rank, n)
            h = torch.randn(rank * (rank + 1) // 2)
            delta = delta_expanded(B, A, h)
            self.assertEqual(expanded_width(rank), rank + rank * (rank + 1) // 2)
            self.assertEqual(parameter_count(m, n, rank), rank * (m + n) + rank * (rank + 1) // 2)
            self.assertLessEqual(int(torch.linalg.matrix_rank(delta)), rank_bound(m, n, rank))

    def test_zero_initialization_has_a_gradient_but_not_b_or_h(self):
        B = torch.randn(6, 2, requires_grad=True)
        A = torch.zeros(2, 5, requires_grad=True)
        h = torch.zeros(3, requires_grad=True)
        target = torch.randn(6, 5)
        loss = (delta_expanded(B, A, h) - target).square().mean()
        gB, gA, gh = torch.autograd.grad(loss, (B, A, h))
        self.assertLess(float(gB.abs().max()), 1e-14)
        self.assertLess(float(gh.abs().max()), 1e-14)
        self.assertGreater(float(gA.abs().max()), 1e-10)

        # After one ordinary A update, the nonlinear coefficients can receive
        # gradient. This is a conditional observation, not a claim of universal
        # nonzero gradients for every target or initialization.
        with torch.no_grad():
            A -= 0.01 * gA
        loss2 = (delta_expanded(B, A, h) - target).square().mean()
        gh2 = torch.autograd.grad(loss2, h)[0]
        self.assertGreater(float(gh2.abs().max()), 1e-14)


if __name__ == "__main__":
    unittest.main()
