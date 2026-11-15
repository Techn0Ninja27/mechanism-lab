import unittest

import torch

from generation.search_sources import (
    autoregressive_sets,
    canonicalize,
    hybrid_sets,
    uniform_exhaustive,
    uniform_sets,
)


class _Mapping:
    bos_token_id = 0
    eos_token_id = 0

    def token_ids_to_cluster_ids(self, token_ids):
        return [value - 1 for value in token_ids if value != self.bos_token_id]


class _TinyModel(torch.nn.Module):
    def forward(self, input_ids, attention_mask=None):
        # Token selection follows sequence position, giving the canonical
        # atom set (0, 1, 2) under top_k=1; EOS (0) is suppressed upstream.
        result = torch.zeros((input_ids.shape[0], input_ids.shape[1], 4), device=input_ids.device)
        for position in range(input_ids.shape[1]):
            result[:, position, min(position + 1, 3)] = 5.0
        return result


class _FailingModel(_TinyModel):
    def forward(self, input_ids, attention_mask=None):
        raise RuntimeError("intentional test failure")


class SearchSourceTests(unittest.TestCase):
    def test_canonicalize_dedupes_and_filters(self):
        self.assertEqual(canonicalize([[3, 1, 2], [2, 1, 3], [1, 1, 2]], 3), ((1, 2, 3),))

    def test_uniform_is_reproducible_and_budgeted(self):
        a = uniform_sets(range(8), num_draws=100, k=3, seed=17)
        b = uniform_sets(range(8), num_draws=100, k=3, seed=17)
        self.assertEqual(a, b)
        self.assertEqual(a.attempted_draws, 100)
        self.assertLessEqual(a.unique_sets, 100)

    def test_exhaustive_count(self):
        pool = uniform_exhaustive(range(6), k=3)
        self.assertEqual(pool.unique_sets, 20)
        self.assertEqual(pool.sets[0], (0, 1, 2))
        with self.assertRaises(ValueError):
            uniform_exhaustive(range(3), k=1)

    def test_hybrid_preserves_matched_budget(self):
        # No model needed for an all-uniform hybrid; useful as a boundary test.
        pool = hybrid_sets(None, None, range(8), num_draws=50, k=3, seed=3, autoregressive_fraction=0)
        self.assertEqual(pool.attempted_draws, 50)
        self.assertEqual(pool.source, "hybrid(ar=0.000)")

    def test_autoregressive_is_deterministic_and_preserves_rng(self):
        mapping = _Mapping()
        model = _TinyModel()
        torch.manual_seed(901)
        expected = torch.get_rng_state()
        pool = autoregressive_sets(model, mapping, num_draws=4, k=3, seed=7, top_k=1)
        self.assertTrue(torch.equal(torch.get_rng_state(), expected))
        again = autoregressive_sets(model, mapping, num_draws=4, k=3, seed=7, top_k=1)
        self.assertEqual(pool, again)
        self.assertEqual(pool.attempted_draws, 4)
        self.assertEqual(pool.sets, ((0, 1, 2),))

    def test_autoregressive_restores_rng_on_failure(self):
        torch.manual_seed(123)
        expected = torch.get_rng_state()
        with self.assertRaises(RuntimeError):
            autoregressive_sets(_FailingModel(), _Mapping(), num_draws=2, k=3, seed=99)
        self.assertTrue(torch.equal(torch.get_rng_state(), expected))

    def test_hybrid_actual_model_half_budget(self):
        pool = hybrid_sets(
            _TinyModel(), _Mapping(), range(4), num_draws=10, k=3,
            seed=11, autoregressive_fraction=0.5, top_k=1,
        )
        self.assertEqual(pool.attempted_draws, 10)
        self.assertEqual(pool.generated_records, 10)
        self.assertTrue(pool.unique_sets)

    def test_hybrid_rejects_empty_budget(self):
        with self.assertRaises(ValueError):
            hybrid_sets(None, None, range(4), num_draws=0, k=3, seed=1)


if __name__ == "__main__":
    unittest.main()
