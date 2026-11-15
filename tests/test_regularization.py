import tempfile
import unittest
from pathlib import Path

from generation.token_mapping import TokenMapping
from scripts.regularization_ablation import run_condition, unigram_nll


class RegularizationTests(unittest.TestCase):
    def test_unigram_baseline_is_finite(self):
        self.assertGreater(unigram_nll([[3, 0, 1, 3]], [[3, 1, 0, 3]], 4), 0)

    def test_initial_checkpoint_exists_even_without_improvement(self):
        mapping = TokenMapping({
            "bos_token_id": 3, "eos_token_id": 3, "n_clusters": 3,
            "cluster_to_token_id": {"0": 0, "1": 1, "2": 2},
            "token_id_to_cluster": {"0": 0, "1": 1, "2": 2},
        })
        with tempfile.TemporaryDirectory() as tmp:
            result = run_condition(
                [(0, 1), (1, 2)], [[3, 0, 1, 3]], [[3, 1, 2, 3]],
                mapping, seed=3, dropout=0.0, weight_decay=0.01,
                steps=100, output_dir=Path(tmp), max_seq_len=8,
            )
            self.assertTrue((Path(tmp) / "model.pt").exists())
            self.assertIn("unigram_test_nll", result)


if __name__ == "__main__":
    unittest.main()
