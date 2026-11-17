import unittest
import torch
from run_arm import eval_batches, evaluate, load_frozen, optimizer_signature


class HeldoutTests(unittest.TestCase):
    def test_eval_windows_are_disjoint_and_tail_explicit(self):
        text = "".join(chr(33 + (i % 90)) for i in range(1000))
        batches, spec = eval_batches(text, seq=4, batch=2)
        self.assertTrue(spec["nonoverlapping"])
        self.assertEqual(spec["candidate_windows"], 200)
        self.assertEqual(spec["used_windows"], 200)
        self.assertEqual(spec["tail_windows"], 0)
        self.assertEqual(len(batches), 100)
        self.assertEqual(int(batches[0][0][0, 0]), ord("!"))
        self.assertEqual(int(batches[1][0][0, 0]), ord("!") + 10)

    def test_tail_not_padded(self):
        _, spec = eval_batches("b" * 11, seq=4, batch=2)
        self.assertEqual(spec["candidate_windows"], 2)
        self.assertEqual(spec["used_windows"], 2)
        self.assertEqual(spec["tail_windows"], 0)

    def test_evaluation_preserves_model_and_optimizer_state(self):
        cp = load_frozen()
        device = torch.device("cpu")
        model, optimizer, _ = cp._make_arm(16, device, 0, "newton_cholesky", 32)
        for module in model.matrix_modules():
            module.capture_input = True
        before_model = optimizer_signature(optimizer)
        before_train = model.training
        batches, _ = eval_batches("abcdef" * 40, seq=4, batch=2)
        result = evaluate(model, batches, device, optimizer)
        self.assertTrue(torch.isfinite(torch.tensor(result["mean_cross_entropy"])))
        self.assertEqual(before_model, optimizer_signature(optimizer))
        self.assertEqual(before_train, model.training)
        self.assertTrue(all(module.capture_input for module in model.matrix_modules()))


if __name__ == "__main__":
    unittest.main()
