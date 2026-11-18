import hashlib
import json
import unittest

import torch

from generation.campaign2_random_controls import random_inventory
from generation.causal_program import CausalState, compile_program


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class Campaign2RandomControlTests(unittest.TestCase):
    def test_shape_determinism_and_compilation(self):
        a, b = random_inventory(), random_inventory()
        self.assertEqual(a, b)
        self.assertEqual(len(a), 24)
        self.assertEqual(len({row["id"] for row in a}), 24)
        self.assertEqual(len({row["program_sha256"] for row in a}), 24)
        for row in a:
            self.assertEqual(row["program_sha256"], digest(row["program"]))
            compile_program(row["program"])
            self.assertEqual(row["arm"], "random")

    def test_bounded_non_hold_predictions_and_finite_runtime(self):
        last = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
        previous = torch.tensor([[0.2, -1.0], [2.0, 1.5]])
        outputs = []
        for row in random_inventory():
            program = compile_program(row["program"])
            state = CausalState((2, 2))
            state.observe(previous, 0, program)
            state.observe(last, 1, program)
            pred = state.predict(2, program)
            self.assertTrue(torch.isfinite(pred).all().item())
            delta = (pred - last).abs()
            self.assertTrue(torch.all(delta <= last.abs() * 0.1 + 1e-6).item())
            self.assertGreater(float(delta.max()), 0.0)
            outputs.append(tuple(round(float(x), 7) for x in pred.flatten()))
        self.assertEqual(len(set(outputs)), 24)

    def test_memory_groups_are_not_permanently_zero(self):
        for row in random_inventory():
            program = compile_program(row["program"])
            state = CausalState((1, 2))
            state.observe(torch.tensor([[2.0, -1.0]]), 0, program)
            self.assertGreater(float(state.memory.abs().sum()), 0.0)

    def test_replicate_parameter_is_frozen(self):
        with self.assertRaises(ValueError):
            random_inventory(3)


if __name__ == "__main__":
    unittest.main()
