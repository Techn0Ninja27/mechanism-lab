import unittest
import torch

from generation.causal_program import CausalState, compile_program, validate_program


def var(name): return {"var": name}
def const(value): return {"const": value}
def op(name, *args): return {"op": name, "args": list(args)}


IDENTITY = {"predict": var("last_sync"), "update": var("memory")}


class CausalProgramTests(unittest.TestCase):
    def test_observe_then_predict_and_snapshot_update(self):
        program = compile_program({
            "predict": op("add", var("last_sync"), var("memory")),
            "update": op("add", var("memory"), const(1)),
        })
        state = CausalState((2, 3)); state.observe(torch.ones(2, 3), 0, program)
        # Predict and update see the same pre-step memory (ones after observe).
        torch.testing.assert_close(state.predict(2, program), torch.full((2, 3), 2.0))
        torch.testing.assert_close(state.memory, torch.full((2, 3), 2.0))

    def test_linear_history_program(self):
        program = compile_program({
            "predict": op("add", var("last_sync"), op("sub", var("last_sync"), var("previous_sync"))),
            "update": var("memory"),
        })
        state = CausalState((1, 2)); state.observe(torch.tensor([[2., 4.]]), 0, program)
        state.observe(torch.tensor([[3., 7.]]), 4, program)
        torch.testing.assert_close(state.predict(8, program), torch.tensor([[4., 10.]]))

    def test_future_prefix_does_not_change_prediction(self):
        program = compile_program(IDENTITY)
        a, b = CausalState((1, 2)), CausalState((1, 2))
        a.observe(torch.tensor([[1., 2.]]), 0, program); b.observe(torch.tensor([[1., 2.]]), 0, program)
        torch.testing.assert_close(a.predict(1, program), b.predict(1, program))

    def test_ops_broadcast_norm_mean_clip_and_divsafe(self):
        p = compile_program({"predict": op("clip", op("divsafe", var("last_sync"), const(0)), const(-1), const(1)), "update": var("memory")})
        s = CausalState((2, 3)); s.observe(torch.tensor([[2., -3., .5], [4., 5., 6.]]), 0, p)
        torch.testing.assert_close(s.predict(1, p), torch.tensor([[1., -1., 1.], [1., 1., 1.]]))
        q = compile_program({"predict": op("norm", var("last_sync")), "update": op("mean", var("last_sync"))})
        s = CausalState((2, 3)); s.observe(torch.ones(2, 3), 0, q)
        torch.testing.assert_close(s.predict(1, q), torch.full((2, 3), 3**.5))

    def test_ast_is_strict_and_bounded(self):
        for bad in (
            {"predict": {"var": "labels"}, "update": var("memory")},
            {"predict": {"var": "memory", "extra": 1}, "update": var("memory")},
            {"predict": const(float("nan")), "update": var("memory")},
            {"predict": op("clip", var("memory"), var("age"), const(1)), "update": var("memory")},
            {"predict": const(5), "update": var("memory")},
        ):
            with self.assertRaises(ValueError): validate_program(bad)
        deep = const(1)
        for _ in range(7): deep = op("neg", deep)
        with self.assertRaises(ValueError): validate_program({"predict": deep, "update": var("memory")})

    def test_runtime_rejects_bad_time_shape_and_observation(self):
        p = compile_program(IDENTITY); s = CausalState((2, 2))
        with self.assertRaises(ValueError): s.predict(1, p)
        with self.assertRaises(ValueError): s.observe(torch.ones(2, 3), 0, p)
        s.observe(torch.ones(2, 2), 2, p)
        with self.assertRaises(ValueError): s.observe(torch.ones(2, 2), 2, p)
        with self.assertRaises(ValueError): s.predict(2, p)

    def test_nonfinite_runtime_result_rejected(self):
        p = compile_program({"predict": op("divsafe", var("last_sync"), const(0)), "update": var("memory")})
        s = CausalState((1, 1)); s.observe(torch.tensor([[1.]]), 0, p)
        self.assertTrue(torch.isfinite(s.predict(1, p)).all())

    def test_compiled_ast_is_immutable_and_scalar_materializes(self):
        raw = {"predict": {"const": 2}, "update": {"const": 3}}
        p = compile_program(raw)
        raw["predict"]["const"] = 99
        s = CausalState((2, 2)); s.observe(torch.ones(2, 2), 0, p)
        out = s.predict(1, p)
        self.assertEqual(tuple(out.shape), (2, 2)); self.assertTrue(torch.all(out == 2))
        with self.assertRaises(TypeError): p.predict["const"] = 5

    def test_operator_nodes_are_counted_per_expression(self):
        leaf = var("memory")
        expr = leaf
        for _ in range(5): expr = op("neg", expr)
        self.assertEqual(compile_program({"predict": expr, "update": expr}).node_count, 12)
        wide = op("add", *[leaf, leaf])
        # A 33-node chain is rejected even though the other expression is tiny.
        for _ in range(32): wide = op("neg", wide)
        with self.assertRaises(ValueError): compile_program({"predict": wide, "update": leaf})

    def test_observe_is_atomic_and_prediction_actions_are_monotone(self):
        failing = compile_program({"predict": var("last_sync"), "update": op("norm", var("age"))})
        identity = compile_program(IDENTITY)
        s = CausalState((1, 1)); s.observe(torch.ones(1, 1), 0, identity)
        before = (s.memory.clone(), s.last_sync.clone(), s.last_t, s.last_action_t)
        with self.assertRaises(Exception): s.observe(torch.full((1, 1), 2.), 1, failing)
        self.assertEqual(s.last_t, before[2]); self.assertEqual(s.last_action_t, before[3]); torch.testing.assert_close(s.memory, before[0]); torch.testing.assert_close(s.last_sync, before[1])
        s.predict(1, identity)
        with self.assertRaises(ValueError): s.predict(1, identity)
        s.observe(torch.full((1, 1), 3.), 1, identity)

    def test_observations_are_detached_from_autograd(self):
        x = torch.ones(1, 2, requires_grad=True); s = CausalState((1, 2)); s.observe(x, 0, compile_program(IDENTITY))
        self.assertFalse(s.last_sync.requires_grad); self.assertFalse(s.memory.requires_grad)


if __name__ == "__main__": unittest.main()
