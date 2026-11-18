import unittest
import numpy as np, torch
import clamp
confirm, gap = clamp.confirm, clamp.gap
X, Y, ADJ = gap.screen.load_cora(gap.DATA); SP = gap.splits(Y); R = clamp.R


class ClampTests(unittest.TestCase):
    def test_unprojected_arms_reproduce_confirmation_loop_exactly(self):
        win = confirm.window_epochs(R["sync"])
        for arm in ("sni", "lag1"):
            coef = R["lag1_coef"] if arm == "lag1" else [0.0, 0.0]
            a = confirm.train(X, Y, ADJ, SP, seed=8, layers=3, parts=4, lr_drop=False, arm=arm, sync=16, coef=coef, window=win, keep_weights=True)
            b = clamp.train(X, Y, ADJ, SP, seed=8, arm=arm, window=win, keep_weights=True)
            for w0, w1 in zip(a["weights"], b["weights"]): self.assertEqual(float(np.abs(w0 - w1).max()), 0.0)
            self.assertEqual(a["window"], b["window"])

    def test_projection_is_identity_on_nonnegative_forecasts_and_modes_differ_otherwise(self):
        g = torch.Generator().manual_seed(0)
        pos = [torch.rand(5, 3, generator=g) * 0.01 + 10 + 0.05 * t for t in range(60)]            # forecasts stay positive
        mixed = [torch.randn(5, 3, generator=g).cumsum(0) * (1 + 0.2 * t) for t in range(60)]       # forecasts go negative
        for base, modes in (("sni", (None, "emit", "state")), ("lag1", (None, "emit"))):
            outs = {m: [] for m in modes}; outs_mixed = {m: [] for m in modes}
            for m in modes:
                e1, e2 = clamp.Projected(base, m, 16, 0.8), clamp.Projected(base, m, 16, 0.8)
                for t in range(60): outs[m].append(e1(pos[t], t)); outs_mixed[m].append(e2(mixed[t], t))
                if m is not None: self.assertTrue(all(float(o.min()) >= 0 for t, o in enumerate(outs_mixed[m]) if t > 20 and (t - 20) % 16 != 0))
            for m in modes[1:]:
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(outs[None], outs[m])))
                self.assertFalse(all(torch.equal(a, b) for a, b in zip(outs_mixed[None], outs_mixed[m])))
            if base == "sni": self.assertFalse(all(torch.equal(a, b) for a, b in zip(outs_mixed["emit"], outs_mixed["state"])))

    def test_emit_keeps_raw_recurrence_state(self):
        """In emit mode the wrapped recurrence state must equal the unprojected one given the same inputs."""
        g = torch.Generator().manual_seed(1); seq = [torch.randn(4, 3, generator=g).cumsum(0) * (1 + 0.2 * t) for t in range(60)]
        a, b = clamp.Projected("sni", None, 16, 0.0), clamp.Projected("sni", "emit", 16, 0.0)
        for t in range(60):
            a(seq[t], t); b(seq[t], t)
            self.assertTrue(torch.equal(a.r.s_last, b.r.s_last) and torch.equal(a.r.s_num, b.r.s_num))

    def test_estimators_ignore_exact_between_syncs_and_log_only_stale_epochs(self):
        g = torch.Generator().manual_seed(2); seq = [torch.randn(4, 3, generator=g) for _ in range(120)]
        for arm, (base, mode) in clamp.ARMS.items():
            e1, e2 = clamp.Projected(base, mode, 16, 0.8), clamp.Projected(base, mode, 16, 0.8)
            for t, v in enumerate(seq):
                s = t <= 20 or (t - 20) % 16 == 0
                self.assertTrue(torch.equal(e1(v, t), e2(v if s else v * 0 + 1e6, t)), (arm, t))
            self.assertEqual(len(e1.log), 93); self.assertTrue(all(t > 20 and (t - 20) % 16 != 0 for t, _, _ in e1.log))


    def test_log_describes_raw_forecast_not_emitted_value(self):
        g = torch.Generator().manual_seed(3); seq = [torch.randn(6, 4, generator=g).cumsum(0) * (1 + 0.3 * t) for t in range(60)]
        for arm, (base, mode) in clamp.ARMS.items():
            e = clamp.Projected(base, mode, 16, 0.8); outs = {}
            for t, v in enumerate(seq): outs[t] = e(v, t)
            shares = {t: a for t, a, _ in e.log}
            self.assertGreater(max(shares.values()), 0.0)                                # raw forecasts do go negative here
            for t, share in shares.items():
                emitted_neg = float((outs[t] < 0).float().mean())
                if mode is None: self.assertAlmostEqual(emitted_neg, share, places=6)   # unprojected: log equals emitted
                else: self.assertEqual(emitted_neg, 0.0)                                 # projected: emitted is clean, log still records raw
        raw, emit = clamp.Projected("sni", None, 16, 0.0), clamp.Projected("sni", "emit", 16, 0.0)
        for t, v in enumerate(seq): raw(v, t); emit(v, t)
        self.assertEqual(raw.log, emit.log)                                              # same raw recurrence, same raw forecasts

    def test_baseline_reproduction_gate_and_reading_labels(self):
        a = {"post_update_exact_eval_120": 0.7, "final_train_loss": 0.1, "window": [{"t": 1, "exact_eval": 0.7}]}
        self.assertTrue(clamp.same_run(a, dict(a)))
        self.assertFalse(clamp.same_run(a, {**a, "final_train_loss": 0.1000001}))
        self.assertFalse(clamp.same_run(a, {**a, "window": [{"t": 1, "exact_eval": 0.701}]}))
        self.assertEqual(clamp.reading(0.3, 0.05, 0.5, 2.0), "within_band_point_estimate")
        self.assertEqual(clamp.reading(-0.4, 1.0, 0.5, 2.0), "within_band_point_estimate")
        self.assertEqual(clamp.reading(1.5, 1.0, 0.5, 2.0), "inconclusive")
        self.assertEqual(clamp.reading(-1.5, 1.0, 0.5, 2.0), "inconclusive")
        self.assertEqual(clamp.reading(1.5, 0.5, 0.5, 2.0), "better")
        self.assertEqual(clamp.reading(-30.0, 2.0, 0.5, 2.0), "worse")


if __name__ == "__main__": unittest.main()
