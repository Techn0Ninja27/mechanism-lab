"""The confirmation loop must be the original loop plus read-only measurements."""
import unittest
import numpy as np, torch
import confirm
gap = confirm.gap
X, Y, ADJ = gap.screen.load_cora(gap.DATA); SP = gap.splits(Y)


class ConfirmTests(unittest.TestCase):
    def test_reproduces_original_loop_exactly_with_and_without_window(self):
        for name, R in confirm.REGIMES.items():
            for arm in confirm.CONFIG["arms"]:
                coef = R["lag1_coef"] if arm == "lag1" else [0.0] * (R["layers"] - 1)
                kw = dict(seed=8, layers=R["layers"], parts=R["parts"], lr_drop=R["lr_drop"], arm=arm, sync=R["sync"], coef=coef)
                ref = gap.run(X, Y, ADJ, SP, trace=True, **kw)
                a = confirm.train(X, Y, ADJ, SP, window=confirm.window_epochs(R["sync"]), keep_weights=True, **kw)
                b = confirm.train(X, Y, ADJ, SP, window=None, keep_weights=True, **kw)
                for w0, w1, w2 in zip(ref["trace"]["weights"], a["weights"], b["weights"]):
                    self.assertEqual(float(np.abs(w0 - w1).max()), 0.0); self.assertEqual(float(np.abs(w0 - w2).max()), 0.0)
                self.assertEqual(a["post_update_exact_eval_120"], ref["eval"]["120"]); self.assertEqual(a["post_update_exact_val_120"], ref["val"]["120"])

    def test_window_and_g1_forward_identity(self):
        R = confirm.REGIMES["moderate_lead"]; win = confirm.window_epochs(R["sync"])
        self.assertEqual(win, list(range(88, 120)))
        kw = dict(seed=8, layers=3, parts=4, lr_drop=False, sync=16, coef=[0.0, 0.0], window=win)
        g1 = confirm.train(X, Y, ADJ, SP, arm="G1", **kw); h = confirm.train(X, Y, ADJ, SP, arm="hold", **kw)
        self.assertEqual([w["t"] for w in g1["window"]], win)
        for w in g1["window"]: self.assertAlmostEqual(w["stale_eval"], w["exact_eval"], places=9)
        for w in h["window"]:
            if w["phase"] == 0: self.assertAlmostEqual(w["stale_eval"], w["exact_eval"], places=9)   # sync epoch: estimate is exact
        self.assertTrue(any(abs(w["stale_eval"] - w["exact_eval"]) > 0 for w in h["window"] if w["phase"] != 0))

    def test_pre_update_exact_matches_previous_post_update(self):
        """Exact accuracy before the update at epoch 119 uses the weights after step 118."""
        kw = dict(seed=8, layers=2, parts=4, lr_drop=False, arm="hold", sync=4, coef=[0.0])
        a = confirm.train(X, Y, ADJ, SP, window=[119], **kw); b = confirm.train(X, Y, ADJ, SP, epochs=119, **kw)
        self.assertAlmostEqual(a["window"][0]["exact_eval"], b["post_update_exact_eval_120"], places=9)


if __name__ == "__main__": unittest.main()
