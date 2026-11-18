"""Semantics tests for gap.py. Run before freezing."""
import unittest
import numpy as np, torch
import gap

X, Y, ADJ = gap.screen.load_cora(gap.DATA); SP = gap.splits(Y)


class GapTests(unittest.TestCase):
    def test_splits_match_screen_and_are_disjoint(self):
        tr, va = gap.screen.split_labels(Y)
        self.assertTrue(np.array_equal(tr, SP[0]) and np.array_equal(va, SP[1]))
        self.assertEqual(len(SP[2]), 1000)
        self.assertFalse(set(SP[0]) & set(SP[1]) or set(SP[0]) & set(SP[2]) or set(SP[1]) & set(SP[2]))

    def test_g0_two_layer_reproduces_original_trajectory(self):
        for drop in (False, True):
            ref, _ = gap.screen.train_trajectory(X, Y, ADJ, 3, drop, epochs=70)
            got = gap.run(X, Y, ADJ, SP, seed=3, arm="G0", lr_drop=drop, epochs=70, trace=True)
            mine = np.stack([h[0] for h in got["trace"]["hidden"]])
            self.assertEqual(float(np.abs(mine - ref).max()), 0.0)

    def test_single_partition_g1_equals_g0(self):
        one = np.zeros(len(Y), dtype=int)
        for layers in (2, 3):
            a = gap.run(X, Y, ADJ, SP, seed=5, arm="G0", layers=layers, epochs=12, trace=True)
            b = gap.run(X, Y, ADJ, SP, seed=5, arm="G1", layers=layers, epochs=12, trace=True, part_vec=one)
            for wa, wb in zip(a["trace"]["weights"], b["trace"]["weights"]):
                self.assertLess(float(np.abs(wa - wb).max()), 1e-6)

    def test_g1_same_forward_but_truncated_gradient(self):
        for layers in (2, 3):
            a = gap.run(X, Y, ADJ, SP, seed=7, arm="G0", layers=layers, epochs=1, trace=True)
            b = gap.run(X, Y, ADJ, SP, seed=7, arm="G1", layers=layers, epochs=1, trace=True)
            self.assertLess(abs(a["final_train_loss"] - b["final_train_loss"]), 1e-6)   # same values
            ga, gb = a["trace"]["grads0"], b["trace"]["grads0"]
            self.assertLess(float(np.abs(ga[-1] - gb[-1]).max()), 1e-6)                 # last layer: no remote gradient path
            self.assertGreater(float(np.abs(ga[0] - gb[0]).max()), 1e-6)                # first layer: path through remote H is cut

    def test_sync_one_equals_g1(self):
        for arm in ("hold", "sni", "lag1"):
            a = gap.run(X, Y, ADJ, SP, seed=9, arm="G1", layers=3, epochs=40, trace=True)
            b = gap.run(X, Y, ADJ, SP, seed=9, arm=arm, layers=3, sync=1, coef=0.7, epochs=40, trace=True)
            for wa, wb in zip(a["trace"]["weights"], b["trace"]["weights"]):
                self.assertEqual(float(np.abs(wa - wb).max()), 0.0)

    def test_hold_uses_last_sync_value_only(self):
        r = gap.run(X, Y, ADJ, SP, seed=11, arm="hold", layers=3, sync=4, epochs=40, trace=True)
        hid, rem = r["trace"]["hidden"], r["trace"]["remote"]
        for t in range(40):
            s = t if t <= gap.WARM else t - ((t - gap.WARM) % 4)
            for k in range(2):
                self.assertEqual(float(np.abs(rem[t][k] - hid[s][k]).max()), 0.0)

    def test_estimator_never_reads_exact_between_syncs(self):
        """Causality: feed garbage as `exact` at non-sync epochs; outputs must not change."""
        g = torch.Generator().manual_seed(0); seq = [torch.randn(6, 3, generator=g) for _ in range(60)]
        for arm in ("hold", "sni", "lag1"):
            for sync in (4, 16):
                e1, e2 = gap.Remote(arm, sync, 0.6), gap.Remote(arm, sync, 0.6)
                for t, v in enumerate(seq):
                    is_sync = t <= gap.WARM or (t - gap.WARM) % sync == 0
                    o1 = e1(v, t); o2 = e2(v if is_sync else v * 0 + 1e6, t)
                    self.assertTrue(torch.equal(o1, o2), (arm, sync, t))

    def test_sni_matches_screen_recurrence(self):
        g = np.random.default_rng(1); a = g.normal(size=(50, 5, 3)).astype(np.float32).cumsum(0)
        last = a[0].copy(); num = np.zeros_like(last); den = np.ones(5, np.float32); ref = {}
        for t in range(1, gap.WARM + 1): last, num, den = gap.screen._observe(last, num, den, a[t], 1)
        for t in range(gap.WARM + 1, 50):
            p = gap.screen._predict(last, num, den, 1)
            if (t - gap.WARM) % 4 == 0: last, num, den = gap.screen._observe(last, num, den, a[t], 1)
            else: ref[t] = p; last, num, den = gap.screen._observe(last, num, den, p, 1)
        e = gap.Remote("sni", 4, 0.0)
        for t in range(50):
            o = e(torch.from_numpy(a[t]), t)
            if t in ref: self.assertLess(float(np.abs(o.numpy() - ref[t]).max()), 1e-4)

    def test_zero_arm_and_lag1_zero_coef(self):
        a = gap.run(X, Y, ADJ, SP, seed=13, arm="hold", sync=4, epochs=40, trace=True)
        b = gap.run(X, Y, ADJ, SP, seed=13, arm="lag1", sync=4, coef=0.0, epochs=40, trace=True)
        self.assertEqual(float(np.abs(a["trace"]["weights"][0] - b["trace"]["weights"][0]).max()), 0.0)
        z = gap.run(X, Y, ADJ, SP, seed=13, arm="zero", layers=3, epochs=24, trace=True)
        g1 = gap.run(X, Y, ADJ, SP, seed=13, arm="G1", layers=3, epochs=gap.WARM + 1, trace=True)
        zw = gap.run(X, Y, ADJ, SP, seed=13, arm="zero", layers=3, epochs=gap.WARM + 1, trace=True)
        for a_, b_ in zip(g1["trace"]["weights"], zw["trace"]["weights"]): self.assertEqual(float(np.abs(a_ - b_).max()), 0.0)  # warmup identical
        for k in range(2): self.assertEqual(float(np.abs(z["trace"]["remote"][23][k]).max()), 0.0)



def traced(**kw):
    """Run and capture the weights after every step via the exact-forward hook."""
    W = []; orig = gap.forward_exact
    def hook(x, adj, ws):
        W.append([w.detach().clone() for w in ws]); return orig(x, adj, ws)
    gap.forward_exact = hook
    try: r = gap.run(X, Y, ADJ, SP, trace=True, **kw)
    finally: gap.forward_exact = orig
    return r, W


def independent_logits(ws):
    A = ADJ.to_dense(); h = torch.relu(A @ (X @ ws[0]))
    if len(ws) == 3: h = torch.relu(A @ (h @ ws[1]))
    return A @ (h @ ws[-1])


class RunLevelTests(unittest.TestCase):
    def test_remote_estimates_rebuilt_from_sync_epochs_only(self):
        """hold and lag1, both hidden layers, sync 4/16/64, lr_drop on: the estimate used in
        training equals one rebuilt from hidden states at warmup/sync epochs alone."""
        c = (0.6, 0.3)
        for arm in ("hold", "lag1"):
            for sync in (4, 16, 64):
                r = gap.run(X, Y, ADJ, SP, seed=1, arm=arm, layers=3, sync=sync, coef=c, lr_drop=True, epochs=100, trace=True)
                hid, rem = r["trace"]["hidden"], r["trace"]["remote"]
                for k in range(2):
                    tp = max(0, 20 - min(sync, 20)); tl = 20; prev, last = hid[tp][k], hid[tl][k]; stale_seen = 0
                    for t in range(100):
                        if t <= 20 or (t - 20) % sync == 0:
                            if t > 20: prev, tp, last, tl = last, tl, hid[t][k], t
                            v = hid[t][k]
                        else:
                            cc = c[k] if arm == "lag1" else 0.0; stale_seen += 1
                            v = last + np.float32(cc) * (last - prev) * np.float32(min(1.0, (t - tl) / (tl - tp)))
                        self.assertLess(float(np.abs(rem[t][k] - v).max()), 1e-5, (arm, sync, k, t))
                    self.assertGreater(stale_seen, 50)

    def test_sni_run_level_matches_original_recurrence(self):
        for sync in (4, 16):
            r = gap.run(X, Y, ADJ, SP, seed=1, arm="sni", layers=3, sync=sync, epochs=70, trace=True)
            hid, rem = r["trace"]["hidden"], r["trace"]["remote"]
            for k in range(2):
                last = hid[0][k].copy(); num = np.zeros_like(last); den = np.ones(len(last), np.float32)
                for t in range(1, 70):
                    p = gap.screen._predict(last, num, den, 1); v = hid[t][k] if (t <= 20 or (t - 20) % sync == 0) else p
                    self.assertLess(float(np.abs(rem[t][k] - v).max() / (np.abs(v).max() + 1e-9)), 1e-4)
                    last, num, den = gap.screen._observe(last, num, den, v, 1)

    def test_sni_matches_replay_gate_debug_predictions(self):
        g = np.random.default_rng(1); a = g.normal(size=(60, 6, 3)).astype(np.float32).cumsum(0); parts = np.array([0, 0, 0, 1, 1, 1])
        for sync in (4, 16):
            dbg = gap.screen.replay_gate(a, parts, boundary_pairs=[(i, 1 - parts[i]) for i in range(6)], sync=sync, debug=True)["debug"]["predictions"]
            e = gap.Remote("sni", sync, 0.0)
            for t in range(60):
                o = e(torch.from_numpy(a[t]), t).numpy()
                if t > 20 and (t - 20) % sync != 0: self.assertLess(float(np.abs(o - np.asarray(dbg[t - 21])).max()), 1e-3)

    def test_fit_lag1_scores_equal_remote_error(self):
        g = np.random.default_rng(2); a = g.normal(size=(120, 6, 3)).astype(np.float32).cumsum(0)
        for sync in (4, 16, 64):
            _, sc = gap.fit_lag1([a], sync)
            for c in (0.0, 0.6, 1.5):
                e = gap.Remote("lag1", sync, c); errs = []
                for t in range(120):
                    o = e(torch.from_numpy(a[t]), t).numpy()
                    if t > 20 and (t - 20) % sync != 0:
                        errs.append(float(np.mean(np.sqrt(((o - a[t]) ** 2).sum(1)) / (np.sqrt((a[t] ** 2).sum(1)) + 1e-8))))
                self.assertAlmostEqual(float(np.mean(errs)), sc[c], places=5)

    def test_metrics_recomputed_independently(self):
        tr, va, ev = SP
        for layers, arm in ((2, "hold"), (3, "G0"), (3, "lag1")):
            r, W = traced(seed=6, arm=arm, layers=layers, sync=16, coef=0.5, epochs=120)
            self.assertEqual(len(W), 120)
            for ep in gap.CHECKPOINTS:
                with torch.no_grad(): pred = independent_logits(W[ep - 1]).argmax(1)
                self.assertAlmostEqual(r["val"][str(ep)], float((pred[va] == Y[va]).float().mean()), places=6)
                self.assertAlmostEqual(r["eval"][str(ep)], float((pred[ev] == Y[ev]).float().mean()), places=6)
            with torch.no_grad(): accs = [float((independent_logits(w).argmax(1)[va] == Y[va]).float().mean()) for w in W]
            first = next((i + 1 for i, a_ in enumerate(accs) if a_ >= 0.75), None)
            self.assertEqual(r["first_epoch_val_0.75"], first)

    def test_g1_gradient_equals_per_partition_reference(self):
        tr = SP[0]; pv = gap.partition(len(Y), 4); A = ADJ.to_dense()
        for layers in (2, 3):
            r = gap.run(X, Y, ADJ, SP, seed=7, arm="G1", layers=layers, epochs=1, trace=True)
            torch.manual_seed(7); dims = [X.shape[1]] + [16] * (layers - 1) + [7]
            sc = lambda i: 0.05 if layers == 2 else float(np.sqrt(2.0 / (dims[i] + dims[i + 1])))
            ws = [torch.nn.Parameter(torch.randn(dims[i], dims[i + 1]) * sc(i)) for i in range(layers)]
            tot = [torch.zeros_like(w) for w in ws]
            for p in range(4):                               # partition p keeps gradient only through its own nodes' states
                m = torch.from_numpy(pv == p).float()[:, None]; h = torch.relu(A @ (X @ ws[0]))
                for k in range(1, layers):
                    hm = h * m + (h * (1 - m)).detach(); h = A @ (hm @ ws[k]); h = torch.relu(h) if k < layers - 1 else h
                idx = torch.from_numpy(tr[pv[tr] == p])
                g = torch.autograd.grad(torch.nn.functional.cross_entropy(h[idx], Y[idx], reduction="sum") / len(tr), ws)
                tot = [a_ + b_ for a_, b_ in zip(tot, g)]
            for a_, b_ in zip(tot, r["trace"]["grads0"]):
                self.assertLess(float(np.abs(a_.numpy() - b_).max()), 1e-6 * max(1.0, float(np.abs(b_).max())) + 1e-7)

    def test_lr_drop_applies_to_partitioned_arms(self):
        one = np.zeros(len(Y), dtype=int)
        a = gap.run(X, Y, ADJ, SP, seed=5, arm="G0", lr_drop=True, epochs=70, trace=True)
        b = gap.run(X, Y, ADJ, SP, seed=5, arm="G1", lr_drop=True, epochs=70, trace=True, part_vec=one)
        c = gap.run(X, Y, ADJ, SP, seed=5, arm="G1", lr_drop=False, epochs=70, trace=True, part_vec=one)
        self.assertLess(float(np.abs(a["trace"]["weights"][0] - b["trace"]["weights"][0]).max()), 1e-5)
        self.assertGreater(float(np.abs(a["trace"]["weights"][0] - c["trace"]["weights"][0]).max()), 1e-4)

    def test_never_resync_is_hold_frozen_at_warmup(self):
        r = gap.run(X, Y, ADJ, SP, seed=3, arm="hold", sync=10 ** 6, epochs=60, trace=True)
        for t in range(21, 60): self.assertEqual(float(np.abs(r["trace"]["remote"][t][0] - r["trace"]["hidden"][20][0]).max()), 0.0)


if __name__ == "__main__": unittest.main()
