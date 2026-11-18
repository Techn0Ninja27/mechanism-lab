"""Free local measurement for the adversarial review (Claude, 2026-11-18).

Imports the frozen gnn-sync-gate trajectory code read-only. Seeds 20-21 are
development, 22-25 fresh. Does not touch the pilot's seeds 10-15.
Questions: (1) noise floor between fixed controls; (2) how much a metric-matched
parameter search over small causal predictor families gains over the fixed
SNI-style control; (3) closed-loop reference gap between fresh, hold and
predicted remote embeddings.
"""
import importlib.util, itertools, json, sys
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("screen", ROOT / "operator-studies/gnn-sync-gate/screen.py")
screen = importlib.util.module_from_spec(spec); spec.loader.exec_module(screen)
DATA = ROOT / "results/gnn-sync-gate-v1/data"
DEV, FRESH = (20, 21), (22, 23, 24, 25)
WARM, SYNC = 20, 4

def rel2(p, t): return np.sqrt(((p - t) ** 2).sum(1)) / (np.sqrt((t ** 2).sum(1)) + 1e-8)

def sni(a, lam, beta, mix=1.0):
    """Self-fed damped trend, same recurrence as screen.py with free lam/beta; mix with hold."""
    last = a[0].copy(); num = np.zeros_like(last); den = np.ones(len(last), np.float32); hold = a[0].copy()
    def obs(v):
        nonlocal last, num, den
        d = np.exp(-lam); s = v - last; last = v.copy(); num = num * d + s; den = den * d + 1.0
    for t in range(1, WARM + 1): obs(a[t]); hold = a[t].copy()
    out = []
    for t in range(WARM + 1, len(a)):
        pred = last + (num / np.maximum(den[:, None], 1e-6)) * np.exp(-beta)
        if (t - WARM) % SYNC == 0: obs(a[t]); hold = a[t].copy()
        else:
            out.append(rel2(mix * pred + (1 - mix) * hold, a[t]).mean()); obs(pred)
    return float(np.mean(out))

def lag_terms(a):
    """For each stale epoch: offset j, truth, last sync value, last sync difference, previous difference."""
    rows = {1: [], 2: [], 3: []}
    syncs = [WARM] + [t for t in range(WARM + 1, len(a)) if (t - WARM) % SYNC == 0]
    for t in range(WARM + 1, len(a)):
        j = (t - WARM) % SYNC
        if j == 0: continue
        s = t - j
        d1 = a[s] - a[s - SYNC]; d2 = a[s - SYNC] - a[s - 2 * SYNC]
        rows[j].append((a[t], a[s], d1, d2))
    return rows

def lag_err(rows_list, coef):
    """coef[j] = (c, e): pred = x_s + c*d1 + e*d2. Mean over stale epochs of mean pair error."""
    errs = []
    for rows in rows_list:
        for j in (1, 2, 3):
            c, e = coef[j]
            errs += [rel2(xs + c * d1 + e * d2, tr).mean() for tr, xs, d1, d2 in rows[j]]
    return float(np.mean(errs))

def fit_lag(rows_list, two):
    cs = np.round(np.arange(-0.2, 1.41, 0.05), 2); es = np.round(np.arange(-0.6, 0.61, 0.1), 2) if two else [0.0]
    coef = {}
    for j in (1, 2, 3):
        best = None
        for c in cs:
            for e in es:
                v = np.mean([rel2(xs + c * d1 + e * d2, tr).mean() for rows in rows_list for tr, xs, d1, d2 in rows[j]])
                if best is None or v < best[0]: best = (v, float(c), float(e))
        coef[j] = (best[1], best[2])
    return coef

def closed_loop(x, y, adj, parts, seed, mode, sync, epochs=120):
    """2-layer GCN; layer-2 messages crossing partitions use a remote estimate (no gradient)."""
    torch.manual_seed(seed); np.random.seed(seed)
    ei = adj.indices(); w = adj.values(); pt = torch.from_numpy(parts)
    cross = pt[ei[0]] != pt[ei[1]]; n = x.shape[0]
    A_in = torch.sparse_coo_tensor(ei[:, ~cross], w[~cross], (n, n)).coalesce()
    A_x = torch.sparse_coo_tensor(ei[:, cross], w[cross], (n, n)).coalesce()
    w1 = torch.nn.Parameter(torch.randn(x.shape[1], 16) * 0.05); w2 = torch.nn.Parameter(torch.randn(16, int(y.max()) + 1) * 0.05)
    opt = torch.optim.Adam([w1, w2], lr=.01, weight_decay=.0005); tr, va = screen.split_labels(y)
    last = None; prev = None; since = 0
    for t in range(epochs):
        h = torch.relu(torch.sparse.mm(adj, x @ w1)); hd = h.detach()
        if mode == "fresh" or t <= WARM or last is None or (t - WARM) % sync == 0:
            prev = last if last is not None else hd; last = hd; since = 0; remote = hd
        else:
            since += 1
            remote = last if mode == "hold" else last + (last - prev) * (since / sync)
        z = torch.sparse.mm(A_in, h @ w2) + torch.sparse.mm(A_x, remote @ w2)
        loss = torch.nn.functional.cross_entropy(z[tr], y[tr]); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        h = torch.relu(torch.sparse.mm(adj, x @ w1)); z = torch.sparse.mm(adj, h @ w2)
    return float((z[va].argmax(1) == y[va]).float().mean()), float(loss)

def main():
    x, y, adj = screen.load_cora(DATA)
    parts = np.random.default_rng(321).integers(0, 4, size=len(y)); ei = adj.indices().numpy()
    src = np.asarray(sorted({(int(i), int(parts[j])) for i, j in zip(ei[0], ei[1]) if parts[i] != parts[j]}))[:, 0]
    out = {"schema": "claude-headroom-check.v1", "dev_seeds": DEV, "fresh_seeds": FRESH, "warmup": WARM, "sync": SYNC, "schedules": {}}
    for drop in (False, True):
        traj = {s: screen.train_trajectory(x, y, adj, s, drop)[0][:, src] for s in DEV + FRESH}
        rows = {s: lag_terms(traj[s]) for s in traj}
        grid = [(l, b, m) for l in (0.0, .05, .1, .2, .4, .8) for b in (0.0, .05, .1, .2, .4, .8) for m in (.5, .75, 1.0)]
        dev_sni = {g: np.mean([sni(traj[s], *g) for s in DEV]) for g in grid}
        g_best = min(dev_sni, key=dev_sni.get)
        c1 = fit_lag([rows[s] for s in DEV], False); c2 = fit_lag([rows[s] for s in DEV], True)
        zero = {j: (0.0, 0.0) for j in (1, 2, 3)}
        per_seed = {}
        for s in DEV + FRESH:
            per_seed[s] = {"hold": lag_err([rows[s]], zero), "sni_fixed_0.2_0.2": sni(traj[s], .2, .2),
                           "sni_dev_tuned": sni(traj[s], *g_best), "lag1_dev_tuned": lag_err([rows[s]], c1),
                           "lag2_dev_tuned": lag_err([rows[s]], c2)}
        fr = lambda k: [per_seed[s][k] for s in FRESH]
        base = np.array(fr("sni_fixed_0.2_0.2"))
        summ = {k: {"fresh_mean": float(np.mean(fr(k))), "rel_change_vs_fixed_sni_mean": float(np.mean(np.array(fr(k)) / base - 1)),
                    "rel_change_vs_fixed_sni_sd": float(np.std(np.array(fr(k)) / base - 1, ddof=1))} for k in per_seed[FRESH[0]]}
        out["schedules"]["lr_drop" if drop else "constant"] = {"sni_dev_best_lam_beta_mix": g_best, "lag1_coef": c1, "lag2_coef": c2,
            "per_seed": per_seed, "fresh_summary": summ,
            "fixed_sni_fresh_cv_across_seeds": float(np.std(base, ddof=1) / np.mean(base))}
    cl = {}
    for sync in (4, 16):
        for mode in ("fresh", "hold", "trend"):
            acc = [closed_loop(x, y, adj, parts, s, mode, sync)[0] for s in range(20, 28)]
            cl[f"{mode}_sync{sync}"] = {"val_acc_mean": float(np.mean(acc)), "val_acc_sd": float(np.std(acc, ddof=1)), "per_seed": acc}
    out["closed_loop_reference"] = {"note": "8 seeds 20-27, validation accuracy of final model with a fresh full forward pass; test labels unused; reference gap, not a ceiling", "arms": cl}
    Path(__file__).with_name("headroom-result.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
    for k, v in out["schedules"].items():
        print(k, "best sni", v["sni_dev_best_lam_beta_mix"], "cv", round(v["fixed_sni_fresh_cv_across_seeds"], 4))
        for n, s in v["fresh_summary"].items(): print("  ", n, round(s["fresh_mean"], 6), round(s["rel_change_vs_fixed_sni_mean"], 4), "+-", round(s["rel_change_vs_fixed_sni_sd"], 4))
    for k, v in cl.items(): print(k, round(v["val_acc_mean"], 4), "+-", round(v["val_acc_sd"], 4))

if __name__ == "__main__": main()
