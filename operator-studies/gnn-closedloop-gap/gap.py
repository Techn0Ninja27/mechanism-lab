"""Closed-loop reference gap on Cora (Claude, 2026-11-18). See PLAN.md.

Single-process emulation of partitioned GCN training in which hidden-state
messages that cross a partition use a remote estimate with no gradient.
Imports the frozen gnn-sync-gate data and split code unmodified.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("screen", ROOT / "operator-studies/gnn-sync-gate/screen.py")
screen = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(screen)
DATA = ROOT / "results/gnn-sync-gate-v1/data"
WARM, EPOCHS, HID = 20, 120, 16
ARMS = ("G0", "G1", "hold", "sni", "lag1", "zero")
CHECKPOINTS = (40, 80, 120)


def splits(y):
    """Train (20/class) and validation (500) exactly as screen.split_labels; plus
    a disjoint 1,000-node development-evaluation split from the same shuffle."""
    g = np.random.default_rng(123); train = []
    for c in np.unique(y.numpy()):
        ix = np.flatnonzero(y.numpy() == c); g.shuffle(ix); train.extend(ix[:20])
    remain = np.setdiff1d(np.arange(len(y)), np.asarray(train)); g.shuffle(remain)
    return np.asarray(train), remain[:500], remain[500:1500]


def partition(n, parts):
    return np.random.default_rng(321).integers(0, parts, size=n)


def split_adj(adj, parts):
    ei = adj.indices(); w = adj.values(); pt = torch.from_numpy(np.asarray(parts)); n = adj.shape[0]
    cross = pt[ei[0]] != pt[ei[1]]
    a_in = torch.sparse_coo_tensor(ei[:, ~cross], w[~cross], (n, n)).coalesce()
    a_x = torch.sparse_coo_tensor(ei[:, cross], w[cross], (n, n)).coalesce()
    return a_in, a_x


def lag_ratio(t, t_last, t_prev):
    """Extrapolation horizon in units of the last observed interval, capped at one interval."""
    return min(1.0, (t - t_last) / (t_last - t_prev))


class Remote:
    """Estimate of remote hidden states for one message layer. Sees only exact
    values at warmup and sync epochs; everything else is its own state."""

    def __init__(self, arm, sync, coef):
        self.arm, self.sync, self.coef = arm, sync, coef
        self.last = self.prev = None; self.t_last = self.t_prev = None
        self.s_last = self.s_num = None; self.s_den = 1.0

    def _sni_observe(self, v):
        d = float(np.exp(-0.2)); slope = v - self.s_last
        self.s_last = v; self.s_num = self.s_num * d + slope; self.s_den = self.s_den * d + 1.0

    def __call__(self, exact, t):
        """`exact` is the detached current hidden state; it may be read only when
        `t` is a warmup or sync epoch (or always, for arm G1)."""
        if self.arm == "G1": return exact
        is_sync = t <= WARM or (t - WARM) % self.sync == 0
        if self.arm == "zero": return exact if t <= WARM else torch.zeros_like(exact)   # same warmup as other arms
        if self.arm == "sni":
            if t == 0:
                self.s_last = exact; self.s_num = torch.zeros_like(exact); self.s_den = 1.0; return exact
            pred = self.s_last + (self.s_num / max(self.s_den, 1e-6)) * float(np.exp(-0.2))
            if is_sync: self._sni_observe(exact); return exact
            self._sni_observe(pred); return pred
        if is_sync:
            if t <= WARM:
                if t == max(0, WARM - min(self.sync, WARM)): self.prev, self.t_prev = exact, t
                if t == WARM: self.last, self.t_last = exact, t
            else:
                self.prev, self.t_prev = self.last, self.t_last; self.last, self.t_last = exact, t
            return exact
        if self.arm == "hold": return self.last
        if self.arm == "lag1":
            return self.last + self.coef * (self.last - self.prev) * lag_ratio(t, self.t_last, self.t_prev)
        raise ValueError(self.arm)


def forward_exact(x, adj, ws):
    h = torch.relu(torch.sparse.mm(adj, x @ ws[0]))
    for k, w in enumerate(ws[1:], 1):
        h = torch.sparse.mm(adj, h @ w)
        if k < len(ws) - 1: h = torch.relu(h)
    return h


def run(x, y, adj, sp, *, seed, layers=2, parts=4, lr_drop=False, arm="G0", sync=4, coef=0.0,
        epochs=EPOCHS, trace=False, part_vec=None):
    """Train one model. Returns metrics; with trace=True also per-epoch hidden
    states, remote estimates used, and first-step gradients."""
    assert arm in ARMS and layers in (2, 3)
    torch.manual_seed(seed); np.random.seed(seed)
    dims = [x.shape[1]] + [HID] * (layers - 1) + [int(y.max()) + 1]
    # 2 layers: the original screen.py initialisation (std 0.05). 3 layers: Glorot-normal,
    # because std 0.05 leaves a 3-layer model at a collapse-to-zero point (see PLAN.md).
    scale = lambda i: 0.05 if layers == 2 else float(np.sqrt(2.0 / (dims[i] + dims[i + 1])))
    ws = [torch.nn.Parameter(torch.randn(dims[i], dims[i + 1]) * scale(i)) for i in range(layers)]
    opt = torch.optim.Adam(ws, lr=.01, weight_decay=.0005); tr, va, ev = sp
    assert not (set(tr) & set(va)) and not (set(tr) & set(ev)) and not (set(va) & set(ev))
    pv = partition(len(y), parts) if part_vec is None else np.asarray(part_vec)
    a_in, a_x = split_adj(adj, pv)
    coefs = [float(coef)] * (layers - 1) if np.isscalar(coef) else [float(c) for c in coef]
    assert len(coefs) == layers - 1
    est = [Remote(arm, sync, coefs[k]) for k in range(layers - 1)]
    out = {"val": {}, "eval": {}, "first_epoch_val_0.75": None}
    tr_h, tr_remote, grads0 = [], [], None
    for t in range(epochs):
        if lr_drop and t == 60:
            for q in opt.param_groups: q["lr"] *= .1
        h = torch.relu(torch.sparse.mm(adj, x @ ws[0]))          # layer 1: raw features, exact in every arm
        hs = [h]; used = []
        for k in range(1, layers):                               # message layers that consume hidden states
            cur = hs[-1]
            if arm == "G0":
                agg = torch.sparse.mm(adj, cur @ ws[k])
            else:
                remote = est[k - 1](cur.detach(), t); used.append(remote)
                agg = torch.sparse.mm(a_in, cur @ ws[k]) + torch.sparse.mm(a_x, remote @ ws[k])
            hs.append(torch.relu(agg) if k < layers - 1 else agg)
        loss = torch.nn.functional.cross_entropy(hs[-1][tr], y[tr])   # training labels only
        opt.zero_grad(); loss.backward()
        if trace:
            tr_h.append([v.detach().numpy().copy() for v in hs[:-1]]); tr_remote.append([v.numpy().copy() for v in used])
            if t == 0: grads0 = [w.grad.detach().numpy().copy() for w in ws]
        opt.step()
        with torch.no_grad():
            z = forward_exact(x, adj, ws); pred = z.argmax(1)
            v = float((pred[va] == y[va]).float().mean())
            if out["first_epoch_val_0.75"] is None and v >= 0.75: out["first_epoch_val_0.75"] = t + 1
            if t + 1 in CHECKPOINTS:
                out["val"][str(t + 1)] = v; out["eval"][str(t + 1)] = float((pred[ev] == y[ev]).float().mean())
    out["final_train_loss"] = float(loss.detach())
    if trace: out["trace"] = {"hidden": tr_h, "remote": tr_remote, "grads0": grads0, "weights": [w.detach().numpy().copy() for w in ws]}
    return out


def fit_lag1(h_traces, sync):
    """One scalar per (hidden layer, sync interval), grid-fit on open-loop G0
    calibration traces of that layer (list of [T, N, D] arrays) against
    stale-epoch relative L2. G0 traces are used because stale-arm hidden states
    are contaminated between syncs."""
    def err(c):
        e = []
        for a in h_traces:
            t_prev = max(0, WARM - min(sync, WARM)); t_last = WARM; prev, last = a[t_prev], a[t_last]
            for t in range(WARM + 1, len(a)):
                if (t - WARM) % sync == 0: prev, t_prev, last, t_last = last, t_last, a[t], t; continue
                p = last + c * (last - prev) * lag_ratio(t, t_last, t_prev)
                e.append(float(np.mean(np.sqrt(((p - a[t]) ** 2).sum(1)) / (np.sqrt((a[t] ** 2).sum(1)) + 1e-8))))
        return float(np.mean(e))
    grid = [round(c, 2) for c in np.arange(0.0, 1.51, 0.05)]
    scores = {c: err(c) for c in grid}
    best = min(scores, key=scores.get)
    return best, scores
