"""Non-negative projection control. See PLAN.md.
Imports ../gnn-closedloop-gap/gap.py and ../gnn-closedloop-confirm/confirm.py unmodified.

  python clamp.py freeze | run [--workers N] | analyse
"""
import argparse, hashlib, itertools, json, math, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
CONF = HERE.parent / "gnn-closedloop-confirm"
sys.path.insert(0, str(CONF)); sys.path.insert(0, str(HERE.parent / "gnn-closedloop-gap"))
import confirm
gap = confirm.gap

R = confirm.REGIMES["moderate_lead"]
ARMS = {"sni": ("sni", None), "sni_emit": ("sni", "emit"), "sni_state": ("sni", "state"), "lag1": ("lag1", None), "lag1_emit": ("lag1", "emit")}
CONFIG = {"seeds": list(range(50, 58)), "arms": list(ARMS), "regime": R, "window_cycles": 2, "baselines_from": "gnn-closedloop-confirm/results/raw.json",
          "reading": {"margin_points": 0.5, "se_multiple": 2.0}}
FILES = [HERE / "clamp.py", HERE / "test_clamp.py", HERE / "PLAN.md", CONF / "confirm.py", CONF / "freeze.json", CONF / "results" / "raw.json",
         HERE.parent / "gnn-closedloop-gap" / "gap.py", gap.ROOT / "operator-studies/gnn-sync-gate/screen.py", gap.DATA / "cora.content", gap.DATA / "cora.cites"]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
hashes = lambda: {str(p.relative_to(gap.ROOT)): sha(p) for p in FILES}


class Projected:
    """Wraps an unmodified gap.Remote. mode None: pass through (with logging).
    'emit': the recurrence is untouched and self-fed with its raw forecast; only the
    emitted value is projected. 'state' (sni only): the projected value is also self-fed."""

    def __init__(self, base, mode, sync, coef):
        assert base in ("sni", "lag1") and mode in (None, "emit", "state") and not (base == "lag1" and mode == "state")
        self.r = gap.Remote(base, sync, coef); self.base, self.mode, self.sync = base, mode, sync; self.log = []

    def _note(self, t, raw):
        neg = torch.clamp(raw, max=0.0)
        self.log.append((t, float((raw < 0).float().mean()), float(neg.norm() / (raw.norm() + 1e-12))))

    def __call__(self, exact, t):
        is_sync = t <= gap.WARM or (t - gap.WARM) % self.sync == 0
        if is_sync or t == 0: return self.r(exact, t)
        if self.mode == "state":
            raw = self.r.s_last + (self.r.s_num / max(self.r.s_den, 1e-6)) * float(np.exp(-0.2))   # same forecast rule as gap.Remote
            self._note(t, raw); out = torch.clamp(raw, min=0.0); self.r._sni_observe(out); return out
        raw = self.r(exact, t); self._note(t, raw)
        return torch.clamp(raw, min=0.0) if self.mode == "emit" else raw


def train(x, y, adj, sp, *, seed, arm, epochs=gap.EPOCHS, window=None, keep_weights=False):
    """confirm.train's loop for the fixed regime, with Projected estimators."""
    base, mode = ARMS[arm]; layers, parts, sync = R["layers"], R["parts"], R["sync"]
    torch.manual_seed(seed); np.random.seed(seed)
    dims = [x.shape[1]] + [gap.HID] * (layers - 1) + [int(y.max()) + 1]
    scale = lambda i: 0.05 if layers == 2 else float(np.sqrt(2.0 / (dims[i] + dims[i + 1])))
    ws = [torch.nn.Parameter(torch.randn(dims[i], dims[i + 1]) * scale(i)) for i in range(layers)]
    opt = torch.optim.Adam(ws, lr=.01, weight_decay=.0005); tr, va, ev = sp
    a_in, a_x = gap.split_adj(adj, gap.partition(len(y), parts))
    coefs = R["lag1_coef"] if base == "lag1" else [0.0] * (layers - 1)
    est = [Projected(base, mode, sync, float(coefs[k])) for k in range(layers - 1)]
    window = set(window or []); rows = []
    acc = lambda z, ix: float((z.argmax(1)[ix] == y[ix]).float().mean())
    for t in range(epochs):
        hs = [torch.relu(torch.sparse.mm(adj, x @ ws[0]))]
        for k in range(1, layers):
            cur = hs[-1]; remote = est[k - 1](cur.detach(), t)
            agg = torch.sparse.mm(a_in, cur @ ws[k]) + torch.sparse.mm(a_x, remote @ ws[k])
            hs.append(torch.relu(agg) if k < layers - 1 else agg)
        if t in window:
            with torch.no_grad():
                zs = hs[-1].detach(); ze = gap.forward_exact(x, adj, ws)
                rows.append({"t": t, "phase": (t - gap.WARM) % sync, "stale_val": acc(zs, va), "stale_eval": acc(zs, ev), "exact_val": acc(ze, va), "exact_eval": acc(ze, ev)})
        loss = torch.nn.functional.cross_entropy(hs[-1][tr], y[tr]); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): z = gap.forward_exact(x, adj, ws)
    neg = [{"hidden_layer": k + 1, "stale_epochs": len(e.log), "mean_negative_share": float(np.mean([a for _, a, _ in e.log])),
            "max_negative_share": float(np.max([a for _, a, _ in e.log])), "mean_negative_norm_ratio": float(np.mean([b for _, _, b in e.log])),
            "max_negative_norm_ratio": float(np.max([b for _, _, b in e.log]))} for k, e in enumerate(est)]
    out = {"post_update_exact_val_120": acc(z, va), "post_update_exact_eval_120": acc(z, ev), "final_train_loss": float(loss.detach()), "window": rows, "raw_forecast_negativity": neg}
    if keep_weights: out["weights"] = [w.detach().numpy().copy() for w in ws]
    return out


def same_run(a, b):
    """Gate: a logged rerun must reproduce the stored confirmation record exactly."""
    return a["post_update_exact_eval_120"] == b["post_update_exact_eval_120"] and a["final_train_loss"] == b["final_train_loss"] and a["window"] == b["window"]


def reading(m, se, margin, k):
    """Descriptive labels. 'within_band' is a point estimate inside +/-margin, not an equivalence claim."""
    if abs(m) <= margin: return "within_band_point_estimate"
    if abs(m) > k * se: return "better" if m > 0 else "worse"
    return "inconclusive"


def freeze():
    p = HERE / "freeze.json"
    if p.exists(): raise FileExistsError("freeze.json exists")
    p.write_text(json.dumps({"schema": "gnn-closedloop-clamp-freeze.v1", "config": CONFIG, "sha256": hashes(),
                             "versions": {"torch": torch.__version__, "numpy": np.__version__, "python": sys.version.split()[0]}}, indent=2) + "\n")


_D = {}
def _init():
    torch.set_num_threads(1); x, y, adj = gap.screen.load_cora(gap.DATA); _D.update(x=x, y=y, adj=adj, sp=gap.splits(y))

def _one(job):
    return {**job, **train(_D["x"], _D["y"], _D["adj"], _D["sp"], seed=job["seed"], arm=job["arm"], window=confirm.window_epochs(R["sync"]))}


def run(workers):
    fz = json.loads((HERE / "freeze.json").read_text())
    if fz["sha256"] != hashes() or fz["config"] != CONFIG: raise RuntimeError("files or config changed since freeze")
    out = HERE / "results" / "raw.json"
    if out.exists(): raise FileExistsError("results/raw.json exists")
    out.parent.mkdir(exist_ok=True); t0 = time.time()
    jobs = [{"seed": s, "arm": a} for s, a in itertools.product(CONFIG["seeds"], CONFIG["arms"])]
    with Pool(workers, initializer=_init) as pool: rows = pool.map(_one, jobs, chunksize=1)
    out.write_text(json.dumps({"schema": "gnn-closedloop-clamp-raw.v1", "freeze_sha256": sha(HERE / "freeze.json"), "n_runs": len(rows), "seconds": round(time.time() - t0, 1), "runs": rows}) + "\n")
    print(len(rows), "runs", round(time.time() - t0, 1), "s")


def analyse():
    out = HERE / "results" / "summary.json"
    if out.exists(): raise FileExistsError("results/summary.json exists")
    fz = json.loads((HERE / "freeze.json").read_text()); C = fz["config"]; raw = json.loads((HERE / "results" / "raw.json").read_text())
    base = json.loads((CONF / "results" / "raw.json").read_text()); seeds = C["seeds"]; n = len(seeds)
    bidx = {(r["seed"], r["arm"]): r for r in base["runs"] if r["regime"] == "moderate_lead"}
    keys = [(r["seed"], r["arm"]) for r in raw["runs"]]; idx = dict(zip(keys, raw["runs"]))
    def finite(r):
        v = [r["post_update_exact_eval_120"], r["final_train_loss"]] + [w[k] for w in r["window"] for k in ("stale_eval", "exact_eval")] + [q[k] for q in r["raw_forecast_negativity"] for k in ("mean_negative_share", "mean_negative_norm_ratio")]
        return all(type(a) is float and math.isfinite(a) for a in v)
    same = same_run
    checks = {"raw_bound_to_freeze": raw["freeze_sha256"] == sha(HERE / "freeze.json"), "frozen_files_unchanged": fz["sha256"] == hashes(),
              "unique_keys": len(keys) == len(set(keys)), "exact_expected_key_set": set(keys) == set(itertools.product(seeds, C["arms"])) and len(keys) == raw["n_runs"],
              "all_metrics_finite": all(finite(r) for r in raw["runs"]),
              "logged_reruns_reproduce_stored_sni_and_lag1": all(same(idx[(s, a)], bidx[(s, a)]) for s in seeds for a in ("sni", "lag1")),
              "stale_epoch_count": all(q["stale_epochs"] == 93 for r in raw["runs"] for q in r["raw_forecast_negativity"])}
    if not all(checks.values()): print(json.dumps(checks, indent=1)); sys.exit("integrity checks failed; no summary written")
    t95 = confirm.t_ppf(0.975, n - 1); M, K = C["reading"]["margin_points"], C["reading"]["se_multiple"]
    wm = lambda r, k: float(np.mean([w[k] for w in r["window"]]))
    def paired(a, b):
        d = (np.asarray(a) - np.asarray(b)) * 100; m = float(d.mean()); se = float(d.std(ddof=1) / math.sqrt(n))
        return {"mean_pt": m, "se_pt": se, "pointwise_ci95_pt": [m - t95 * se, m + t95 * se], "n_positive": int((d > 0).sum()), "n": n, "reading_vs_reference": reading(m, se, M, K)}
    hold_e = [wm(bidx[(s, "hold")], "exact_eval") for s in seeds]; hold_o = [wm(bidx[(s, "hold")], "stale_eval") for s in seeds]; g1 = [wm(bidx[(s, "G1")], "exact_eval") for s in seeds]
    summ = {"schema": "gnn-closedloop-clamp-summary.v1", "status": "one regime, exposed development seeds 50-57, pointwise intervals; not held-out evidence",
            "integrity_checks": checks, "raw_sha256": sha(HERE / "results" / "raw.json"), "freeze_sha256": sha(HERE / "freeze.json"),
            "baseline_raw_sha256": sha(CONF / "results" / "raw.json"), "reference_means": {"G1_exact": float(np.mean(g1)), "hold_exact": float(np.mean(hold_e)), "hold_own": float(np.mean(hold_o))}, "arms": {}}
    for a in C["arms"]:
        e = [wm(idx[(s, a)], "exact_eval") for s in seeds]; o = [wm(idx[(s, a)], "stale_eval") for s in seeds]
        neg = {f"hidden{k + 1}": {q: float(np.mean([idx[(s, a)]["raw_forecast_negativity"][k][q] for s in seeds])) for q in ("mean_negative_share", "max_negative_share", "mean_negative_norm_ratio", "max_negative_norm_ratio")} for k in range(R["layers"] - 1)}
        summ["arms"][a] = {"mean_exact": float(np.mean(e)), "mean_own": float(np.mean(o)), "endpoint_exact_120": float(np.mean([idx[(s, a)]["post_update_exact_eval_120"] for s in seeds])),
                           "minus_hold_exact": paired(e, hold_e), "minus_hold_own_forward": paired(o, hold_o), "minus_G1_exact": paired(e, g1),
                           "minus_raw_predictor_exact": None if ARMS[a][1] is None else paired(e, [wm(idx[(s, ARMS[a][0])], "exact_eval") for s in seeds]),
                           "raw_forecast_negativity": neg}
    out.write_text(json.dumps(summ, indent=2) + "\n")
    f = lambda p: f"{p['mean_pt']:+.2f}±{p['se_pt']:.2f} ({p['n_positive']}/{p['n']}) [{p['reading_vs_reference']}]"
    print("reference", {k: round(v, 4) for k, v in summ["reference_means"].items()})
    for a, v in summ["arms"].items():
        print(f"{a:10s} exact {v['mean_exact']:.3f} own {v['mean_own']:.3f} | vs hold {f(v['minus_hold_exact'])} | vs G1 {f(v['minus_G1_exact'])} | vs raw {f(v['minus_raw_predictor_exact']) if v['minus_raw_predictor_exact'] else '-'} | own-fwd vs hold {f(v['minus_hold_own_forward'])}")
        print("            neg share/norm ratio (mean):", {k: (round(q['mean_negative_share'], 3), round(q['mean_negative_norm_ratio'], 3)) for k, q in v["raw_forecast_negativity"].items()})
    print("checks", checks)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["freeze", "run", "analyse"]); ap.add_argument("--workers", type=int, default=4); a = ap.parse_args()
    {"freeze": freeze, "run": lambda: run(a.workers), "analyse": analyse}[a.cmd]()
