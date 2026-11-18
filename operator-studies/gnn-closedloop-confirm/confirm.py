"""Bounded confirmation and phase check for the closed-loop staleness lead.
See PLAN.md. Imports ../gnn-closedloop-gap/gap.py unmodified.

  python confirm.py freeze | run [--workers N] | analyse
"""
import argparse, hashlib, itertools, json, math, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
GAPDIR = HERE.parent / "gnn-closedloop-gap"
sys.path.insert(0, str(GAPDIR))
import gap

# Coefficients are copied from gnn-closedloop-gap/results/raw.json calibration and checked against it at freeze.
REGIMES = {"negative_control": {"parts": 4, "layers": 2, "lr_drop": False, "sync": 4, "lag1_coef": [0.95]},
           "moderate_lead": {"parts": 4, "layers": 3, "lr_drop": False, "sync": 16, "lag1_coef": [0.8, 0.9]}}
CONFIG = {"seeds": list(range(50, 58)), "arms": ["G1", "hold", "sni", "lag1"], "regimes": REGIMES, "window_cycles": 2,
          "rule": {"min_points": 0.5, "min_se_multiple": 2.0}}
FILES = [HERE / "confirm.py", HERE / "test_confirm.py", HERE / "PLAN.md", GAPDIR / "gap.py", GAPDIR / "freeze.json", GAPDIR / "results" / "raw.json",
         gap.ROOT / "operator-studies/gnn-sync-gate/screen.py", gap.DATA / "cora.content", gap.DATA / "cora.cites"]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
hashes = lambda: {str(p.relative_to(gap.ROOT)): sha(p) for p in FILES}


def train(x, y, adj, sp, *, seed, layers, parts, lr_drop, arm, sync, coef, epochs=gap.EPOCHS, window=None, keep_weights=False):
    """Same training loop as gap.run. Additionally, for epochs in `window`, records accuracy of
    (a) the forward pass the model actually trained with (stale remote estimates) and
    (b) an exact forward on the same weights, both BEFORE the optimiser update.
    The stale logits are the training forward itself, so no estimator is called twice."""
    torch.manual_seed(seed); np.random.seed(seed)
    dims = [x.shape[1]] + [gap.HID] * (layers - 1) + [int(y.max()) + 1]
    scale = lambda i: 0.05 if layers == 2 else float(np.sqrt(2.0 / (dims[i] + dims[i + 1])))
    ws = [torch.nn.Parameter(torch.randn(dims[i], dims[i + 1]) * scale(i)) for i in range(layers)]
    opt = torch.optim.Adam(ws, lr=.01, weight_decay=.0005); tr, va, ev = sp
    a_in, a_x = gap.split_adj(adj, gap.partition(len(y), parts))
    coefs = [float(c) for c in coef]; est = [gap.Remote(arm, sync, coefs[k]) for k in range(layers - 1)]
    window = set(window or []); rows = []
    acc = lambda z, ix: float((z.argmax(1)[ix] == y[ix]).float().mean())
    for t in range(epochs):
        if lr_drop and t == 60:
            for q in opt.param_groups: q["lr"] *= .1
        hs = [torch.relu(torch.sparse.mm(adj, x @ ws[0]))]
        for k in range(1, layers):
            cur = hs[-1]; remote = est[k - 1](cur.detach(), t)
            agg = torch.sparse.mm(a_in, cur @ ws[k]) + torch.sparse.mm(a_x, remote @ ws[k])
            hs.append(torch.relu(agg) if k < layers - 1 else agg)
        if t in window:
            with torch.no_grad():
                zs = hs[-1].detach(); ze = gap.forward_exact(x, adj, ws)
                rows.append({"t": t, "phase": (t - gap.WARM) % sync, "stale_val": acc(zs, va), "stale_eval": acc(zs, ev),
                             "exact_val": acc(ze, va), "exact_eval": acc(ze, ev)})
        loss = torch.nn.functional.cross_entropy(hs[-1][tr], y[tr]); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): z = gap.forward_exact(x, adj, ws)
    out = {"post_update_exact_val_120": acc(z, va), "post_update_exact_eval_120": acc(z, ev), "final_train_loss": float(loss.detach()), "window": rows}
    if keep_weights: out["weights"] = [w.detach().numpy().copy() for w in ws]
    return out


def window_epochs(sync): return list(range(gap.EPOCHS - CONFIG["window_cycles"] * sync, gap.EPOCHS))


def freeze():
    p = HERE / "freeze.json"
    if p.exists(): raise FileExistsError("freeze.json exists")
    cal = json.loads((GAPDIR / "results" / "raw.json").read_text())["calibration"]
    for name, r in REGIMES.items():
        want = [cal[f"layers{r['layers']}_drop{int(r['lr_drop'])}_sync{r['sync']}_hidden{k + 1}"]["coef"] for k in range(r["layers"] - 1)]
        if want != r["lag1_coef"]: raise RuntimeError(f"{name}: locked coefficient differs from original calibration")
    p.write_text(json.dumps({"schema": "gnn-closedloop-confirm-freeze.v1", "config": CONFIG, "sha256": hashes(),
                             "versions": {"torch": torch.__version__, "numpy": np.__version__, "python": sys.version.split()[0]}}, indent=2) + "\n")


_D = {}
def _init():
    torch.set_num_threads(1); x, y, adj = gap.screen.load_cora(gap.DATA); _D.update(x=x, y=y, adj=adj, sp=gap.splits(y))

def _one(job):
    r = REGIMES[job["regime"]]
    res = train(_D["x"], _D["y"], _D["adj"], _D["sp"], seed=job["seed"], layers=r["layers"], parts=r["parts"], lr_drop=r["lr_drop"], arm=job["arm"],
                sync=r["sync"], coef=r["lag1_coef"] if job["arm"] == "lag1" else [0.0] * (r["layers"] - 1), window=window_epochs(r["sync"]))
    print(json.dumps({"done": job}), flush=True); return {**job, **res}


def run(workers):
    fz = json.loads((HERE / "freeze.json").read_text())
    if fz["sha256"] != hashes() or fz["config"] != CONFIG: raise RuntimeError("files or config changed since freeze")
    out = HERE / "results" / "raw.json"
    if out.exists(): raise FileExistsError("results/raw.json exists")
    out.parent.mkdir(exist_ok=True); t0 = time.time()
    jobs = [{"regime": g, "seed": s, "arm": a} for g, s, a in itertools.product(REGIMES, CONFIG["seeds"], CONFIG["arms"])]
    with Pool(workers, initializer=_init) as pool: rows = pool.map(_one, jobs, chunksize=1)
    out.write_text(json.dumps({"schema": "gnn-closedloop-confirm-raw.v1", "freeze_sha256": sha(HERE / "freeze.json"), "n_runs": len(rows),
                               "seconds": round(time.time() - t0, 1), "runs": rows}) + "\n")
    print(len(rows), "runs", round(time.time() - t0, 1), "s")


def t_ppf(q, df):
    xs = np.linspace(-60, 60, 2_400_001); pdf = (1 + xs ** 2 / df) ** (-(df + 1) / 2); cdf = np.cumsum(pdf); cdf /= cdf[-1]
    return float(xs[np.searchsorted(cdf, q)])


def analyse():
    out = HERE / "results" / "summary.json"
    if out.exists(): raise FileExistsError("results/summary.json exists")
    fz = json.loads((HERE / "freeze.json").read_text()); C = fz["config"]; raw = json.loads((HERE / "results" / "raw.json").read_text())
    seeds = C["seeds"]; n = len(seeds)
    keys = [(r["regime"], r["seed"], r["arm"]) for r in raw["runs"]]; expected = set(itertools.product(C["regimes"], seeds, C["arms"]))
    def finite(r):
        v = [r["post_update_exact_eval_120"], r["post_update_exact_val_120"], r["final_train_loss"]] + [w[k] for w in r["window"] for k in ("stale_val", "stale_eval", "exact_val", "exact_eval")]
        return all(type(a) is float and math.isfinite(a) for a in v)
    checks = {"raw_bound_to_freeze": raw["freeze_sha256"] == sha(HERE / "freeze.json"), "frozen_files_unchanged": fz["sha256"] == hashes(),
              "unique_keys": len(keys) == len(set(keys)), "exact_expected_key_set": set(keys) == expected and len(keys) == raw["n_runs"] == len(expected),
              "all_metrics_finite": all(finite(r) for r in raw["runs"]),
              "window_lengths": all(len(r["window"]) == C["window_cycles"] * C["regimes"][r["regime"]]["sync"] for r in raw["runs"]),
              "g1_stale_equals_exact": all(abs(w["stale_eval"] - w["exact_eval"]) < 1e-9 for r in raw["runs"] if r["arm"] == "G1" for w in r["window"])}
    if not all(checks.values()): print(json.dumps(checks, indent=1)); sys.exit("integrity checks failed; no summary written")
    idx = dict(zip(keys, raw["runs"])); t95 = t_ppf(0.975, n - 1)
    def paired(a, b):
        d = (np.asarray(a) - np.asarray(b)) * 100; m = float(d.mean()); se = float(d.std(ddof=1) / math.sqrt(n))
        return {"mean_pt": m, "se_pt": se, "pointwise_ci95_pt": [m - t95 * se, m + t95 * se], "n_positive": int((d > 0).sum()), "n": n,
                "passes_rule": bool(m > C["rule"]["min_points"] and m > C["rule"]["min_se_multiple"] * se)}
    wm = lambda g, s, a, k: float(np.mean([w[k] for w in idx[(g, s, a)]["window"]]))
    summ = {"schema": "gnn-closedloop-confirm-summary.v1", "status": "bounded confirmation on new seeds; two fixed regimes; pointwise intervals",
            "integrity_checks": checks, "raw_sha256": sha(HERE / "results" / "raw.json"), "freeze_sha256": sha(HERE / "freeze.json"), "regimes": {}}
    for g, R in C["regimes"].items():
        row = {"config": R, "arm_means": {}, "comparisons": {}, "phase_profile": {}}
        for a in C["arms"]:
            row["arm_means"][a] = {"endpoint_exact_eval_120": float(np.mean([idx[(g, s, a)]["post_update_exact_eval_120"] for s in seeds])),
                                   "window_exact_eval": float(np.mean([wm(g, s, a, "exact_eval") for s in seeds])),
                                   "window_stale_eval": float(np.mean([wm(g, s, a, "stale_eval") for s in seeds]))}
            ph = {}
            for p in range(R["sync"]):
                ph[str(p)] = {k: float(np.mean([w[k] for s in seeds for w in idx[(g, s, a)]["window"] if w["phase"] == p])) for k in ("exact_eval", "stale_eval")}
            row["phase_profile"][a] = ph
        g1 = lambda k: [wm(g, s, "G1", k) for s in seeds]
        row["comparisons"]["PRIMARY_G1_minus_hold_window_exact"] = paired(g1("exact_eval"), [wm(g, s, "hold", "exact_eval") for s in seeds])
        row["comparisons"]["G1_minus_hold_window_own_forward"] = paired(g1("exact_eval"), [wm(g, s, "hold", "stale_eval") for s in seeds])
        row["comparisons"]["G1_minus_hold_endpoint_exact_120"] = paired([idx[(g, s, "G1")]["post_update_exact_eval_120"] for s in seeds], [idx[(g, s, "hold")]["post_update_exact_eval_120"] for s in seeds])
        row["comparisons"]["hold_own_forward_minus_hold_exact"] = paired([wm(g, s, "hold", "stale_eval") for s in seeds], [wm(g, s, "hold", "exact_eval") for s in seeds])
        for a in ("sni", "lag1"):
            row["comparisons"][f"{a}_minus_hold_window_exact"] = paired([wm(g, s, a, "exact_eval") for s in seeds], [wm(g, s, "hold", "exact_eval") for s in seeds])
            row["comparisons"][f"{a}_minus_hold_window_own_forward"] = paired([wm(g, s, a, "stale_eval") for s in seeds], [wm(g, s, "hold", "stale_eval") for s in seeds])
        summ["regimes"][g] = row
    lead = summ["regimes"]["moderate_lead"]["comparisons"]; neg = summ["regimes"]["negative_control"]["comparisons"]
    summ["decision"] = {"lead_confirmed_exact_forward": lead["PRIMARY_G1_minus_hold_window_exact"]["passes_rule"],
                        "lead_survives_own_forward": lead["G1_minus_hold_window_own_forward"]["passes_rule"],
                        "negative_control_stays_null": not neg["PRIMARY_G1_minus_hold_window_exact"]["passes_rule"]}
    out.write_text(json.dumps(summ, indent=2) + "\n")
    f = lambda p: f"{p['mean_pt']:+.2f}±{p['se_pt']:.2f} ({p['n_positive']}/{p['n']}){'*' if p['passes_rule'] else ''}"
    for g, row in summ["regimes"].items():
        print(g, row["config"]); print("  means", {a: {k: round(v, 4) for k, v in m.items()} for a, m in row["arm_means"].items()})
        for k, v in row["comparisons"].items(): print("   ", k, f(v))
        print("   hold by phase (exact, own):", {p: (round(v["exact_eval"], 3), round(v["stale_eval"], 3)) for p, v in row["phase_profile"]["hold"].items()})
    print("checks", checks); print("decision", summ["decision"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["freeze", "run", "analyse"]); ap.add_argument("--workers", type=int, default=4); a = ap.parse_args()
    {"freeze": freeze, "run": lambda: run(a.workers), "analyse": analyse}[a.cmd]()
