"""Freeze, run and analyse the closed-loop reference-gap study. See PLAN.md.

  python run.py freeze            # writes freeze.json (hashes + config); refuses to overwrite
  python run.py run --workers 8   # requires matching freeze.json; writes results/raw.json once
  python run.py analyse           # writes results/summary.json once
"""
import argparse, hashlib, itertools, json, math, os, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gap

CONFIG = {"calibration_seeds": [20, 21], "measurement_seeds": list(range(30, 46)), "parts": [4, 16], "layers": [2, 3],
          "lr_drop": [False, True], "sync": [4, 16, 64], "stale_arms": ["hold", "sni", "lag1"], "never_resync_sync": 10 ** 6,
          "primary_metric": "eval@120", "rule": {"min_points": 0.5, "min_se_multiple": 2.0}}
FILES = ["gap.py", "run.py", "test_gap.py", "PLAN.md", "PREFREEZE.md"]
EXTERNAL = [gap.ROOT / "operator-studies/gnn-sync-gate/screen.py", gap.DATA / "cora.content", gap.DATA / "cora.cites"]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def hashes():
    h = {f: sha(HERE / f) for f in FILES}; h.update({str(p.relative_to(gap.ROOT)): sha(p) for p in EXTERNAL}); return h


def freeze():
    p = HERE / "freeze.json"
    if p.exists(): raise FileExistsError("freeze.json exists")
    p.write_text(json.dumps({"schema": "gnn-closedloop-gap-freeze.v1", "config": CONFIG, "sha256": hashes(),
                             "versions": {"torch": torch.__version__, "numpy": np.__version__, "python": sys.version.split()[0]}}, indent=2) + "\n")


_D = {}
def _init():
    torch.set_num_threads(1); x, y, adj = gap.screen.load_cora(gap.DATA); _D.update(x=x, y=y, adj=adj, sp=gap.splits(y))

def _one(job):
    r = gap.run(_D["x"], _D["y"], _D["adj"], _D["sp"], **job); return {**job, **r}


def run(workers):
    fz = json.loads((HERE / "freeze.json").read_text())
    if fz["sha256"] != hashes() or fz["config"] != CONFIG: raise RuntimeError("files or config changed since freeze")
    out = HERE / "results" / "raw.json"
    if out.exists(): raise FileExistsError("results/raw.json exists; results are immutable")
    out.parent.mkdir(exist_ok=True); t0 = time.time(); _init(); C = CONFIG
    coef = {}; cal = {}
    for layers, drop in itertools.product(C["layers"], C["lr_drop"]):
        runs = [gap.run(_D["x"], _D["y"], _D["adj"], _D["sp"], seed=s, layers=layers, lr_drop=drop, arm="G0", trace=True)["trace"]["hidden"] for s in C["calibration_seeds"]]
        for s in C["sync"]:
            per_layer = []
            for k in range(layers - 1):
                best, scores = gap.fit_lag1([np.stack([h[k] for h in run_]) for run_ in runs], s); per_layer.append(best)
                cal[f"layers{layers}_drop{int(drop)}_sync{s}_hidden{k + 1}"] = {"coef": best, "scores": scores}
            coef[(layers, drop, s)] = per_layer
    jobs = []
    for seed, layers, drop in itertools.product(C["measurement_seeds"], C["layers"], C["lr_drop"]):
        base = dict(seed=seed, layers=layers, lr_drop=drop)
        jobs.append({**base, "arm": "G0", "parts": 4, "sync": 1})          # centralized: independent of parts and sync
        for parts in C["parts"]:
            b = {**base, "parts": parts}
            jobs.append({**b, "arm": "G1", "sync": 1}); jobs.append({**b, "arm": "zero", "sync": 1})
            jobs.append({**b, "arm": "hold", "sync": C["never_resync_sync"]})
            for s in C["sync"]:
                for arm in C["stale_arms"]:
                    jobs.append({**b, "arm": arm, "sync": s, "coef": coef[(layers, drop, s)] if arm == "lag1" else [0.0] * (layers - 1)})
    with Pool(workers, initializer=_init) as pool: rows = pool.map(_one, jobs, chunksize=8)
    out.write_text(json.dumps({"schema": "gnn-closedloop-gap-raw.v1", "freeze_sha256": sha(HERE / "freeze.json"), "calibration": cal,
                               "n_runs": len(rows), "seconds": round(time.time() - t0, 1), "runs": rows}) + "\n")
    print(len(rows), "runs", round(time.time() - t0, 1), "s")


def analyse():
    def t_ppf(q, df):
        """Student-t quantile by bisection on a numerically integrated CDF (no SciPy in this venv)."""
        xs = np.linspace(-60, 60, 2_400_001); pdf = (1 + xs ** 2 / df) ** (-(df + 1) / 2); cdf = np.cumsum(pdf); cdf /= cdf[-1]
        return float(xs[np.searchsorted(cdf, q)])
    raw = json.loads((HERE / "results" / "raw.json").read_text()); out = HERE / "results" / "summary.json"
    if out.exists(): raise FileExistsError("results/summary.json exists")
    C = CONFIG; seeds = C["measurement_seeds"]; n = len(seeds)
    idx = {(r["arm"], r["seed"], r["layers"], r["lr_drop"], r["parts"], r["sync"]): r for r in raw["runs"]}
    def vec(arm, layers, drop, parts, sync, metric):
        k, ep = metric.split("@"); return np.array([idx[(arm, s, layers, drop, parts, sync)][k][ep] for s in seeds])
    m_tests = len(C["parts"]) * len(C["layers"]) * len(C["lr_drop"]) * len(C["sync"])
    t95 = t_ppf(0.975, n - 1); t_sim = t_ppf(1 - 0.025 / m_tests, n - 1)
    def paired(a, b):
        d = (a - b) * 100; mean = float(d.mean()); se = float(d.std(ddof=1) / math.sqrt(n))
        return {"mean_pt": mean, "se_pt": se, "ci95_pt": [mean - t95 * se, mean + t95 * se],
                "simultaneous_ci_pt": [mean - t_sim * se, mean + t_sim * se],
                "rule_unadjusted": bool(mean > C["rule"]["min_points"] and mean > C["rule"]["min_se_multiple"] * se),
                "rule_simultaneous": bool(mean > C["rule"]["min_points"] and mean - t_sim * se > 0)}
    regimes = []
    for parts, layers, drop in itertools.product(C["parts"], C["layers"], C["lr_drop"]):
        for metric in ("eval@120", "val@120"):
            g0 = vec("G0", layers, drop, 4, 1, metric); g1 = vec("G1", layers, drop, parts, 1, metric)
            row = {"parts": parts, "layers": layers, "lr_drop": drop, "metric": metric,
                   "means": {"G0": float(g0.mean()), "G1": float(g1.mean())},
                   "truncation_G0_minus_G1": paired(g0, g1),
                   "destructive_G1_minus_never_resync": paired(g1, vec("hold", layers, drop, parts, C["never_resync_sync"], metric)),
                   "destructive_G1_minus_zero": paired(g1, vec("zero", layers, drop, parts, 1, metric)), "by_sync": {}}
            for s in C["sync"]:
                h = vec("hold", layers, drop, parts, s, metric)
                row["by_sync"][str(s)] = {"means": {a: float(vec(a, layers, drop, parts, s, metric).mean()) for a in C["stale_arms"]},
                                          "staleness_G1_minus_hold": paired(g1, h),
                                          "recovered_sni_minus_hold": paired(vec("sni", layers, drop, parts, s, metric), h),
                                          "recovered_lag1_minus_hold": paired(vec("lag1", layers, drop, parts, s, metric), h)}
            regimes.append(row)
    prim = [(r, s) for r in regimes if r["metric"] == C["primary_metric"] for s in r["by_sync"]]
    summ = {"schema": "gnn-closedloop-gap-summary.v1", "raw_sha256": sha(HERE / "results" / "raw.json"), "n_seeds": n,
            "staleness_tests": m_tests, "t95": float(t95), "t_simultaneous_bonferroni": float(t_sim),
            "regimes_passing_unadjusted_rule": [[r["parts"], r["layers"], r["lr_drop"], int(s)] for r, s in prim if r["by_sync"][s]["staleness_G1_minus_hold"]["rule_unadjusted"]],
            "regimes_passing_simultaneous_rule": [[r["parts"], r["layers"], r["lr_drop"], int(s)] for r, s in prim if r["by_sync"][s]["staleness_G1_minus_hold"]["rule_simultaneous"]],
            "status": "exploratory regime-selection development measurement; any passing regime needs independent confirmation on new seeds",
            "regimes": regimes}
    out.write_text(json.dumps(summ, indent=2) + "\n")
    f = lambda p: f"{p['mean_pt']:+.2f}±{p['se_pt']:.2f}"
    for r in regimes:
        if r["metric"] != C["primary_metric"]: continue
        print(f"P{r['parts']} L{r['layers']} drop{int(r['lr_drop'])} G0 {r['means']['G0']:.3f} G1 {r['means']['G1']:.3f} trunc {f(r['truncation_G0_minus_G1'])} never {f(r['destructive_G1_minus_never_resync'])} zero {f(r['destructive_G1_minus_zero'])}")
        for s, v in r["by_sync"].items():
            print(f"    sync{s}: stale(G1-hold) {f(v['staleness_G1_minus_hold'])} {'*' if v['staleness_G1_minus_hold']['rule_unadjusted'] else ''}{'**' if v['staleness_G1_minus_hold']['rule_simultaneous'] else ''} | sni-hold {f(v['recovered_sni_minus_hold'])} | lag1-hold {f(v['recovered_lag1_minus_hold'])}")
    print("unadjusted:", summ["regimes_passing_unadjusted_rule"], "simultaneous:", summ["regimes_passing_simultaneous_rule"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["freeze", "run", "analyse"]); ap.add_argument("--workers", type=int, default=8); a = ap.parse_args()
    {"freeze": freeze, "run": lambda: run(a.workers), "analyse": analyse}[a.cmd]()
