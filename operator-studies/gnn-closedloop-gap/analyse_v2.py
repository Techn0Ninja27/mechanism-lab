"""Audited summary (post-freeze analysis code; written while the frozen run was
in progress, after Codex's integrity review aba96c62, before any result was seen).

Differs from `run.py analyse` only in integrity checks and labelling:
- verifies raw.json is bound to freeze.json and that frozen files are unchanged;
- validates the exact expected job-key set, uniqueness, 16 seeds, finite metrics;
- binds the calibration coefficients into the summary;
- the Bonferroni simultaneous interval is attached ONLY to the primary family
  (24 staleness comparisons, G1 - hold, eval@120). Everything else is pointwise.
Refuses to overwrite its output. Does not import or modify run.py's analyse.
"""
import hashlib, itertools, json, math, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def t_ppf(q, df):
    xs = np.linspace(-60, 60, 2_400_001); pdf = (1 + xs ** 2 / df) ** (-(df + 1) / 2); cdf = np.cumsum(pdf); cdf /= cdf[-1]
    return float(xs[np.searchsorted(cdf, q)])


def main():
    out = HERE / "results" / "summary-v2.json"
    if out.exists(): raise FileExistsError("results/summary-v2.json exists")
    fz = json.loads((HERE / "freeze.json").read_text()); C = fz["config"]; raw = json.loads((HERE / "results" / "raw.json").read_text())
    checks = {}
    checks["raw_bound_to_freeze"] = raw["freeze_sha256"] == sha(HERE / "freeze.json")
    root = HERE.parents[1]
    cur = {k: sha(HERE / k if (HERE / k).exists() else root / k) for k in fz["sha256"]}
    checks["frozen_files_unchanged"] = cur == fz["sha256"]
    seeds = C["measurement_seeds"]; n = len(seeds); NR = C["never_resync_sync"]
    cal = raw["calibration"]
    expected = set(); coef_expected = {}
    for seed, layers, drop in itertools.product(seeds, C["layers"], C["lr_drop"]):
        expected.add(("G0", seed, layers, drop, 4, 1))
        for parts in C["parts"]:
            expected |= {("G1", seed, layers, drop, parts, 1), ("zero", seed, layers, drop, parts, 1), ("hold", seed, layers, drop, parts, NR)}
            for s in C["sync"]:
                for arm in C["stale_arms"]: expected.add((arm, seed, layers, drop, parts, s))
    for layers, drop, s in itertools.product(C["layers"], C["lr_drop"], C["sync"]):
        coef_expected[(layers, drop, s)] = [cal[f"layers{layers}_drop{int(drop)}_sync{s}_hidden{k + 1}"]["coef"] for k in range(layers - 1)]
    keys = [(r["arm"], r["seed"], r["layers"], r["lr_drop"], r["parts"], r["sync"]) for r in raw["runs"]]
    checks["n_seeds_16"] = n == 16 and seeds == list(range(30, 46))
    checks["unique_keys"] = len(keys) == len(set(keys))
    checks["exact_expected_key_set"] = set(keys) == expected and len(keys) == len(expected) == raw["n_runs"]
    def finite(r):
        vals = [r["val"].get(e) for e in ("40", "80", "120")] + [r["eval"].get(e) for e in ("40", "80", "120")] + [r["final_train_loss"]]
        return all(isinstance(v, float) and math.isfinite(v) for v in vals)
    bad = [k for k, r in zip(keys, raw["runs"]) if not finite(r)]
    checks["all_required_metrics_finite"] = not bad
    checks["lag1_coefs_match_calibration"] = all(list(r["coef"]) == coef_expected[(r["layers"], r["lr_drop"], r["sync"])] for r in raw["runs"] if r["arm"] == "lag1")
    if not all(checks.values()):
        print(json.dumps({"checks": checks, "nonfinite_examples": [list(map(str, b)) for b in bad[:10]], "n_nonfinite": len(bad)}, indent=1)); sys.exit("integrity checks failed; no summary written")
    idx = dict(zip(keys, raw["runs"]))
    def vec(arm, layers, drop, parts, sync, metric):
        k, ep = metric.split("@"); return np.array([idx[(arm, s, layers, drop, parts, sync)][k][ep] for s in seeds])
    m = len(C["parts"]) * len(C["layers"]) * len(C["lr_drop"]) * len(C["sync"])
    t95 = t_ppf(0.975, n - 1); t_sim = t_ppf(1 - 0.025 / m, n - 1)
    def paired(a, b, primary=False):
        d = (a - b) * 100; mean = float(d.mean()); se = float(d.std(ddof=1) / math.sqrt(n))
        o = {"mean_pt": mean, "se_pt": se, "pointwise_ci95_pt": [mean - t95 * se, mean + t95 * se], "n_positive_of_16": int((d > 0).sum()),
             "interval_type": "pointwise, descriptive"}
        if primary:
            o.update({"interval_type": "primary family: pointwise and Bonferroni-24 simultaneous",
                      "simultaneous_ci_pt": [mean - t_sim * se, mean + t_sim * se],
                      "flag_exploratory_unadjusted": bool(mean > C["rule"]["min_points"] and mean > C["rule"]["min_se_multiple"] * se),
                      "flag_simultaneous": bool(mean > C["rule"]["min_points"] and mean - t_sim * se > 0)})
        return o
    regimes = []
    for parts, layers, drop in itertools.product(C["parts"], C["layers"], C["lr_drop"]):
        for metric in ("eval@120", "val@120"):
            prim = metric == C["primary_metric"]
            g0 = vec("G0", layers, drop, 4, 1, metric); g1 = vec("G1", layers, drop, parts, 1, metric)
            row = {"parts": parts, "layers": layers, "lr_drop": drop, "metric": metric, "means": {"G0": float(g0.mean()), "G1": float(g1.mean())},
                   "truncation_G0_minus_G1": paired(g0, g1), "destructive_G1_minus_never_resync": paired(g1, vec("hold", layers, drop, parts, NR, metric)),
                   "destructive_G1_minus_zero": paired(g1, vec("zero", layers, drop, parts, 1, metric)), "by_sync": {}}
            for s in C["sync"]:
                h = vec("hold", layers, drop, parts, s, metric)
                row["by_sync"][str(s)] = {"means": {a: float(vec(a, layers, drop, parts, s, metric).mean()) for a in C["stale_arms"]},
                                          "lag1_coefs": coef_expected[(layers, drop, s)],
                                          "staleness_G1_minus_hold": paired(g1, h, primary=prim),
                                          "recovered_sni_minus_hold": paired(vec("sni", layers, drop, parts, s, metric), h),
                                          "recovered_lag1_minus_hold": paired(vec("lag1", layers, drop, parts, s, metric), h)}
            regimes.append(row)
    P = [(r, s) for r in regimes if r["metric"] == C["primary_metric"] for s in r["by_sync"]]
    summ = {"schema": "gnn-closedloop-gap-summary.v2", "status": "exploratory regime-selection development measurement; flagged regimes require independent confirmation on new seeds",
            "integrity_checks": checks, "raw_sha256": sha(HERE / "results" / "raw.json"), "freeze_sha256": sha(HERE / "freeze.json"), "analysis_code_sha256": sha(__file__),
            "n_seeds": n, "primary_family": {"comparison": "G1 - hold", "metric": C["primary_metric"], "size": m, "t95": t95, "t_bonferroni": t_sim},
            "other_families": "truncation, destructive controls, predictor recovery and val@120 carry pointwise intervals only; no multiplicity coverage",
            "flagged_exploratory_unadjusted": [[r["parts"], r["layers"], r["lr_drop"], int(s)] for r, s in P if r["by_sync"][s]["staleness_G1_minus_hold"]["flag_exploratory_unadjusted"]],
            "flagged_simultaneous": [[r["parts"], r["layers"], r["lr_drop"], int(s)] for r, s in P if r["by_sync"][s]["staleness_G1_minus_hold"]["flag_simultaneous"]],
            "regimes": regimes}
    out.write_text(json.dumps(summ, indent=2) + "\n")
    f = lambda p: f"{p['mean_pt']:+.2f}±{p['se_pt']:.2f}"
    for r in regimes:
        if r["metric"] != C["primary_metric"]: continue
        print(f"P{r['parts']} L{r['layers']} drop{int(r['lr_drop'])} G0 {r['means']['G0']:.3f} G1 {r['means']['G1']:.3f} | trunc {f(r['truncation_G0_minus_G1'])} | never {f(r['destructive_G1_minus_never_resync'])} | zero {f(r['destructive_G1_minus_zero'])}")
        for s, v in r["by_sync"].items():
            st = v["staleness_G1_minus_hold"]
            print(f"    sync{s:>2}: G1-hold {f(st)} {'*' if st['flag_exploratory_unadjusted'] else ' '}{'*' if st['flag_simultaneous'] else ' '} | sni-hold {f(v['recovered_sni_minus_hold'])} | lag1-hold {f(v['recovered_lag1_minus_hold'])} c={v['lag1_coefs']}")
    print("checks", checks); print("flag unadjusted:", summ["flagged_exploratory_unadjusted"]); print("flag simultaneous:", summ["flagged_simultaneous"])


if __name__ == "__main__": main()
