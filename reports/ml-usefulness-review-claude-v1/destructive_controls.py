"""Persist the two destructive controls and paired differences cited in README.

Exploratory, post hoc: written after headroom.py's first run, at Codex's review
request. Refuses to overwrite. Constant learning rate only, seeds 20-27.
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import headroom as H

def zero_remote(x, y, adj, parts, seed, epochs=120):
    torch.manual_seed(seed); np.random.seed(seed)
    ei = adj.indices(); w = adj.values(); pt = torch.from_numpy(parts); cross = pt[ei[0]] != pt[ei[1]]; n = x.shape[0]
    A_in = torch.sparse_coo_tensor(ei[:, ~cross], w[~cross], (n, n)).coalesce()
    w1 = torch.nn.Parameter(torch.randn(x.shape[1], 16) * 0.05); w2 = torch.nn.Parameter(torch.randn(16, int(y.max()) + 1) * 0.05)
    opt = torch.optim.Adam([w1, w2], lr=.01, weight_decay=.0005); tr, va = H.screen.split_labels(y)
    for _ in range(epochs):
        h = torch.relu(torch.sparse.mm(adj, x @ w1)); z = torch.sparse.mm(A_in, h @ w2)
        loss = torch.nn.functional.cross_entropy(z[tr], y[tr]); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        h = torch.relu(torch.sparse.mm(adj, x @ w1)); z = torch.sparse.mm(adj, h @ w2)
    return float((z[va].argmax(1) == y[va]).float().mean())

def paired(a, b):
    d = np.asarray(a) - np.asarray(b); se = d.std(ddof=1) / np.sqrt(len(d)); t = 2.365  # t_{0.975, 7}
    return {"mean_diff": float(d.mean()), "sd_diff": float(d.std(ddof=1)), "ci95": [float(d.mean() - t * se), float(d.mean() + t * se)], "n": len(d)}

def main():
    out_path = HERE / "destructive-controls.json"
    if out_path.exists(): raise FileExistsError("refusing to overwrite")
    x, y, adj = H.screen.load_cora(H.DATA); parts = np.random.default_rng(321).integers(0, 4, size=len(y))
    ei = adj.indices(); pt = torch.from_numpy(parts)
    seeds = list(range(20, 28))
    arms = json.loads((HERE / "headroom-result.json").read_text())["closed_loop_reference"]["arms"]
    extra = {"hold_never_resync": [H.closed_loop(x, y, adj, parts, s, "hold", 10 ** 6)[0] for s in seeds],
             "zero_remote": [zero_remote(x, y, adj, parts, s) for s in seeds]}
    ref = arms["fresh_sync4"]["per_seed"]
    pairs = {f"{k}_minus_fresh_detached": paired(v["per_seed"], ref) for k, v in arms.items() if k != "fresh_sync4"}
    pairs.update({f"{k}_minus_fresh_detached": paired(v, ref) for k, v in extra.items()})
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    out = {"schema": "claude-destructive-controls.v1", "status": "exploratory, post hoc, single run, not preregistered",
           "scope": "constant learning rate only; seeds 20-27; 'fresh' means exact remote values with detached gradient, not full-gradient training",
           "cross_edge_weight_share": float(adj.values()[pt[ei[0]] != pt[ei[1]]].sum() / adj.values().sum()),
           "per_seed": extra, "summary": {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))} for k, v in extra.items()},
           "paired_val_accuracy_differences": pairs,
           "receipts_sha256": {"headroom.py": sha(HERE / "headroom.py"), "headroom-result.json": sha(HERE / "headroom-result.json"),
                               "screen.py": sha(H.ROOT / "operator-studies/gnn-sync-gate/screen.py"),
                               "cora.content": sha(H.DATA / "cora.content"), "cora.cites": sha(H.DATA / "cora.cites")},
           "versions": {"torch": torch.__version__, "numpy": np.__version__, "python": sys.version.split()[0]}}
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    for k, v in pairs.items(): print(k, round(v["mean_diff"], 4), [round(c, 4) for c in v["ci95"]])

if __name__ == "__main__": main()
