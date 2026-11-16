"""Verify compact trace artifacts without loading the model."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np

def main():
    p = argparse.ArgumentParser(); p.add_argument("run", type=Path); a = p.parse_args()
    m = json.loads((a.run / "manifest.json").read_text())
    assert m["status"] == "completed_trace_smoke_only"
    assert len(m["cases"]) == 2 and m["device"] in {"mps", "cpu"}
    for row in m["cases"]:
        path = a.run / row["npz"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["npz_sha256"]
        z = np.load(path)
        assert z["visual_positions"].shape == (66,)
        assert z["prefill_visual_attention_mean"].shape == (30,)
        assert z["decode_attention_mean"].shape == (4, 30)
        assert np.isfinite(z["prefill_visual_attention_mean"]).all()
        assert np.isfinite(z["decode_attention_mean"]).all()
    print(json.dumps({"status": "verified", "cases": len(m["cases"])}))

if __name__ == "__main__": main()
