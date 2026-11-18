"""Rerun headroom.py into a fresh directory without touching the frozen result.

headroom.py is left byte-identical to the hash in destructive-controls.json.
It writes next to its own __file__, so this wrapper points that at --output.
"""
import argparse, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ap = argparse.ArgumentParser(); ap.add_argument("--output", type=Path, required=True); a = ap.parse_args()
out = a.output.resolve()
if out.exists(): raise FileExistsError("choose a fresh output directory")
if out == HERE or HERE in out.parents: raise ValueError("output must be outside the frozen report directory")
out.mkdir(parents=True)
import headroom
headroom.__file__ = str(out / "headroom.py")   # main() writes headroom-result.json beside __file__
headroom.main()
print("wrote", out / "headroom-result.json")
