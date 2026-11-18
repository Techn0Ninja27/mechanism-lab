"""Reproduce in a fresh minimal workspace, preserving all published outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def prepare(output):
    output = output.resolve()
    if output.exists():
        raise FileExistsError('Choose a new output directory; existing paths are never overwritten')
    freeze = json.loads((HERE/'freeze.json').read_text())
    sources = []
    for name, digest in freeze['sha256'].items():
        src = HERE/name if (HERE/name).is_file() else ROOT/name
        if hashlib.sha256(src.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Frozen input changed: {name}')
        sources.append(src)
    # These are post-freeze analysis/entry-point files, not changes to training.
    sources += [HERE/'analyse_v2.py', HERE/'requirements.txt']
    for src in sources:
        dest = output/src.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
    return output/HERE.relative_to(ROOT)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 8: parser.error('workers must be between 1 and 8')
    study = prepare(args.output)
    subprocess.run([sys.executable, 'run.py', 'freeze'], cwd=study, check=True)
    if not args.prepare_only:
        subprocess.run([sys.executable, 'run.py', 'run', '--workers', str(args.workers)], cwd=study, check=True)
        subprocess.run([sys.executable, 'analyse_v2.py'], cwd=study, check=True)
    print(study)
