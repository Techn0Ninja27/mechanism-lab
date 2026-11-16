#!/usr/bin/env python3
"""Explicit, pinned public inference-asset installer; never runs on app startup."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile


def main():
    here = Path(__file__).resolve().parent
    root = Path(os.environ.get('ALIEN_REPO', here.parent / 'upstream' if (here.parent / 'upstream/alien.py').exists() else here.parent)).resolve()
    manifest = root / 'local_runner/assets-manifest.json'
    records = json.loads(manifest.read_text())
    downloads = root / 'downloads'
    downloads.mkdir(exist_ok=True)
    for record in records:
        repo, revision = record['repo'], record['revision']
        is_data = 'alien-data' in repo
        for item in record['files']:
            name = item['path']
            rel = Path(name)
            if rel.is_absolute() or '..' in rel.parts: raise ValueError('Unsafe manifest path')
            target = downloads / rel if is_data else root / 'models' / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            def valid():
                if not target.is_file(): return False
                with target.open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest() == item['sha256']
            if not valid():
                prefix = 'datasets/' if is_data else ''
                url = f'https://huggingface.co/{prefix}{repo}/resolve/{revision}/{name}'
                part = target.with_suffix(target.suffix + '.part')
                subprocess.run(['curl', '-fL', '--retry', '2', '-o', str(part), url], check=True)
                with part.open('rb') as stream: digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                if digest != item['sha256']: raise ValueError('Checksum mismatch: ' + name)
                part.replace(target)
            print('Verified', name, flush=True)
            if is_data and name.endswith('.tar.gz'):
                with tarfile.open(target) as archive: archive.extractall(root, filter='data')
    print('Assets installed. Legacy provenance warnings still apply.')

if __name__ == '__main__': main()
