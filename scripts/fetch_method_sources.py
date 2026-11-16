"""Fetch only pinned official source files, verifying recorded byte hashes."""
import argparse
import hashlib
import json
import pathlib
import re
import sys
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
MAX_BYTES = 5_000_000

def fetch(manifest, root=ROOT):
    completed = []
    root = pathlib.Path(root).resolve()
    for snapshot in manifest['snapshots']:
        repository = snapshot['repository']
        if not re.fullmatch(r'https://github.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repository):
            raise ValueError('Expected public GitHub repository URL')
        commit = snapshot['commit']
        if not re.fullmatch(r'[a-f0-9]{40}', commit):
            raise ValueError('Exact commit required')
        relative = pathlib.Path(snapshot['local_directory'])
        if relative.is_absolute() or '..' in relative.parts or relative.parts[0] != 'results':
            raise ValueError('Snapshots must stay in ignored results storage')
        for name, expected in snapshot['files'].items():
            child = pathlib.Path(name)
            if child.is_absolute() or '..' in child.parts or not re.fullmatch(r'[a-f0-9]{64}', expected):
                raise ValueError('Unsafe filename or invalid SHA256')
            path = (root / relative / child).resolve()
            if root not in path.parents:
                raise ValueError('Source path escapes root')
            if path.exists():
                if path.stat().st_size > MAX_BYTES or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    raise ValueError('Existing source does not match pinned hash: ' + str(path))
                completed.append({'path':str(path.relative_to(root)), 'status':'verified_cache'})
                continue
            url = 'https://raw.githubusercontent.com/' + repository.split('github.com/')[1] + '/' + commit + '/' + urllib.parse.quote(name)
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read(MAX_BYTES + 1)
            if len(data) > MAX_BYTES or hashlib.sha256(data).hexdigest() != expected:
                raise ValueError('Downloaded source fails size/hash verification: ' + name)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as f: f.write(data)
            completed.append({'path':str(path.relative_to(root)), 'status':'downloaded_verified'})
    return completed

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=pathlib.Path,default=ROOT/'docs/method-grounding/sourcepack-manifest.json')
    a=p.parse_args()
    print(json.dumps(fetch(json.loads(a.manifest.read_text())),indent=2))
