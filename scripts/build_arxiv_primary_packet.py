"""Build a small primary-abstract packet from explicitly pinned arXiv versions.

Fetches only arxiv.org/abs/<id>v<version>, preserves HTML and hashes, and can
rebuild offline from those exact response bytes. No generated summaries.
"""
import argparse
import hashlib
import json
import re
import time
import urllib.request
from datetime import date
from html.parser import HTMLParser
from pathlib import Path

class Meta(HTMLParser):
    def __init__(self):
        super().__init__(); self.values = {}
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'meta' and attrs.get('name', '').startswith('citation_'):
            self.values.setdefault(attrs['name'], []).append(attrs.get('content', ''))

def parse(payload, pinned_id, url):
    meta = Meta(); meta.feed(payload.decode('utf-8')); fields = meta.values
    def one(key):
        values = fields.get('citation_' + key, [])
        if len(values) != 1 or not values[0].strip():
            raise ValueError('missing/ambiguous citation metadata: ' + key)
        return re.sub(r'\s+', ' ', values[0]).strip()
    bare_id = re.sub(r'v\d+$', '', pinned_id)
    if one('arxiv_id') != bare_id:
        raise ValueError('response paper ID mismatch')
    published = date.fromisoformat(one('date').replace('/', '-')).isoformat()
    # Explicit v1 pins avoid conflating first submission with later revisions.
    if not pinned_id.endswith('v1'):
        raise ValueError('this packet extractor supports first-version pins only')
    return {'id': pinned_id, 'arxiv_id': bare_id, 'title': one('title'),
            'abstract': one('abstract'), 'authors': fields.get('citation_author', []),
            'published': published, 'updated': published, 'url': url,
            'pdf_url': 'https://arxiv.org/pdf/' + pinned_id,
            'corpus_group': 'ml', 'retrieval_method': 'arxiv_primary_html',
            'retrieved_at': date.today().isoformat(),
            'source_hash': hashlib.sha256(payload).hexdigest()}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--id', action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--offline', action='store_true')
    a = p.parse_args()
    if not 1 <= len(a.id) <= 20 or len(set(a.id)) != len(a.id) or any(not re.fullmatch(r'\d{4}\.\d{4,5}v1', i) for i in a.id):
        p.error('provide 1–20 distinct modern arXiv IDs pinned to v1')
    a.output.mkdir(parents=True, exist_ok=True)
    if (a.output/'records.jsonl').exists() or (a.output/'manifest.json').exists():
        raise FileExistsError('preserve completed packet; choose a fresh output')
    records = []; receipts = []
    for index, aid in enumerate(a.id):
        url = 'https://arxiv.org/abs/' + aid; path = a.output/(aid+'.html')
        if path.exists():
            payload = path.read_bytes()
        else:
            if a.offline: raise FileNotFoundError(path)
            if index: time.sleep(3)
            req = urllib.request.Request(url, headers={'User-Agent':'MechanismLab/1.0 primary-source audit'})
            with urllib.request.urlopen(req, timeout=30) as response:
                if response.geturl() != url: raise ValueError('unexpected redirect')
                payload = response.read(1_000_001)
            if len(payload) > 1_000_000: raise ValueError('response size bound exceeded')
            parse(payload, aid, url)
            path.write_bytes(payload)
        records.append(parse(payload, aid, url))
        receipts.append({'id':aid,'url':url,'path':path.name,'sha256':hashlib.sha256(payload).hexdigest(),'bytes':len(payload)})
    packet = ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True)+'\n' for r in records)
    (a.output/'records.jsonl').write_text(packet)
    manifest = {'schema':'arxiv-primary-packet.v1','date':date.today().isoformat(),
                'records':len(records),'responses':receipts,
                'packet_sha256':hashlib.sha256(packet.encode()).hexdigest(),
                'extractor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'scope':'Primary abstracts, not full methods or novelty judgments.'}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'records':len(records),'packet_sha256':manifest['packet_sha256']}))

if __name__ == '__main__': main()
