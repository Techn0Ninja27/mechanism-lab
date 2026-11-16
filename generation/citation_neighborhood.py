"""Bounded OpenAlex neighborhood snapshots, with an offline replay path.

Edges are retrieval leads, not evidence of contribution overlap. This is not
Connected Papers' similarity algorithm and not a historical citation snapshot.
"""
import argparse
import hashlib
import json
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

FIELDS = 'id,title,doi,publication_date,abstract_inverted_index,referenced_works,related_works'


def work_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'(https://openalex.org/)?W[0-9]+', value):
        raise ValueError('Expected an explicit OpenAlex work ID; resolve seed identity first')
    return value.rsplit('/', 1)[-1]


def abstract(index):
    if not index:
        return ''
    positions = {}
    for word, offsets in index.items():
        for offset in offsets:
            if not isinstance(offset, int) or not 0 <= offset < 100_000 or offset in positions:
                raise ValueError('Invalid abstract positions')
            positions[offset] = word
    return ' '.join(positions[i] for i in sorted(positions))


def snapshot(seed, cache_dir, *, allow_network=False, limit=40, opener=None):
    seed = work_id(seed)
    if type(limit) is not int or not 1 <= limit <= 50:
        raise ValueError('limit must be 1..50 per edge type')
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    receipts = []

    def fetch(path, params):
        url = 'https://api.openalex.org/' + path + '?' + urllib.parse.urlencode(params)
        key = hashlib.sha256(url.encode()).hexdigest()
        body_path, receipt_path = cache_dir / (key + '.json'), cache_dir / (key + '.receipt.json')
        if body_path.exists() or receipt_path.exists():
            body, receipt = body_path.read_bytes(), json.loads(receipt_path.read_text())
            if receipt['url'] != url or receipt['sha256'] != hashlib.sha256(body).hexdigest() or receipt['bytes'] != len(body):
                raise ValueError('Citation cache integrity mismatch')
        else:
            if not allow_network:
                raise FileNotFoundError('Missing citation snapshot; network is opt-in')
            request = urllib.request.Request(url, headers={'User-Agent': 'MechanismLab/CitationNeighborhood/1'})
            with (opener or urllib.request.urlopen)(request, timeout=30) as response:
                body = response.read(2_000_001)
            if len(body) > 2_000_000:
                raise ValueError('Citation response exceeds 2 MB')
            payload = json.loads(body)
            receipt = {'url': url, 'sha256': hashlib.sha256(body).hexdigest(), 'bytes': len(body),
                       'retrieved_at': datetime.now(timezone.utc).isoformat(),
                       'provider_reported_cost_usd': payload.get('meta', {}).get('cost_usd')}
            body_path.write_bytes(body)
            receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
        receipts.append(receipt)
        return json.loads(body)

    source = fetch('works/' + seed, {'select': FIELDS})
    if work_id(source['id']) != seed:
        raise ValueError('Seed was redirected/merged: explicitly resolve new identity')
    groups = {name: sorted({work_id(x) for x in source.get(field, [])})
              for name, field in [('references', 'referenced_works'), ('related', 'related_works')]}
    selected = sorted(set(groups['references'][:limit] + groups['related'][:limit]))
    records = {seed: source}
    if selected:
        batch = fetch('works', {'filter': 'openalex:' + '|'.join(selected), 'per_page': 100, 'select': FIELDS})
        for row in batch['results']:
            wid = work_id(row['id'])
            if wid not in selected:
                raise ValueError('Unexpected work in batch response')
            records[wid] = row
    citing = fetch('works', {'filter': 'cites:' + seed, 'per_page': limit,
                             'sort': 'publication_date:desc', 'select': FIELDS})
    incoming = []
    for row in citing['results']:
        wid = work_id(row['id'])
        if seed not in [work_id(x) for x in row.get('referenced_works', [])]:
            raise ValueError('Citing result lacks reported citation edge')
        records[wid] = row
        incoming.append(wid)
    edges = ([{'source': seed, 'target': x, 'type': 'cites'} for x in groups['references'][:limit]] +
             [{'source': seed, 'target': x, 'type': 'provider_related'} for x in groups['related'][:limit]] +
             [{'source': x, 'target': seed, 'type': 'cites'} for x in incoming])
    return {'schema': 'citation-neighborhood.v1', 'seed': seed, 'limit_per_edge_type': limit,
            'scope': 'Current bounded provider metadata; not historical, exhaustive, novelty or usefulness evidence.',
            'selection': 'References and related IDs sorted lexically; citing works newest first.',
            'coverage': {'reference_ids': len(groups['references']), 'related_ids': len(groups['related']),
                         'citing_total': citing['meta']['count'], 'citing_retained': len(incoming),
                         'unresolved_selected_ids': sorted(set(selected) - set(records))},
            'records': [{'id': wid, 'title': r['title'], 'doi': r.get('doi'),
                         'publication_date': r.get('publication_date'),
                         'abstract': abstract(r.get('abstract_inverted_index')),
                         'evidence_level': 'abstract_or_metadata_only'} for wid, r in sorted(records.items())],
            'edges': edges, 'receipts': receipts, 'novelty_established': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=40)
    parser.add_argument('--allow-network', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Output already exists; choose a separate replay output')
    result = snapshot(args.seed, args.cache, allow_network=args.allow_network, limit=args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'records': len(result['records']), 'edges': len(result['edges']), 'coverage': result['coverage']}))


if __name__ == '__main__':
    main()
