"""Offline, bounded selection of citation-neighborhood abstracts for generation.

Validation checks shape and binds the exact snapshot, not source authenticity,
semantic relevance, exhaustive retrieval, or historical citation availability.
"""
import hashlib
import json
from pathlib import Path
from generation.citation_neighborhood import work_id

MAX_CHARS = 16000
SCOPE = ('Selected provider abstracts from a current citation snapshot; citation and '
         'related links are retrieval leads, not evidence of overlap or novelty. '
         'Not a historical backtest or exhaustive prior-art search.')


def _read(path, cap):
    with Path(path).open('rb') as stream:
        raw = stream.read(cap + 1)
    if len(raw) > cap:
        raise ValueError('Citation evidence exceeds byte limit')
    return raw, json.loads(raw)


def load_citation_evidence(config_path=None):
    if config_path is None:
        return {'summary': {'status': 'not_requested'}, 'chunks': []}
    config_path = Path(config_path)
    _, config = _read(config_path, 10000)
    if not isinstance(config, dict) or set(config) != {'snapshot_path', 'selected_ids'}:
        raise ValueError('Citation config requires snapshot_path and selected_ids')
    selected = config['selected_ids']
    if (not isinstance(selected, list) or not 1 <= len(selected) <= 8 or
            any(not isinstance(x, str) or work_id(x) != x for x in selected) or len(set(selected)) != len(selected)):
        raise ValueError('Choose 1..8 unique bare OpenAlex work IDs')
    if not isinstance(config['snapshot_path'], str) or not config['snapshot_path'].strip():
        raise ValueError('Citation snapshot path required')
    path = config_path.parent / config['snapshot_path']
    raw, snapshot = _read(path, 5_000_000)
    if not isinstance(snapshot, dict) or snapshot.get('schema') != 'citation-neighborhood.v1':
        raise ValueError('Unsupported citation snapshot')
    seed = work_id(snapshot.get('seed'))
    rows, edges = snapshot.get('records'), snapshot.get('edges')
    if not isinstance(rows, list) or len(rows) > 151 or not isinstance(edges, list) or len(edges) > 150:
        raise ValueError('Unbounded citation snapshot')
    records = {}
    for row in rows:
        if not isinstance(row, dict): raise ValueError('Invalid citation record')
        wid = work_id(row.get('id'))
        if wid in records: raise ValueError('Duplicate citation record')
        for key, limit in [('title', 2000), ('abstract', 100000)]:
            if not isinstance(row.get(key), str) or len(row[key]) > limit:
                raise ValueError('Invalid citation text')
        records[wid] = row
    graph_ids = {seed}
    normalized_edges = []
    for edge in edges:
        if not isinstance(edge, dict) or edge.get('type') not in ('cites', 'provider_related'):
            raise ValueError('Invalid citation edge')
        source, target = work_id(edge.get('source')), work_id(edge.get('target'))
        if seed not in (source,target) or (edge['type']=='provider_related' and source!=seed):
            raise ValueError('Expected a one-hop seed neighborhood')
        graph_ids.update((source,target))
        normalized_edges.append({'source':source,'target':target,'type':edge['type']})
    if seed not in records or not set(selected) <= graph_ids:
        raise ValueError('Selected IDs must belong to the supplied seed neighborhood')
    chunks, omitted, used = [], [], 0
    for wid in selected:
        row = records.get(wid)
        if row is None or not row['abstract'].strip() or used >= MAX_CHARS:
            omitted.append(wid); continue
        text = row['abstract'][:min(4000, MAX_CHARS-used)]
        source_hash = hashlib.sha256((row['title']+'\n'+row['abstract']).encode()).hexdigest()
        # Canonical provider IDs avoid following arbitrary URLs from imported data.
        chunks.append({'id': 'citation:'+wid+':'+source_hash[:16], 'source_id': 'openalex:'+wid,
                       'title': row['title'], 'text': text, 'url': 'https://openalex.org/'+wid,
                       'evidence_level': 'provider_abstract', 'source_hash': source_hash,
                       'truncated': len(text) < len(row['abstract']),
                       'relations': [edge for edge in normalized_edges if wid in (edge['source'], edge['target'])][:10]})
        used += len(text)
    summary = {'status': 'included', 'snapshot_sha256': hashlib.sha256(raw).hexdigest(), 'seed': seed,
               'selected_ids': selected, 'included_ids': [c['source_id'].split(':')[1] for c in chunks],
               'omitted_ids': omitted, 'included_chars': used, 'scope': SCOPE}
    return {'summary': summary, 'chunks': chunks}
