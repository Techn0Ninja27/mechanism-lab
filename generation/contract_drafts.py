"""Fail-closed validation for model-proposed operational contract drafts.

Drafts are evidence-linked hypotheses. They are deliberately separate from the
curated interface-contract registry and can never establish compatibility or
novelty by passing this module.
"""
from __future__ import annotations

from collections.abc import Mapping
from .interface_contracts import validate_contracts
from .sourcepack import validate_source_pack

MAX_SOURCES = 5
FIELDS = {'artifact','representation','granularity','phase','role','exposure','conditions','operation_scope'}
DECISIONS = {'accept','revise','reject','insufficient_evidence'}


def _bounded_text(value, name, limit):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f'invalid {name}')


def validate_drafts(value, pack, source_ids):
    validate_source_pack(pack)
    if not isinstance(source_ids, list) or not 1 <= len(source_ids) <= MAX_SOURCES:
        raise ValueError('source_ids must contain 1-5 sources')
    if len(set(source_ids)) != len(source_ids):
        raise ValueError('duplicate source_ids')
    known = {s['id'] for s in pack['sources']}
    if any(x not in known for x in source_ids):
        raise ValueError('unknown source id')
    if not isinstance(value, Mapping) or set(value) != {'records'} or not isinstance(value['records'], list):
        raise ValueError('draft envelope schema')
    if len(value['records']) != len(source_ids):
        raise ValueError('one draft record required per source')
    by_source = {}; contract_ids = set(); result = []
    for record in value['records']:
        if not isinstance(record, Mapping) or set(record) != {'source_id','contract','unknowns'}:
            raise ValueError('draft record schema')
        sid = record['source_id']
        if sid not in source_ids or sid in by_source:
            raise ValueError('draft source identity or duplicate')
        unknowns = record['unknowns']
        if not isinstance(unknowns, list) or not 1 <= len(unknowns) <= 8:
            raise ValueError('draft unknowns required')
        for item in unknowns: _bounded_text(item, 'unknown', 500)
        card = record['contract']
        checked = None
        if card is not None:
            if not isinstance(card, Mapping): raise ValueError('contract must be object or null')
            if card.get('source_id') != sid: raise ValueError('contract source mismatch')
            cid = card.get('id')
            if not isinstance(cid, str) or cid in contract_ids: raise ValueError('contract id missing or duplicate')
            contract_ids.add(cid)
            checked = validate_contracts({'schema':'interface-contracts.v1','contracts':[dict(card)]}, pack,
                                         expected_interpretation='model_proposed_not_curated')[cid]
        row = {'source_id': sid, 'contract': checked, 'unknowns': list(unknowns),
               'interpretation':'model_proposed_not_curated', 'executable_compatibility_proven':False,
               'novelty_established':False}
        by_source[sid] = row; result.append(row)
    if set(by_source) != set(source_ids): raise ValueError('missing source draft')
    return result


def validate_review(value, records, pack):
    validate_source_pack(pack)
    if not isinstance(value, Mapping) or set(value) != {'assessments'} or not isinstance(value['assessments'], list):
        raise ValueError('draft review schema')
    source_ids = [r['source_id'] for r in records]
    if len(value['assessments']) != len(source_ids): raise ValueError('one review per draft')
    by_source = {r['source_id']: r for r in records}; seen = set(); output=[]
    chunks = {c['id']: c for c in pack['chunks']}
    for assessment in value['assessments']:
        if not isinstance(assessment, Mapping) or set(assessment) != {'source_id','contract_id','decision','reason','issues'}:
            raise ValueError('assessment schema')
        sid = assessment['source_id']
        if sid not in by_source or sid in seen: raise ValueError('assessment source identity')
        seen.add(sid); _bounded_text(assessment['reason'], 'review reason', 2400)
        if assessment['decision'] not in DECISIONS: raise ValueError('invalid review decision')
        card = by_source[sid]['contract']; cid = assessment['contract_id']
        if cid != (card['id'] if card is not None else None): raise ValueError('review contract identity')
        issues = assessment['issues']
        if not isinstance(issues, list) or len(issues) > 16: raise ValueError('invalid review issues')
        ports = {p['id'] for side in ('inputs','outputs') for p in (card or {}).get(side, [])}
        clean=[]
        for issue in issues:
            if not isinstance(issue, Mapping) or set(issue) != {'port_id','field','reason','evidence'}:
                raise ValueError('issue schema')
            if issue['port_id'] is not None and issue['port_id'] not in ports: raise ValueError('unknown issue port')
            if issue['field'] not in FIELDS: raise ValueError('unknown issue field')
            _bounded_text(issue['reason'], 'issue reason', 1500)
            evidence=issue['evidence']
            if not isinstance(evidence,list) or len(evidence)>8: raise ValueError('invalid issue evidence')
            ev=[]
            for item in evidence:
                if not isinstance(item,Mapping) or set(item)!={'chunk_id','quote'}: raise ValueError('evidence schema')
                chunk=chunks.get(item['chunk_id'])
                if not chunk or chunk['source_id'] != sid or not isinstance(item['quote'],str) or not item['quote'].strip() or len(item['quote'])>600 or item['quote'] not in chunk['text']:
                    raise ValueError('issue evidence quote mismatch')
                ev.append(dict(item))
            clean.append({'port_id':issue['port_id'],'field':issue['field'],'reason':issue['reason'],'evidence':ev})
        if assessment['decision']=='accept' and (card is None or clean):
            raise ValueError('accept requires contract and no issues')
        output.append({'source_id':sid,'contract_id':cid,'decision':assessment['decision'],'reason':assessment['reason'],'issues':clean})
    if seen != set(source_ids): raise ValueError('review missing source')
    return {'assessments':output,'scope':'Model-proposed contract review; semantic truth, executable compatibility, usefulness and novelty remain unverified.', 'curated':False, 'executable_compatibility_proven':False, 'novelty_established':False}
