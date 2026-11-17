"""Bounded claim/evidence contracts. Coverage is reported, never proved novelty."""
CLAIM_SCHEMA = {'id':'stable claim ID','text':'one proposed contribution',
                'role':'central|supporting','kind':'problem|method|resource|expected_result'}
ASSESSMENT_SCHEMA = {'claim_id':'exact proposal claim ID','coverage':'covered|partial|not_found|unknown',
                     'checked_chunk_ids':['supplied evidence chunk IDs examined'],
                     'evidence':[{'chunk_id':'supplied chunk ID','quote':'exact short substring','supports':'specific overlap premise'}],
                     'reason':'explain the overlap or missing evidence; expected results are hypotheses, not observations'}

def _text(value,limit):
    return isinstance(value,str) and bool(value.strip()) and len(value)<=limit

def _bounded_values(values, limit=3, width=160):
    """Render malformed evidence IDs without echoing unbounded model output."""
    out=[]
    for value in values[:limit]:
        if isinstance(value,str):
            out.append(repr(value[:width] + ('…' if len(value)>width else '')))
        else:
            out.append(repr(value)[:width])
    if len(values)>limit: out.append(f'… and {len(values)-limit} more')
    return ', '.join(out)

def _checked_id_error(claim_id, checked, source):
    prefix=f'Claim {claim_id!r}: '
    if not isinstance(checked,list):
        return prefix+'checked_chunk_ids must be a list'
    if len(checked)>30:
        return prefix+f'checked_chunk_ids has {len(checked)} IDs; maximum is 30'
    bad_types=[x for x in checked if not isinstance(x,str)]
    if bad_types:
        return prefix+'checked_chunk_ids contains non-string values: '+_bounded_values(bad_types)
    duplicates=[];seen=set()
    for value in checked:
        if value in seen and value not in duplicates: duplicates.append(value)
        seen.add(value)
    if duplicates:
        return prefix+'checked_chunk_ids contains duplicate IDs: '+_bounded_values(duplicates)
    unknown=[x for x in checked if x not in source]
    if unknown:
        return prefix+'checked_chunk_ids contains unknown IDs: '+_bounded_values(unknown)
    return None

def validate_claims(claims):
    if not isinstance(claims,list) or not 1<=len(claims)<=6:
        raise ValueError('One to six contribution claims required')
    ids=set()
    for c in claims:
        if not isinstance(c,dict) or set(c)!=set(CLAIM_SCHEMA):raise ValueError('Claim schema mismatch')
        if not _text(c['id'],80) or c['id'] in ids:raise ValueError('Unique bounded claim IDs required')
        ids.add(c['id'])
        if not _text(c['text'],1200) or c['role'] not in ('central','supporting') or c['kind'] not in ('problem','method','resource','expected_result'):
            raise ValueError('Invalid contribution claim')
    if not any(c['role']=='central' for c in claims):raise ValueError('At least one central claim required')
    return claims

def assess(claims,assessments,chunks):
    validate_claims(claims)
    if not isinstance(assessments,list) or len(assessments)!=len(claims):raise ValueError('Assessment required for every claim')
    by_id={c['id']:c for c in claims};source={c['id']:c['text'] for c in chunks};seen=set();rows=[]
    for a in assessments:
        if not isinstance(a,dict) or set(a)!=set(ASSESSMENT_SCHEMA):raise ValueError('Claim assessment schema mismatch')
        cid=a['claim_id']
        if not isinstance(cid,str) or cid not in by_id or cid in seen:raise ValueError('Unknown or repeated claim assessment')
        seen.add(cid)
        if a['coverage'] not in ('covered','partial','not_found','unknown') or not _text(a['reason'],2400):raise ValueError('Invalid coverage assessment')
        checked=a['checked_chunk_ids']
        checked_error=_checked_id_error(cid,checked,source)
        if checked_error: raise ValueError(checked_error)
        if a['coverage']!='unknown' and not checked:raise ValueError('Missing evidence cannot establish coverage or a completed search')
        ev=a['evidence']
        if not isinstance(ev,list) or len(ev)>5:raise ValueError('Invalid claim evidence')
        if a['coverage'] in ('covered','partial') and not ev:raise ValueError('Overlap requires source evidence')
        for e in ev:
            if not isinstance(e,dict) or set(e)!={'chunk_id','quote','supports'}:raise ValueError('Invalid claim evidence schema')
            if e['chunk_id'] not in checked or not _text(e['quote'],600) or e['quote'] not in source[e['chunk_id']] or not _text(e['supports'],800):
                raise ValueError('Claim evidence must cite an examined exact span')
        rows.append({**a,'claim':by_id[cid]})
    core=[a['coverage'] for a in rows if a['claim']['role']=='central']
    if any(x=='unknown' for x in core):status='central_evidence_unresolved'
    elif all(x=='covered' for x in core):status='central_overlap_reported'
    elif all(x=='not_found' for x in core):status='no_match_in_examined_evidence'
    else:status='mixed_or_partial_overlap_reported'
    return {'status':status,'claims':rows,'novelty_estimate':None,'novelty_established':False,'train_eligible':False,
            'scope':'Model-reported coverage in explicitly examined evidence only. Exact quotes do not prove entailment; no-match is not novelty. Expected-result claims remain hypotheses. No automatic proposal rejection.'}
