"""Source-linked declarative interface checks, not executable type proofs."""
import re
from .sourcepack import validate_source_pack

SCHEMA='interface-contracts.v1'
TOKEN=re.compile(r'^[a-z][a-z0-9_]{0,79}$')

def _token(value, nullable=False):
    if nullable and value is None:return
    if not isinstance(value,str) or not TOKEN.fullmatch(value):raise ValueError('invalid contract identifier')
    if nullable and value in {'unknown','unspecified','not_stated','not_specified','none','null'}:raise ValueError('Use JSON null for an unknown contract field')

def validate_contracts(registry,pack, *, expected_interpretation='curated_not_formally_verified'):
    if expected_interpretation not in {'curated_not_formally_verified','model_proposed_not_curated'}:raise ValueError('unsupported interpretation mode')
    validate_source_pack(pack)
    if not isinstance(registry,dict) or set(registry)!={'schema','contracts'} or registry['schema']!=SCHEMA:raise ValueError('contract schema')
    if not isinstance(registry['contracts'],list) or not 1<=len(registry['contracts'])<=32:raise ValueError('contract count')
    chunks={c['id']:c for c in pack['chunks']};validated={}
    for c in registry['contracts']:
        if not isinstance(c,dict) or set(c)!={'id','source_id','inputs','outputs','evidence','interpretation'}:raise ValueError('contract fields')
        _token(c['id'])
        if c['id'] in validated:raise ValueError('duplicate contract ID')
        if not isinstance(c['source_id'],str) or not c['source_id'].strip() or len(c['source_id'])>200:raise ValueError('contract source ID')
        if c['interpretation']!=expected_interpretation:raise ValueError('unsupported verification claim')
        if not isinstance(c['evidence'],list) or not 1<=len(c['evidence'])<=12:raise ValueError('evidence count')
        evidence={}
        for e in c['evidence']:
            if not isinstance(e,dict) or set(e)!={'id','chunk_id','source_sha256','quote'}:raise ValueError('evidence fields')
            _token(e['id'])
            if e['id'] in evidence:raise ValueError('duplicate evidence ID')
            if not isinstance(e['chunk_id'],str) or not isinstance(e['source_sha256'],str):raise ValueError('evidence references')
            chunk=chunks.get(e['chunk_id'])
            if not chunk or chunk['source_id']!=c['source_id'] or chunk['source_sha256']!=e['source_sha256']:raise ValueError('evidence source mismatch')
            if not isinstance(e['quote'],str) or not e['quote'].strip() or len(e['quote'])>250 or e['quote'] not in chunk['text']:raise ValueError('non-exact evidence quote')
            evidence[e['id']]=e
        for side in ['inputs','outputs']:
            ports=c[side]
            if not isinstance(ports,list) or not 1<=len(ports)<=8:raise ValueError('port count')
            seen=set()
            for port in ports:
                if not isinstance(port,dict) or set(port)!={'id','artifact','representation','granularity','phase','role','exposure','conditions','evidence_ids'}:raise ValueError('port fields')
                _token(port['id'])
                if port['id'] in seen:raise ValueError('duplicate port ID')
                seen.add(port['id'])
                for field in ['artifact','representation','granularity','phase']:_token(port[field],nullable=True)
                if not isinstance(port['role'],str) or port['role'] not in {'artifact','metric'}:raise ValueError('port role')
                if not isinstance(port['exposure'],str) or port['exposure'] not in ({'required'} if side=='inputs' else {'returned','intermediate','unknown'}):raise ValueError('port exposure')
                for field,limit in [('conditions',12),('evidence_ids',12)]:
                    if not isinstance(port[field],list) or len(port[field])>limit or any(not isinstance(v,str) for v in port[field]) or len(set(port[field]))!=len(port[field]):raise ValueError('port annotations')
                    for v in port[field]:_token(v)
                if not port['evidence_ids'] or not set(port['evidence_ids'])<=set(evidence):raise ValueError('unlinked port evidence')
        validated[c['id']]=c
    return validated

def inspect_contracts(registry,pack):
    """Return only cards whose exact cited source chunks are present."""
    scope='Partial curated source declarations; not executable compatibility, novelty or correctness proofs.'
    if not isinstance(registry,dict) or set(registry)!={'schema','contracts'} or registry.get('schema')!=SCHEMA or not isinstance(registry.get('contracts'),list) or len(registry['contracts'])>32:
        return {'scope':scope,'contracts':[],'warnings':['contract registry unavailable or invalid'],'matched_count':0}
    ids=[c.get('id') for c in registry['contracts'] if isinstance(c,dict) and isinstance(c.get('id'),str)]
    duplicate={i for i in ids if ids.count(i)>1};cards=[];warnings=[]
    for card in registry['contracts']:
        try:
            if isinstance(card,dict) and isinstance(card.get('id'),str) and card['id'] in duplicate:raise ValueError('duplicate contract ID')
            checked=validate_contracts({'schema':SCHEMA,'contracts':[card]},pack)
            cards.extend(checked.values())
        except (ValueError,TypeError) as exc:warnings.append(str(exc)[:200])
    return {'scope':scope,'contracts':cards,'warnings':warnings,'matched_count':len(cards)}

def assess_connection(contracts,producer_id,output_id,consumer_id,input_id):
    try:
        producer=contracts[producer_id];consumer=contracts[consumer_id]
        output=next(p for p in producer['outputs'] if p['id']==output_id)
        target=next(p for p in consumer['inputs'] if p['id']==input_id)
    except (KeyError,StopIteration) as exc:raise ValueError('unknown contract or port') from exc
    mismatches=[];unknown=[]
    for field in ['artifact','representation','granularity','phase','role']:
        if output[field] is None or target[field] is None:unknown.append(field)
        elif output[field]!=target[field]:mismatches.append({'field':field,'produced':output[field],'required':target[field]})
    missing_conditions=sorted(set(target['conditions'])-set(output['conditions']))
    other_inputs=[p['id'] for p in consumer['inputs'] if p['id']!=input_id]
    adapter_needed=output['exposure']!='returned'
    status='requires_bridge' if mismatches or adapter_needed else ('unresolved_requirements' if unknown or missing_conditions or other_inputs else 'declared_match_unverified')
    return {'producer':producer_id,'output_port':output_id,'consumer':consumer_id,'input_port':input_id,
            'producer_required_inputs':[p['id'] for p in producer['inputs']],
            'status':status,'mismatches':mismatches,'unknown_fields':unknown,
            'unestablished_conditions':missing_conditions,'other_required_inputs':other_inputs,
            'output_exposure':output['exposure'],'adapter_or_exposure_required':adapter_needed,
            'novelty_established':False,'executable_compatibility_proven':False,
            'scope':'Comparison of curated declarations. Quotes establish provenance, not entailment. Differing phases may admit adaptation; this is not a universal impossibility result.'}
