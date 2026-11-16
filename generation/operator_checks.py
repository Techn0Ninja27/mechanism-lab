"""Bounded local checks of allowlisted lossless integer compositions."""
import hashlib
import json
import time
from .integer_codecs import registry
from .operator_composition import compose_chain


def check_chain(operator_ids, values):
    if not isinstance(operator_ids,list) or not 1<=len(operator_ids)<=4 or any(not isinstance(x,str) or not x for x in operator_ids):
        raise ValueError('Choose 1-4 registered operator IDs')
    if not isinstance(values,list) or not 1<=len(values)<=4096:
        raise ValueError('Provide 1-4096 signed integer values')
    if any(isinstance(x,bool) or not isinstance(x,int) or not -(1<<63)<=x<(1<<63) for x in values):
        raise ValueError('Values must be signed64 integers; booleans/floats are not accepted')
    chain=compose_chain(registry(),operator_ids)
    if chain.input_type!='signed_ints' or chain.output_type!='bytes':
        raise ValueError('Codec chain must map signed_ints to bytes')
    started=time.perf_counter();payload=chain.execute(values);encode_seconds=time.perf_counter()-started
    started=time.perf_counter();decoded=chain.decode(payload);decode_seconds=time.perf_counter()-started
    if decoded!=values:raise ValueError('Round-trip contract failed')
    return {'schema_version':'operator-check.v1','operator_ids':operator_ids,'chain_id':chain.content_id,
            'molecule':chain.to_molecule(),'input_count':len(values),
            'input_sha256':hashlib.sha256(json.dumps(values,separators=(',',':')).encode()).hexdigest(),
            'payload_bytes':len(payload),'raw_signed64_bytes':len(values)*8,'payload_sha256':hashlib.sha256(payload).hexdigest(),
            'round_trip':True,'encode_seconds':encode_seconds,'decode_seconds':decode_seconds,
            'scientific_novelty_claim':False,'macro_atom_promoted':False,
            'scope':'Single supplied-input roundtrip check only. Payload size excludes the study framing header; not a research benchmark or general correctness proof.'}
