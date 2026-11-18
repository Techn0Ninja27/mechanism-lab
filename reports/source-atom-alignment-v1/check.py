"""Offline exact-quote overlap diagnostic; does not establish source identity."""
import argparse
import hashlib
import json
from pathlib import Path


def inspect(pack_path, atoms_path):
    pack_bytes=pack_path.read_bytes(); atom_bytes=atoms_path.read_bytes()
    pack=json.loads(pack_bytes)
    atoms=[json.loads(line) for line in atom_bytes.decode().splitlines() if line.strip()]
    rows=[]
    for index,atom in enumerate(atoms):
        quote=atom.get('source_quote')
        matches=[c['id'] for c in pack['chunks'] if isinstance(quote,str) and quote.strip() and quote in c['text']]
        rows.append({'atom_index':index,'atom_source_id':atom.get('source_id'),'matching_chunk_ids':matches})
    return {'schema':'exact-atom-quote-overlap.v1','pack_sha256':hashlib.sha256(pack_bytes).hexdigest(),
            'atoms_sha256':hashlib.sha256(atom_bytes).hexdigest(),'atom_count':len(atoms),
            'chunk_count':len(pack['chunks']),'atoms_with_quote_match':sum(bool(r['matching_chunk_ids']) for r in rows),
            'rows':rows,'limitations':['Exact text overlap only; matching quotes would still require source-identity and interpretation checks.',
                                      'No matches means these inputs cannot support an exact-quote-grounded matched-context comparison without new extraction.'],
            'paid_calls':0}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pack',type=Path,required=True);p.add_argument('--atoms',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=inspect(a.pack,a.atoms)
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:result[k] for k in ('atom_count','chunk_count','atoms_with_quote_match','paid_calls')}))
