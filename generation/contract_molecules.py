"""Connect source-linked interface declarations to hypothesis molecules.

This is a declarative bridge, never an executable-compatibility or novelty proof.
It does not consume lexical similarity scores or legacy learned atom IDs.
"""
import hashlib
import json
from .interface_contracts import validate_contracts, assess_connection
from .molecules import atom_record, molecule_record


def contract_connection(registry, pack, producer, output_port, consumer, input_port,
                        *, mechanism, assumptions, falsifiable_prediction):
    cards = validate_contracts(registry, pack)
    assessment = assess_connection(cards, producer, output_port, consumer, input_port)
    sources = {s['id']: s for s in pack['sources']}
    atoms = []
    for cid in (producer, consumer):
        card = cards[cid]
        digest = hashlib.sha256(json.dumps(card, sort_keys=True).encode()).hexdigest()
        atoms.append(atom_record('contract:' + cid,
                     f"Source-linked substep {cid}; inputs: " + ', '.join(p['id'] for p in card['inputs']) +
                     '; outputs: ' + ', '.join(p['id'] for p in card['outputs']),
                     [{'source_id': card['source_id'], 'url': sources[card['source_id']]['url'],
                       'contract_id': cid, 'contract_sha256': digest,
                       'evidence_chunk_ids': sorted({e['chunk_id'] for e in card['evidence']})}]))
    molecule = None
    if assessment['status'] == 'declared_match_unverified':
        molecule = molecule_record(atoms, [{'source': atoms[0]['id'], 'target': atoms[1]['id'],
                                            'type': 'declared_output_to_input'}],
                                   mechanism=mechanism, assumptions=assumptions,
                                   falsifiable_prediction=falsifiable_prediction,
                                   status='combination', provenance=[{'interface_assessment': assessment}])
    return {'schema': 'contract-molecule-connection.v1', 'atoms': atoms, 'assessment': assessment,
            'molecule': molecule, 'train_eligible': False, 'novelty_established': False,
            'executable_compatibility_proven': False,
            'scope': 'Hypothesis representation from validated source declarations. Unmet inputs or mismatched ports require a separately specified bridge.'}
