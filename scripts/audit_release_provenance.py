"""Offline audit of released scorer assignments versus semantic catalogue."""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generation.provenance import audit_provenance
from generation.token_mapping import TokenMapping


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assignments', type=Path, default=Path('papers/clusters_80.json'))
    p.add_argument('--canonical', type=Path, default=Path('papers/clusters.json'))
    p.add_argument('--dataset', type=Path, default=Path('data/coherence_dataset'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    mapping_path = args.dataset / 'token_mapping.json'
    report = audit_provenance(args.assignments, args.canonical, mapping_path)
    mapping = TokenMapping.load(mapping_path)
    paper_sets = {}
    paths = sorted(args.dataset.glob('coherence_*.jsonl'))
    if not paths:
        raise ValueError('No coherence JSONL files found')
    for path in paths:
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                atoms = set(mapping.token_ids_to_cluster_ids(row['input_ids']))
                pid = row['paper_id']
                if pid in paper_sets and paper_sets[pid] != atoms:
                    raise ValueError('Inconsistent atom set for paper ' + pid)
                paper_sets[pid] = atoms
    report['dataset_assignment_comparison'] = {}
    for path in [args.assignments, args.canonical]:
        data = json.loads(path.read_text()); assignments = defaultdict(set)
        for row in data['atom_assignments']:
            if row['cluster'] != -1:
                assignments[row['paper_id']].add(row['cluster'])
        matches = sum(assignments[pid] == atoms for pid, atoms in paper_sets.items())
        report['dataset_assignment_comparison'][str(path)] = {
            'dataset_papers': len(paper_sets), 'exact_atom_set_matches': matches}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({'safe_for_canonical_join': report['safe_for_canonical_join'],
                      'label_agreement': report['canonical_label_agreement'],
                      'dataset_assignment_comparison': report['dataset_assignment_comparison']}, indent=2))


if __name__ == '__main__':
    main()
