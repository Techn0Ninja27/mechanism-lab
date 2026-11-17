"""Validate matched controls and summarize descriptive costs; no quality gate."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path


def summarize(folder):
    rows = []
    hashes = {}
    for width in (256, 512):
        width_rows = []
        for refresh in (16, 32, 64):
            path = folder / f'd{width}-r{refresh}.json'
            raw = path.read_bytes()
            x = json.loads(raw)
            p = x['protocol']
            assert (p['width'], p['refresh'], p['warmup'], p['steps'], p['window']) == (width, refresh, 8, 128, 32)
            assert x['loss_finite'] and not x['warnings'] and x['cholesky_info_max'] == 0
            assert x['device'] == 'mps' and x['fallback_env'] == '0'
            assert x['refresh_counts'] == [128 // refresh]
            assert len(x['losses']) == 136 and len(x['windows']) == 4
            assert sum(w['refreshes'] for w in x['windows']) == 128 // refresh
            costs = [1000 * w['seconds'] / w['steps'] for w in x['windows']]
            assert costs == x['ms_per_step']
            hashes[path.name] = hashlib.sha256(raw).hexdigest()
            width_rows.append(x)
            rows.append({'width': width, 'refresh': refresh, 'median_ms_per_step': statistics.median(costs),
                         'window_ms_per_step': costs, 'loss_last': x['loss_last']})
        for field in ('initial_model_state_sha256', 'batch_stream_sha256', 'text_sha256', 'frozen_cost_profile_sha256'):
            assert len({x[field] for x in width_rows}) == 1, field
    comparisons = []
    for width in (256, 512):
        r = {x['refresh']: x for x in rows if x['width'] == width}
        comparisons.append({'width': width, 'refresh64_to32_cost_ratio': r[64]['median_ms_per_step'] / r[32]['median_ms_per_step']})
    return {'status': 'mixed_short_run_cost_no_quality_conclusion', 'arms': rows, 'comparisons': comparisons,
            'raw_sha256': hashes, 'runner_sha256_at_aggregation': hashlib.sha256((Path(__file__).parent / 'refresh_control.py').read_bytes()).hexdigest(),
            'limitations': ['One process per arm, fixed order, four dependent windows.', 'No held-out quality or time-to-target measurement.',
                            'Generic FP32 MPS development harness; not source optimized CUDA kernels.'],
            'novelty_established': False, 'train_eligible': False}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve existing result')
    args.output.write_text(json.dumps(summarize(args.input), indent=2) + '\n')
