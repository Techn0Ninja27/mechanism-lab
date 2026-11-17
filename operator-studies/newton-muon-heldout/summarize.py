"""Predeclared descriptive paired screen, with no extrapolation or target tuning."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def curve_stats(checkpoints, horizon):
    t = np.array([c['training_seconds'] for c in checkpoints], dtype=float)
    y = np.array([c['heldout']['mean_cross_entropy'] for c in checkpoints], dtype=float)
    if not (np.isfinite(t).all() and np.isfinite(y).all() and t[0] == 0 and np.all(np.diff(t) > 0) and 0 < horizon <= t[-1]):
        raise ValueError('Invalid curve or extrapolation requested')
    grid = np.r_[t[t < horizon], horizon]
    interpolated = np.interp(grid, t, y)
    last = int(np.flatnonzero(t <= horizon)[-1])
    return {'fixed_step_endpoint': float(y[-1]), 'common_time_interpolated': float(interpolated[-1]),
            'common_time_last_observed': float(y[last]), 'last_observed_step': checkpoints[last]['step'],
            'normalized_time_area': float(np.trapezoid(interpolated, grid) / horizon)}


def summarize(folder, freeze):
    frozen = json.loads(freeze.read_text())
    rows = []
    refs = {}
    for seed in (0, 1):
        arms = {}
        for mode in ('muon', 'newton_cholesky'):
            path = folder / f'{mode}-seed{seed}.json'
            raw = path.read_bytes(); x = json.loads(raw)
            if x['status'] != 'completed' or x['warnings'] or x['cholesky_info_max'] != 0:
                raise ValueError('Incomplete or invalid arm')
            if x['runner_sha256'] != frozen['hashes']['run_arm.py']:
                raise ValueError('Runner differs from pre-run freeze')
            p = x['protocol']
            if (p['seed'], p['mode'], p['width'], p['steps'], p['refresh']) != (seed, mode, 256, 256, 32):
                raise ValueError('Protocol mismatch')
            if [c['step'] for c in x['checkpoints']] != list(range(0, 257, 32)):
                raise ValueError('Missing checkpoint')
            if x['device'] != 'mps' or x['fallback_env'] != '0' or x['internal_cpu_fallback_detected']:
                raise ValueError('Device or fallback mismatch')
            if x['optimizer_refresh_counts'] != ([] if mode == 'muon' else [8]):
                raise ValueError('Wrong refresh counts')
            for c in x['checkpoints']:
                h = c['heldout']
                if h['model_hash_before'] != h['model_hash_after'] or h['tokens'] != x['heldout_spec']['used_windows'] * 64:
                    raise ValueError('Evaluation changed weights or token population')
            arms[mode] = x; refs[path.name] = hashlib.sha256(raw).hexdigest()
        for key in ('initial_model_sha256','train_batch_stream_sha256','heldout_text_sha256','train_text_sha256','split','heldout_spec','frozen_cost_profile_sha256'):
            if arms['muon'][key] != arms['newton_cholesky'][key]:
                raise ValueError(f'Unmatched {key}')
        horizon = min(x['checkpoints'][-1]['training_seconds'] for x in arms.values())
        stats = {mode: curve_stats(x['checkpoints'], horizon) for mode, x in arms.items()}
        differences = {key: stats['newton_cholesky'][key] - stats['muon'][key] for key in
                       ('fixed_step_endpoint','common_time_interpolated','common_time_last_observed','normalized_time_area')}
        passes = all(differences[key] < 0 for key in ('fixed_step_endpoint','common_time_interpolated','normalized_time_area'))
        rows.append({'seed': seed, 'common_training_seconds': horizon, 'arms': stats,
                     'newton_minus_muon': differences, 'escalation_rule_passed': passes})
    return {'status': 'eligible_for_larger_baseline_study' if all(x['escalation_rule_passed'] for x in rows) else 'close_local_candidate_at_fixed_settings',
            'paired_seeds': rows, 'raw_sha256': refs, 'freeze_sha256': hashlib.sha256(freeze.read_bytes()).hexdigest(),
            'scope': 'Two seeds, one tiny byte workload, fixed settings; allocation rule, not a significance test or universal optimizer verdict.',
            'limitations': ['Piecewise-linear interpolation between 32-step checkpoints is not a measured endpoint.',
                            'Evaluation time excluded; initial evaluation warms forward kernels before timed training.',
                            '256 steps can miss late benefits; learning rates were not tuned independently.',
                            'No low-rank candidate, novelty, or general usefulness established.'],
            'novelty_established': False, 'train_eligible': False}


if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--freeze',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists(): raise FileExistsError('Preserve existing result')
    a.output.write_text(json.dumps(summarize(a.input,a.freeze),indent=2)+'\n')
