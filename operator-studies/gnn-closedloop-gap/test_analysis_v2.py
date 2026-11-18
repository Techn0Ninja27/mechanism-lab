"""Codex integrity tests using synthetic records, never actual study outcomes."""
import contextlib
import hashlib
import importlib.util
import io
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('gap_analysis_audit', HERE/'analyse_v2.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def fixture(root):
    config = json.loads((HERE/'freeze.json').read_text())['config']
    (root/'results').mkdir()
    (root/'bound.txt').write_text('fixed fixture bytes')
    freeze = {'config': config, 'sha256': {'bound.txt': a.sha(root/'bound.txt')}}
    (root/'freeze.json').write_text(json.dumps(freeze))
    cal = {}
    for layers, drop, sync in itertools.product(config['layers'], config['lr_drop'], config['sync']):
        for k in range(layers-1):
            cal[f'layers{layers}_drop{int(drop)}_sync{sync}_hidden{k+1}'] = {'coef': .5}
    rows = []
    for seed, layers, drop in itertools.product(config['measurement_seeds'], config['layers'], config['lr_drop']):
        jobs = [('G0',4,1)]
        for parts in config['parts']:
            jobs += [('G1',parts,1),('zero',parts,1),('hold',parts,config['never_resync_sync'])]
            jobs += [(arm,parts,sync) for sync in config['sync'] for arm in config['stale_arms']]
        for arm,parts,sync in jobs:
            accuracy = .7 if arm == 'hold' else .8
            r = dict(arm=arm, seed=seed, layers=layers, lr_drop=drop, parts=parts, sync=sync,
                     val={str(t):accuracy for t in (40,80,120)}, eval={str(t):accuracy for t in (40,80,120)}, final_train_loss=.5)
            if arm == 'lag1': r['coef'] = [.5]*(layers-1)
            rows.append(r)
    assert len(rows) == 1600
    return dict(freeze_sha256=a.sha(root/'freeze.json'), calibration=cal, n_runs=len(rows), runs=rows)


class AnalysisIntegrityTests(unittest.TestCase):
    def execute(self, mutation=None, change_file=False):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = fixture(root)
            if mutation: mutation(raw)
            if change_file: (root/'bound.txt').write_text('changed bytes')
            (root/'results/raw.json').write_text(json.dumps(raw))
            output = root/'results/summary-v2.json'
            with patch.object(a,'HERE',root), patch.object(a,'t_ppf',return_value=2.0), contextlib.redirect_stdout(io.StringIO()):
                if mutation or change_file:
                    with self.assertRaises(SystemExit): a.main()
                    self.assertFalse(output.exists())
                    return
                a.main()
                result = json.loads(output.read_text())
                with self.assertRaises(FileExistsError): a.main()
            self.assertTrue(all(result['integrity_checks'].values()))
            self.assertEqual(result['primary_family']['size'],24)
            self.assertEqual(len(result['flagged_simultaneous']),24)
            for r in result['regimes']:
                self.assertNotIn('simultaneous_ci_pt',r['truncation_G0_minus_G1'])
                for item in r['by_sync'].values():
                    self.assertEqual('simultaneous_ci_pt' in item['staleness_G1_minus_hold'],r['metric']=='eval@120')
                    self.assertNotIn('simultaneous_ci_pt',item['recovered_sni_minus_hold'])

    def test_complete_fixture_and_primary_family_only(self): self.execute()
    def test_duplicate_key_rejected(self): self.execute(lambda r:r['runs'].__setitem__(1,r['runs'][0]))
    def test_missing_job_rejected(self): self.execute(lambda r:r['runs'].pop())
    def test_nonfinite_metric_rejected(self): self.execute(lambda r:r['runs'][0]['eval'].__setitem__('120',float('nan')))
    def test_boolean_metric_rejected(self): self.execute(lambda r:r['runs'][0]['eval'].__setitem__('120',True))
    def test_freeze_binding_rejected(self): self.execute(lambda r:r.__setitem__('freeze_sha256','0'*64))
    def test_changed_frozen_file_rejected(self): self.execute(change_file=True)
    def test_calibration_mismatch_rejected(self):
        def mutate(r): next(x for x in r['runs'] if x['arm']=='lag1')['coef']=[999.]
        self.execute(mutate)


if __name__ == '__main__': unittest.main()
