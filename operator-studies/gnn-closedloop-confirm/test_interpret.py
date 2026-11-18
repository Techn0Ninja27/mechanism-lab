import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('confirmation_interpretation',HERE/'interpret.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class InterpretationTests(unittest.TestCase):
    def test_allocation_pass_is_not_positive_95(self):
        with tempfile.TemporaryDirectory() as t:
            out=Path(t)/'interpretation.json'
            result=m.interpret(HERE/'results/summary.json',out)
            self.assertTrue(result['allocation_rule_passed'])
            self.assertFalse(result['primary_positive_at_pointwise_95'])
            self.assertFalse(result['auto_execute'])
            with self.assertRaises(FileExistsError):m.interpret(HERE/'results/summary.json',out)

    def test_changed_raw_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t);shutil.copyfile(HERE/'results/summary.json',d/'summary.json')
            (d/'raw.json').write_text('{}');out=d/'out.json'
            with self.assertRaises(ValueError):m.interpret(d/'summary.json',out)
            self.assertFalse(out.exists())

    def test_misleading_legacy_rule_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            d=Path(t);shutil.copyfile(HERE/'results/raw.json',d/'raw.json')
            s=json.loads((HERE/'results/summary.json').read_text());s['decision']['lead_confirmed_exact_forward']=False
            (d/'summary.json').write_text(json.dumps(s));out=d/'out.json'
            with self.assertRaises(ValueError):m.interpret(d/'summary.json',out)
            self.assertFalse(out.exists())

if __name__=='__main__':unittest.main()
