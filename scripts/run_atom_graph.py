"""Run the frozen atom graph study into a fresh output directory."""
import argparse
import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    spec=importlib.util.spec_from_file_location('atom_graph_study',ROOT/'operator-studies/atom-graph/experiment.py')
    study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)
    study.OUT=a.output.resolve()
    study.main()
