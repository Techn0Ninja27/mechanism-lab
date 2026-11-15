"""Represent a generated mechanism as a molecule; do not auto-admit it."""
import argparse,json,pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from generation.molecules import atom_record,molecule_record,promote,MoleculeError

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=pathlib.Path,required=True);a=p.parse_args()
 case=a.run/'candidates/digits_shift-world-3.json';proposal=json.loads(case.read_text())['proposal']
 refs=[{'kind':'generated_hypothesis','path':str(case),'verified_prior_art':False}]
 atoms=[atom_record('block_dropout','Train on digit images with contiguous pixel blocks removed.',refs),atom_record('kernel_classification','Fit separate RBF-SVM classifiers on transformed or augmented training features.',refs),atom_record('margin_weighting','Weight classifier predictions by held-out validation decision margins.',refs)]
 mol=molecule_record(atoms,[{'source':'block_dropout','target':'kernel_classification','type':'changes_training_distribution'},{'source':'kernel_classification','target':'margin_weighting','type':'supplies_predictions_and_margins'}],mechanism=proposal['mechanism'],assumptions=[proposal['assumptions']],falsifiable_prediction=proposal['falsification'],status='combination',provenance=refs)
 # No independent second-task validation is available. Fail closed, even if
 # one benchmark happens to improve. Promotion is not a novelty certificate.
 try:promote(mol,{})
 except MoleculeError as exc:status={'promoted':False,'reason':str(exc)}
 out=a.run/'molecule-demo.json';out.write_text(json.dumps({'molecule':mol,'promotion':status,'construction':'Explicit researcher-authored decomposition of one generated candidate; not automatic semantic discovery.'},indent=2));print(json.dumps(status))
if __name__=='__main__':main()
