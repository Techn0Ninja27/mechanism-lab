"""Replay exact-source contract validation and the attempted direct connection."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from generation.contract_molecules import contract_connection
ap=argparse.ArgumentParser();ap.add_argument('--pack',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
if a.output.exists():raise FileExistsError('Preserve existing result')
r=contract_connection(json.loads(Path(__file__).with_name('contracts.json').read_text()),json.loads(a.pack.read_text()),
    'chen_cancellation_length_bound','updated_bound','dmrs_peak_demand','output_length',
    mechanism='Attempt to use a cancellation lower bound as exact DMRS output length',
    assumptions=['Would require exact output length and prompt size before admission'],
    falsifiable_prediction='A lower bound cannot generally determine exact peak demand without an additional rule')
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({'status':r['assessment']['status'],'atoms':len(r['atoms']),'molecule_created':r['molecule'] is not None}))
