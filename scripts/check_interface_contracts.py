"""Check explicitly selected source-linked port connections offline."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from generation.interface_contracts import validate_contracts,assess_connection

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pack',type=Path,required=True)
    p.add_argument('--contracts',type=Path,default=ROOT/'docs/method-grounding/interface-contracts.json')
    p.add_argument('--connection',action='append',required=True,help='producer:output:consumer:input')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve existing assessment')
    if len(a.connection)>16 or any(len(c.split(':'))!=4 for c in a.connection):p.error('provide 1–16 four-part connections')
    pack=json.loads(a.pack.read_text());cards=validate_contracts(json.loads(a.contracts.read_text()),pack)
    results=[assess_connection(cards,*c.split(':')) for c in a.connection]
    report={'scope':'Curated declarations, not automatic extraction or executable compatibility proof.',
            'input_sha256':{'pack':hashlib.sha256(a.pack.read_bytes()).hexdigest(),'contracts':hashlib.sha256(a.contracts.read_bytes()).hexdigest()},
            'code_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['generation/interface_contracts.py','scripts/check_interface_contracts.py']},
            'assessments':results}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps([{'output':r['output_port'],'input':r['input_port'],'status':r['status']} for r in results]))
if __name__=='__main__':main()
