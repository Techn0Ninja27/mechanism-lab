"""Extract only the two named LINQS data files; download stays explicit via curl."""
import hashlib,json,tarfile,argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);a=p.parse_args()
f=a.data/'cora.tar.gz';rows=[]
with tarfile.open(f,mode='r:gz') as t:
 for name in ['cora.content','cora.cites']:
  matches=[m for m in t.getmembers() if m.isfile() and Path(m.name).name==name]
  if len(matches)!=1 or matches[0].size>12_000_000:raise ValueError('Unexpected archive layout/size')
  b=t.extractfile(matches[0]).read();(a.data/name).write_bytes(b)
  rows.append({'name':name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
r={'url':'https://linqs-data.soe.ucsc.edu/public/datasets/cora/cora.tar.gz','archive_bytes':f.stat().st_size,'archive_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'files':rows}
(a.data/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
