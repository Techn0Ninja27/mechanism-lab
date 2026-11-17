"""Freeze small real-text profiling inputs from the existing local corpus."""
import argparse,hashlib,json,sqlite3
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--corpus',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
with sqlite3.connect(a.corpus.resolve().as_uri()+'?mode=ro',uri=True) as db:
 rows=db.execute("SELECT canonical_id,title,abstract FROM works WHERE canonical_id LIKE 'arxiv:2501.%' AND length(abstract)>500 ORDER BY canonical_id LIMIT 128").fetchall()
if len(rows)!=128:raise ValueError('Expected 128 frozen source records')
for name,records in [('text.txt',rows[:96]),('unused-validation.txt',rows[96:])]:
 data='\n\n'.join(title+'\n'+abstract for _,title,abstract in records).encode()
 (a.output/name).write_bytes(data)
manifest={'scope':'Real research abstracts for runtime profiling only; not a general language-model quality benchmark. Deterministic January2025 selection, no random representativeness claim. Last32documents reserved and unused by cost screen.','selection_query':'January2025 arxiv canonical IDs, abstract length>500, canonical-ID order first128','files':{name:{'bytes':(a.output/name).stat().st_size,'sha256':hashlib.sha256((a.output/name).read_bytes()).hexdigest()} for name in ['text.txt','unused-validation.txt']},'sources':[{'id':i,'text_sha256':hashlib.sha256((title+'\n'+abstract).encode()).hexdigest(),'split':'profile_train' if n<96 else 'reserved_unused'} for n,(i,title,abstract) in enumerate(rows)]}
(a.output/'text-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(manifest['files'])
