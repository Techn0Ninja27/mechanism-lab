"""Rebuild the five-source pack from existing, inspected local excerpts."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from generation.sourcepack import build_from_file,validate_source_pack
ap=argparse.ArgumentParser();ap.add_argument('--source-root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
if a.output.exists():raise FileExistsError('Preserve existing output')
base=json.loads((a.source_root/'manifest.json').read_text())['sources']
for row in json.loads((a.source_root/'adjacent/manifest.json').read_text())['sources']:
 base.append({k:('adjacent/'+v if k=='path' else v) for k,v in row.items() if k in ('id','title','url','path','evidence_level')})
a.output.mkdir(parents=True)
manifest=a.output/'manifest.json';manifest.write_text(json.dumps({'sources':base},indent=2)+'\n')
pack=build_from_file(manifest,a.source_root,max_total_chars=80000,chunk_chars=4000);validate_source_pack(pack)
if pack['limits']['truncated_source_count']:raise ValueError('Inspect source selection instead of truncating')
(a.output/'pack.json').write_text(json.dumps(pack,indent=2,ensure_ascii=False)+'\n');print(pack['digest'])
