from pathlib import Path
import json,hashlib
root=Path(__file__).resolve().parent
f=(root/'sources/2606.22180v1.txt').read_text()
start=f.index('\n5.2. Activity-aware Communication Mechanism\n');end=f.index('\n6. Experimental Results\n',start)
e=(root/'sources/2605.01938v1.txt').read_text()
a=e.index('\nIII Measurement Methodology\n', e.index('III Measurement Methodology')+1);b=e.index('\nIV Experiments\n',a)
c=e.index('\nV Discussion and Outcomes\n', e.index('V Discussion and Outcomes')+1);d=e.index('\nVI Conclusion\n',c)
selected={'felog-methods.txt':('2606.22180v1',f[start:end]), 'energy-methods.txt':('2605.01938v1',e[a:b]+'\n[Sections IV experimental tables omitted; Section V follows.]\n'+e[c:d])}
manifest={'sources':[]};ranges=[]
for name,(sid,text) in selected.items():
 text='Selected method sections, not a complete paper. Equations retain HTML alttext; images are omitted.\n'+text
 (root/'sources'/name).write_text(text)
 manifest['sources'].append({'id':sid,'title':('FeLoG sections 5.2–5.4' if name.startswith('felog') else 'GH200 energy analysis sections III and V'),'url':'https://arxiv.org/html/'+sid,'path':'sources/'+name,'evidence_level':'full_text'})
 ranges.append({'path':'sources/'+name,'sha256':hashlib.sha256(text.encode()).hexdigest(),'characters':len(text),'scope':'selected sections only'})
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(root/'selection.json').write_text(json.dumps(ranges,indent=2)+'\n');print(ranges)
