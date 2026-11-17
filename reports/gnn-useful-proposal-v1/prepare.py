from html.parser import HTMLParser
from pathlib import Path
import urllib.request,json,hashlib,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from generation.sourcepack import build_source_pack
root=Path(__file__).resolve().parent
class Text(HTMLParser):
 def __init__(self):super().__init__();self.parts=[];self.skip=None
 def handle_starttag(self,tag,attrs):
  if self.skip:return
  attrs=dict(attrs)
  if tag in ('script','style'):self.skip=tag;return
  if tag=='math':
   self.parts.append(' $'+attrs.get('alttext','[equation unavailable]')+'$ ');self.skip='math';return
  if tag in ('p','div','h1','h2','h3','h4','section','li','tr'):self.parts.append('\n')
 def handle_endtag(self,tag):
  if self.skip:
   if tag==self.skip:self.skip=None
  elif tag in ('p','div','h1','h2','h3','h4','section','li','tr'):self.parts.append('\n')
 def handle_data(self,data):
  if not self.skip:self.parts.append(data)
url='https://arxiv.org/html/2608.06441v1';p=root/'sources/sni.html'
if not p.exists():
 with urllib.request.urlopen(url,timeout=30) as r:raw=r.read(2_000_001)
 if len(raw)>2_000_000:raise ValueError('download cap')
 p.write_bytes(raw)
parser=Text();parser.feed(p.read_text());text='\n'.join(' '.join(x.split()) for x in ''.join(parser.parts).splitlines() if x.strip())
(root/'sources/sni-full.txt').write_text(text)
start=text.rindex('\nIV Design Details of SNI-GNN\n');end=text.index('\nV Evaluation\n',start)
selected='Selected Section IV only; figures omitted; equations from HTML alttext.\n'+text[start:end]
(root/'sources/sni-method.txt').write_text(selected)
manifest={'sources':[{'id':'sni-method','title':'SNI-GNN Section IV methods and analysis','url':url,'path':'sources/sni-method.txt','evidence_level':'full_text'}]}
neighbors=json.loads((root.parent/'citation-neighborhood-v1/neighborhood.json').read_text())
for row in neighbors['records']:
 if row['id'] not in ['W4407013152','W4396628137','W7160826828','W4409983225']:continue
 name=row['id'];file='sources/'+name+'.txt'
 (root/file).write_text(row['title']+'\n'+row['abstract']+'\n')
 manifest['sources'].append({'id':name,'title':row['title'],'url':row['doi'] or 'https://openalex.org/'+name,'path':file,'evidence_level':'abstract'})
(root/'manifest.json').write_text(json.dumps(manifest,indent=2))
(root/'pack.json').write_text(json.dumps(build_source_pack(manifest,root,max_total_chars=28000,chunk_chars=3500),indent=2))
(root/'source-receipt.json').write_text(json.dumps({'url':url,'html_bytes':p.stat().st_size,'html_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'method_chars':len(selected),'section':'IV','abstract_scope':'Four provider abstracts from saved citation snapshot, not independently verified full texts'},indent=2))
print('method chars',len(selected))
