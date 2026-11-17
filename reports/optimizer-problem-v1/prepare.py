from html.parser import HTMLParser
from pathlib import Path
import urllib.request,json,hashlib,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
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

receipts=[]
for sid in ['2604.01472v1','2608.26288v1','2604.09967v1','2609.33047v1','2610.26742v1','2606.16371v1','2606.27216v1']:
 url='https://arxiv.org/html/'+sid;p=root/'sources'/f'{sid}.html'
 if not p.exists():
  with urllib.request.urlopen(url,timeout=30) as response:raw=response.read(2_000_001)
  if len(raw)>2_000_000:raise ValueError('Source exceeds 2MB cap')
  p.write_bytes(raw)
 parser=Text();parser.feed(p.read_text());text='\n'.join(' '.join(x.split()) for x in ''.join(parser.parts).splitlines() if x.strip())
 p.with_suffix('.txt').write_text(text)
 receipts.append({'id':sid,'url':url,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'text_sha256':hashlib.sha256(text.encode()).hexdigest()})
 print(sid,len(text),flush=True)
(root/'receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')
