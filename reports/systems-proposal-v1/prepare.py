from html.parser import HTMLParser
from pathlib import Path
import urllib.request,json,hashlib,re
ROOT=Path(__file__).resolve().parent
class Text(HTMLParser):
 def __init__(self):super().__init__();self.parts=[];self.skip=0
 def handle_starttag(self,tag,attrs):
  attrs=dict(attrs)
  if self.skip:self.skip+=1;return
  if tag in ('script','style'):self.skip=1;return
  if tag=='math' and attrs.get('alttext'):
   self.parts.append(' $'+attrs['alttext']+'$ ');self.skip=1;return
  if tag in ('p','div','h1','h2','h3','h4','section','li','tr'):self.parts.append('\n')
 def handle_endtag(self,tag):
  if self.skip:self.skip-=1
  elif tag in ('p','div','h1','h2','h3','h4','section','li','tr'):self.parts.append('\n')
 def handle_data(self,data):
  if not self.skip:self.parts.append(data)
rows=[]
for sid in ['2606.22180v1','2605.01938v1']:
 url='https://arxiv.org/html/'+sid;p=ROOT/'sources'/f'{sid}.html'
 if not p.exists():
  with urllib.request.urlopen(url,timeout=30) as r:data=r.read(2_000_001)
  if len(data)>2_000_000:raise ValueError('source too large')
  p.write_bytes(data)
 parser=Text();parser.feed(p.read_text());text='\n'.join(' '.join(x.split()) for x in ''.join(parser.parts).splitlines() if x.strip())
 (p.with_suffix('.txt')).write_text(text+'\n')
 rows.append({'id':sid,'url':url,'bytes':p.stat().st_size,'html_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'text_sha256':hashlib.sha256(p.with_suffix('.txt').read_bytes()).hexdigest()})
(ROOT/'source-receipts.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows))
