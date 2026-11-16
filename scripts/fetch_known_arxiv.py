"""Fetch a bounded explicit list of primary arXiv abstract records (no search)."""
import argparse,hashlib,json,re,time,urllib.request
from html.parser import HTMLParser
from pathlib import Path

class AbstractPage(HTMLParser):
    def __init__(self):
        super().__init__();self.meta={};self.depth=0;self.abstract=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='meta' and a.get('name','').startswith('citation_'):
            self.meta.setdefault(a['name'],[]).append(a.get('content',''))
        if tag=='blockquote' and 'abstract' in a.get('class','').split():self.depth=1
        elif self.depth and tag=='blockquote':self.depth+=1
    def handle_endtag(self,tag):
        if tag=='blockquote' and self.depth:self.depth-=1
    def handle_data(self,data):
        if self.depth:self.abstract.append(data)

def main():
    p=argparse.ArgumentParser();p.add_argument('--id',action='append',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if not 1<=len(a.id)<=10 or any(not re.fullmatch(r'\d{4}\.\d{4,5}(?:v\d+)?',s) for s in a.id):p.error('Provide 1–10 explicit modern arXiv IDs')
    if a.output.exists():raise FileExistsError('Preserve existing source packet')
    a.output.mkdir(parents=True);records=[];receipts=[]
    for i,aid in enumerate(a.id):
        if i:time.sleep(3)
        url='https://arxiv.org/abs/'+aid
        req=urllib.request.Request(url,headers={'User-Agent':'MechanismLab-primary-records/1.0'})
        with urllib.request.urlopen(req,timeout=30) as response:
            payload=response.read(2_000_001)
        if len(payload)>2_000_000:raise ValueError('Source exceeds size cap')
        page=AbstractPage();page.feed(payload.decode('utf-8'))
        title=page.meta.get('citation_title',[''])[0]
        abstract=re.sub(r'^\s*Abstract:\s*','',re.sub(r'\s+',' ',' '.join(page.abstract))).strip()
        if not title or not abstract:raise ValueError('Primary title/abstract extraction failed')
        (a.output/(aid+'.html')).write_bytes(payload)
        records.append({'id':aid,'title':title,'abstract':abstract,'authors':page.meta.get('citation_author',[]),
          'published':page.meta.get('citation_date',[''])[0].replace('/','-'),'url':url,'evidence_level':'primary_abstract'})
        receipts.append({'id':aid,'url':url,'response_sha256':hashlib.sha256(payload).hexdigest(),'bytes':len(payload)})
    (a.output/'packet.json').write_text(json.dumps({'papers':records},indent=2)+'\n')
    (a.output/'receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')
    print(json.dumps({'records':len(records),'output':str(a.output)}))
if __name__=='__main__':main()
