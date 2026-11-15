"""Bounded read-only client for Lacuna's documented public API.

Uses the authors' production research map for live navigation. It does not
mirror, cache or ingest Lacuna content into the local training corpus.
"""
import argparse,json,re,urllib.parse,urllib.request
BASE='https://lacuna.tiptreesystems.com'
def search_url(query,kind='paper',limit=5):
 if kind not in {'paper','cluster','hypothesis','resource'}:raise ValueError('Unsupported type')
 if not query.strip() or not 1<=limit<=10:raise ValueError('Nonempty query and limit1..10 required')
 return BASE+'/api/v1/search?'+urllib.parse.urlencode({'q':query,'type':kind,'limit':limit})
def context_url(kind,ident):
 if kind not in {'paper','direction','hypothesis'} or not re.fullmatch(r'[A-Za-z0-9_-]+',ident):raise ValueError('Use a returned ID, not a URL or path')
 return BASE+'/api/v1/context/'+kind+'/'+ident+'?view=compact'
def fetch(url):
 req=urllib.request.Request(url,headers={'User-Agent':'mechanism-lab-research/0.1'})
 with urllib.request.urlopen(req,timeout=40) as r:return json.load(r)
def main():
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
 q=sub.add_parser('search');q.add_argument('query');q.add_argument('--type',default='paper',choices=['paper','cluster','hypothesis','resource']);q.add_argument('--limit',default=5,type=int)
 q=sub.add_parser('context');q.add_argument('kind',choices=['paper','direction','hypothesis']);q.add_argument('id')
 a=p.parse_args();url=search_url(a.query,a.type,a.limit) if a.mode=='search' else context_url(a.kind,a.id)
 print(json.dumps(fetch(url),indent=2,ensure_ascii=False))
if __name__=='__main__':main()
