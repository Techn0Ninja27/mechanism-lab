"""Verify saved provider responses and the provenance of collected records."""
import argparse,gzip,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.fetch_openalex_corpus import normalize

def verify(folder):
    observed=set();ok=failed=0;requests=set()
    with (folder/'receipts.jsonl').open() as stream:
        for line in stream:
            receipt=json.loads(line);number=int(receipt['request'])
            if number in requests:raise ValueError('Duplicate request receipt')
            requests.add(number)
            if receipt.get('error') or receipt.get('status')!=200:failed+=1;continue
            with gzip.open(folder/'responses'/f'{number:06d}.json.gz','rb') as f:raw=f.read(15_000_001)
            if len(raw)>15_000_000 or hashlib.sha256(raw).hexdigest()!=receipt['response_sha256']:raise ValueError('Response integrity mismatch')
            if len(raw)!=receipt['bytes']:raise ValueError('Response size mismatch')
            for row in json.loads(raw)['results']:
                normalized=normalize(row,{'id':receipt['slot']},receipt['timestamp'][:10])
                if normalized is not None:observed.add(hashlib.sha256(json.dumps(normalized,sort_keys=True).encode()).hexdigest())
            ok+=1
    count=0
    with (folder/'records.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line)
            if hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest() not in observed:raise ValueError('Collected record lacks matching raw response')
            count+=1
    return {'successful_responses_verified':ok,'failed_requests_preserved':failed,'collected_records_verified':count,
            'scope':'Provider response integrity and record provenance, not scientific truth or exhaustive corpus coverage.'}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve existing verification')
    r=verify(a.folder)
    with a.output.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps(r))
