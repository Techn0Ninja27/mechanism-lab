"""Download bounded public SNAP inputs, or verify cached bytes; no model calls."""
import argparse,gzip,hashlib,json,urllib.request
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent

def prepare(out):
    protocol=json.loads((HERE/'protocol.json').read_text());out.mkdir(parents=True,exist_ok=True)
    receipt_path=out/'sources.json'
    previous=json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    receipts={}
    for source in protocol['datasets']:
        sid=source['id'];path=out/(sid+'.txt.gz')
        if not path.exists():
            with urllib.request.urlopen(source['url'],timeout=60) as response:
                data=response.read(protocol['download_cap_bytes_per_file']+1)
            if len(data)>protocol['download_cap_bytes_per_file']:raise ValueError('download size cap')
            with path.open('xb') as f:f.write(data)
        raw=path.read_bytes();sha=hashlib.sha256(raw).hexdigest()
        if sid in previous and previous[sid]['sha256']!=sha:raise ValueError('cached source changed')
        with gzip.open(path,'rb') as f:plain=f.read(100_000_001)
        if len(plain)>100_000_000:raise ValueError('uncompressed source size cap')
        # Only integer edge pairs; no downloaded code is executed.
        import io
        edges=np.loadtxt(io.BytesIO(plain),dtype=np.uint32,comments='#',ndmin=2)
        if edges.ndim!=2 or edges.shape!=(source['edges'],2):raise ValueError('unexpected edge count/shape')
        ids,inverse=np.unique(edges,return_inverse=True)
        if len(ids)!=source['nodes']:raise ValueError('unexpected node count')
        e=inverse.reshape(-1,2).astype(np.uint32)
        if np.any(e[:,0]==e[:,1]):raise ValueError('self-loop requires declared policy')
        src=np.concatenate([e[:,0],e[:,1]]);dst=np.concatenate([e[:,1],e[:,0]])
        order=np.lexsort((dst,src));src=src[order];dst=dst[order]
        if np.any((src[1:]==src[:-1])&(dst[1:]==dst[:-1])):raise ValueError('duplicate directed edges')
        counts=np.bincount(src,minlength=len(ids));indptr=np.concatenate([[0],np.cumsum(counts)]).astype(np.int64)
        if np.any(counts==0):raise ValueError('isolated node policy unspecified')
        np.savez(out/(sid+'-csr.npz'),indptr=indptr,indices=dst,original_ids=ids)
        receipts[sid]={**source,'sha256':sha,'compressed_bytes':len(raw),'uncompressed_sha256':hashlib.sha256(plain).hexdigest(),
                       'csr_arrays_sha256':hashlib.sha256(indptr.tobytes()+dst.tobytes()+ids.tobytes()).hexdigest()}
        receipt_path.write_text(json.dumps(receipts,indent=2)+'\n')
        print(json.dumps({'id':sid,'bytes':len(raw),'nodes':len(ids),'edges':len(edges),'sha256':sha}),flush=True)
    return receipts

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();prepare(a.output)
