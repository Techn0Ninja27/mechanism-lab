"""Fetch pinned public assets and reproduce the frozen processed smoke images."""
import argparse, hashlib, json
from io import BytesIO
from pathlib import Path
import requests
from PIL import Image
from huggingface_hub import snapshot_download
from capture import REV, verify_model

SOURCES={
 'candy':'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG',
 'statue':'https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg',
}
def get(url,cap):
    chunks=[];size=0
    with requests.get(url,timeout=60,stream=True) as response:
        response.raise_for_status()
        for chunk in response.iter_content(65536):
            size+=len(chunk)
            if size>cap:raise ValueError('asset response size cap')
            chunks.append(chunk)
    return b''.join(chunks)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('fresh assets output required')
    plan=json.loads(Path(__file__).with_name('plan.json').read_text())
    a.output.mkdir(parents=True);(a.output/'images').mkdir();(a.output/'original-images').mkdir()
    data=get(f'https://huggingface.co/api/models/HuggingFaceTB/SmolVLM-256M-Instruct/revision/{REV}?blobs=true',2_000_000)
    meta=json.loads(data)
    if meta['sha']!=REV:raise ValueError('model revision mismatch')
    names=[r['file'] for r in plan['model_files']];refs={r['rfilename']:r for r in meta['siblings']}
    if sum(refs[n]['size'] for n in names)>1_000_000_000:raise ValueError('model size cap')
    (a.output/'model-api.json').write_bytes(data)
    snapshot_download('HuggingFaceTB/SmolVLM-256M-Instruct',revision=REV,allow_patterns=names,local_dir=a.output/'model')
    verify_model(a.output/'model',a.output/'model-api.json')
    receipts=[]
    for expected in plan['images']:
        name=expected['name'];raw=get(SOURCES[name],20_000_000)
        (a.output/'original-images'/f'{name}.bin').write_bytes(raw)
        dest=a.output/'images'/f'{name}.jpg'
        Image.open(BytesIO(raw)).convert('RGB').resize((512,512)).save(dest,format='JPEG',quality=95)
        actual=hashlib.sha256(dest.read_bytes()).hexdigest()
        if actual!=expected['sha256']:raise ValueError('source image changed; do not silently replace frozen input: '+name)
        receipts.append({'name':name,'url':SOURCES[name],'original_sha256':hashlib.sha256(raw).hexdigest(),'processed_sha256':actual})
    (a.output/'image-receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')
    print(json.dumps({'status':'assets_prepared','model_revision':REV,'images':len(receipts)}))
if __name__=='__main__':main()
