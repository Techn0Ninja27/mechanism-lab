"""Bounded real-model Q/K/V capture. Exposed smoke fixtures, not a benchmark."""
import argparse, hashlib, json, shutil, subprocess, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb

REV='d039ab0d4b9311b617e8ffec1d06c8464364aeb3'
PROMPTS={'candy':'What animal is on the candy?','statue':'What landmark is shown in this image?'}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arr(t): return t.detach().float().cpu().numpy()
def finite(t):
    if not torch.isfinite(t).all().item():raise AssertionError('nonfinite tensor')

def check_attention(t,start):
    finite(t)
    if t.ndim!=4 or t.shape[0]!=1:raise AssertionError('attention shape')
    error=float((t.sum(-1)-1).abs().max().item())
    if error>2e-5 or t.min().item() < -1e-7:raise AssertionError('attention normalization')
    future=torch.arange(t.shape[-1],device=t.device)[None,:]>torch.arange(start,start+t.shape[-2],device=t.device)[:,None]
    leak=float(t.masked_select(future[None,None,:,:]).abs().max().item()) if future.any().item() else 0.
    if leak>1e-7:raise AssertionError('causal leakage')
    return error,leak

def verify_model(folder,metadata):
    data=json.loads(metadata.read_text())
    if data['sha']!=REV:raise ValueError('revision mismatch')
    refs={r['rfilename']:r for r in data['siblings']};files=[]
    for p in sorted(folder.iterdir()):
        if not p.is_file() or p.name.startswith('.'):continue
        ref=refs[p.name];raw=p.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        actual=digest if ref.get('lfs') else hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        expected=ref['lfs']['sha256'] if ref.get('lfs') else ref['blobId']
        if actual!=expected:raise ValueError('pinned file mismatch: '+p.name)
        files.append({'file':p.name,'bytes':len(raw),'sha256':digest})
    if 'model.safetensors' not in {r['file'] for r in files}:raise ValueError('missing weights')
    return files

def capture_case(name,model,processor,folder,device):
    image=Image.open(folder/'images'/f'{name}.jpg').convert('RGB')
    message=[{'role':'user','content':[{'type':'image'},{'type':'text','text':PROMPTS[name]}]}]
    prompt=processor.apply_chat_template(message,add_generation_prompt=True)
    inputs={k:v.to(device) for k,v in processor(text=prompt,images=[image],return_tensors='pt').items()}
    ids=inputs['input_ids'][0];visual=torch.where(ids==model.config.image_token_id)[0]
    if visual.numel()!=64:raise AssertionError('expected 64 actual visual feature tokens')
    specials=set(processor.tokenizer.all_special_ids)
    text=[i for i in range(int(visual.max())+1,ids.numel()) if int(ids[i]) not in specials]
    if not text:raise AssertionError('no prompt queries')
    captured={};handles=[]
    def hook(module,args,kwargs):
        hidden=kwargs['hidden_states'];cos,sin=kwargs['position_embeddings'];shape=(*hidden.shape[:-1],-1,module.head_dim)
        q=module.q_proj(hidden).view(shape).transpose(1,2);k=module.k_proj(hidden).view(shape).transpose(1,2)
        v=module.v_proj(hidden).view(shape).transpose(1,2);q,k=apply_rotary_pos_emb(q,k,cos,sin)
        for t in (q,k,v):finite(t)
        captured[module.layer_idx]=(arr(q[0]),arr(k[0]),arr(v[0]))
    for module in model.modules():
        if isinstance(module,LlamaAttention):handles.append(module.register_forward_pre_hook(hook,with_kwargs=True))
    layers=model.config.text_config.num_hidden_layers
    if len(handles)!=layers:raise AssertionError('layer coverage')
    def stack(index):return np.stack([captured[i][index] for i in range(layers)])
    try:
        with torch.inference_mode():first=model(**inputs,output_attentions=True,output_hidden_states=True,use_cache=True,return_dict=True)
        finite(first.logits);errors=[check_attention(t,0) for t in first.attentions]
        vis=visual.cpu().numpy();pre_att=np.stack([arr(t[0]) for t in first.attentions])
        traces={'input_ids':ids.cpu().numpy(),'visual_positions':vis,'prompt_text_positions':np.array(text),
                'aligned_input_embeddings':arr(first.hidden_states[0][0]),'prefill_q':stack(0),'prefill_k':stack(1),'prefill_v':stack(2),
                'prefill_attention':pre_att,'prefill_text_visual_attention':pre_att[:,:,text,:][:,:,:,vis]}
        current=first.logits[:,-1:].argmax(-1);past=first.past_key_values;mask=inputs['attention_mask']
        generated=[];qs=[];ks=[];vs=[];visual_attention=[]
        for step in range(4):
            generated.append(int(current.item()));mask=torch.cat([mask,torch.ones((1,1),device=device,dtype=mask.dtype)],1)
            with torch.inference_mode():out=model(input_ids=current,attention_mask=mask,past_key_values=past,output_attentions=True,use_cache=True,return_dict=True)
            finite(out.logits);past=out.past_key_values;qs.append(stack(0));ks.append(stack(1));vs.append(stack(2))
            for t in out.attentions:
                if t.shape[-2:]!=(1,ids.numel()+step+1):raise AssertionError('decode dimensions')
                errors.append(check_attention(t,ids.numel()+step))
            att=np.stack([arr(t[0,:,0,:]) for t in out.attentions]);traces[f'decode_attention_{step}']=att
            visual_attention.append(att[:,:,vis]);current=out.logits[:,-1:].argmax(-1)
        traces.update(decode_q=np.stack(qs),decode_k_new=np.stack(ks),decode_v_new=np.stack(vs),
                      decode_visual_attention=np.stack(visual_attention),generated_ids=np.array(generated))
        path=folder/f'{name}.npz';np.savez_compressed(path,**traces)
        if path.stat().st_size>50_000_000:raise AssertionError('trace size cap')
        return {'image':name,'input_tokens':int(ids.numel()),'visual_tokens':len(vis),'prompt_text_tokens':len(text),
                'image_token_id':model.config.image_token_id,'generated_ids':generated,'generated_text':processor.decode(generated),
                'max_row_sum_error':max(x[0] for x in errors),'max_causal_leakage':max(x[1] for x in errors),
                'npz':path.name,'npz_sha256':sha(path),'bytes':path.stat().st_size,'shapes':{k:list(v.shape) for k,v in traces.items()}}
    finally:
        for h in handles:h.remove()

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--model-metadata',type=Path,required=True)
    p.add_argument('--images-from',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['mps','cpu'],default='mps');a=p.parse_args()
    if a.output.exists():raise FileExistsError('fresh output required')
    files=verify_model(a.model,a.model_metadata);a.output.mkdir(parents=True);(a.output/'images').mkdir();images=[]
    for name in PROMPTS:
        src=a.images_from/f'{name}.jpg';dst=a.output/'images'/src.name;shutil.copyfile(src,dst)
        images.append({'name':name,'sha256':sha(dst),'bytes':dst.stat().st_size,
                       'provenance':'Previously archived processed v2 image. Original download bytes not retained; exposed development input.'})
    lock=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True);(a.output/'dependencies.lock').write_text(lock)
    plan={'schema':'smolvlm-trace-plan.v2','seed':4811,'model':'HuggingFaceTB/SmolVLM-256M-Instruct','revision':REV,'model_files':files,
          'model_metadata_sha256':sha(a.model_metadata),'script_sha256':sha(__file__),'prompts':PROMPTS,'images':images,'device':a.device,
          'steps':4,'image_splitting':False,'dependencies_sha256':sha(a.output/'dependencies.lock'),
          'scope':'Smoke inputs already exposed; Q/K/V hooks duplicate projection computation, no timing claim.'}
    (a.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');torch.manual_seed(4811)
    processor=AutoProcessor.from_pretrained(a.model,local_files_only=True,trust_remote_code=False);processor.image_processor.do_image_splitting=False
    model=AutoModelForVision2Seq.from_pretrained(a.model,local_files_only=True,trust_remote_code=False,dtype=torch.float32,attn_implementation='eager').eval().to(a.device)
    rows=[capture_case(name,model,processor,a.output,a.device) for name in PROMPTS]
    manifest={'schema':'smolvlm-trace.v2','status':'captured_pending_independent_replay','device':a.device,'cases':rows,
              'plan_sha256':sha(a.output/'plan.json'),'script_sha256':sha(__file__),'torch':torch.__version__,
              'transformers':__import__('transformers').__version__,'scope':'Development traces, not quality, latency or novelty evidence.'}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'status':manifest['status'],'device':a.device,'cases':len(rows)}))
if __name__=='__main__':main()
