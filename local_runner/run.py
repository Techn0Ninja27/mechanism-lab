#!/usr/bin/env python3
"""Local upstream sampling + bounded OpenRouter reconstruction; no secret persistence."""
import argparse, hashlib, html, json, os, pathlib, subprocess, sys, urllib.request, urllib.error
ROOT=pathlib.Path(os.environ.get('ALIEN_WORKSPACE', pathlib.Path(__file__).resolve().parent))
UP=pathlib.Path(os.environ.get('ALIEN_REPO', ROOT/'upstream'))
sys.path.insert(0,str(UP))
def key():
    value=os.environ.get('OPENROUTER_API_KEY')
    if value: return value
    value=None
    try:
        for line in (pathlib.Path.home()/'.config/openrouter/env').read_text().splitlines():
            line=line.strip()
            if line.startswith('export '): line=line[7:].lstrip()
            name,sep,candidate=line.partition('=')
            if sep and name.strip()=='OPENROUTER_API_KEY':
                value=candidate.strip().strip('"\'')
                break
    except OSError:
        pass
    if not value: raise SystemExit('No OPENROUTER_API_KEY in environment or ~/.config/openrouter/env')
    return value

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n-gen',type=int,default=1000)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--top-k',type=int,default=20)
    p.add_argument('--betas',nargs='+',type=float,default=[0.0,0.4,0.7])
    p.add_argument('--decode',type=int,default=0,help='Paid calls: decode this many top candidates per beta; default offline')
    p.add_argument('--model',default='anthropic/claude-haiku-5.5')
    p.add_argument('--device',choices=['cpu','mps','cuda','auto'],default='cpu')
    p.add_argument('--generation-batch-size',type=int,default=1)
    p.add_argument('--candidate-source',choices=['autoregressive','uniform','hybrid'],default='autoregressive')
    p.add_argument('--skip-sample',action='store_true')
    p.add_argument('--output',type=pathlib.Path,default=ROOT/'results/demo')
    a=p.parse_args()
    if a.decode<0 or a.decode>10: p.error('--decode must be 0..10 per beta')
    out=a.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    if a.skip_sample:
        saved=json.loads((out/'beta_sweep_summary.json').read_text())['config']
        if saved['seed']!=a.seed or saved['n_gen']!=a.n_gen:
            p.error('Seed or n-gen differs from saved run; supply the original values or remove --skip-sample')
        a.device=saved.get('device','cpu')
        a.generation_batch_size=saved.get('generation_batch_size',1)
        a.candidate_source=saved.get('candidate_source','autoregressive')
    if not a.skip_sample:
        subprocess.run([sys.executable,'alien.py','sample','--device',a.device,'--generation-batch-size',str(a.generation_batch_size),'--candidate-source',a.candidate_source,'--batch-size','128','--n-gen',str(a.n_gen),'--seed',str(a.seed),'--top-k',str(a.top_k),'--betas',*[str(b) for b in a.betas],'--output-dir',str(out)],cwd=UP,check=True,env={**os.environ,'OMP_NUM_THREADS':'4','VECLIB_MAXIMUM_THREADS':'4'})
    from generation.prompts import RECONSTRUCTION_PROMPT
    catalogue=json.loads((UP/'papers/clusters_80.json').read_text())['clusters']
    cards=[]; count=0
    for beta in a.betas:
        payload=json.loads((out/f'k3/beta_{beta:g}/results.json').read_text())
        cards.append(f'<h2>β = {beta:g}</h2><p>{payload["config"]["n_candidate_sets"]} distinct candidates. Higher β favors unavailability.</p>')
        for i,item in enumerate(payload['top_k_sequences']):
            atoms=[catalogue[str(c)]['name'] for c in item['cluster_ids']]
            prompt=RECONSTRUCTION_PROMPT.format(super_atoms_text='\n'.join(f'{j+1}. {s}' for j,s in enumerate(atoms)))
            identity=hashlib.sha256((a.model+'\n'+prompt+'\nmax_tokens=4096;temperature=0.7').encode()).hexdigest()[:20]
            dest=out/'proposals'/identity; dest.mkdir(parents=True,exist_ok=True)
            token_cap=4096
            old_receipt=dest/'receipt.json'
            if old_receipt.exists() and json.loads(old_receipt.read_text()).get('finish_reason')=='length':
                token_cap=8192
                identity=hashlib.sha256((a.model+'\n'+prompt+'\nmax_tokens=8192;temperature=0.7').encode()).hexdigest()[:20]
                dest=out/'proposals'/identity; dest.mkdir(parents=True,exist_ok=True)
            (dest/'prompt.txt').write_text(prompt)
            if i<a.decode and not (dest/'proposal.md').exists():
                request=urllib.request.Request('https://openrouter.ai/api/v1/chat/completions',data=json.dumps({'model':a.model,'messages':[{'role':'user','content':prompt}],'max_tokens':token_cap,'temperature':0.7}).encode(),headers={'Authorization':'Bearer '+key(),'Content-Type':'application/json'})
                try:
                    with urllib.request.urlopen(request,timeout=180) as r: response=json.load(r)
                except urllib.error.HTTPError as e: raise SystemExit(f'OpenRouter HTTP {e.code}; response body omitted to avoid credential disclosure')
                choice=response['choices'][0]; content=choice['message'].get('content')
                receipt_text=json.dumps({'max_tokens':token_cap,'model':response.get('model'),'id':response.get('id'),'usage':response.get('usage'),'finish_reason':choice.get('finish_reason'),'cluster_ids':item['cluster_ids']},indent=2)
                import time
                (dest/f'receipt-attempt-{time.time_ns()}.json').write_text(receipt_text)
                (dest/'receipt.json').write_text(receipt_text)
                if not content or choice.get('finish_reason')=='length': raise SystemExit('Incomplete decoder output; receipt saved; no success claimed')
                (dest/'proposal.md').write_text(content); count+=1
                print('Decoded',item['cluster_ids'],flush=True)
            body=(dest/'proposal.md').read_text() if (dest/'proposal.md').exists() else 'Not decoded. Run with --skip-sample --decode 1 for a paid reconstruction.'
            scores=f'coherence {item["coherence"]:.3f} · availability {item["availability"]:.3f} · fused {item["fusion_score"]:.3f}'
            cards.append(f'<article><h3>#{i+1} · atoms {html.escape(str(item["cluster_ids"]))}</h3><p>{scores}</p><ul>'+''.join('<li>'+html.escape(s)+'</li>' for s in atoms)+f'</ul><details><summary>Methodology sketch</summary><pre>{html.escape(body)}</pre></details></article>')
    page='<!doctype html><meta charset="utf-8"><title>Alien Science — local run</title><style>body{max-width:1000px;margin:40px auto;padding:0 24px;font:17px/1.6 system-ui;background:#f7f5ee;color:#192a28}article{background:white;padding:24px;margin:20px 0;border:1px solid #ccc}pre{white-space:pre-wrap;font:inherit}h2{margin-top:50px}</style><h1>Alien Science · local run</h1><p>Real released scorers; sampled candidate pool; generated sketches are untested hypotheses, not established findings. Scores are relative to this candidate pool, not probabilities of scientific success.</p>'+''.join(cards)
    (out/'report.html').write_text(page)
    (out/'run.json').write_text(json.dumps({'device':a.device,'generation_batch_size':a.generation_batch_size,'candidate_source':a.candidate_source,'seed':a.seed,'betas':a.betas,'n_gen':a.n_gen,'decode_per_beta':a.decode,'model':a.model,'new_calls':count,'upstream_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=UP,text=True).strip()},indent=2))
    print('Report:',out/'report.html')
if __name__=='__main__': main()
