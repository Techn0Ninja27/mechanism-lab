"""Cost screen for source-aligned Muon/Newton--Muon on a tiny causal LM.

This module intentionally separates synchronized end-to-end timing from an
instrumented diagnostic pass.  It is a development harness, not a claim about
large-model training or optimizer quality.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
import warnings
import hashlib
import platform
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F

torch.set_num_threads(1)

NS_COEFFS = (3.4445, -4.7750, 2.0315)


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def require_no_fallback(device: torch.device) -> None:
    if device.type == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise RuntimeError("MPS fallback must be disabled for a capability/timing claim")


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    require_no_fallback(device)
    return device


def matrix_sign(x: torch.Tensor) -> torch.Tensor:
    transposed=x.shape[0]>x.shape[1]
    z=x.T if transposed else x
    z=z/(z.norm()+1e-7)
    a,b,c=NS_COEFFS
    for _ in range(5):
        gram=z@z.T
        polynomial=b*gram+c*(gram@gram)
        z=a*z+polynomial@z
    return z.T if transposed else z


def right_precondition(grad: torch.Tensor, inverse: torch.Tensor | None, blocks: int = 1) -> torch.Tensor:
    """Apply a right inverse to a raw gradient before momentum."""
    if inverse is None:
        return grad
    if blocks == 1:
        return grad @ inverse
    m, width = grad.shape
    if width % blocks:
        raise ValueError("block count must divide gradient input width")
    chunk = width // blocks
    return torch.cat([grad[:, i * chunk:(i + 1) * chunk] @ inverse[i] for i in range(blocks)], dim=1)


@dataclass
class CovarianceState:
    k: torch.Tensor
    inverse: torch.Tensor | None = None
    last_refresh: int = -1
    refresh_count: int = 0
    block_count: int = 1


class TrackedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, *, block_count: int = 1):
        super().__init__(in_features, out_features)
        self.block_count = block_count
        self.last_input: torch.Tensor | None = None
        self.capture_input = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_input = x.detach().reshape(-1, x.shape[-1]) if self.capture_input else None
        return F.linear(x, self.weight, self.bias)


class TinyBlock(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__(); self.width = width; self.heads = heads
        self.norm1 = nn.LayerNorm(width); self.qkv = TrackedLinear(width, 3 * width)
        self.proj = TrackedLinear(width, width); self.norm2 = nn.LayerNorm(width)
        self.fc1 = TrackedLinear(width, 4 * width); self.fc2 = TrackedLinear(4 * width, width, block_count=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape; h = self.heads; dh = d // h
        qkv = self.qkv(self.norm1(x)).view(b, t, 3, h, dh).transpose(1, 3)
        q, k, v = qkv.unbind(2)
        mask = torch.ones(t, t, device=x.device, dtype=torch.bool).tril()
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        a = a.transpose(1, 2).contiguous().view(b, t, d)
        x = x + self.proj(a)
        x = x + self.fc2(F.gelu(self.fc1(self.norm2(x))))
        return x


class TinyCausalTransformer(nn.Module):
    def __init__(self, vocab: int, width: int = 256, layers: int = 6, heads: int = 4, seq: int = 64):
        super().__init__(); self.width = width; self.seq = seq
        self.token = nn.Embedding(vocab, width); self.pos = nn.Parameter(torch.zeros(1, seq, width))
        self.blocks = nn.ModuleList([TinyBlock(width, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(width); self.lm_head = nn.Linear(width, vocab)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.token(tokens) + self.pos[:, :tokens.shape[1]]
        for block in self.blocks: x = block(x)
        return self.lm_head(self.norm(x))

    def matrix_modules(self) -> list[TrackedLinear]:
        return [b.qkv for b in self.blocks] + [b.proj for b in self.blocks] + [b.fc1 for b in self.blocks] + [b.fc2 for b in self.blocks]


class SourceAlignedMuon:
    """Muon or full-Cholesky Newton--Muon for tracked matrix parameters."""
    def __init__(self, modules: Iterable[TrackedLinear], *, mode: str, beta: float = .95, gamma: float = .2, refresh: int = 32, nesterov: bool = True):
        if mode not in {"muon", "newton_cholesky"}: raise ValueError("unknown optimizer mode")
        self.modules = list(modules); self.mode = mode; self.beta = beta; self.gamma = gamma; self.refresh = refresh; self.nesterov = nesterov; self.step_index = 0
        self.momentum = {id(m.weight): torch.zeros_like(m.weight) for m in self.modules}
        self.states: dict[int, CovarianceState] = {}
        self.factor_infos = []
        for m in self.modules:
            n = m.in_features
            if n % m.block_count: raise ValueError("block count must divide input width")
            chunk = n // m.block_count
            if mode == "newton_cholesky":
                eye = torch.eye(chunk, device=m.weight.device, dtype=m.weight.dtype) * 1e-3
                k = eye.unsqueeze(0).repeat(m.block_count, 1, 1)
                inverse = torch.eye(chunk, device=m.weight.device, dtype=m.weight.dtype).unsqueeze(0).repeat(m.block_count, 1, 1)
                self.states[id(m.weight)] = CovarianceState(k, inverse=inverse[0] if m.block_count == 1 else inverse, last_refresh=0, block_count=m.block_count)

    def _inverse(self, state: CovarianceState) -> tuple[torch.Tensor | None, float]:
        k = state.k; n = k.shape[-1]
        ridges = self.gamma * torch.diagonal(k, dim1=-2, dim2=-1).sum(-1) / n
        eye = torch.eye(n, device=k.device, dtype=k.dtype).expand(k.shape[0], -1, -1)
        damped = k + ridges[:, None, None] * eye
        factor, info = torch.linalg.cholesky_ex(damped, check_errors=False)
        self.factor_infos.append(info.detach())
        inv = torch.cholesky_inverse(factor)
        return (inv[0] if state.block_count == 1 else inv), ridges

    @torch.no_grad()
    def step(self, *, synchronize_components: bool = False) -> dict[str, float]:
        self.step_index += 1
        timing={k:0.0 for k in ('covariance','refresh','apply','muon')}
        refreshed=0
        def measure(name,fn,device):
            if not synchronize_components:return fn()
            synchronize(device);start=time.perf_counter();result=fn();synchronize(device)
            timing[name]+=time.perf_counter()-start
            return result
        for m in self.modules:
            grad=m.weight.grad
            if grad is None:continue
            state=self.states.get(id(m.weight));device=m.weight.device
            if state is not None and self.step_index%self.refresh==0:
                if m.last_input is None:raise ValueError('Missing refresh activations')
                def covariance():
                    chunks=m.last_input.split(m.in_features//state.block_count,dim=1)
                    for i,chunk in enumerate(chunks):
                        state.k[i].mul_(self.beta).add_((chunk.T@chunk)/chunk.shape[0],alpha=1-self.beta)
                measure('covariance',covariance,device)
                state.inverse,_=measure('refresh',lambda:self._inverse(state),device)
                state.last_refresh=self.step_index;state.refresh_count+=1;refreshed+=1
            pre=measure('apply',lambda:right_precondition(grad,state.inverse,state.block_count),device) if state else grad
            def update():
                mom=self.momentum[id(m.weight)];mom.mul_(self.beta).add_(pre)
                direction=pre.add(mom,alpha=self.beta) if self.nesterov else mom
                if m.out_features==3*m.in_features:
                    direction=torch.cat([matrix_sign(block) for block in direction.chunk(3,dim=0)],dim=0)
                    scale=math.sqrt(m.in_features)
                else:
                    direction=matrix_sign(direction);scale=math.sqrt(max(m.weight.shape))
                m.weight.mul_(1-.001*.01).add_(direction,alpha=-.001*scale)
                m.weight.grad=None
            measure('muon',update,device)
        return dict(timing,refreshed_modules=refreshed)


def make_batches(text: str, vocab: int, seq: int = 64, batch: int = 4, steps: int = 16, seed: int = 0) -> list[tuple[torch.Tensor, torch.Tensor]]:
    g = torch.Generator().manual_seed(seed); ids = torch.tensor(list(text.encode("utf-8")), dtype=torch.long)
    if ids.numel() < seq+2: raise ValueError("Real text input too short")
    batches = []
    for _ in range(steps):
        starts = torch.randint(0, ids.numel() - seq - 1, (batch,), generator=g); x = torch.stack([ids[s:s + seq] for s in starts]); y = torch.stack([ids[s + 1:s + seq + 1] for s in starts]); batches.append((x, y))
    return batches


def _make_arm(width: int, device: torch.device, seed: int, mode: str, refresh: int):
    torch.manual_seed(seed)
    model = TinyCausalTransformer(256, width=width).to(device)
    modules = model.matrix_modules(); matrix_ids = {id(m.weight) for m in modules}
    nonmatrix = [p for p in model.parameters() if id(p) not in matrix_ids]
    return model, SourceAlignedMuon(modules, mode=mode, refresh=refresh), torch.optim.AdamW(nonmatrix, lr=1e-3, weight_decay=.01)


def _train_step(model,opt,adam,x,y,*,sync_components=False):
    device=x.device
    for m in opt.modules:m.capture_input=(opt.mode=='newton_cholesky' and (opt.step_index+1)%opt.refresh==0)
    def boundary():
        if sync_components:synchronize(device)
    boundary();total_start=time.perf_counter();start=total_start
    model.zero_grad(set_to_none=True)
    logits=model(x);loss=F.cross_entropy(logits.reshape(-1,256),y.reshape(-1));loss.backward()
    boundary();forward=time.perf_counter()-start
    timing=opt.step(synchronize_components=sync_components)
    boundary();start=time.perf_counter();adam.step();adam.zero_grad(set_to_none=True)
    boundary();adam_time=time.perf_counter()-start
    timing.update(forward_backward=forward,adam=adam_time,total=time.perf_counter()-total_start)
    return loss.detach(),timing


def _run_arm(batches,*,width,device,seed,mode,refresh,instrumented=False):
    model,opt,adam=_make_arm(width,device,seed,mode,refresh)
    warmup=8;window_steps=32;windows=[];losses=[];components=[]
    for i in range(warmup):
        loss,_=_train_step(model,opt,adam,*batches[i]);losses.append(loss)
    for w in range(1 if instrumented else 3):
        synchronize(device);start=time.perf_counter()
        for i in range(warmup+w*window_steps,warmup+(w+1)*window_steps):
            loss,timing=_train_step(model,opt,adam,*batches[i],sync_components=instrumented)
            losses.append(loss)
            if instrumented:components.append(timing)
        synchronize(device);windows.append({'window':w,'steps':window_steps,'seconds':time.perf_counter()-start})
    values=[float(x.cpu()) for x in losses]
    info=[int(x.max().cpu()) for x in opt.factor_infos]
    if not all(math.isfinite(x) for x in values) or any(info):raise ValueError('Nonfinite loss or failed Cholesky')
    memory={'current_allocated_bytes':torch.mps.current_allocated_memory(),'driver_allocated_bytes':torch.mps.driver_allocated_memory(),'scope':'end-of-arm snapshots, NOT peak memory'} if device.type=='mps' else {}
    result={'windows':windows,'loss_finite':True,'loss_first':values[0],'loss_last':values[-1],
            'diagnostic_components':components,'refresh_counts':[x.refresh_count for x in opt.states.values()],
            'covariance_state_bytes':sum(x.k.numel()*x.k.element_size() for x in opt.states.values()),
            'inverse_state_bytes':sum(x.inverse.numel()*x.inverse.element_size() for x in opt.states.values()),
            'cholesky_info_max':max(info,default=0),'memory_snapshot':memory}
    del model,opt,adam,losses;gc.collect()
    if device.type=='mps':torch.mps.empty_cache()
    return result


def run_profile(text,*,width=256,device_name='cpu',seed=0,refresh=32):
    device=resolve_device(device_name)
    batches=[(x.to(device),y.to(device)) for x,y in make_batches(text,256,seq=64,batch=4,steps=104,seed=seed)]
    arms={}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for mode in ('muon','newton_cholesky'):
            e2e=_run_arm(batches,width=width,device=device,seed=seed,mode=mode,refresh=refresh)
            diag=_run_arm(batches,width=width,device=device,seed=seed,mode=mode,refresh=refresh,instrumented=True)
            components=diag.pop('diagnostic_components')
            sums={k:sum(x[k] for x in components) for k in ('covariance','refresh','apply','muon','forward_backward','adam','total')}
            fraction=sum(sums[k] for k in ('covariance','refresh','apply'))/sums['total']
            arms[mode]={'end_to_end':e2e,'instrumented':dict(diag,steps=32,component_seconds=sums,precondition_fraction_total=fraction,
                scope='Separately initialized and warmed diagnostic; synchronization changes overlap. Not additive to uninstrumented windows.')}
    warning_text=[str(w.message)[:500] for w in caught]
    fallback=any('fall' in x.lower() and 'cpu' in x.lower() for x in warning_text)
    return {'schema':'newton-muon-cost-screen.v3','width':width,'device':str(device),'dtype':'float32','seed':seed,'refresh':refresh,'arms':arms,
            'fallback_env':os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'),'warnings':warning_text,'internal_cpu_fallback_detected':fallback,
            'valid_mps_timing':device.type=='mps' and not fallback,'torch_version':torch.__version__,'platform':platform.platform(),
            'scope':'Necessary cost screen only; no target-loss, utility or novelty claim'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--text',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--width',type=int,choices=[256,512],default=256);p.add_argument('--device',default='cpu');p.add_argument('--refresh',type=int,default=32)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve existing run; choose a fresh output')
    raw=a.text.read_bytes();result=run_profile(raw.decode('utf-8'),width=a.width,device_name=a.device,refresh=a.refresh)
    result.update(text_sha256=hashlib.sha256(raw).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'width':a.width,'valid_mps_timing':result['valid_mps_timing'],'fraction':result['arms']['newton_cholesky']['instrumented']['precondition_fraction_total']}),flush=True)
if __name__=='__main__':main()
