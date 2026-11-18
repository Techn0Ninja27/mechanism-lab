"""Closed-loop GCN evaluator for causal-program candidates (local only).

The final-seed path is intentionally disabled until a frozen development
selection manifest exists. This module never calls a model or a network.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Any
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
GAPDIR = ROOT / "operator-studies/gnn-closedloop-gap"
sys.path[:0] = [str(ROOT), str(GAPDIR)]
import gap  # type: ignore
from generation.causal_program import CausalState, CompiledProgram, compile_program

class ProgramRemote:
    def __init__(self, program: CompiledProgram, shape: tuple[int,int], sync: int):
        self.program, self.state, self.sync = program, CausalState(shape), sync
    def __call__(self, exact: torch.Tensor, t: int) -> torch.Tensor:
        observed = t <= gap.WARM or (t-gap.WARM) % self.sync == 0
        if observed:
            return self.state.observe(exact.detach(), t, self.program)
        return self.state.predict(t, self.program)

def run_one(x, y, adj, sp, candidate: dict[str,Any], *, seed: int, epochs: int=120,
            parts: int=4, layers: int=3, sync: int=16, window: tuple[int,...]|None=None) -> dict[str,Any]:
    metadata = {k: candidate.get(k) for k in ("id", "call_id", "arm")} if isinstance(candidate, dict) else {"id": None, "call_id": None, "arm": None}
    try:
        if not isinstance(candidate, dict): raise ValueError("candidate must be an object")
        if not isinstance(seed, int) or isinstance(seed, bool): raise ValueError("invalid seed")
        if not isinstance(epochs, int) or isinstance(epochs, bool) or not 1 <= epochs <= 120: raise ValueError("invalid epochs")
        if (parts, layers, sync) != (4, 3, 16): raise ValueError("only the fixed pilot regime is implemented")
        program = compile_program(candidate["program"])
        torch.manual_seed(seed); np.random.seed(seed)
        dims=[x.shape[1]]+[gap.HID]*(layers-1)+[int(y.max())+1]
        scale=lambda i: .05 if layers==2 else float(np.sqrt(2/(dims[i]+dims[i+1])))
        ws=[torch.nn.Parameter(torch.randn(dims[i],dims[i+1]) * scale(i)) for i in range(layers)]
        opt=torch.optim.Adam(ws,lr=.01,weight_decay=.0005); tr,va,ev=sp
        if set(tr)&set(va) or set(tr)&set(ev) or set(va)&set(ev): raise ValueError("label splits overlap")
        a_in,a_x=gap.split_adj(adj,gap.partition(len(y),parts)); est=[ProgramRemote(program,(len(y),dims[k+1]),sync) for k in range(layers-1)]
        rows=[]; wanted=set(range(max(0,epochs-32),epochs) if window is None else window)
        if any(type(t) is not int or not 0 <= t < epochs for t in wanted): raise ValueError("invalid evaluation window")
        for t in range(epochs):
            hs=[torch.relu(torch.sparse.mm(adj,x@ws[0]))]
            for k in range(1,layers):
                cur=hs[-1]; remote=est[k-1](cur.detach(),t)
                agg=torch.sparse.mm(a_in,cur@ws[k])+torch.sparse.mm(a_x,remote@ws[k])
                hs.append(torch.relu(agg) if k<layers-1 else agg)
            if t in wanted:
                with torch.no_grad():
                    exact=gap.forward_exact(x,adj,ws)
                    rows.append({"t":t,"own_eval":float((hs[-1].argmax(1)[ev]==y[ev]).float().mean()),"exact_eval":float((exact.argmax(1)[ev]==y[ev]).float().mean())})
            loss=torch.nn.functional.cross_entropy(hs[-1][tr],y[tr])
            if not torch.isfinite(loss).item() or not torch.isfinite(hs[-1]).all().item(): raise ValueError("non-finite training loss or logits")
            opt.zero_grad(); loss.backward()
            if any(w.grad is None or not torch.isfinite(w.grad).all().item() for w in ws): raise ValueError("non-finite gradient")
            opt.step()
            if any(not torch.isfinite(w).all().item() for w in ws): raise ValueError("non-finite model weights")
        with torch.no_grad(): final=gap.forward_exact(x,adj,ws)
        if not torch.isfinite(final).all().item(): raise ValueError("non-finite evaluation logits")
        return {**metadata,"status":"complete","seed":seed,"epochs":epochs,"window":rows,"post_update_exact_eval":float((final.argmax(1)[ev]==y[ev]).float().mean()),"final_train_loss":float(loss.detach()),"program_nodes":program.node_count}
    except Exception as exc:
        return {**metadata,"status":"failed","seed":seed,"epochs":epochs,"error":f"{type(exc).__name__}: {str(exc)[:300]}"}

def evaluate_candidates(candidates: list[dict[str,Any]], *, seed: int=98, epochs: int=32) -> list[dict[str,Any]]:
    x,y,adj=gap.screen.load_cora(gap.DATA); sp=gap.splits(y)
    return [run_one(x,y,adj,sp,candidate,seed=seed,epochs=epochs,window=tuple(range(max(0,epochs-2),epochs))) for candidate in candidates]

def select_development(result: dict[str,Any]) -> dict[str,Any]:
    raise NotImplementedError("Development selection requires a validated campaign manifest and complete per-candidate seed sets; it is not enabled in the smoke evaluator.")


def main():
    p=argparse.ArgumentParser();p.add_argument("--bundle",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--execute",action="store_true");p.add_argument("--seed",type=int,default=98);a=p.parse_args()
    if a.output.exists(): raise FileExistsError("output exists; choose a fresh directory")
    bundle=json.loads(a.bundle.read_text()); candidates=bundle.get("candidates") if isinstance(bundle,dict) else None
    if not isinstance(candidates,list) or not 1 <= len(candidates) <= 128: raise ValueError("bundle candidates missing or outside bounds")
    if a.execute and a.seed != 98: raise ValueError("only smoke seed 98 is enabled; dev/final execution is disabled")
    a.output.mkdir(parents=True)
    if not a.execute: result={"schema":"closed-loop-candidate-plan.v1","execute_required":True,"final_evaluation":"disabled_until_frozen_dev_selection","paid_calls":0}
    else:
        if a.seed != 98: raise ValueError("only smoke seed 98 is enabled; dev/final execution is disabled")
        result={"schema":"closed-loop-candidate-smoke.v1","runs":evaluate_candidates(candidates,seed=a.seed,epochs=32),"paid_calls":0,"auto_execute":False,"novelty_established":False,"train_eligible":False}
    (a.output/"result.json").write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps({"schema":result["schema"],"output":str(a.output),"paid_calls":0}))
if __name__=="__main__": main()
