"""Fail-closed development selection and final reporting for candidate campaigns.

This module is deliberately model-agnostic and performs no training or network
calls.  It treats calls as the experimental unit: candidates from one call are
correlated and are never counted as independent replicates.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

DEV_SEEDS = tuple(range(100, 116))
FINAL_SEEDS = tuple(range(200, 232))


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def _base(schema: str) -> dict[str, Any]:
    return {"schema": schema, "auto_execute": False, "train_eligible": False, "novelty_established": False}


def _inventory(inventory: Any) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if not isinstance(inventory, list) or not inventory:
        raise ValueError("candidate inventory must be a non-empty list")
    by_id: dict[str, dict[str, Any]] = {}
    for row in inventory:
        if not isinstance(row, Mapping): raise ValueError("inventory candidate must be an object")
        cid, call, arm, ph = row.get("id"), row.get("call_id"), row.get("arm"), row.get("program_sha256")
        if not all(isinstance(x, str) and x for x in (cid, call, arm, ph)):
            raise ValueError("inventory candidate identifiers and program hash must be strings")
        if len(ph) != 64 or any(c not in "0123456789abcdef" for c in ph): raise ValueError("invalid program_sha256")
        if cid in by_id: raise ValueError("duplicate inventory candidate")
        by_id[cid] = dict(row)
    return [by_id[k] for k in sorted(by_id)], by_id


def _window_score(run: Mapping[str, Any], expected_window: Sequence[int] = tuple(range(88, 120))) -> float:
    window = run.get("window")
    if not isinstance(window, list) or tuple(sorted({r.get("t") for r in window if isinstance(r, Mapping)})) != tuple(expected_window): raise ValueError("evaluation window must contain each expected epoch exactly once")
    vals = []
    for row in window:
        if not isinstance(row, Mapping) or not _finite(row.get("exact_eval")):
            raise ValueError("window exact_eval is malformed or non-finite")
        value = float(row["exact_eval"])
        if not 0 <= value <= 1: raise ValueError("accuracy must be in [0,1]")
        vals.append(value)
    return sum(vals) / len(vals)


def _optional_window_score(run: Mapping[str, Any], key: str, expected_window: Sequence[int]) -> float | None:
    window = run.get("window")
    if not isinstance(window, list) or tuple(sorted({r.get("t") for r in window if isinstance(r, Mapping)})) != tuple(expected_window): return None
    vals = [r.get(key) for r in window if isinstance(r, Mapping)]
    if len(vals) != len(expected_window) or any(not _finite(v) or not 0 <= float(v) <= 1 for v in vals): return None
    return sum(float(v) for v in vals) / len(vals)


def _optional_endpoint(run: Mapping[str, Any]) -> float | None:
    value = run.get("post_update_exact_eval", run.get("post_update_exact_eval_120"))
    return float(value) if _finite(value) and 0 <= float(value) <= 1 else None


def _run_key(row: Mapping[str, Any]) -> tuple[str, int]:
    cid, seed = row.get("candidate_id", row.get("id")), row.get("seed")
    if not isinstance(cid, str) or not isinstance(seed, int) or isinstance(seed, bool): raise ValueError("malformed run key")
    return cid, seed


def select_development(inventory: Any, runs: Any, *, expected_seeds: Sequence[int] = DEV_SEEDS,
                       inventory_sha256: str | None = None, code_sha256: str | None = None,
                       expected_window: Sequence[int] = tuple(range(88, 120))) -> dict[str, Any]:
    """Select the highest mean exact-window accuracy per call on development seeds."""
    candidates, by_id = _inventory(inventory)
    seeds = tuple(expected_seeds)
    if not seeds or any(not isinstance(s, int) or isinstance(s, bool) for s in seeds) or len(set(seeds)) != len(seeds): raise ValueError("invalid expected seeds")
    if not isinstance(runs, list): raise ValueError("development runs must be a list")
    expected = {(c["id"], s) for c in candidates for s in seeds}
    seen: set[tuple[str, int]] = set(); by_candidate: dict[str, dict[int, Mapping[str, Any]]] = defaultdict(dict); errors: list[str] = []
    for row in runs:
        if not isinstance(row, Mapping): raise ValueError("development run must be an object")
        key = _run_key(row); cid, seed = key
        if cid not in by_id: errors.append(f"foreign candidate {cid}"); continue
        if key in seen: errors.append(f"duplicate run {cid}/{seed}"); continue
        if seed not in seeds: errors.append(f"unexpected seed {seed}"); continue
        if row.get("call_id") != by_id[cid]["call_id"] or row.get("arm") != by_id[cid]["arm"]:
            errors.append(f"candidate metadata mismatch {cid}/{seed}"); continue
        seen.add(key); by_candidate[cid][seed] = row
    missing = sorted(expected - seen)
    if missing: errors.append(f"missing development runs: {missing[:8]}")
    if errors:
        out = _base("causal-candidate-selection.v1"); out.update({"status":"invalid", "errors":errors, "source_inventory_sha256": inventory_sha256, "code_sha256": code_sha256, "expected_runs":len(expected), "observed_unique_runs":len(seen)})
        return out
    scored: list[dict[str, Any]] = []
    for c in candidates:
        rows = by_candidate[c["id"]]; scores=[]; failures=[]
        for seed in seeds:
            r=rows[seed]
            if r.get("status") != "complete": failures.append({"seed":seed,"error":r.get("error","run failed")}); continue
            try: scores.append({"seed":seed,"score":_window_score(r, expected_window)})
            except ValueError as exc: failures.append({"seed":seed,"error":str(exc)})
        scored.append({"candidate_id":c["id"],"call_id":c["call_id"],"arm":c["arm"],"program_sha256":c["program_sha256"],"scores":scores,"failures":failures,"eligible":len(failures)==0 and len(scores)==len(seeds),"mean_score":sum(x["score"] for x in scores)/len(scores) if len(scores)==len(seeds) else None})
    selected=[]
    for call in sorted({x["call_id"] for x in scored}):
        valid=[x for x in scored if x["call_id"]==call and x["eligible"]]
        win=min(valid,key=lambda x:(-x["mean_score"],x["candidate_id"])) if valid else None
        selected.append({"call_id":call,"candidate_id":win["candidate_id"] if win else None,"program_sha256":win["program_sha256"] if win else None,"status":"selected" if win else "no_eligible_candidate"})
    out=_base("causal-candidate-selection.v1"); out.update({"status":"complete","split":"dev","expected_seeds":list(seeds),"source_inventory_sha256":inventory_sha256 or canonical_hash(candidates),"candidate_inventory_sha256":canonical_hash(candidates),"code_sha256":code_sha256,"candidate_scores":scored,"selected":selected,"descriptive_call_level_only":True,"no_between_condition_test":True}); return out


def final_report(inventory: Any, selection: Mapping[str, Any], runs: Any, hold_runs: Any, *, expected_seeds: Sequence[int] = FINAL_SEEDS,
                 inventory_sha256: str | None = None, expected_window: Sequence[int] = tuple(range(88, 120)),
                 dev_hold_runs: Any = None, control_runs: Any = None, control_id: str | None = None) -> dict[str, Any]:
    """Score only candidates frozen by development selection on final seeds."""
    candidates, by_id = _inventory(inventory); seeds=tuple(expected_seeds)
    if selection.get("schema") != "causal-candidate-selection.v1" or selection.get("status") != "complete": raise ValueError("invalid selection manifest")
    computed_inventory_hash = canonical_hash(candidates)
    if selection.get("candidate_inventory_sha256") != computed_inventory_hash: raise ValueError("selection candidate inventory hash mismatch")
    if inventory_sha256 is not None and selection.get("source_inventory_sha256") != inventory_sha256: raise ValueError("selection source inventory hash mismatch")
    selected={x.get("candidate_id"):x for x in selection.get("selected",[]) if isinstance(x,Mapping) and isinstance(x.get("candidate_id"),str)}
    chosen=[x for x in selected.values() if x.get("candidate_id") in by_id]
    no_eligible = [{"call_id":x.get("call_id"),"status":"no_eligible_candidate"} for x in selection.get("selected",[]) if isinstance(x,Mapping) and x.get("candidate_id") is None]
    allruns=[runs,hold_runs]
    indexed=[]
    for source in allruns:
        if not isinstance(source,list): raise ValueError("final runs must be lists")
        indexed.append(source)
    def index(rows, *, hold=False):
        out={}
        for r in rows:
            if not isinstance(r,Mapping): raise ValueError("final run must be object")
            cid=r.get("candidate_id", r.get("id"))
            if hold and (cid in (None, "__hold__") or r.get("arm") == "hold"): cid="hold"
            seed=r.get("seed")
            if not isinstance(cid,str) or not isinstance(seed,int) or isinstance(seed,bool): raise ValueError("malformed final key")
            key=(cid,seed)
            if key in out: raise ValueError("duplicate final run")
            out[key]=r
        return out
    idx=index(runs); hidx=index(hold_runs,hold=True); expected_hold=set(seeds)
    if set((cid,s) for cid,s in hidx) != {("hold",s) for s in seeds}: raise ValueError("hold final runs incomplete or foreign")
    for s in seeds:
        h=hidx[("hold",s)]
        if h.get("status") != "complete" or h.get("arm", "hold") != "hold": raise ValueError("hold final run is not complete hold metadata")
    cidx = None
    if control_runs is not None:
        if not isinstance(control_id, str) or not control_id: raise ValueError("control_id is required with control runs")
        cidx = index(control_runs)
        if set(cidx) != {(control_id, s) for s in seeds}: raise ValueError("control final runs incomplete or foreign")
        for s in seeds:
            c = cidx[(control_id, s)]
            if c.get("status") != "complete" or c.get("control_id", c.get("id")) != control_id: raise ValueError("control final metadata mismatch")
    allowed = {(x["candidate_id"], s) for x in chosen for s in seeds}
    if set(idx) != allowed: raise ValueError("final runs contain foreign or unselected candidate keys")
    reports=list(no_eligible)
    hashes=defaultdict(list)
    for c in candidates: hashes[(c["arm"], c["program_sha256"])].append(c["id"])
    dev_hold_by_seed={}
    if dev_hold_runs is not None:
        if not isinstance(dev_hold_runs,list): raise ValueError("development hold runs must be a list")
        for h in dev_hold_runs:
            if not isinstance(h,Mapping) or not isinstance(h.get("seed"),int) or h.get("status") != "complete": raise ValueError("malformed development hold run")
            dev_hold_by_seed[h["seed"]]=_window_score(h,expected_window)
    for chosenrow in chosen:
        cid=chosenrow["candidate_id"]
        if chosenrow.get("program_sha256") != by_id[cid]["program_sha256"]: raise ValueError(f"selected program hash mismatch: {cid}")
        expected={(cid,s) for s in seeds}; got={(k[0],k[1]) for k in idx if k[0]==cid}
        if got != expected: raise ValueError(f"selected candidate final runs incomplete: {cid}")
        values=[]; holds=[]; failures=[]; secondary_values=[]; secondary_holds=[]; endpoint_values=[]; endpoint_holds=[]
        for s in seeds:
            r=idx[(cid,s)]; h=hidx[("hold",s)]
            if r.get("status") != "complete": failures.append(s); continue
            if r.get("candidate_id",cid) != cid or r.get("call_id") != chosenrow.get("call_id") or r.get("arm") != by_id[cid].get("arm"): failures.append(s); continue
            try:
                values.append((s,_window_score(r,expected_window))); holds.append((s,_window_score(h,expected_window)))
                secondary_values.append((s,_optional_window_score(r,"own_eval",expected_window))); secondary_holds.append((s,_optional_window_score(h,"own_eval",expected_window)))
                endpoint_values.append((s,_optional_endpoint(r))); endpoint_holds.append((s,_optional_endpoint(h)))
            except ValueError: failures.append(s)
        # Bounds impute failed candidate accuracy by the metric's valid range [0,1].
        value_map=dict(values); hold_map=dict(holds); hold_values=[_window_score(hidx[("hold",s)],expected_window) for s in seeds]
        diffs=[value_map[s]-hold_map[s] for s in sorted(value_map)]
        control_values={}
        control_diffs=[]
        if cidx is not None:
            for s in sorted(value_map):
                control_values[s]=_window_score(cidx[(control_id,s)],expected_window)
            control_diffs=[value_map[s]-control_values[s] for s in sorted(value_map)]
        lower=sum((value_map[s]-hold_values[i] if s in value_map else -hold_values[i]) for i,s in enumerate(seeds))/len(seeds)
        upper=sum((value_map[s]-hold_values[i] if s in value_map else 1-hold_values[i]) for i,s in enumerate(seeds))/len(seeds)
        mean=sum(diffs)/len(diffs) if not failures and diffs else None
        se=(sum((x-mean)**2 for x in diffs)/(len(diffs)-1)/len(diffs))**.5 if mean is not None and len(diffs)>1 else None
        control_mean=sum(control_diffs)/len(control_diffs) if cidx is not None and not failures and control_diffs else None
        control_se=(sum((x-control_mean)**2 for x in control_diffs)/(len(control_diffs)-1)/len(control_diffs))**.5 if control_mean is not None and len(control_diffs)>1 else None
        low_precision = bool(failures or se is None or se > .005)
        better_hold=bool(not low_precision and not failures and mean is not None and mean>.005 and se is not None and mean>2*se)
        control_low=bool(failures or control_se is None or control_se>.005)
        better_control=bool(cidx is not None and not control_low and control_mean is not None and control_mean>.005 and control_se is not None and control_mean>2*control_se)
        own_map=dict(secondary_values); own_hold_map=dict(secondary_holds); end_map=dict(endpoint_values); end_hold_map=dict(endpoint_holds)
        def secondary(metric_map, baseline_map):
            pairs=[metric_map[s]-baseline_map[s] for s in metric_map if metric_map[s] is not None and baseline_map.get(s) is not None]
            return {"candidate_mean":sum(v for v in metric_map.values() if v is not None)/len([v for v in metric_map.values() if v is not None]) if any(v is not None for v in metric_map.values()) else None,"baseline_mean":sum(v for v in baseline_map.values() if v is not None)/len([v for v in baseline_map.values() if v is not None]) if any(v is not None for v in baseline_map.values()) else None,"difference":sum(pairs)/len(pairs) if pairs else None}
        secondary_metrics={"own_forward_window_vs_hold":secondary(own_map,own_hold_map),"endpoint_exact_vs_hold":secondary(end_map,end_hold_map)}
        if cidx is not None:
            c_own={s:_optional_window_score(cidx[(control_id,s)],"own_eval",expected_window) for s in seeds}; c_end={s:_optional_endpoint(cidx[(control_id,s)]) for s in seeds}
            secondary_metrics["own_forward_window_vs_control"]=secondary(own_map,c_own); secondary_metrics["endpoint_exact_vs_control"]=secondary(end_map,c_end)
        reports.append({"candidate_id":cid,"call_id":chosenrow.get("call_id"),"arm":by_id[cid]["arm"],"program_sha256":by_id[cid]["program_sha256"],"exact_duplicate_count":len(hashes[(by_id[cid]["arm"],by_id[cid]["program_sha256"])]),"failed_final_seeds":sorted(failures),"complete_final_seeds":len(values),"paired_mean_difference":mean,"paired_se":se,"paired_mean_difference_control":control_mean,"paired_se_control":control_se,"pointwise_ci95":([mean-2.04*se,mean+2.04*se] if mean is not None and se is not None and len(seeds)==32 else None),"secondary_metrics":secondary_metrics,"low_precision":low_precision,"low_precision_control":control_low,"missing_failure_difference_bounds":[lower,upper],"positive_requires_zero_failures":True,"better_than_hold":better_hold,"better_than_control":better_control,"comparison_baseline":"best_control" if cidx is not None else "hold_diagnostic"})
    positive=[r for r in reports if r.get("better_than_control") if cidx is not None] if cidx is not None else [r for r in reports if r.get("better_than_hold")]
    distinct_positive={(r.get("arm"),r.get("program_sha256")) for r in positive}
    task_count=len({h for a,h in distinct_positive if a in {"task-only","task"}})
    source_counts={arm:len({h for a,h in distinct_positive if a==arm}) for arm in ("raw","atoms","distill")}
    source_count=max(source_counts.values(), default=0)
    random_positive=any(r.get("better_than_control" if cidx is not None else "better_than_hold") and r.get("arm") in {"random","control","linear","linear-control"} for r in reports)
    primary_low_precision = any(r.get("status")=="no_eligible_candidate" or r.get("low_precision_control" if cidx is not None else "low_precision", True) for r in reports)
    closure_eligible = cidx is not None and bool(reports) and not positive and not primary_low_precision
    out=_base("causal-campaign-result.v1"); out.update({"status":"complete","split":"final","expected_seeds":list(seeds),"reports":reports,"selection_manifest_hash":canonical_hash(selection),"no_selection_on_final":True,"descriptive_call_level_only":True,"no_between_condition_test":True,"exact_duplicate_groups":{arm:len({h for a,h in hashes if a==arm}) for arm in {a for a,h in hashes}},"decision":{"comparison_baseline":"best_control" if cidx is not None else "hold_diagnostic","distinct_positive_task_or_control":task_count,"distinct_positive_source":source_count,"random_or_control_positive":random_positive,"task_or_dsl_lead":task_count>=2,"distinct_positive_by_source_arm":source_counts,"representation_replication_candidate":source_count>=2 and task_count<2 and not random_positive and cidx is not None,"low_precision_present":primary_low_precision,"closure_eligible":closure_eligible}}); return out
