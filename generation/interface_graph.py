"""Bounded, evidence-only interface matching for recovered atoms.

The scores are lexical proposal heuristics.  An edge is not a claim that the
two methods compose, are novel, or are experimentally compatible.
"""
from __future__ import annotations
import hashlib, json, math, random, re
from collections import Counter
from pathlib import Path

TOKEN = re.compile(r"[a-z0-9]+")

def _tokens(s): return TOKEN.findall(str(s or "").casefold())

def _vecs(texts):
    docs = [Counter(_tokens(x)) for x in texts]
    df = Counter(t for d in docs for t in d)
    n = len(docs)
    out=[]
    for d in docs:
        z=sum(d.values()) or 1
        out.append({t:(c/z)*math.log((n+1)/(df[t]+1)) for t,c in d.items()})
    return out

def _cos(a,b):
    den=math.sqrt(sum(x*x for x in a.values())*sum(x*x for x in b.values()))
    return sum(a.get(k,0)*v for k,v in b.items())/den if den else 0.0

def _sid(a):
    raw=f"{a['source_id']}:{a['statement_sha256']}".encode()
    return "atom-"+hashlib.sha256(raw).hexdigest()[:16]

def load_atoms(path, review_path, papers_path):
    atoms=[json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    papers={x["source_id"]:x for x in json.loads(Path(papers_path).read_text())}
    review=json.loads(Path(review_path).read_text())
    records=list(review.get("atoms",[]))+list(review.get("recovered_batch2_atoms",[]))
    bykey={(x.get("source_id"),x.get("source_hash"),x.get("statement_sha256")):x for x in records}
    rejected={(x.get("source_id"),x.get("source_hash"),x.get("statement_sha256")) for x in records if x.get("flag_type")=="semantic"}
    clean=[]; rejected_missing=0
    for a in atoms:
        if not isinstance(a.get("source_id"),str) or not isinstance(a.get("source_hash"),str) or not isinstance(a.get("statement"),str):
            rejected_missing+=1; continue
        required=(a.get("source_id"),a.get("source_hash"),a.get("statement"),a.get("kind"),a.get("inputs"),a.get("outputs"))
        if (not all(required) or a.get("kind") not in {"method","finding","limitation"} or
            not isinstance(a["inputs"],list) or not isinstance(a["outputs"],list) or
            any(not isinstance(v,str) or not v.strip() for v in a["inputs"]+a["outputs"])):
            rejected_missing+=1; continue
        h=hashlib.sha256(a["statement"].encode()).hexdigest()
        key=(a["source_id"],a.get("source_hash"),h)
        matches=[x for x in records if x.get("source_id")==a["source_id"] and x.get("source_hash")==a.get("source_hash") and x.get("statement_sha256")==h]
        if any(x.get("flag_type") not in {"none","semantic","relevance_only"} for x in matches): rejected_missing+=1; continue
        if (not matches or any(x.get("reviewed") is not True for x in matches) or
            len({(x.get("verdict"),x.get("flag_type")) for x in matches})>1): rejected_missing+=1; continue
        a=dict(a, statement_sha256=h, stable_atom_id=_sid(dict(a,statement_sha256=h)),
               corpus_group=papers.get(a["source_id"],{}).get("corpus_group"),
               source_url=papers.get(a["source_id"],{}).get("url"))
        if key in rejected: continue
        if a["corpus_group"] not in {"ml","systems","math"}: rejected_missing+=1; continue
        clean.append(a)
    return clean, review, {"semantic_rejections":len(rejected),"missing_or_ambiguous_rejections":rejected_missing}

def build(atoms, review, rejection_counts, *, seed=73031, limit=20):
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 20:
        raise ValueError("limit must be an integer in [1, 20]")
    # Any group may supply a method or finding producer. Consumers
    # are methods in a different paper and different corpus group.
    producers=[a for a in atoms if a["kind"] in {"method","finding"}]
    targets=[a for a in atoms if a["kind"]=="method"]
    pairs=[]
    for p in producers:
        for t in targets:
            if p["source_id"]==t["source_id"] or p["corpus_group"]==t["corpus_group"]: continue
            if "ml" not in {p["corpus_group"],t["corpus_group"]}: continue
            if not any(_tokens(x) for x in p["outputs"]) or not any(_tokens(x) for x in t["inputs"]): continue
            pairs.append((p,t))
    texts=[" ".join(a["outputs"]+a["inputs"]+[a["statement"]]) for a in atoms]
    # IDF is corpus-wide and deterministic; pair vectors use the same corpus.
    all_tokens=[Counter(_tokens(x)) for x in texts]; df=Counter(t for d in all_tokens for t in d); n=len(all_tokens)
    def v(s):
        d=Counter(_tokens(s)); z=sum(d.values()) or 1
        return {t:c/z*math.log((n+1)/(df[t]+1)) for t,c in d.items()}
    scored=[]
    for p,t in pairs:
        iv=_cos(v(" ".join(p["outputs"])),v(" ".join(t["inputs"])))
        sv=_cos(v(p["statement"]),v(t["statement"]))
        scored.append({"producer":p,"target":t,"interface_tfidf_cosine":iv,
                       "statement_tfidf_cosine":sv})
    scored.sort(key=lambda x:(-x["interface_tfidf_cosine"],-x["statement_tfidf_cosine"],x["producer"]["stable_atom_id"],x["target"]["stable_atom_id"]))
    rng=random.Random(seed); random_short=rng.sample(scored,min(limit,len(scored))) if scored else []
    def render(x):
        p,t=x["producer"],x["target"]
        return {"producer_atom_id":p["stable_atom_id"],"target_atom_id":t["stable_atom_id"],
          "producer_source":p["source_id"],"target_source":t["source_id"],
          "producer_group":p["corpus_group"],"target_group":t["corpus_group"],
          "producer_url":p.get("source_url"),"target_url":t.get("source_url"),
          "producer_statement":p["statement"],"target_statement":t["statement"],
          "producer_assumptions":p.get("assumptions",[]),"target_assumptions":t.get("assumptions",[]),
          "producer_outputs":p["outputs"],"target_inputs":t["inputs"],
          "interface_tfidf_cosine":round(x["interface_tfidf_cosine"],8),
          "statement_tfidf_cosine":round(x["statement_tfidf_cosine"],8),
          "source_refs":{"producer":p["source_hash"],"target":t["source_hash"]},
          "unresolved_compatibility":["type, scale, training/inference, and runtime assumptions are unverified"],
          "status":"textual proposal heuristic; not a typed proof or molecule promotion"}
    return {"candidates":[render(x) for x in scored[:limit]],
            "seeded_random_shortlist":[render(x) for x in random_short],
            "candidate_pool_size":len(scored),"rejections":rejection_counts,
            "assumptions":["different source papers and corpus groups required","one endpoint is ML and the other is non-ML; target is always a method","semantic review flags excluded by exact source_id+source_hash+computed statement_sha256","relevance_only review flags retained","no novelty, compatibility, or usefulness claim"],
            "seed":seed,"limit":limit}
