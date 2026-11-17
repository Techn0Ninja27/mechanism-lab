"""Reproducible unranked baseline from source-backed atom IDs.

No scorer values, previously scored pools, or semantic quality claims enter this
selector. It is a comparison baseline while the learned ranking is unvalidated.
"""
import math
import random


def select_seed_triplets(catalogue, count=6, *, seed=45, excluded=()):
    if type(count) is not int or not 0 <= count <= 100:
        raise ValueError('count must be an integer from 0 to 100')
    if type(seed) is not int:
        raise ValueError('seed must be an integer')
    clusters, assignments = catalogue['clusters'], catalogue['atom_assignments']
    valid = set()
    for record in assignments:
        cid = record.get('cluster')
        if (type(cid) is int and cid >= 0 and str(cid) in clusters and
                isinstance(record.get('paper_id'),str) and record['paper_id'].strip() and
                isinstance(record.get('text'),str) and record['text'].strip()):
            valid.add(cid)
    ids = sorted(valid)
    excluded = {tuple(sorted(x)) for x in excluded}
    # Enumerate combination ranks, not millions of materialized triples.
    n = len(ids); total = math.comb(n,3) if n >= 3 else 0
    def unrank(rank):
        for i in range(n-2):
            size = math.comb(n-i-1,2)
            if rank < size: break
            rank -= size
        for j in range(i+1,n-1):
            size = n-j-1
            if rank < size: return (ids[i],ids[j],ids[j+1+rank])
            rank -= size
        raise ValueError('Invalid combination rank')
    # A lazy Fisher-Yates permutation gives no repeats and bounded memory for
    # ordinary small selections. Exclusions never alter the initial draw order.
    rng=random.Random(seed); swaps={}; remaining=total; selected=[]
    while remaining and len(selected)<count:
        index=rng.randrange(remaining); rank=swaps.get(index,index)
        remaining-=1; swaps[index]=swaps.get(remaining,remaining); swaps.pop(remaining,None)
        triple=unrank(rank)
        if triple not in excluded: selected.append(list(triple))
    return {'method':'uniform_source_backed_triplets','seed':seed,'eligible_atoms':len(ids),
            'candidate_combinations':total,'requested':count,'returned':len(selected),
            'ranking_claim_allowed':False,'triplets':selected}
