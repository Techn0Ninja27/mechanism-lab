# Graph-walk codec development test

2026-11-17. This tests the known frequency-rank/LEB128 composition proposed in
[the Opus systems run](../../reports/systems-proposal-v1/README.md). It is a
bounded byte-cost and reconstruction test, not FeLoG reproduction or a novel
compression algorithm. No graph model is trained and no GPU/link/energy result
is claimed.

## Frozen question

Does the candidate save enough bytes versus both an explicitly specified FaBC
baseline and stronger packed controls to justify implementing GPU decoders?
`protocol.json` was frozen before downloading or inspecting either graph. The
root did not change its sampling sizes, seeds, thresholds or graphs after seeing
results. Both graphs and all seeds are development data, not a held-out test.

SNAP's [Amazon](https://snap.stanford.edu/data/com-Amazon.html) and
[DBLP](https://snap.stanford.edu/data/com-DBLP.html) provide undirected edge lists.
`prepare.py` verifies the advertised counts, rejects duplicate directed edges
and self-loops, maps sorted original node IDs to dense trainer IDs, and constructs
sorted-neighbor CSR. It executes no downloaded code. Compressed files are bounded
to 15 MB each and decompressed files to 100 MB. Cached source hashes must match
prior receipts. Subsequent runs verify CSR array hashes.

For each of five fixed seeds, round 1 contains 4,096 unbiased random walks of
length 80, with uniform starts and uniform neighbor transitions. The frequency
ranking from that round is frozen for four equally sized evaluation rounds.
All nodes, including unobserved nodes, are in the rank dictionary; ties use the
dense trainer ID. This is not ten walks per node, node2vec with nonuniform p/q,
FeeST active-frontier sampling, synchronization, or convergence testing.

The reference codecs restore the identical dense trainer-ID matrix. Common
original-ID/dense-ID setup is excluded for all arms because these are the IDs
consumed by the hypothetical trainer. Extra frequency rank-to-trainer mapping
is charged once per four evaluated rounds for varint. Every encoded corpus is
fully decoded and equality checked; saved payload and walk hashes permit replay.

The two FaBC selection interpretations are explicit in the protocol: greedy
marginal addition and the global last passing addition. Both use the original
uint32-tail selection equation, including when the tail representation is later
packed. The smaller observed-byte variant is a deliberately favorable baseline
sensitivity control, not a deployable selector. The mathematically inconsistent
alternative bitmap extent in the source is not treated as a valid codec.

The gate requires median varint total bytes (including one rank dictionary) to
be at most 0.85 times FaBC and 0.90 times the best packed control on **both**
graphs, with exact round trips. These are predeclared engineering gates, not
thresholds established by a paper. We also report optimistic resident-dictionary
ratios; loss against packing even with a free map cannot be fixed merely by
amortizing map storage over more rounds. This says nothing about a better profile
or a different graph/sampler.

## Reproduce

Use Python 3.12.14 and NumPy 2.5.4 (`requirements.txt`), matching the existing
project environment. From the engine root:

```sh
../.venv/bin/python operator-studies/graph-walk-codecs/prepare.py \
  --output results/graph-walk-codecs-fresh/data
../.venv/bin/python -m unittest discover \
  -s operator-studies/graph-walk-codecs -p 'test_*.py'
../.venv/bin/python operator-studies/graph-walk-codecs/run.py \
  --data results/graph-walk-codecs-fresh/data \
  --output results/graph-walk-codecs-fresh/run
```

The first command downloads only the two named graph files if absent. A rerun
can point `--data` at the original verified cache to avoid network usage. Use a
fresh run directory: existing results are never overwritten. There are no paid
model calls. The runner records protocol/code/source hashes and runtime versions.
Source sampling, codec computation and result analysis run on CPU; implementation
timings are deliberately not used as throughput claims.

## Pre-run design review

A separate native agent reviewed the protocol and runner before results. Uniform
neighbor transitions do not imply uniformly visited nodes: visitation becomes
degree-biased. Rounds are contiguous draws from one PCG64 stream per seed, not
separately seeded replicates. Profile coverage and top-two mass are reported,
since finite profiling and zero-frequency tie-breaking can hurt the candidate.
The resident dictionary scenario is an optimistic counterfactual (the lookup map
would still need to exist at the consumer); it does not demonstrate a deployable
GPU layout. Passing the byte gate would only justify considering implementation,
not prove GPU feasibility. Public URLs are not version pins; the retained gzip
archives' recorded hashes identify the actual inputs. Source receipts do not
include server HTTP metadata. This review is model-assisted, not human validation.

## Result

**Failed the byte gate on both development graphs.** Median ratios over five
seeds (lower is better; 1.0 means equal bytes to the named control):

| Dataset | Varint / FaBC, map charged | Varint / packed, map charged | Varint / packed, map resident |
|---|---:|---:|---:|
| Amazon | 0.9735 | 1.6806 | 1.2503 |
| DBLP | 0.9374 | 1.6182 | 1.2107 |

Plain fixed-width trainer-ID packing was the best packed control. All ten
profiles selected **k=0** under both FaBC rules, so their differences do not
affect this result. The four evaluated rounds have 1,310,720 positions per seed.
Median profile coverage was 38.85% of Amazon nodes and 38.42% of DBLP nodes;
median top-two probability mass was only 0.04364% and 0.02899%, respectively.
These are materially different from the source illustration's concentrated
walks, and unobserved nodes make the profile a weak global rank estimate.

All **280 encoded corpora** (10 graph/seed cases × 4 rounds × 7 codec/selection
arms) reconstructed identical trainer IDs. Six codec boundary/invalid-stream
tests passed. A fresh run using the same verified CSR cache produced
byte-identical result JSON, including walk and encoded-payload hashes. This is
same-code reproducibility, not an independent implementation check.

The measured formats have 48-byte headers. FaBC additionally stores per-sequence
hot IDs, byte-padded length-L bitmaps and uint64 tail-item offsets; varint uses
uint64 byte offsets. Packed values form one continuous bitstream with only final
byte padding. Both graphs need 19 bits per dense trainer ID. This simple control
beats the candidate even without charging its rank dictionary. The candidate's
extra table contains 4 bytes per graph node and is charged once per four rounds.
All encoded bytes include the specified framing; codec runtime, allocation cost,
and histogram/sort runtime were not benchmarked.

**Disposition:** do not implement GPU decoders for this tested configuration.
Do not infer that every rank codec, profile, graph or FeLoG workload fails.
A different configuration would need a new justified hypothesis and a fresh
protocol. This result is attached to the originating proposal in the webapp and
included as scoped feedback for future FeLoG source packs; it is not a training
label or a ban on the source paper.

Downloads totaled 8,639,898 compressed bytes. No paid model calls were made.
`results.json`, `freeze.json`, `source-receipts.json`, `protocol-freeze.json` and
`reproduction.json` retain the measurements and provenance. Raw graphs remain in
ignored local storage, not in Git.

At k=0 our specified FaBC frame still carries sequence offsets; a practical raw
fallback could remove them. That would strengthen the baseline rather than
rescue the candidate. The raw and packed arms already avoid those offsets. The
real-graph replay therefore exercises no hot-node bitmap compression; unit cases
exercise nonzero-k and all-hot reconstruction separately.
