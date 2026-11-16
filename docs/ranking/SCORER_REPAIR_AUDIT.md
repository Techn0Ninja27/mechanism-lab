# Scorer repair audit (read-only, 2026-11-16)

## Decision

Do **not** present `results/retraining-corrected/*` as a scorer for
source-consistent/original concepts. It is a valid small coherence language
model for the expanded assignment corpus, but its token IDs are local labels
whose observed co-occurrence semantics come from `clusters_80.json`. The
original `clusters.json` has the same 273 numeric keys and the same copied
cluster names, but its units are assigned differently. Numeric-ID matching is
therefore not a semantic repair.

The cheapest defensible repair is to rebuild only the coherence side from
`clusters.json` (same 273-token architecture and pilot script), then verify a
small seeded checkpoint and run the existing availability model only after its
input lineage is resolved. There is no evidence in this tree for a
source-consistent availability checkpoint.

## What the retrained pilot actually trained

`upstream/scripts/train_scaling_pilot.py` (`sha256 d952db7cb742beacfe009cefebe5c7b920e31a833c5b56d01173039631e65b59`) reads
`data/coherence_dataset/coherence_{train,val}.jsonl`, not either cluster file
directly. Those JSONL files are generated from the expanded corpus: metadata
records `clustering_eval/reassigned/clusters_0.80.json`, and the dataset has
136,827 train rows / 15,203 validation rows over 15,203 papers. Each row is a
random atom permutation with BOS/EOS; the loader deduplicates the tokenized
sets and then compacts IDs. The resulting pilot mapping is an identity map
`cluster 0..272 -> token 0..272`, vocabulary 274 including the shared
BOS/EOS ID 50256 (mapping hash
`3db43c55b7ca89b28814f4817c12525e648ade028bf65413dfe983b5c6fc865d`;
checkpoint copy hash
`2e86436c9835f0136795707de111a53131590b83a098e0e2d0b43e7f52a13fe2d`).

The seed-42 corrected run has 14,587 unique atom sets, an 11,670/2,917
disjoint set split (`split.json` hash
`ec05e491df49acbb693395c9c5c06177f11efd4b57df5f9a537ced7693c854c`), and 433,152 parameters. For the largest condition,
`n8000-steps400`, the model hash is
`e23a92ddd915639d855170968c36fb645f0e6a0f4c2baff78ff4370343fed85c`; the reported held-out NLL is 4.4582008082. This is a
co-occurrence scorer for expanded sets, not a quality or temporal model.

The older `results/retraining-pilot/summary.json` is also a pilot summary
(vocabulary 274; 14,587 canonical sets), not evidence of original-label
training. It has no checkpoint metadata that changes this lineage.

## Assignment/source mismatch (measured)

I compared the exact key `(paper_id, text)` in the two 82,255-unit assignment
files:

| quantity | value |
|---|---:|
| units in each file / exact shared keys | 82,255 / 82,255 |
| equal numeric assignment labels | 10,628 (12.9208%) |
| any noise/non-noise transition | 47,302 |
| original non-noise -> expanded noise | 6,039 |
| original noise -> expanded non-noise | 41,263 |
| original non-noise assignments | 30,580 |
| expanded non-noise assignments | 65,804 |

Both files contain 273 cluster keys and their `name` strings are bytewise
identical by numeric key (name-list hash
`2dbdb1ff6c72072a839737993e3cede67bb9387728a98c83204a6bfaea9ac667`). That is copied description text, not proof that the
supporting units retained their meaning. For example, exact unit membership
intersection/Jaccard for numeric IDs 0, 1, 2, 100, and 200 is respectively
0/0.000, 4/0.005, 0/0.000, 0/0.000, and 0/0.000. The expanded corpus has
15,203 papers with at least two assigned atoms and 14,587 unique sets; the
original corpus has 6,541 such papers and only 4,503 unique sets.

`upstream/PROVENANCE.md` (hash `b893cdd2fae94d2d5267a4bbd1705411d66e64df7736e0324f07e48ad99b3a83`) records the same release
audit: all 15,203 coherence papers exactly match expanded assignments and none
exactly matches original assignments; a majority remap is unsafe because
expanded clusters mix many original labels. The mismatch is thus at the
assignment/source-unit layer, not a missing token offset.

## Availability lineage and joins

`generation/make_availability.py` defaults to `papers/clusters_80.json`, the
same expanded 273-token mapping. The database itself is available and
reproducible (`data/llm-papers.db`, 43,839,488 bytes, SHA-256
`16d53fdaa176a6bcaedf9ba3c46808c9b2b31885720c0aa777219dceece0afb1`). Its tables contain 16,068 papers, 46,412 authors, and
96,060 paper-author links.

Recomputing the author joins from the two assignment files with the code's
slash/`__` paper-ID aliases gives:

| source assignments | mapped papers | author profiles | profiles with >=2 atoms | >=3 atoms | unresolved papers |
|---|---:|---:|---:|---:|---:|
| original `clusters.json` | 13,419 | 41,691 | 27,768 | 15,951 | 0 |
| expanded `clusters_80.json` | 16,020 | 46,329 | 44,809 | 39,794 | 0 |

The released availability model is visibly the expanded regime: its mapping
is the same 274-vocabulary shape (hash `e6e8208466749758cf3f8aca0ffb0e4f8e540343d5c88aa4b43c1907bf5b261a`), its author index has
39,791 entries (hash `ad4b12e283e60e94db3cee5b1d2cbece76c42241731b09173a8f680f58feb14d`), and `run_config.json` records
`vocab_size:274`, 512-wide two-tower training, seed 42 and dataset hash
`a701c0b511adae786e7895e954bece8eae427a6054ad64cc9c35865419054529`. The release documentation reports 39,791 training authors.
No original-label availability dataset, author embedding bank, or matching
dataset hash is present under `data/`; the DB and generator are sufficient to
recreate one, but the expensive full run is not needed for the first audit.

## Recommended cheap reconstruction

1. Build `coherence_original.jsonl` directly from `clusters.json` using the
   existing `generation/datasets/coherence.py` protocol (10 permutations,
   seed 42, min 2 atoms), and record source hashes, row/paper counts, and the
   generated mapping. Do not reuse expanded JSONL rows or the existing split.
2. Run the existing pilot with that corpus at the smallest falsifying grid
   (for example 500/2,000 sets × 100 steps, one seed), saving the new mapping,
   split hash, and NLL against an original-corpus unigram baseline. This
   produces an explicitly source-consistent coherence scorer without paid
   calls.
3. Before availability retraining, run `profiles_from_sources` against
   original assignments and the DB, and persist the exact profile count and
   token-mapping hash. A tiny two-tower smoke test can check serialization and
   join correctness; it cannot certify ranking quality.
4. Only if coherence and availability are both rebuilt from the same original
   assignment snapshot should fused rankings be called repaired. Until then,
   use original assignments/descriptions for unranked source evidence and keep
   the existing models explicitly labeled expanded-lineage scorers.

This path preserves the original 273-unit vocabulary and author links; it does
not claim that the original clustering itself is scientifically correct.
