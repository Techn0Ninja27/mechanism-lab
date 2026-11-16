# Local search improvements

Fork of [Alien Space of Science](https://github.com/alejandrohdez00/alien-space-of-science), preserving the authors' MIT license and attribution. Based on commit `a55d660de75c99afe9e0948781cdb828fe3798eb`.

This branch improves inference infrastructure. It does not claim improved scientific idea quality or reproduction of the paper's backtest.

## Batched generation and Apple GPU

```sh
python alien.py sample --device mps --generation-batch-size 128 \
  --batch-size 128 --n-gen 10000 --betas 0 0.4 0.7 \
  --top-k 30 --output-dir results/mps-search
```

`--generation-batch-size 1` preserves the original serial sampler. The new batched path uses the same per-token temperature, EOS suppression and top-k rule. Batched execution changes random-number consumption and can change floating-point results; a seed alone does not guarantee identical candidates across batch sizes or devices. Record both settings. The original `auto` policy is unchanged (CUDA if available, otherwise CPU); select `mps` explicitly.

## Compare changes on a fixed pool

Every run now saves **all candidates and their raw scores**, not only the winning shortlist, to `k3/scored_candidates.json`. Re-score that pool on another backend:

```sh
python alien.py sample \
  --candidate-pool results/mps-search/k3/scored_candidates.json \
  --device cpu --batch-size 128 --betas 0 0.4 0.7 \
  --output-dir results/cpu-same-pool
```

This validates IDs/set sizes, canonicalizes the sets, recomputes scores, and normalizes against the same full pool. Do not compare fusion scores across different pools as if they shared a scale. Reusing a pool controls candidates, not the model weights: retain model/token mappings and source revision alongside results.

## Validation

```sh
python -m unittest discover -s tests -v
```

Tests cover batch-one equivalence to serial token sampling, partial batches, seeded repeatability, EOS exclusion/top-k behavior, invalid batch sizes, and ranking extremes. Actual released-checkpoint tests and timing observations are in `LOCAL_BENCHMARK.md`.

## Research progression

1. Establish same-pool numerical equivalence and throughput (this branch).
2. Compare matched-budget candidate sources: autoregressive, uniform and a mixture, keeping scorers and decoder fixed.
3. Measure portfolio diversity separately from blind quality ratings and prior-art overlap.
4. Only then change atom extraction or scoring objectives. Updated corpora need new model training and temporally isolated evaluation.

No generated proposal should be labeled novel merely because it scores highly. Reusing authors' precomputed checkpoints is inference reproduction, not training reproduction.

## Candidate-source experiment

Two optional sources now supplement the original autoregressive generator:

```sh
python alien.py sample --device mps --batch-size 128 --generation-batch-size 128 \
  --candidate-source hybrid --autoregressive-fraction 0.5 --n-gen 10000 \
  --betas 0 0.4 0.7 --top-k 30 --output-dir results/hybrid-10000
python alien.py sample --device mps --batch-size 128 \
  --candidate-source uniform --n-gen 10000 --betas 0 0.4 0.7 \
  --top-k 30 --output-dir results/uniform-10000
python scripts/compare_search_sources.py --runs results/mps-10000 \
  results/hybrid-10000 results/uniform-10000
```

Each source attempts 10,000 draws; duplicate/repeated-atom losses are visible in accounting. Equal draws are not equal compute or equal unique-set budgets. The comparison script uses a shared union's normalization to avoid attributing normalization changes to candidate quality. It reports diversity and absence from the released paper atom sets, neither of which establishes scientific novelty.

On this one-seed experiment at beta 0.7/top30, the autoregressive pool had 18 combinations absent from corpus papers, hybrid 25, uniform 30. Uniform also had lower mean coherence. These are descriptive search tradeoffs, not proof that random sampling produces better science. Future evaluation must match costs, use several seeds and assess quality blind to method.

## Evidence and evaluation extensions

- `alien.py rerank` applies optional coherence quantiles and pair-diversity limits to a saved pool. Normalization is computed before filtering; these guards do not establish novelty.
- `generation/evidence_review.py` constructs bounded literature-comparison prompts and validates source IDs and exact abstract quotes. Model output cannot mark itself human-adjudicated or training-eligible.
- `generation/temporal_data.py` constructs publication-year partitions and strict-new triple counts. The released vocabulary includes future information, so this is a fixed-vocabulary diagnostic. See TEMPORAL_EVALUATION.md.
- See DATASET_ROADMAP.md for a source-backed dataset plan and REGULARIZATION.md for the small-model pilot and its reproduction caveat.

Source assignment provenance must be validated before interpreting scorer IDs or presenting canonical descriptions as evidence-grounded. Numeric-ID agreement alone is insufficient across clustering artifacts.

## Executable composition and ranking results — 2026-11-16

The typed operator registry now supports bounded composition and local round-trip checks in the workbench. A 115-chain compression search produced a small positive synthetic result; a separately trained utility ranker failed its unseen-family evaluation and is not promoted. See the [results overview](README.md#results-including-what-failed), [utility report](reports/utility-ranker/README.md), and [scorer provenance limits](reports/source-coherence/README.md). The three-axis ranking scaffold separates novelty evidence, feasibility, and comparable task-specific utility.
