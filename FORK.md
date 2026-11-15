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
