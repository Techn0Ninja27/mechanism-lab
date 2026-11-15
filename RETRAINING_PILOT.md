# Retraining pilot — 2026-11-16

This replaces preliminary pilot outputs. Actual saved models were trained from scratch on CPU; released checkpoints remain untouched.

## Protocol

- Compact 274-token vocabulary (273 atoms plus shared BOS/EOS).
- Globally deduplicated canonical atom sets, then 80/20 split; no identical set crosses the split.
- Training atom order randomized on every draw; validation uses a fixed randomized order.
- 128-width, two-layer transformer: 433,152 parameters. Same initialization within each seed across all data/step conditions.
- Seeds42,43,44; nested data budgets500/2000/8000;100/400 optimizer steps;64 examples/step sampled with replacement; AdamW lr0.001.
- Heldout metric is per-token NLL including EOS; it differs from inference coherence which excludes EOS.
- Unigram baseline fitted on training targets only with add-one smoothing.

## Results

Mean heldout NLL across three seeds; lower is better.

| Unique training sets | 100 steps | 400 steps | Unigram |
|---:|---:|---:|---:|
| 500 | 4.657 | 5.090 | 4.696 |
| 2000 | 4.486 | 4.517 | 4.662 |
| 8000 | 4.463 | 4.456 | 4.653 |

At500 sets,400 steps worsened heldout NLL in all three seeds despite lower training loss, consistent with overfitting. At8000 sets,400 steps slightly improved heldout NLL in all three seeds. At2000 the longer run was slightly worse in all three. More data and more training are distinct interventions.

This is a small-model optimization/data pilot, not evidence of better scientific ideas, a scaling law, or future discovery forecasting. Random heldout sets are not a temporal split, and the fixed atom vocabulary was created from the full corpus. Neither scorer expansion nor availability retraining was attempted.

## Reproduce

```sh
python scripts/train_scaling_pilot.py --seed 42 --output results/retraining-corrected
python scripts/train_scaling_pilot.py --seed 43 --output results/retraining-corrected-seed43
python scripts/train_scaling_pilot.py --seed 44 --output results/retraining-corrected-seed44
```

Every condition saves model.pt and token_mapping.json; summary.json retains losses, timing and configuration; split.json records exact disjoint atom sets. `results/retraining-corrected/n8000-steps400` can be passed as `--coherence-model` for exploratory sampling, with the original availability checkpoint kept fixed. This mixes training regimes and is not a controlled model-quality comparison.

Useful next conditions to test: temporal holdout with frozen vocabulary, matched capacity/compute curves, recovery against popularity/bigram baselines, then blinded proposal quality. Merely feeding more papers into273 existing atoms cannot create new atom primitives.
