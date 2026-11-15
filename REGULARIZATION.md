# Regularization ablation — 2026-11-16

This diagnostic tests the small-corpus overfit observed in the retraining pilot. It compares the released architecture family at tiny scale with three conditions: baseline dropout 0 / weight decay .01, dropout .2 / weight decay .01, and dropout 0 / weight decay .1. Each condition uses the same fixed sampled minibatches within a seed and data size. Validation selects the checkpoint; the held-out test split is evaluated once afterward.

The split is by globally deduplicated canonical atom sets, not by permutation rows. It is random, not temporal, and uses the fixed 273-atom vocabulary. It therefore measures interpolation to unseen combinations within the same corpus, not future research prediction.

Run:

```sh
python scripts/regularization_ablation.py
```

Results and checkpoints are in `results/regularization-ablation/`. Compare `best_val_nll` to `test_nll`; a condition that only improves training loss or validation while failing test is not an improvement. The test set is not used for selection.

No result should be treated as evidence of better generated research ideas. The next useful test after this ablation is a temporal split with the atom vocabulary frozen before the test period.

## Result

Three seeds (42, 43, 44), 400 maximum steps, and the same fixed minibatch stream within each seed/data-size condition were run on CPU. Mean scores:

| training sets | condition | best validation NLL | test NLL | mean selected step |
|---:|---|---:|---:|---:|
| 500 | baseline | 4.628 | 4.643 | 100 |
| 500 | dropout 0.2 | **4.590** | **4.609** | 100 |
| 500 | weight decay 0.1 | 4.625 | 4.640 | 100 |
| 2,000 | baseline | 4.476 | 4.486 | 100 |
| 2,000 | dropout 0.2 | **4.473** | **4.484** | 200 |
| 2,000 | weight decay 0.1 | 4.475 | 4.486 | 100 |

Dropout was the only tested intervention with a consistent, though small, test improvement. The effect is clearer at 500 sets than 2,000. Stronger weight decay did not materially improve the baseline. This is a tiny model and random canonical split; it justifies testing dropout in a larger or temporal pilot, not changing the released checkpoint.

### Reproduction caveat
The table above predates the fixes that save the initial checkpoint and derive maximum sequence length from data (formerly hardcoded16). The full18-run matrix has not been rerun after these changes; changed initialization may change the exact values. Treat the table as historical pilot evidence, not verified output of the latest script.
