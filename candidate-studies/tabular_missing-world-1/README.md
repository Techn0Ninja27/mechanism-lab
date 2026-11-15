# tabular_missing-world-1: reconstruction-regularized missingness-aware classifier

## Proposal and verdict

The candidate appends a 20-bit missingness mask to each example, randomly masks standardized features during training at per-row rates `p = 0.5 * Beta(2,1)`, and trains a shared MLP encoder with a binary classification head plus a reconstruction head. The auxiliary MSE is calculated only on values artificially masked during that minibatch, with weight 0.2. The reconstruction head is not used to score examples. This matches the proposal's causal mechanism: reconstruction regularization plus high-severity-skewed masking and explicit missingness indicators.

**Verdict: inconclusive for the preregistered performance threshold, and no novelty established.** Held-out mean AP was 0.841534 for the candidate and 0.838100 for the same MLP and masking schedule without reconstruction (paired mean +0.003434; preregistered threshold +0.005). The candidate won two seeds and lost one. Uniform severity sampling did better on the held-out aggregate (0.845857). The explicit-mask ablation did better in one seed and worse in two, so its contribution is also unstable. All deep models beat the median-imputation HGB baseline on mean aggregate AP in this synthetic test, but the candidate-vs-neural-control contrast is the causal comparison for the auxiliary loss.

## Prior art checked

A direct primary precedent is the **Denoising Fully Connected Network (DFCN)**. It jointly trains a classification network and a decoder, applies reconstruction regularization while training the classifier, randomly masks inputs, and calculates reconstruction error only on known values hidden for training. The authors compare the auxiliary-loss architecture with a classifier without reconstruction and report robustness to missing inputs. This is the same central mechanism, so joint masked reconstruction as a classification regularizer is not a new mechanism. The candidate's specific combination of a tabular synthetic task, explicit mask concatenation, and Beta-skewed severity schedule remains a variant to test, not an established novelty claim. [Primary DFCN study](https://pmc.ncbi.nlm.nih.gov/articles/PMC8323880/).

A second nearby primary precedent is **MFAN: Learning on Missing Tabular Data: Attention with Self-Supervision, Not Imputation, Is All You Need**. Its abstract describes auxiliary self-supervised tasks that randomly hide observed values and encourage inference of hidden features, together with explicit treatment of missingness. This is further evidence that masked feature reconstruction and missingness-aware representation learning in tabular classification are established directions. [MFAN paper DOI](https://doi.org/10.1145/3729241).

The proposal correctly identifies denoising autoencoders and masked self-supervision as precedent. The precise Beta(2,1) schedule is an implementation choice rather than a demonstrated new relationship. On this benchmark, uniform corruption performed better than the Beta schedule.

## Design and deviations

- Data: `make_classification(5000, 20, 5 informative, 5 redundant, weights=[.8,.2], class_sep=1)`, dataset seed equal to split seed.
- Seeds: 137, 271, 389. Stratified 60/20/20 split; validation and test partition uses `seed+1`.
- Standardization uses training rows only. All methods receive identical pre-generated MCAR draws for clean, 10%, 30%, and 50% evaluation.
- Candidate and primary control have identical MLP structure (40 input, hidden widths 64/32, binary classifier); only the reconstruction loss differs. The no-mask ablation has 20 inputs, as required to remove the mask signal.
- Additional controls: uniform corruption `p~U[0,.5]`, no explicit mask input, and train-median imputation plus indicators with HGB (`max_iter=120`).
- Fixed candidate hyperparameters: 30 epochs, Adam `lr=.002`, weight decay `1e-4`, batch size 128, reconstruction weight .2. No tuning was performed.
- The original proposal allowed a gradient-boosted or neural classifier. A neural classifier was necessary for the shared representation/reconstruction head; HGB is retained as the requested baseline. The HGB fit and prediction were run with the benchmark environment, while the PyTorch MLP used the main environment.
- Evaluation corruption is MCAR; it does not model MNAR mechanisms. Feature standardization makes zero the training mean, while the explicit mask disambiguates missing from observed mean-valued inputs.

## Reproduction

From `experiments/alien-science/upstream`, prepare the validation fixtures using benchmark sklearn, then train the network using main-environment PyTorch:

```sh
../.venv-benchmark/bin/python scripts/with_compute_slot.py -- ../.venv-benchmark/bin/python candidate-studies/tabular_missing-world-1/experiment.py prepare validation
../.venv-benchmark/bin/python scripts/with_compute_slot.py -- ../.venv/bin/python candidate-studies/tabular_missing-world-1/experiment.py train validation
```

After validation, `preregistration.json` froze the source SHA-256, seeds, split hashes, primary metric, control, and decision threshold before any test split was scored. Final files were then generated once:

```sh
../.venv-benchmark/bin/python scripts/with_compute_slot.py -- ../.venv-benchmark/bin/python candidate-studies/tabular_missing-world-1/experiment.py prepare final
../.venv-benchmark/bin/python scripts/with_compute_slot.py -- ../.venv/bin/python candidate-studies/tabular_missing-world-1/experiment.py train final
```

The final `results.json` contains per-seed, per-condition AP and training durations for neural methods. The HGB fit time and neural inference time were not measured. Do not rerun final mode over the frozen artifacts; the current script does not enforce a separate overwrite flag, which is a reproducibility limitation.

## Results

The primary mean AP averages the four conditions equally. Per-seed values and per-condition APs are in `validation.json` and `results.json`.

| Seed | Reconstruction + Beta | No reconstruction + Beta | Paired Δ | Uniform + reconstruction | No mask + reconstruction | Median HGB |
|---:|---:|---:|---:|---:|---:|---:|
| 137 | 0.745259 | 0.740145 | +0.005114 | 0.750637 | 0.752663 | 0.715993 |
| 271 | 0.889143 | 0.882861 | +0.006282 | 0.889223 | 0.875344 | 0.854558 |
| 389 | 0.890200 | 0.891295 | −0.001095 | 0.897710 | 0.867627 | 0.785138 |
| **Mean** | **0.841534** | **0.838100** | **+0.003434** | **0.845857** | **0.831878** | **0.785230** |

Candidate MLP train time was 0.170–0.176 seconds per seed, versus 0.113–0.130 seconds for the no-reconstruction network in this environment. These are small-run local timings and not portable performance claims.

Validation candidate-minus-control deltas were +0.001843, +0.004239, and −0.001090 by seed; the held-out result is directionally similar but does not meet the preregistered +0.005 mean threshold. Clean-condition candidate AP did not fall by more than 0.01 versus the neural control on any seed. Uniform scheduling slightly beat the candidate in all three held-out seeds, so the Beta schedule has no evidence of benefit here.

## Smallest falsification test and limitations

The smallest direct falsifier is the paired comparison already run: on fixed data and corruption draws, compare the candidate to the same encoder, augmentation and mask input with reconstruction weight set to zero. The candidate missed its predeclared +0.005 mean AP threshold. Three seeds are too few to estimate a stable effect, and the synthetic distribution does not establish behavior on real tabular data or non-MCAR missingness. Results are exploratory replications, not statistical proof. Novelty is not established and the central mechanism has close direct prior art.
