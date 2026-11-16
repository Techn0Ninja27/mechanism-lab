# Mechanism-review instruction diagnostic

2026-11-17. **Development diagnostic, not a research-proposal quality result.**
Six root-authored analytic cases were classified with two review instructions.
The old instruction matched 4/6 reference statuses; the added specialization
instruction matched 5/6. Neither correctly labeled the fixed-kernel redundancy.
One old-arm status also contradicts its own correct derivation. These results
are not evidence of a reliable improvement in scientific judgment.

## Question and method

Can a cheap reviewer distinguish an operation that changes its stated target,
a redundant operation, and a transfer that cannot be decided from the evidence?
“Nontrivial” here means a target changes; it does not mean novel, useful or fast.

Cases include positive scaling before normalization with and without a fixed
additive constant, deterministic and stochastic covariance evolution, an
unspecified compressed representation, and a transfer outside a theorem's
stated regime. There are two reference cases per class. The root wrote analytic
labels before the calls. A separate native agent read only the case file and
agreed on all six classifications with derivations. This is same-family model
review, not independent human validation or a formal proof audit.

Both arms use Haiku 5.5, temperature zero, reasoning disabled, the same six cases
in the same order, the same classification schema and a 4,096-token output cap.
There is one batched call per arm, old first. The second arm adds specialization,
assumption-scope and uncertainty instructions. This is a compact instruction
comparison, **not** the full protocol-4-versus-5 production workflow: there is no
proposal generation, full literature packet, structured production mechanism
schema, or Opus reviewer. Labels are not included in either request.

## Results and audit

| Case | Reference | Old instruction | Added specialization |
|---|---|---|---|
| Positive scaling, no additive constant | redundant | nontrivial | redundant |
| Positive scaling, fixed additive constant | nontrivial | nontrivial | nontrivial |
| Fixed-kernel residual covariance | redundant | nontrivial | unresolved |
| Brownian covariance forcing | nontrivial | nontrivial | nontrivial |
| Unspecified compression/decompression | unresolved | unresolved | unresolved |
| Fixed-fraction theorem to fixed-degree masks | unresolved | unresolved | unresolved |

No arm falsely classified a reference nontrivial/unresolved case as redundant.
That is four exposed cases, not a reliable estimate of false rejection rates.

The old arm's first reason proves scale invariance and explicitly calls the
operation redundant, while its machine-readable status says nontrivial. Its
4/6 score is literal status agreement, not a claim that it failed to derive that
identity. Do not silently repair the stored response or train on its status.

Both arms mishandle the fixed-kernel mapping. Mapping the stated target with
A=-K determines F=0 along its evolution. Inventing a new nonzero correction
changes the target problem, not its exact solution. The new arm recognizes the
exact baseline and F=0 but treats an unspecified alternative F as unresolved.
This shows why a derivation plus a status is still not verified reasoning.
There can be ambiguity about off-trajectory extensions of F; those cannot add a
necessary correction to the specified exact target.

The new arm's second reason incorrectly suggests its denominator comparison has
an exception at zero variance: sqrt(1) and 2*sqrt(1) also differ there. Zero
variance is outside the case's nonconstant-input domain, and the output there
would be zero regardless. The error does not change this reference status.

The compression-case reasons omit an important possibility: a changed recovered
vector can still produce the same second-layer output. Both unresolved statuses
remain appropriate. Specific models and observations are needed.

## Limits and next action

These cases are deliberately small and exposed; the motivating graphon case
and earlier normalization work informed their design. No held-out set, order
counterbalancing, model replication, seed replication, significance claim, or
literature novelty evaluation was performed. The shared category definitions
already encourage uncertainty in both arms. Batched cases are not independent
samples. The native audit is additional model assistance, not scientific truth.

Keep the new fields for inspectability, but do not promote this result as a
validated ranker improvement. Preserve raw judgments and inspect status/reason
consistency before using labels. A broader evaluation must include grounded
proposals, strong controls, and independent adjudication of false rejection and
missed redundancy. No automatic training or proposal promotion follows here.

## Reproduce

From the engine root using the project's Python 3.12.14 environment:

```sh
../.venv/bin/python operator-studies/mechanism-review/run.py \
  --output results/mechanism-review-fresh
```

This freezes cases, labels, code hashes and exact prompts without a paid call.
Add `--execute` to make two calls under a $0.05 run cap and the shared $10 cap.
Previous attempts cannot be overwritten or retried automatically. There is no
random local sampling; hosted outputs may vary despite temperature zero, and
the provider model name does not guarantee immutable model weights.

`plan.json` is the original frozen plan; `result.json` preserves parsed outputs;
`receipts.json` records usage and hashes; `independent-review.json` contains the
label-blind native review. `output-audit.json` is that agent's subsequent audit;
its “no material error” judgment for case 02 is qualified by the denominator
exception error noted above. This report preserves both assessments.
The webapp lists this study as `not_independent_quality_evidence`.
Calls completed without truncation: 3,375 input and
1,884 output tokens. Provider usage cost was $0.001266705; conservative ledger
charge was **$0.0012795**, bringing the phase total to **$2.1719695 / $10**.
