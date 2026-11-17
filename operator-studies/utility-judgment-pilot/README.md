# Can a cheap judge predict which method helps?

2026-11-17. **This exposed retrospective probe does not support using unassisted
model confidence as a usefulness score.** It is not a benchmark of general model
accuracy, novel research, or a trained ranking system.

We selected three method pairs from already completed local experiments: graph
ID compression, visual KV-window precision assignment and stale GNN embedding
prediction. Each model request received the task, matched metric, methods and
explicitly missing evidence. It did not receive measured values, winner labels,
experiment verdicts or subsequent reasoning corrections. Each pair was presented
in both A/B orders. Outcomes and input files were hashed before execution; the
runner opened outcomes only after persisting all predictions.

Selection was informed by known outcomes. These are familiar development
examples, not a held-out or prospective test. A hosted model may also know the
underlying methods. Six calls represent only three paired cases, not six
independent experiments. No source retrieval or tools were available to this
judge. It was allowed to abstain and to leave its probability unknown.

## Results

| Pair | First presentation | Reversed presentation | Measured winner |
|---|---|---|---|
| Fixed-width packing / rank-varint | Abstain | Rank-varint, 0.80 | Fixed-width packing |
| Exact softmax mass / one-bit mean logits | Abstain | Exact mass, 0.60 | Exact mass |
| Predictor / adaptive Fixed Share mixture | Mixture, 0.65 | Abstain | Predictor |

All six responses completed normally without truncation. Three abstained. Of
three non-abstaining responses, one matched the observed narrow-metric winner.
All three pairs changed between abstention and a preference when reordered;
none produced two opposing non-abstaining choices. This is presentation
sensitivity in this run, not a proven causal estimate of position bias: hosted
inference need not be deterministic even with temperature zero.

Conditional mean Brier error over the three numeric answers was 0.4075. An
uninformative 0.5 on those same three outcomes yields 0.25, but the tiny,
selectively answered sample does not support population calibration claims.
Abstentions were not silently filled with 0.5 or counted as incorrect answers.

## Reasoning audit

The wrong codec preference assumed typical graph degree skew would be enough.
For 1,310,720 evaluation IDs and a uint32 dictionary over roughly 317k–335k nodes,
beating 19-bit packing requires mean varint payload below about **1.35–1.41 bytes
per ID**, before differences in framing. Saying that many IDs take two bytes or
less is not sufficient. The response's own approximate arithmetic could have
made this uncertainty much clearer. The input did not supply an actual histogram.

The wrong GNN preference said the adaptive mixture generalizes the predictor and
that convex mixing bounds the downside. Convexity does not guarantee lower
error than the better fixed expert. With nonzero share and finite feedback,
the gate cannot simply choose exactly weight one at every step. These are
mechanistic objections available before seeing the observed failure.

Neither correction proves another judge would predict the outcome correctly.
The paired responses often acknowledged missing empirical evidence; abstention
can be appropriate. The conclusion is narrower: these uncalibrated probabilities
should not populate the production ranker's expected-gain or confidence fields.
The next evaluation should supply independently checked cost/observability
constraints and compare stronger reviewers on prospectively fixed cases.
Do not tune on these cases and then call them held-out validation.

## Artifacts and reproduction

`cases.json` is exactly the outcome-free input set. `labels.json` binds the local
experiment artifacts by hash; those labels concern only the specified metrics.
`predictions.json`, `evaluation.json`, `summary.json`, `plan.json` and
`receipts.json` preserve the six calls and outcome comparison. Raw requests and
responses remain under local `results/utility-judgment-pilot-v1/calls/`.

```sh
python -m unittest discover -s operator-studies/utility-judgment-pilot -p 'test_*.py'
python operator-studies/utility-judgment-pilot/run.py --output results/new-utility-probe
# Paid execution of that unchanged plan:
python operator-studies/utility-judgment-pilot/run.py --output results/new-utility-probe --execute
```

The runner refuses a directory containing an existing attempt, has no automatic
retries, and enforces a $0.04 local cap plus the shared $10 phase cap. Hosted
responses are not deterministic. Two tests verify probability validation,
abstention handling and reversed-order label mapping. An initial test-import
name collision was fixed without changing the judging run or its prompts.

Haiku 5.5 used **3,252 input + 2,648 output tokens**. Conservative cost:
**$0.0016492**. Phase total: **$2.8189165 / $10**. No corpus/model downloads,
training, real-world experiments, or novelty claims were added by this probe.
