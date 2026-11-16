# Claim-only replay of the existing graph-codec proposal

2026-11-17. One cheap model call examined three **manually decomposed** claims
from the already exposed codec proposal against one primary abstract:
[frequency-ordered tokenization](https://arxiv.org/abs/2602.22958v1). This is a
posthoc integration diagnostic, not a new proposal or a novelty benchmark.
The evaluator did not receive the graph-codec experiment outcome.

| Claim | Model coverage |
|---|---|
| Apply rank/varint coding to graph-walk transfer instead of FaBC | not_found |
| Achieve at most 0.85 times FaBC bytes on new graphs | unknown |
| Rank frequent symbols low and use variable-length integers | covered |

The aggregate status is `central_evidence_unresolved`; the novelty estimate is
null and training/novelty promotion remains false. The reviewer recognized the
known ingredient and did not infer graph performance from text compression.

The application label is debatable. Its explanation acknowledges shared coding
ingredients; `partial` would also be defensible. Domain relabeling must not turn
an unchanged known mechanism into confident novelty. A manually assigned
supporting role also does not independently establish which contribution is
central. This single response provides no accuracy or calibration estimate.

Separately, the actual [graph-walk experiment](../../operator-studies/graph-walk-codecs/README.md)
already failed against packed controls. That negative outcome is not reversed
by a literature no-match label. The app presents a separately labeled claim-only
replay, its root audit, and the original experiment link. The original proposal
and review files were not changed.

`profile.json` preserves the response with exact source quotes replaced by hashes.
`plan.json`, `receipt.json` and the historical pilot script document the call.
Full input/response artifacts are local under `results/claim-assessment-pilot-v1`.
To repeat, copy `run.py` to a fresh directory under `results/` with the original
primary packet available. It makes one paid call under a $0.03 cap and the shared
$10 cap, refusing an existing attempt. The script uses the current validator;
for exact historical software replay use this recorded repository revision.
Hosted output is not deterministic.

The call completed without truncation: 1,067 input and 806 output tokens.
Conservative charge: **$0.0005097**. Phase total: **$2.5158393 / $10**.
