# Generation audit limitations

2026-11-19. The independent generation audit reproduced all 192 inventory
labels, 49 receipt bindings (including preparation), model assignments,
reported costs, and the two 124-row evaluation exports. All generation calls
finished normally and reported BYOK routing. This validates bookkeeping and
mechanical classification, not scientific usefulness.

The prompt says “depth 6” without defining whether depth counts levels or
edges. The frozen compiler counts the root as level 1. Of 23 depth rejects,
19 have seven levels, equivalent to six edges, with no other violation:
17 Opus programs and two Fable programs. They remain invalid in this run.
This ambiguity limits interpretation of the decoder validity difference;
no alternative-depth results enter candidate selection or final evaluation.

Three Opus task-only calls fail JSON parsing and one Opus raw-source call has
no valid program. Those four no-eligible calls make the Opus condition
incomplete under the frozen decision rule. Individual final-seed effects
will still be reported, but cannot repair the missing calls or support a
complete cross-decoder attribution.

Mechanical validity per arm (out of 24 slots):

| Decoder | Task only | Raw | Atoms | Prose |
|---|---:|---:|---:|---:|
| Opus 5.5 | 7 | 12 | 14 | 18 |
| Fable 5.2 | 23 | 19 | 23 | 23 |

The 139 mechanically valid slots contain 115 distinct program hashes.
Hash uniqueness does not establish semantic uniqueness or novelty. The
audit found averaging language in 135 valid programs; the task explicitly
disclosed the averaging comparator. This is a test of improvement on that
named baseline, not independent discovery of averaging.

Per-call mechanical counts are in `generation-per-call-validity.json`.
Development eligibility is recorded separately after all development seeds
finish. Report both beside final picks: the number of choices per call
differs, and format compliance must not be mistaken for research utility.

Audit source: `reports/generation-comparison-review-claude-v1/fixed-artifact-generation-audit.md`.
Original programs, receipts, compiler, prompts and selection rules remain unchanged.
