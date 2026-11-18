# End-to-end representation comparison (draft)

Date: 2026-11-19. This is a design memo for a new campaign. It does not reopen
the stopped Campaign 2 directory and makes no API calls.

## Why this design

Campaign 2 answered only whether a preparation draw could pass a strict review
gate. It stopped after one atom draw cost $0.106524: 23 of 24 record reviews
were mechanically valid, and one unknown field was marked supported without the
required `absence_reason`. Seven of the 23 valid reviews also flagged six
fields as unsupported. The raw atom output, reviews, hashes and stop marker are
preserved. This is a contract failure, not evidence that atoms are useful or
useless.

The next useful question is end-to-end: when the preparation output is used as
it was produced, does it help a generator produce candidates that execute and
beat a fixed control? A fidelity gate that halts before candidate generation cannot answer that
question. Campaign 2 preserved its failures; it did not discard them silently. The next analysis should retain the status of every attempted preparation
output, including malformed or unsupported records.

## Frozen comparison

Use a new canonical output directory and a new phase-ledger chain. The stopped
directory remains immutable and is never copied into a selected-preparation
slot. The four generator conditions are task-only, raw excerpts, typed atoms,
and prose distillation. Hold the task, source packet, source header, model,
temperature, output limit, candidate grammar, six calls per condition, four
candidates per call, development seeds 100–115, final seeds 200–231, controls,
selection code and decision thresholds fixed before any new call.

The existing 12 atom records may be reused only as an explicitly labeled
intention-to-treat atom artifact, with their original hashes and failed review
record attached. They must not be repaired, filtered, or presented as a
faithful source representation. If a matched prose artifact is generated, it
is frozen independently under the same source and task contract. A fairer
follow-up would generate both representations before candidate generation and
include their preparation failures in the same denominator.

This is initially a **fixed-artifact, as-produced pilot**, not a population
estimate of an end-to-end pipeline. There is one shared preparation artifact
per prepared arm, and the atom artifact is reused after a failed gate. Record
that selection history explicitly. Six calls do not create six independent
preparation replicates. A larger end-to-end study would need independent
preparation draws assigned prospectively.

The generation unit remains the call, not its four candidates. For each
condition, report all six calls, including calls whose preparation input is
malformed, candidates fail parsing, or execution produces non-finite results.
No call is removed because its representation looks poor. A failed call gets a
predeclared score of failure for the end-to-end metric and retains its raw
artifacts and reason.

## What is measured

The primary outcome is the paired candidate-minus-frozen-control score on the
exact-forward accuracy window after development selection, with the existing
failure bounds (accuracy 0 and 1) and no survivor-only averages. Secondary
outcomes are candidate-minus-hold, own-forward accuracy, endpoint accuracy,
candidate parse/failure rate, and preparation support rate.

Report two estimands together:

1. **Intention-to-treat:** every requested call in an arm, with preparation and
   generation failures counted as failures. This describes this fixed-artifact pipeline under its recorded failures;
   it does not estimate variation across preparation draws.
2. **As-prepared descriptive analysis:** calls are grouped by the recorded
   preparation status and support counts, without selecting the best subgroup.
   This describes how preparation quality relates to execution; it is not a
   causal effect of fidelity.

The representation comparison is positive only if the predeclared rule is met
on the intention-to-treat estimand: at least three distinct positive source-arm
calls, no more than one task-only positive, no positive random/control call,
and the paired precision rule. A result driven by dropping failed preparation
calls is explicitly not a representation result. A task-only lead, a control
positive, or a null caused by widespread preparation failure is reported as
such. No novelty claim follows from this benchmark.

## Minimal implementation changes

Create a new runner that shares Campaign 2's frozen evaluator and inventory
schema but does not call `select_pair` as a prerequisite for generation. It
should:

- freeze the four arm inputs and their hashes before generation;
- write one immutable preparation-status record per arm and call;
- pass raw, atom and prose inputs to the generator even when support review
  reports failures, while marking those failures in the call metadata;
- retain invalid candidates and assign failure outcomes rather than dropping
  them;
- use the existing development-only picker and final-seed validator;
- never execute or modify the stopped Campaign 2 directory; copy its atom
  artifact as immutable data into the new input bundle with its original hash;
- bind the task, representations, controls, code, prompts, seeds and phase
  ledger to the new plan; and
- produce one report containing both estimands and all denominators.

The old strict gate remains valuable as a separate fidelity diagnostic. It must
not be bypassed inside the old runner, and this experiment must not relabel its
stopped result as a pass. The new runner is a distinct, predeclared experiment
with its own budget and provenance chain.

## Cheapest falsification

Before paying for six calls per arm, run a free harness with the already
generated atom artifact and synthetic raw/prose fixtures. Verify that a failed
preparation status reaches the generator, remains in the denominator, and is
reported as an end-to-end failure. Verify that a malformed candidate cannot be
silently removed and that the decision classifier counts the call. If this
harness cannot preserve failures, the design is not ready for API spending.

The user's goal is operational usefulness, so this question is relevant. It cannot establish that atomization itself is better than
raw text when the atom and prose preparations differ in quality; it can show
whether the complete pipelines differ under their declared costs and failure
rates. A cleaner representation effect requires a second study with matched,
successful preparation artifacts, which should be planned separately after
this intention-to-treat result.

## Before implementation

Freeze mechanical failure handling explicitly. Unsupported but schema-valid
records remain unmodified input data. Mechanically unusable preparation should
produce an unavailable-arm outcome and retained planned-slot denominators, not
be silently repaired or rendered by ad hoc rules. No-eligible calls must remain
incomplete for representation attribution; failure bounds remain 0 and 1.
The fidelity audit is descriptive and cannot select which artifact is used.
One shared artifact per arm precludes estimating a fidelity/performance relation.

No budget has been allocated or spent for this draft. Existing phase accounting
is $5.1417516 of $30. Any execution requires a separate prospectively frozen
protocol and reservation calculation within that remaining allowance.
