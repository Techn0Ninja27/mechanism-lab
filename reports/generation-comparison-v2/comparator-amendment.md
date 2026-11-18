# Stronger comparator amendment, 2026-11-18

This amendment is written after the development-only control screen, before
paid preparation, generation or final evaluation. The original stage-0 protocol
and code remain unchanged so its frozen hashes stay verifiable.

The primary comparison for every generated candidate is now against the fixed
mean-sync control in `control-selection.json`. It was selected by maximum mean
development accuracy among the four predeclared controls. Evaluate that exact
program on seeds 200–231, alongside each development-selected candidate. Do not
reselect the control after final results. Candidate-minus-hold remains secondary
and cannot qualify a source condition for replication.

The allocation rule remains an improvement above 0.5 percentage points and
above two paired standard errors, with zero failures and SE at most 0.5 points,
now applied to candidate-minus-control. The distinct-program, task-only and
random-control checks apply to this stronger comparison. All comparisons remain
descriptive; three generation calls per condition are not a significance test.

Keep the 32-seed allocation from the prewritten G1-minus-hold variance rule.
Do not recompute a more favourable allocation after seeing development means.
That rule gives no power guarantee for candidate-minus-mean-sync. Small gains
may remain unresolved. Report actual paired uncertainty and do not call an
uncertain result evidence that atomization cannot help.

The development gap between G1 and mean-sync is about 0.58 points. This is a
reference difference, not a ceiling: detachment, stale-state smoothing and
optimization can change downstream accuracy in either direction. Do not claim
that a candidate must close a fixed fraction of that gap to be useful.

All generation conditions are told the primary comparator averages the last two
observed states, without its measured performance or development results. This
intentionally supplies a strong baseline to improve upon; it is identical across
conditions. The random sampler description is part of the common grammar, not
evidence of its performance. Machine-readable `closure_eligible` requires a
complete, adequately precise primary comparison with no positives; failure or
low precision blocks closure.

The comparator disclosed to the model was chosen by the stage-0 development
screen. The task statement is therefore informed by that screen even though it
contains no measured performance figures. Naming the two-observation average
steers every arm toward the damping/averaging neighbourhood. The question is
whether a source representation helps improve a disclosed strong baseline,
not whether it helps discover that averaging can outperform holding.
