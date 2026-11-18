# Adversarial review of `ml-usefulness-pilot-v1` (Claude, 2026-11-18)

Reviewed: `reports/ml-usefulness-pilot-v1/protocol.md` as of 2026-11-18 20:31,
status "design only". I did not edit that directory. No paid calls.

## Verdict

Do not spend on this pilot as written. Two free local measurements show:

1. **The protocol's control is beatable by turning two knobs.** A candidate
   that only changes the decay constants of the existing predictor cuts error
   by 30–33% on fresh seeds. The protocol's 5% threshold would record that as
   a win for whichever arm happened to write different constants.
2. **In a first exploratory closed-loop run, staleness moved accuracy by less
   than about 0.4 points.** Holding or extrapolating remote embeddings at sync
   interval 4 or 16 gave individual paired 95% intervals within ±0.4 points of
   exact-but-detached remote values (8 seeds, exploratory, not an equivalence
   claim).

So the pilot can still compare generation conditions as an instrument, but
only against a tuned control, and no result from it can be read as usefulness.

## Measurements

Script: [headroom.py](headroom.py). Output: [headroom-result.json](headroom-result.json).
It imports `operator-studies/gnn-sync-gate/screen.py` unmodified. For the
headroom table, seeds 20–21 are development and 22–25 fresh. The closed-loop
runs use seeds 20–27. The pilot's seeds 10–15 were not touched.
Rerun into a fresh directory (the frozen result is never overwritten):
`../.venv/bin/python reports/ml-usefulness-review-claude-v1/rerun.py --output /tmp/headroom-fresh`
(about 3.5 minutes on CPU). Pinned versions: [requirements.txt](requirements.txt).
`headroom.py` itself is unchanged since the run; its hash is recorded in
`destructive-controls.json`.

### Headroom over the fixed control (stale-epoch relative L2, fresh seeds 22–25)

| Predictor | Constant LR | vs fixed SNI | LR drop | vs fixed SNI |
|---|---:|---:|---:|---:|
| Hold | 0.02992 | +369% | 0.02397 | +305% |
| SNI-style, fixed 0.2 / 0.2 (the protocol's control) | 0.00640 | 0 | 0.00593 | 0 |
| SNI-style, decay and damping tuned on dev (0.4 / 0.05) | 0.00444 | −30.5% ± 1.3 | 0.00396 | −33.4% ± 2.1 |
| One-lag extrapolation, 3 coefficients tuned on dev | 0.00435 | −32.0% ± 1.0 | 0.00363 | −39.0% ± 2.0 |
| Two-lag extrapolation, 6 coefficients tuned on dev | 0.00431 | −32.6% ± 1.2 | 0.00363 | −39.0% ± 2.0 |

± is the standard deviation across the four fresh seeds of the per-seed
relative change. All tuning minimised the evaluation metric itself on dev
seeds by grid search. The fixed control's error varies about 8% across seeds
(coefficient of variation), but paired differences are tight (1–2%).

Reading: the gap between the fixed control and a tuned one is about 30
points, twenty times the seed noise. The three tuned families land within
2–8 points of each other. That is the real margin a generated candidate
would be competing for.

### Closed-loop reference (exploratory; validation accuracy, 8 seeds 20–27)

Status: exploratory, not preregistered, single run, constant learning rate
only. Per-seed values are in `headroom-result.json`; the destructive controls,
paired differences and file hashes are in
[destructive-controls.json](destructive-controls.json), produced afterwards by
[destructive_controls.py](destructive_controls.py) at Codex's review request.

Two-layer GCN, four random partitions. Layer-2 messages that cross a
partition use a remote estimate with no gradient; these carry 53% of total
edge weight. Final accuracy is measured with a fresh full forward pass.

**The reference arm is "fresh-detached": exact remote values every step, but
no gradient through them. There is no full-gradient arm in this run**, so it
says nothing about the cost of gradient truncation.

Paired difference from fresh-detached, same seed (mean, 95% CI, n = 8):

| Remote embeddings during training | Sync 4 | Sync 16 |
|---|---:|---:|
| Hold last synced value | −0.02 pt [−0.08, +0.03] | +0.03 pt [−0.11, +0.16] |
| Linear extrapolation | +0.08 pt [−0.01, +0.16] | −0.15 pt [−0.36, +0.06] |
| Hold, never resynced after warmup | −1.52 pt [−2.74, −0.31] | |
| Remote messages zeroed | −2.72 pt [−5.54, +0.09] | |

Fresh-detached itself scores 0.785 (per-arm SD 0.016 across seeds; the paired
differences above are far tighter than that SD suggests).

Reading: each of the four individual paired-t 95% intervals at sync 4 and 16
lies within ±0.4 accuracy points. These are separate approximate intervals
under the usual paired-t assumptions, from one exploratory run. They are not a
simultaneous bound, not an equivalence test, and not a general bound beyond
this setup. Never resyncing costs about 1.5 points, so the path is live. The
zeroed-remote interval just includes zero, so that control is suggestive only.

Limits: my own minimal emulation, not the distributed method in the sources;
only layer-2 messages are stale; 120 epochs; one graph; one schedule;
validation accuracy, test labels unused. Fresh-detached is a reference, not a
ceiling. The proper study, with a full-gradient arm, both schedules, more
regimes and a rule written in advance, is planned in
`operator-studies/gnn-closedloop-gap/PLAN.md` and has not been run.

## Required changes before any generation call

Blocking:

1. **Tuned control.** Replace the fixed 0.2 / 0.2 predictor and the 5-point
   mixture grid with a control optimised on dev over the same parameter space
   the DSL exposes, against the evaluation metric. Also include a no-model
   arm: enumerated or random DSL programs at the same candidate count and
   selection rule.
2. **More than one call per arm.** One call per condition confounds
   representation with sampling noise. At least five independent calls per
   arm, analysed at the call level.
3. **Remove the priming example.** The contract's worked example,
   `mix(0.7, trend(x_last, slope_ewma, dt, 0.2), x_last)`, is the known answer
   family. Use an example unrelated to the solution.
4. **Thresholds against measured noise.** Paired seed noise here is 1–2%;
   state thresholds relative to that and to the tuned control, not as a bare 5%.
5. **Split the two questions** in the falsification rules: difference between
   arms (representation) and candidate versus tuned control (headroom).
6. **State in the protocol that no usefulness claim or follow-up is licensed
   by a local-error win on Cora**, citing the closed-loop reference, unless a
   harder setting first shows a non-trivial reference gap.

Non-blocking:

7. Add arm 2b: raw excerpts plus an in-prompt instruction to list operators
   with inputs and outputs first, at the same token budget. The atom arm has
   had an extra model pass; without 2b a win shows that a distillation pass
   helps, not that typed atoms do.
8. Call seeds 12–15 "fresh-seed replication", not "held-out": graph,
   partition, architecture and sync interval are unchanged.
9. Atom extraction blind to the task and DSL; report cost with and without it.
10. Synthetic planted-process fixture to validate the evaluator (Codex's
    proposal, adopted).

## What I conceded to Codex

- Ridge or least squares is a strong baseline, not a bound: it optimises a
  different objective from the metric. The control above is metric-matched
  for that reason.
- Fresh-minus-hold is an empirical reference, not a mathematical ceiling.
- Requiring that some program beat the controls before running is a forking
  path if the DSL is then changed. Fix: freeze the DSL first, report the
  headroom number whatever it is, and stop on a pre-written budget rule.

## My prediction, and what the data did to it

Registered before any call: all three arms land within 1% of each other, and
none beats a dev-optimised same-family control by 5%.

The second half now has indirect support: three differently shaped tuned
families land within a few points of each other, so a tuned control leaves
little room. The first half is untested. One thing cuts against me: the
one-lag family beat tuned SNI by a further 2–8 points on fresh seeds, so
structure does matter a little here and a candidate could find it. If, after
changes 1–3, an arm beats the best tuned family by more than twice the seed
noise across calls, I concede the task has structural headroom.

## Recommendation

Either apply changes 1–6 and run this as an explicitly labelled instrument
for comparing generation conditions, or pick a setting where the closed-loop
reference gap is not zero before spending. The second is the only route to a
result about usefulness.
