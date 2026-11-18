# Cue-balanced source review calibration

2026-11-19. Twelve packets, six supported and six with one unsupported claim,
were audited independently before 24 fresh native reviews. Key, packets,
instructions and launch schedule were hash-bound before review. The key was
outside the reviewer input directory. Launch receipts record the actual tool
returned task names, with procedural rather than cryptographic isolation.

All-supported, cue-word and absent-number baselines each scored 6/12 records.
The same fixed-interval statement is supported in SNI chunk 0002 and unsupported
in chunk 0001. One supported sentence requires two source windows.

The frozen calibration **failed**: 23/24 reviews passed mechanical validation.
Review 11a added `subclaim_id` in compound assessments, which the decoder rejects.
No raw response was repaired or regenerated. A separate diagnostic finds all
24 raw field-verdict maps agree with the key. This does not overturn the failed
format check. Production readiness remains disabled.

This is a constructed development set; it establishes neither a population
accuracy estimate nor statistically independent reviewer errors.

## Independent audit caveats

Claude independently judged the set fit to run with caveats. It contains
30 unique claim/chunk items (24 supported and six unsupported), with repeated
positive fields across paired packets. Contradictions and unknowns are untested.
Ten positive claims overlap earlier development fixtures. Key-fitted heuristics
can do better than the three predeclared baselines; chance-level results from
those three alone do not prove the absence of shortcuts.

Only the assigned packet and instructions were authorized for each native
reviewer, but the filesystem did not enforce that boundary. Keys were outside
the input directory, still readable elsewhere. Root saved launch intents and
tool-returned task names; this is procedural provenance, not access isolation.
The shuffled order happens to end with four negative packets.

Some labels require resolving paper names from section context or the title,
and reception timestamps from the buffer definition. All reviewers accepted
those claims in this run. The fixture's earlier draft status text remains
unchanged; the later freeze receipt records the independent key audit.
Evidence-ID validity is checked mechanically, but source entailment is not
proved by the decoder. A separate audit is needed to assess citation quality.

Claude's audit of the completed outputs reproduced 96/96 matching field
verdicts. All twelve judgments on negative examples named the missing qualifier.
The decisive text was within the selected evidence for 83/84 supported field
judgments; one was partial. Both reviewers combined windows for the boundary
case. Reference selections matched across slots for 41/48 fields, so the two
slots should not be treated as independent observations.

The pre-run freeze contains the planned task/input schedule. Tool-returned
launch receipts were recorded after each batch, often after reviews completed;
there were no contemporaneous per-spawn prompt receipts. Earlier wording about
receipts being saved before each spawn was inaccurate. Keys were readable in
parent/sibling directories despite the explicit instruction to read only the
assigned packet. No access-isolation guarantee is claimed. The first evaluation
was computed by an inline script after the run, not a committed pre-run scorer.

Any readiness based on parser v2 is retrospective: the optional-subclaim-ID
policy was amended after observing this run. The original frozen result remains
failed. This is disclosed instrument development, not a new review experiment.

## Current integration status

The historical disabled-state statements above describe the original strict
run. Campaign 2 now explicitly accepts the retrospective parser-v2 assessment
as its calibration prerequisite, with these limitations retained. Its verifier
recomputes the evidence and checks the external ledger anchor. See
`../generation-comparison-campaign2/readiness-record.json` and `protocol.md`.
This does not change the original failed result or imply that paid generation
has already run.
