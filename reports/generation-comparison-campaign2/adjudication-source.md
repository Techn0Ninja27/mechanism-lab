# Adjudication of the six flagged fields in the stopped campaign-2 draw

Date: 2026-11-19. Author: Claude, from an independent agent's read-only
adjudication. This does not reopen the stopped run. Each field was read
against the full text of the chunks its own record cites (the injected first
line of `sni-method:chunk:0001` ignored). Quote and hash validation was not
re-run.

| Record, field | Atom's value | Reviewer a / b | Adjudication | Class |
|---|---|---|---|---|
| 1 conditions[0] | "each node has a GPU and a SmartNIC" | unsupported / unsupported | Unsupported by the cited chunks 0002 and 0003. The fact is stated in the uncited chunk 0001 | Real atom error |
| 1 unknowns[0] | "value of s used" | unsupported / supported | The unknown is correct: the cited chunks give only the symbol s. Reviewer a's own reason argues the value is absent; its verdict label is inverted | Reviewer false flag |
| 2 unknowns[0] | "the sampling_ratio used in the experiments" | unsupported / supported | The unknown is correct: 1/3 appears only in a figure's toy example, not as an experimental setting | Reviewer false flag |
| 3 unknowns[0] | "full proofs (only proof sketches are given)" | supported / unsupported | Full proofs are absent; the parenthetical is a true positive statement. The rule does not say whether an unknown may carry one | Rule ambiguity |
| 5 outputs[0] | "the trained GNN" | unsupported / supported | Supported: "an online algorithm to train the embedding predictor and the distributed GNN" | Reviewer false flag |
| 8 outputs[0] | "channel-wise historical embedding choices" | unsupported / unsupported | Unsupported, borderline: the chunk describes channel-wise partial historical embedding and an evaluator, not emitted choices | Real atom error, minor |

**Tally.** Two real atom errors (both where the reviewers agreed), three
reviewer false flags (all where the reviewers disagreed), one rule ambiguity.
Ten of twelve atoms are clean on support by this reading.

**Unflagged fields.** No clear reviewer miss was found. Every number, name and
qualifier checked is in the cited chunks. Three soft spots: self-contradictory
wording in one statement, one mildly inferred input, one unknown that
presupposes thresholds exist.

**The malformed review** (slot b, record 10) was substantively right on its
unknown field; the defect was purely the missing key.

**Reading.** The main problem in this draw was reviewer noise concentrated on
`unknowns`, made worse by an underspecified rule for how an unknown may be
phrased and scoped. Atom quality was secondary: one fact taken from an
uncited chunk and one over-inferred output. Where both reviewers agreed, the
flag was real; where only one flagged, it was not.
