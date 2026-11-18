# Field-level cited-chunk calibration

This packet was built on 2026-11-19 from the closed v5 preparation draw. It is
for offline calibration of the campaign-2 field-review rubric only. The records
must not be used as campaign-2 generation inputs.

`inputs.json` contains the 24 original records plus four planted negatives. It
contains no verdict labels. The source pack is referenced by path and SHA-256;
the original full chunks remain in the private local source-pack artifact.
`answer-key.json` keeps the labels separate.

The frozen calibration check is deliberately narrow: a reviewer should flag all
four planted bad records—two explicit contradictions and two unsupported
numeric/interface claims—and should not flag the two named, directly supported
positive checks (`fresh-2` and `fresh-2-results`). Verdicts on the other 24
records are descriptive and are not treated as ground truth.

The packet is a rubric check, not a reviewer accuracy estimate. It contains
known negatives by construction and cannot establish population false-flag or
miss rates. The closed v5 attempt remains closed and is not reopened by this
fixture.

## Recorded outcome

Both reviewers detected all four planted negatives and kept the two known
positive records supported. Both nevertheless marked the original `sni-1`
fixed-interval condition supported, even though its cited chunk does not state
that condition. Each assessed all 28 records and all 133 field values; the input
and instruction hashes match the calibration freeze.

The fixed pass mark required detecting that real coverage failure as well as
the planted negatives. Both reviewers therefore failed calibration. The runner
now blocks paid campaign execution on this failed calibration. This is evidence
of correlated reviewer mistakes on one case, not a population error estimate.
Reviewer B used `fields` instead of `field_items`; the scoring adapter records
that syntax difference and preserves both original outputs.
