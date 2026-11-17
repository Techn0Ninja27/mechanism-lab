# Full-method contract draft extraction

2026-11-17. A bounded two-source pilot completed extraction and a separate
semantic review using Opus. Both calls ended with `stop`; exact evidence and
schema checks passed. Cost: **$0.232392**. Including the earlier failed Haiku
attempt, this work cost **$0.2350711**; shared total is **$4.5191076/$20**.

| Source | Extracted operation | Semantic review |
|---|---|---|
| DMRS | LP-guided dyadic-group admission scheduling | Accept as a limited draft; acknowledges missing pseudocode/proof details |
| A_min | Lower-bound batch formation | Revise: missing prompt-size input; output/update belongs to the full loop, not just batch formation |

The two known manual source contracts were not supplied to the models. The
models chose broader operations than those manual substeps, so there is no
aligned accuracy or recall score. Opus reviewed a separate Opus extraction;
this is not independent human validation. The successful attempt also changed
instructions/validation after a failed Haiku attempt, so it does not establish
that one model is more accurate than the other.

No card was written to the curated registry or converted to a molecule.
`accept` means only the reviewing model supports the limited draft. Unknowns,
missing evidence, inaccurate types and hidden inputs remain material risks.
The pilot demonstrates the path and one useful semantic catch, not scalable
extraction accuracy or a new research mechanism.

## Pipeline

`scripts/extract_contract_drafts.py` consumes1–5 sources marked full text or
official code, at most40,000 included characters. This pilot used29,760
characters in8chunks from [DMRS](https://arxiv.org/html/2610.15579v1) and
[Chen's A_min paper](https://arxiv.org/html/2508.14544v2). Metadata alone does not
prove full-method coverage; the selected excerpts were inspected beforehand.

Extraction emits one model-proposed card or an explicit null per source, plus
unknowns. Exact source/chunk hashes and spans are checked. A separate review
checks port meanings, hidden inputs, output exposure and scope. Model drafts
use their own interpretation marker and fail the curated registry's default
validator. Unknown typed values must be JSON null, not concrete strings such
as `unknown`. No automatic repair, curation or molecule creation occurs.

Both stages use the shared locked client and the same **$1 run ceiling** under
the **$20 phase cap**. Failed attempts remain intact; execution refuses previous
calls/results. The default extractor remains Haiku; `--extractor opus` selects
the stronger bounded variant. Abstract notes remain a separate data type.

Offline planning (a fresh output path):

```sh
../.venv/bin/python scripts/extract_contract_drafts.py --pack results/full-contract-extraction-v1/pack.json --output /tmp/full-contract-plan --extractor opus
```

Adding `--execute` makes paid calls; unit tests use mocked responses. The
[source-linked draft export](result.json), [plan](plan.json) and [receipts](call-receipts.json)
are public. Verbatim evidence spans stay in the local original; public quote
hashes are not an exact-span replay fixture. The webapp displays both the failed
attempt and these reviewed drafts separately from executable operators.

The next meaningful test is a preselected, unseen full-method packet with
explicit operation scopes and independent field-level adjudication. More
papers should not be typed automatically from abstract similarity alone.
