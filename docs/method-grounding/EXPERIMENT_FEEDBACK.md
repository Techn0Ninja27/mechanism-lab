# Scoped memory of completed experiments

2026-11-16. Grounded proposal generation now reads the curated
`experiment-feedback.json` registry before paid execution. The first record is
the [FIER → WindowQuant trace pilot](../../operator-studies/trace-window-ablation/README.md).
Its 1-bit ranking failed a fixed-budget fidelity gate against stronger controls.

The purpose is to expose previous evidence when the engine revisits the same
source combination. This does not retrain a model, prove improved proposal
quality, establish novelty, or ban the whole pair of papers. A new experiment
may test a materially different operator, data regime or systems tradeoff.

`generation/experiment_feedback.py` accepts a record only when every source URL
matches the source pack after canonicalizing supported HTTPS arXiv abs/pdf
URLs. It then verifies report/result paths beneath `operator-studies/`, rejects
traversal and symlink escapes, and hashes the bounded result file against the
registry. Matching a digest verifies artifact identity, not scientific validity.
Registry fields must explicitly disallow training eligibility and novelty
promotion. Invalid or conflicting records are quarantined with bounded warnings.
Unrelated valid records remain usable.

The registry is limited to 100 records; at most five enter a prompt, with bounded
text fields, warnings and 20 MB result reads. Modern arXiv identifiers are the
currently supported matching scheme; this is deliberately partial coverage.
The pipeline additionally bounds the registry file to 1 MB.

The grounded runner supplies accepted feedback as citable text chunks inside
the existing untrusted-evidence boundary. The proposal must describe its
difference from a matching local experiment in `prior_art_difference`. Both the
proposal and critical reviewer receive the scoped result. Exact quote validation
applies to feedback chunks as it does to supplied papers. Frozen run plans bind
the registry, resolved feedback, reader code and other dependencies. Changed
evidence cannot silently alter an already planned paid run.

Verification includes matching source pairs, version normalization, malformed
URLs, tampered hashes, path/symlink escapes, conflicting duplicate IDs, malformed
records, bounded warnings, prompt inclusion and citation validation, and refusal
to execute a paid plan after feedback changes. An offline plan using the actual
FIER/WindowQuant source pack matched the saved pilot record without warnings.
No paid generation was used to validate this integration. Its effect on future
proposal quality remains to be measured.

2026-11-17 update: a real generation/review run confused ranking inputs with
evaluation inputs, proposed an already-tested two-bit arm, and inferred equal
measurements from rounded summaries. The feedback now explicitly records
prefill-only selection, already-tested arms, more precise values and the fact
that exact attention mass is not a theoretical fidelity ceiling. The max-
aggregation follow-up is a second scoped record. These corrections do not
establish that future reviews are reliable; see the post-run factual audit.
