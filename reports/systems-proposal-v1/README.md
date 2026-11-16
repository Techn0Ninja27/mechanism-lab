# Broader ML-systems proposal: graph-walk transfer coding

Subsequent [development codec replay](../../operator-studies/graph-walk-codecs/README.md)
failed against packed controls on both graphs, even with a free rank dictionary.
The record below describes the earlier generation/review stage.

2026-11-17. **A candidate for a revised byte-accounting test, not a novel codec
or measured systems improvement.** The updated Opus generator proposed replacing
FeLoG's frequent-node bitmaps plus uint32 tail with frequency-ranked LEB128
coding. It explicitly identified this as a known composition. Review identified
fixed-width packing and a packed FaBC tail as stronger missing controls.

## Evidence and run

The selected method packet contains FeLoG sections 5.2–5.4 and the GH200 energy
paper's measurement/discussion sections (III and V): 33,055 characters, 13 chunks.
They are selected full-method excerpts, not complete papers or author code.
The two downloaded HTML snapshots totaled 740,578 bytes. Equations used their
HTML TeX alttext; figures were not included. Original HTML, extracted text and
unredacted model outputs stay in local `results/systems-proposal-v1/`.

Primary sources:

- [FeLoG](https://arxiv.org/html/2606.22180v1): compression, synchronization and
  pipeline methods. Its cost equations and subsequent bitmap-length description
  leave a baseline ambiguity. The authors' runtime results were not replicated.
- [GH200 energy analysis](https://arxiv.org/html/2605.01938v1): measurements are
  hardware/workload-specific. They do not license energy claims on an Apple Mac.
- [FedAIS](https://arxiv.org/abs/2409.14655) and
  [SAT](https://arxiv.org/abs/2308.13466): primary abstracts establish relevant
  adaptive-sampling/synchronization and staleness-prediction prior work. Their
  full methods were not checked or supplied to this decoder.

Five local retrieval records were supplied, including Roaring bitmap work.
Exact-name queries returned no FedAIS/SAT records. The earlier unused plan also
revealed that Boolean-looking `OR` is tokenized as an ordinary required word by
this retrieval interface; it is not a Boolean query language. Separate plain
queries still missed those two papers. Empty retrieval is a coverage gap, not
novelty evidence. Initial unused retrieval/plan files are preserved locally.

After generation, a further primary search found
[Frequency-Ordered Tokenization for Better Text Compression](https://arxiv.org/abs/2602.22958),
which applies frequency-based ID reassignment and variable-length integers
before standard compression. The application differs, but this reinforces the
proposal's own concession that the coding ingredients are known. This source
was **not** supplied to the paid run; no exhaustive novelty assessment was done.

## Validation failure and explicit intervention

The proposal finished normally, without truncation, but its evidence list
removed TeX dollar delimiters from one cited phrase. Four other spans matched.
Strict validation rejected it, and the normal pipeline did not run a review.
The failed `decode-v2` result and original API receipt remain unchanged.

The root restored only those delimiters in a separate `review-after-relink`
artifact. All scientific claims remained unchanged. The corrected proposal
passed structural/quote checks and received one separate Opus review. This
review used the production review schema plus explicit cost-accounting prompts;
it is not a pristine automatic end-to-end success. Its status is
`reviewed_after_manual_quote_relink`. Exact spans are redacted to hashes in the
public JSON; this redacted copy is not valid generator input.

## What needs revision

1. **Mapping costs:** sorting embedding rows does not establish that sampled IDs
   already share the codec's rank mapping. Account for inverse maps, lookup,
   offsets, profile construction and refresh. If a mapping is already resident
   and shared by both baselines, demonstrate that and avoid charging it twice.
2. **Stronger controls:** compare plain fixed-width packing and FaBC with a
   packed tail, not only the original uint32 tail. Freeze a decodable FaBC
   serialization and the precise interpretation of its marginal selection rule.
   An inconsistent bitmap extent is a specification gap, not a second valid
   codec unless exact reconstruction can actually be demonstrated.
3. **Applicability:** both proposer and reviewer call low-skew graphs “excluded.”
   The supplied source does not establish that exclusion. Retain them as
   prespecified applicability checks.
4. **Measurement scope:** nominal-link break-even equations are extrapolations.
   A Mac microbenchmark can measure its own codec time and bytes, not discrete
   PCIe/NVLink speed, GH200 energy, or distributed graph-training improvement.

Under the proposal's illustrative aggregate assumptions, relative to uint32:

| Encoding | Modeled ratio |
|---|---:|
| Original FaBC | 0.4875 |
| Proposed varint, with sequence offsets | 0.4125 |
| FaBC with a three-byte tail | 0.3875 |

Thus the worked example does not beat that stronger control. These are formulas
on assumed mass bands, not observed walks, complete serialization, or a realized
monotone rank histogram. Omitted dictionary/header costs and actual rank masses
can change results. The table does not falsify the codec on arbitrary graphs.
See `illustrative-costs.json`.

The native agent separately recommended revision. Its audit is preserved, with
quotation text withheld; some strings were paraphrases rather than exact spans.
Its recommendation to remove all chosen throughput thresholds is too strong:
a prespecified design criterion is legitimate, but must not be presented as a
measured hardware fact. Both native and paid reviews remain model judgments.

**Next experiment:** freeze exact serialization and cost accounting, then replay
byte costs on development walk histograms against the stronger controls. Only a
surviving benefit justifies GPU decoder implementation and later untouched graph
validation. No new graph dataset or model was downloaded in this run. No local
codec benchmark or graph-task experiment has yet been performed.

## Artifacts, cost and reproduction

`proposal-and-review.json` retains the claims and review with evidence hashes.
`source-receipts.json`, `selection.json`, `manifest.json`, `receipts.json`, and
native audits document scope and provenance. The local webapp shows the full
result with a separate root factual audit, including the manual quote repair.

Both paid calls ended with `stop`: 41,480 input and 8,725 output tokens, including
provider-reported reasoning output. Conservative charge **$0.34042**; phase total
**$2.5123895 / $10**. No retries or reservations remain for this run.

For source reconstruction, copy `prepare.py` and `select.py` to a fresh directory
under `results/`, create its `sources/` subdirectory, and run them with the project
Python 3.12.14 environment. Compare reconstructed hashes against the saved
receipts: an arXiv version URL does not pin the HTML renderer. Build the pack with
`scripts/build_sourcepack.py`, 40,000 total characters and 3,000-character chunks.
Use the saved question/search configuration with `scripts/grounded_research.py`
in a fresh output directory; planning is offline and `--execute` spends API
credits. Hosted outputs need not reproduce. `review_relinked.py` records the
specific historical intervention and expects its original local inputs; it is
not a general automatic repair/retry tool.
