# Jev / typed-decision routing note

**Verified 2026-11-16.** This is a routing primitive, not a truth oracle. The
useful question for upstream is whether a cheap typed gate can decide which
expensive operation is worth running, or whether an evidence item is relevant
enough to send onward. It should not decide mathematical novelty, factual
truth, or whether a proposed mechanism is publishable.

## What Jev actually exposes

OpenRouter exposes Jev through `POST https://openrouter.ai/api/alpha/decisions`
and also documents a System One-compatible endpoint for the official SDK. The
request accepts a `state` (string, object, or array), named `questions`, and a
model. Questions are:

* `noul`: probability that a yes/no statement is true;
* `choice`: distribution over caller-defined named options, returning the
  highest-probability option and its confidence;
* `score`: distribution over an ordered list of rubric levels and an expected
  score; and, in the current schema, `box` for image localization.

Multiple questions share one state and one request. The response contains
typed answers, probabilities/confidence, model name, and input/output token
usage. A minimal request is:

```json
{"model":"jev-1.14","state":{"task":"...","evidence":"..."},
 "questions":{"route":{"type":"choice","instructions":"Which next stage?",
 "criteria":{"cheap_check":"bounded/local check","strong_review":"send to Opus",
 "defer":"insufficient evidence; do not guess"}}}}
```

Pin a version for an experiment (`jev-1.14`, released 2026-10-14), rather than
the moving `jev-latest` alias. TypeSafe's architecture, weights, data, and
calibration procedure are undisclosed. “Zero hallucinations” and “calibrated
probabilities” on the marketing site mean typed output and a confidence field,
not proof of domain accuracy or calibration on this corpus.

Sources: [OpenRouter Jev documentation hub](https://openrouter.ai/docs/guides/community/jev),
[OpenRouter Jev tutorial](https://openrouter.ai/docs/guides/community/jev-tutorial),
[OpenRouter Jev 1.14 model page](https://openrouter.ai/typesafe/jev-1.14-20261014).

## Access and cost (no call made here)

An existing OpenRouter key is the least-friction path and does not require a
separate TypeSafe account. Use `POST https://openrouter.ai/api/alpha/decisions`, with model
`typesafe/jev-1.14` (or `~typesafe/jev-latest`). OpenRouter currently lists
$0.042 per million input tokens and $0 output tokens for Jev 1.14. The state
is charged on every request, so a long evidence packet can dominate the
nominally tiny price. Do not use `jev-router` for this pilot: it is a paid
router whose serving model and price vary. No key was read and no paid request
was made for this note. If the existing OpenRouter key is unavailable or
restricted from the Decisions API, obtain a separate key/permission before
testing.

Sources: [OpenRouter Jev documentation hub](https://openrouter.ai/docs/guides/community/jev),
[OpenRouter Jev tutorial](https://openrouter.ai/docs/guides/community/jev-tutorial),
[OpenRouter model pricing](https://openrouter.ai/typesafe/jev-1.14-20261014).

## Evidence and limits

The benchmark by Deußer, Ramamurthy, Sparrenberg & Sifa (arXiv:2609.37647, v2)
evaluated Jev 1.13 on 37 datasets/346,009 requests: strong results on several
standard classification datasets, but degradation on low-resource languages,
fine-grained/noisy labels, and rubric judgments. Binary probabilities ranked
well but were poorly placed at a fixed 0.5 threshold; tuning thresholds on
training data materially changed F1. This supports selective routing only after
local threshold fitting, not the vendor's universal calibration claim.

The evidence audit (Tang & Zheng, arXiv:2609.32160) concludes that typed
readouts have not independently demonstrated an accuracy advantage over
label-probability readouts; the clearest gains are latency and cost. The newer
controlled study (Rastogi, arXiv:2611.15592) finds that, under one shared
protocol, readout complexity barely affects accuracy (81.3–82.0% on Qwen3-0.6B),
while backbone scale matters (Qwen3-0.6B to 4B: +4 in-domain, +11 held-out).
This makes Jev interesting as a productized low-latency endpoint, not evidence
that a proprietary “System One” mechanism is intrinsically superior.

A newer audit (Delaunay, arXiv:2611.12521) is relevant to the pilot but does
not directly evaluate hosted Jev: it evaluates five open-weight System-1-style
models against generative baselines. It reports that confidence can be a
dispersion transform rather than empirical accuracy, that bundling questions
can cause cross-question interference in small models, and that larger open
models can be better calibrated under syntactic shifts. Treat these as reasons
to log each question separately and test multi-question bundling, not as claims
about Jev's internals.

Sources: [benchmark](https://arxiv.org/abs/2609.37647), [evidence audit](https://arxiv.org/abs/2609.32160),
[open-model audit](https://arxiv.org/abs/2611.12521), [controlled readout study](https://arxiv.org/abs/2611.15592).

## Open alternatives

For a no-paid-call control, **Nimble** is a local typed scorer (Qwen3.5-9B,
text-only, flat boolean/enum schema; logits over one-token answer codes), with
weights/code and a reproducible Ollama path. **Clef/Clef Flash** are Cloudflare's
open decision models; Clef is multimodal but substantially larger, and can be
served locally or through Workers AI. The **Decision 1.0–3.0** family from vLLM
Semantic Router supplies open readout variants. The controlled study suggests
that a minimal cosine/letter head is a viable baseline; compare backbone and
data, not just head names.

Sources: [Nimble repository](https://github.com/bespokelabsai/nimble),
[Ollama decision capability](https://github.com/ollama/ollama/blob/main/docs/capabilities/decision.mdx),
[Clef model metadata](https://github.com/cloudflare/cloudflare-docs/blob/production/src/content/workers-ai-models/clef.json),
[Small Heads study's taxonomy](https://arxiv.org/abs/2611.15592).

## Cheapest 10–30-case pilot

Question: *does a typed gate reduce expensive/repeated work while preserving
operator-review recall?* The gate's allowed outputs should be exactly
`cheap_local_check`, `send_strong_review`, and `defer_insufficient_evidence`.

1. Construct 10–30 cases from existing known operator failures (loops,
   irrelevant evidence, malformed/overconfident proposals) plus genuinely
   unknown cases. Freeze case IDs and the raw state/evidence packet before any
   model call. Keep the gold label, if one exists, in a separate file never sent
   to the model; for unknowns record “unresolved” rather than manufacturing a
   label.
2. Ask one Jev request per case with two questions: the three-way `choice` above
   and a `noul` safety question (“Is there enough explicit evidence to act
   without strong review?”). Include a neutral `defer` option and concise
   criteria; do not include prior outcome, case name, or “known failure” text.
   Log raw probabilities, selected option, token usage, latency, version, and
   request hash. Do not let Jev rewrite evidence or produce the action.
3. Replay thresholds offline. For each confidence threshold, report coverage,
   error among autonomous cases, recall of known-failure cases sent to review,
   defer rate, latency, tokens/case, and estimated cost. Compare against (a)
   always-send-strong, (b) fixed heuristic, and (c) a local Nimble/Clef/ordinary
   classifier if available. Use bootstrap intervals; with 10–30 cases this is a
   falsification probe, not a performance estimate.
4. Kill the routing idea if any high-confidence autonomous case is an unsafe
   known failure, if `defer` is not used on unknowns, if confidence is not
   monotonic with correctness on the frozen labels, or if tokenizing the evidence
   packet erases the cost/latency advantage. A pass only licenses a larger,
   independently labelled study.

The appropriate first integration is therefore a **shadow-mode relevance and
escalation scorer** before Opus or a costly search/test. Preserve the full
evidence ledger and let deterministic code enforce every threshold and action.
Jev may prioritize attention; it must not be the project's novelty or truth
arbiter.
