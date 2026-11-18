# Decoder replication: cost note, not an execution plan

OpenRouter's official Fable 5.2 page, checked 2026-11-19, lists standard rates
of $8 per million input tokens and $40 per million output tokens. Fast routing
is more expensive. Source: https://openrouter.ai/anthropic/claude-fable-5.2

At the current 12,288-output-token cap, 24 calls reserve $11.79648 for output
alone, before input. A later 24-call replication therefore needs its own cost
calculation from the actual frozen prompt bytes; the existing $12 campaign cap
cannot simply be reused. Cache discounts and BYOK credits must not be assumed
when reserving spend. No request was sent for this note.

A useful follow-up would hold task, selected representations, controls, seeds
and call counts fixed while changing the generator model. It would test model
sensitivity on this task, not establish general superiority. Register it and
its budget separately; leave the current Opus campaign unchanged. Reusing the
prepared representations also carries their shared-preparation limitation.

Current confirmed accounting remains $30 total, with $5.0352276 already charged.
The current campaign remains capped at $12. This note does not raise either cap
or treat relayed credit balances as permission for unbounded spending.
