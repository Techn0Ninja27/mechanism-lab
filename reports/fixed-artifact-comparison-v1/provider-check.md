# Provider and token-budget check

2026-11-19. Read-only documentation and API metadata queries; no completion
request or account-setting change. The public snapshot is
[provider-metadata.json](provider-metadata.json).

The OpenRouter models and endpoint APIs advertise `low` effort, mandatory
reasoning, `max_tokens`, and a 128,000-token maximum completion for both decoders.
Standard Anthropic endpoint prices are $4/$20 per million input/output tokens
for Opus 5.5 and $8/$40 for Fable 5.2; their fast endpoints cost more and are not
part of the proposed configuration. Anthropic independently lists Fable's
price, output limit, always-on adaptive thinking, and requirement to omit sampling
parameters in its [model documentation](https://platform.claude.com/docs/en/models/fable-5-2/overview).

The requested 12,288-token cap covers reasoning and visible output together.
Equal caps therefore do not guarantee equal visible answer capacity. Low effort
is a supported setting, not a guarantee against truncation. This check establishes
advertised compatibility, not successful access by this account; only an eventual
receipt can establish actual provider/model/routing behavior.
[OpenRouter reasoning documentation](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens).

OpenRouter charges its credit-purchase platform fee when buying credits, not on
each inference request. No credit purchase is part of this campaign. BYOK has a
separate 5% fee after the workspace's free allowance; current workspace-wide
allowance was not established. Reserve that full 5% on all new inference, including
preparation, without treating promotional credits as a budget exemption.
[Pricing](https://openrouter.ai/pricing),
[BYOK documentation](https://openrouter.ai/docs/guides/overview/auth/byok).

The resulting conservative new reservation is $24.8005002 and the phase maximum
is $29.9422518. The prospective campaign subcap becomes $24.85, inside the unchanged
$30 phase cap. This preflight revision happens before any call in this campaign;
no generation result, retry or new spending counter is involved. The earlier
$24.50 subcap and standard-price calculation remain in Git history.

Provider ordering can prefer an Anthropic BYOK endpoint but does not itself
prove the account has a matching usable key or forbid shared capacity. A receipt
must report actual BYOK status. Do not claim that pinning `anthropic` alone
spends the user's promotional credits. No account fallback setting was changed.
