# Request-client implementation status

2026-11-19. `generation/fixed_artifact_client.py` is a transport-injected
component, not a command that can contact a provider. It has no credential
lookup or default HTTP implementation and is now used by the audited campaign lifecycle through an explicit adapter.

Each request uses its model's fixed prices, the original system prompt,
12,288 tokens, low reasoning, no sampling parameters, and an Anthropic-only
provider policy with a price ceiling. All extra payload fields are rejected.
Reservations use UTF-8 input bytes plus framing, the full output cap, and 5%
fee contingency. Successful calls retain their complete reservation; actual
usage is stored separately. A reported cost exceeding the reservation stops
the campaign and is recorded upward rather than concealed.

Tests cover two different models sharing one receipt directory, reuse of a
completed bound receipt without transport, interrupted transport, deleted stop
markers, changed cached requests, missing receipts, fabricated cache entries,
invalid provider/model/usage data, reservation denial, phase-cap denial, and
parity with the planner's reservation. All tests use temporary ledgers and fake
transports. The focused client suite has 15 tests; the combined lifecycle and analysis
suite has 100 tests. Completed receipt checksums are anchored in the phase ledger,
so changing response text or usage without changing that anchor is detected.

This is ordinary failure detection, not cryptographic proof of provider origin.
A coordinated rewrite of local receipts and the ledger is outside the boundary.
Interrupted requests are terminal rather than automatically resent. The caller
still must enforce the single canonical campaign directory, immutable execution
plan, request order, artifact bindings, inventory preservation, and final
analysis. That orchestration is implemented and covered by fake-provider lifecycle tests;
no paid call has been made through this client.
