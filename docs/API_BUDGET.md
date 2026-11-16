# Shared paid-call budget

`generation.api_budget.BudgetLedger` is a run-local, cross-process ledger for a hard API ceiling. Workers reserve a conservative maximum before sending a request. Pending and unknown calls remain charged. Only a verified provider receipt can replace a reservation with actual usage; failed or ambiguous calls must not be released automatically.

```python
from generation.api_budget import BudgetLedger, request_id
ledger = BudgetLedger("results/run/api-budget.json", cap_usd=10.0)
rid = request_id({"model": model, "messages": messages, "max_tokens": 4096})
ledger.reserve(rid, reserve_usd, {"model": model})
try:
    response = call_provider()
except Exception:
    ledger.mark_unknown(rid)
    raise
else:
    receipt = response["usage"]["cost_details"]["upstream_inference_cost"]
    ledger.reconcile(rid, receipt, verified=True, receipt_id=response.get("id"))
```

The file and its `.lock` sibling contain no API key. A changed reservation or changed reconciled amount for an existing request is rejected. A ledger cap cannot be changed by reopening it with another value. The current client in `scripts/world_search.py` can adopt this API at its existing reservation point; this module intentionally does not modify that historical runner.

Client integration uses provider price ceilings in USD per million tokens, verified against [OpenRouter official routing documentation](https://github.com/OpenRouterTeam/docs/blob/main/guides/routing/provider-selection.mdx) on 2026-11-16. Opus 5.5 uses a $4 prompt/$20 completion ceiling and a conservative UTF-8 byte input bound. Both top-level billed cost and upstream cost are inspected, and the larger is charged. BYOK is not assumed or excluded from this phase ledger.
