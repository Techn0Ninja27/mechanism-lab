# Authorized total API cap: $20

2026-11-17. The user explicitly raised the shared phase cap from $10 to $20. A file-locked migration preserved all 36 reconciled reservations, amounts and provider receipt IDs. No pending or unknown calls existed. Charged total at migration: **$3.2803365**; remaining allowance **$16.7196635**. This is a total phase cap, not $20 additional.

The live ledger and immutable pre-migration backup remain local. `audit.json` records before/after hashes and the authorization. The active grounded pipeline now requests the $20 ledger; its per-run $2 ceiling is unchanged. Historical plans and reports retain their original $10 cap. Old-cap clients still fail closed rather than silently expanding their budget. Normal reservation calls cannot change a cap.
