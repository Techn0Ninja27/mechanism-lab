# Execution lifecycle

2026-11-19. Implemented and tested with fake transports. No new paid preparation,
generation, or Cora training has run under this campaign.

The offline plan is immutable and grants no execution authority. A separate
initial freeze binds its code, runtime, model schedule and limits before
preparation. Both paid library stages and the command entry point require an
independent audit receipt matching that freeze. The command previews by default;
credentials are loaded only after audit checks and explicit `--execute`.

Preparation has one request. Its raw response and validated prose are stored
under `prepared/`, leaving the base plan untouched. An invalid prose artifact
stops the whole comparison with all 192 model slots recorded as unattempted.
Source-only packets are exported for 24 separate native review contexts.
Launch intents are written before spawning; completion records are separate.
Their timestamps and hashes document the procedure, not proof of independence.
Missing or malformed review output remains missing fidelity measurement.

Generation freezes the actual four prompts and the fidelity table before
requesting candidates. It checks the full remaining reservation, current caps,
and frozen inputs before each request. A full response containing invalid
candidate JSON contributes four invalid attempted slots and the schedule
continues. A provider error or truncation stops further spending. Completed
receipts reconstruct checkpoints without resending requests; receipts are
anchored by checksum in the phase ledger. A crash before that anchor can leave
an unverifiable response, which fails closed with its reservation retained.

Each model contributes 96 slots. The 28 shared controls give two 124-row local
inventories compatible with the existing evaluator. Export verifies candidate
rows against the provider receipts. The analysis adapter keeps decoder results
separate, retains missing calls, and attaches fidelity coverage and flags to
positive prepared-arm picks. These tests establish software behavior, not a
usefulness or novelty result.

From the repository root:

```sh
../.venv/bin/python -m scripts.fixed_artifact_campaign plan
../.venv/bin/python -m scripts.fixed_artifact_campaign freeze
../.venv/bin/python -m scripts.fixed_artifact_campaign prepare
```

The final command is a preview. After the independent audit has been recorded,
`prepare --execute` performs the single preparation. The root agent then uses
`generation.fixed_artifact_review_launch` to record and collect native reviews.
`finalize` binds the actual prompts and measurements. `generate --execute` runs
the audited schedule; `export` produces evaluator inventories. No command
raises the phase cap or resets an attempt.

The local evaluator remains `scripts.run_causal_campaign`; development uses
seeds 100–115, selection is frozen before final seeds 200–231, and the frozen
mean-of-two control is required for reporting. Runtime versions are captured
in the initial freeze, with dependencies pinned in `requirements.txt`.
