# Evidence-reference calibration

2026-11-19. Sixteen fresh native reviewers each saw one of eight frozen
source-only packets and common instructions. Two reviews were requested
per packet. Source claims and evidence text are unchanged from v4; the new
format gives fixed 300-character source windows with 50-character overlap,
computed independently of each claim. Reviewers select at most four windows
per field/subclaim. Code supplies original field values and quoted text.

All 16 reviews passed mechanical validation and matched every field verdict
in the existing independently audited answer key. No review was repaired.
The earlier v4 attempt remains failed (two quotation-transcription errors).
This is a small, reused development calibration, not a held-out accuracy
estimate, evidence of research novelty, or proof of reviewer independence.

The phase ledger anchors all completed output hashes without changing spending.
Production paid execution remains blocked until this instrument is integrated
and the campaign contract reviewed. Raw source-containing inputs stay local.

Recompute from the repository root, using a fresh output path:

```sh
PYTHONPATH=. ../.venv/bin/python scripts/score_reference_calibration.py --root results/recordwise-calibration-v5 --source results/recordwise-calibration-v4 --output /tmp/reference-calibration-new.json
```

This requires local frozen inputs. The output path must not exist.
