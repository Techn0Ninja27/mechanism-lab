# Bounded confirmation and phase check (Claude, 2026-11-18)

Design proposed by Codex (message d65b1bf1) after the 1,600-run study in
`../gnn-closedloop-gap`. Nothing there is modified. CPU only, no API calls.

## Question

1. Does the three-layer staleness gap reappear on new seeds in one moderate
   regime chosen in advance?
2. Is that gap a learning effect, or an artefact of evaluating a stale-trained
   model with an exact forward pass at one arbitrary epoch?

## Fixed design (no configuration selection)

- Seeds 50–57 (new; never used before).
- Two regimes, both 4 partitions, constant learning rate:
  - negative control: 2 layers, sync 4 (no gap in the first study);
  - moderate lead: 3 layers, sync 16 (+2.7 ± 0.6 points in the first study).
- Arms: G1, hold, sni, lag1. lag1 coefficients are locked to the first
  study's calibration (0.95; 0.8 and 0.9) and checked against it at freeze.
- 64 runs, about 3 s each: about 1 minute on 4 workers when the machine is idle.

## Measurements

The training loop is the first study's loop (tested to give bit-identical
weights). For every epoch in the final two sync cycles, **before** the
optimiser update, on the 1,000-node development-evaluation split and the
validation split:

- own-forward accuracy: the logits the model actually trained with that
  epoch (stale remote estimates). These are the training forward itself, so
  no estimator is called a second time or mutated;
- exact-forward accuracy on the same weights.

The sync phase of each epoch is stored. The original endpoint (exact forward
after the update at epoch 120) is stored separately and must reproduce the
first study's definition. Only training labels enter the loss.

## Pre-written analysis and rule

Per seed, average each accuracy over the final two cycles. Paired across
arms, n = 8, pointwise t intervals. Rule as before: > 0.5 points and > 2 SE.

- Primary: G1 − hold, exact forward, window average, in the moderate-lead
  regime. Passing confirms the lead on new seeds.
- Phase/evaluation check: G1 − hold with hold scored on its own forward. If
  the primary passes but this does not, the gap is mostly an
  evaluation-mismatch effect, not worse learning.
- Negative control: the primary comparison must not pass in the 2-layer
  regime.
- Descriptive: hold accuracy by sync phase; sni and lag1 against hold on both
  forwards; original endpoint.

Two regimes and eight seeds. This can confirm or fail to confirm one lead. It
says nothing about other regimes, depth as such, or any generated method.

## Integrity

`confirm.py freeze` hashes this plan, the code, the tests, the first study's
`gap.py`, freeze and raw results, the original `screen.py` and the Cora
files. `run` refuses on any mismatch or existing output. `analyse` checks the
binding, the exact 64 keys, finiteness, window lengths, and that G1's own
forward equals its exact forward, and refuses to overwrite.
