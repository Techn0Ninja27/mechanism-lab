# Local benchmark — 2026-11-16

Apple Silicon Mac, 24 GiB unified memory; Python 3.12.14, PyTorch 2.15.0; released checkpoints. Four CPU threads. Timings are single-run observations, exclude model loading and are not a hardware-wide speed claim.

| Run | Generation | Coherence | Availability |
|---|---:|---:|---:|
| Original CPU serial, 1,000 draws | 17.7 s | 2.6 s | 0.3 s |
| Batched CPU (128), 1,000 draws | 0.5 s | 2.2 s | 0.2 s |
| Batched MPS (128), 1,000 draws | 0.9 s | 1.4 s | 0.4 s |

The main gain is batching, not GPU alone. CPU is faster at generation in this comparison; MPS wins coherence scoring. End-to-end times are similar after batching. Different sampler layouts/backends yield different candidate pools (973 CPU vs 980 MPS unique sets), so these timings aren't exact work-matched speedups.

Re-scored all 980 MPS candidates on CPU: identical top-20 atom-set order at beta 0, 0.4 and 0.7. Maximum absolute raw-score differences: {"coherence": 1.2715657549122739e-06, "unavailability": 1.233816146850586e-05}. These are numerical agreement checks, not evidence of research quality.

Five unit tests passed. Original serial generation remains default. No paid API calls are needed for these checks.

Larger run: 10,000 draws with MPS and generation/scoring batch size 128 produced 8,124 unique triplets. Generation 2.7 s, coherence 9.8 s, availability 1.4 s (13.9 s total). Earlier serial CPU run: 8,105 unique triplets, 145.5 + 21.0 + 2.1 = 168.6 s. About 12x observed search-stage speedup, combining batching and backend changes; this does not isolate a GPU-only benefit.
