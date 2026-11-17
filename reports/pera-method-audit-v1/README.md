# PERA: source fidelity, algebra and claim validation

2026-11-17. The previous extraction run accepted a limited PERA update card as
source-supported. We then checked the equations, related methods and published
code before treating the card as a building block for new research.

The most substantive result is a [counterexample to Appendix A.3’s approximation
bound](../../operator-studies/pera-rank-counterexample/README.md). The paper uses
containment in a rank-bounded family in the wrong direction. A specific 3-by-3
projector has rank two, yet every real PERA update with nominal rank one stays
at spectral distance at least `1/sqrt(3)` from it, where the stated theorem
would require zero error. The written proof was checked independently by two
native agents and the parent; exact integer tests verify the target matrix,
not the universal proof. We make no priority or formal-verification claim.

Other checks:

- [Algebra and initialization](../../operator-studies/pera-algebra/README.md):
  concatenated and rank-one forms agree numerically; the nonlinear branch can
  activate after the first update. The strict absolute gradient discrepancy
  gate failed in 2/9 arms. Its failed status is preserved alongside a separate
  relative-error diagnostic consistent with floating-point accumulation.
- [Pinned author-layer probe](../../operator-studies/pera-author-probe/README.md):
  the isolated public layer lacks its coefficient entry. A diagnostic injection
  enables the bias-free path; a non-null-bias fixture exposes a separate shape
  issue. This is not a full training reproduction.
- [Prior art](prior-art.md): FedPara/LoHa, HiRA, ABBA and related adapters already
  use Hadamard structure. PERA’s particular tied polynomial map should be
  distinguished from these, without treating a rank bound as useful expressivity.
- [Equation audit](equations-audit.md): initialization, scaling and rank limits.

The engine now records all three studies in its evidence ledger and webapp,
with no automatic novelty, training or experiment-eligibility promotion. An
[integration receipt](integration-receipt.json) verifies that subsequent
source-matched generation receives all three records as prompt evidence; it
does not prove that a model will reason correctly from them. The
original accepted extraction remains unchanged: it captured a mathematical
operation, not a verified theorem or executable implementation.

No model weights or datasets were downloaded. No paid calls were made for
these checks; the phase ledger remains **$4.8168796/$20**.

## Consequence for subsequent generation

Do not feed the invalid approximation bound back as a proven capability.
The equation itself remains a usable mathematical definition, and the paper’s
empirical usefulness remains an open reproduction question. Any proposed
improvement needs matched parameter/compute baselines and the related Hadamard
adapters, not merely a comparison against rank-r LoRA. These findings do not
justify closing the entire method or asserting that a replacement is novel.
