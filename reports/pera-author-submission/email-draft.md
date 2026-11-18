Subject: Counterexample to PERA Appendix A.3, Theorem 1

Dear PERA authors,

We found a counterexample to the approximation bound in Appendix A.3, Theorem 1 of your ACL 2026 paper, and would appreciate your assessment.

For nominal rank r=1, take the real 3×3 projector P=I−11ᵀ/3. Its third singular value is zero, so the stated bound would require zero error. However, every update M=baᵀ+h(b⊙b)(a⊙a)ᵀ satisfies ||P−M||₂ ≥ 1/√3, including the infimum over all parameters. The proof appears to reverse the inequality obtained by restricting optimization to a subset of rank-bounded matrices.

We attached a short proof and suggested corrected bound. The concrete counterexample has also been checked in Lean, with pinned dependencies and verification logs:
https://github.com/Techn0Ninja27/mechanism-lab/tree/e6125df573b337559257e61038e402ccb3c6b605/operator-studies/pera-lean

Could you clarify whether the theorem was intended for the tied polynomial family or a larger independent-summand relaxation? This observation concerns the theoretical bound, not your reported empirical results. We have not found a public correction and would welcome any clarification or earlier discussion we missed.

Best,
[Your name / affiliation]

Attachment: technical-note.md

---
Draft only. No message has been sent.
