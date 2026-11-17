# PERA prior-art and equivalence audit

Date: 2026-11-17

This note audits the operation extracted from PERA (arXiv:2604.11841v1), rather
than accepting its “high-order” label as evidence of a new mechanism. The local
source excerpt defines, for (B\in\mathbb R^{m\times r}) and
(A\in\mathbb R^{r\times n}),

\[
\Delta W = \operatorname{Poly}^2(B)\operatorname{Poly}^2_H(A)
 = BA + \sum_{i\leq j}h_{ij}
 (b_i\odot b_j)(a_i\odot a_j)^T.
\]

The paper initializes (B) from a Gaussian distribution, (A=0), and the
Hadamard coefficients (h_{ij}=0). It claims the expanded factors have width
(R=2r+\binom r2), while the trainable objects remain (A,B,h). The selected
text is not a complete paper, so this is a parameterization audit, not a
reimplementation or a confirmation of its benchmark claims.

## What is already in the literature

**LoRA** is the baseline additive factorization \(\Delta W=BA\): Hu et al.,
[arXiv:2106.09685](https://arxiv.org/abs/2106.09685) and the authors’
[Microsoft implementation](https://github.com/microsoft/LoRA). If the (h)
terms are zero, PERA is exactly LoRA at the level of the represented update.

**FedPara** predates PERA and explicitly uses Hadamard products of two learned
low-rank matrices to obtain a higher-rank parameterization. Its paper states the
rank bound and full-rank condition for the product and gives the parameter-count
trade-off: [arXiv:2108.06098](https://arxiv.org/abs/2108.06098), especially its
method and proposition sections. PERA is not the same parameterization: its
Hadamard products are formed inside tied polynomial features of the two LoRA
factors, and its coefficient vector gates the quadratic terms.

**LoHa** is the closest established family in the Stable Diffusion adapter
ecosystem. The maintained [LyCORIS implementation](https://github.com/KohakuBlueleaf/LyCORIS)
lists LoHa and LoKr as implemented methods, and the [Hugging Face PEFT LoHa
documentation](https://github.com/huggingface/peft/blob/main/docs/source/package_reference/loha.md)
describes LoHa as combining low-rank approximations with a Hadamard product.
The exact implementation should be treated as the authority for tensor shapes;
the README is an inventory, not a proof of equivalence to PERA.

**HiRA** uses a different source of high rank: its update is
\(\Delta W=W_0\odot(BA)\), coupling the adapter to the frozen pretrained weight.
The ICLR paper gives this equation and the rank-product argument:
[ICLR 2025 paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/48c368f105e8145b945227b73255635a-Paper-Conference.pdf),
and the official [authors’ repository](https://github.com/hqsiswiliam/hira)
contains the training implementation. HiRA therefore differs from PERA even
though both use Hadamard structure: HiRA modulates by (W_0); PERA constructs
new tied polynomial features from (A) and (B).

**ABBA** is a particularly important overlap. Its paper presents a learned
Hadamard-product adapter and an exact Khatri–Rao reformulation, retaining
low-rank factors while representing higher-rank updates:
[arXiv:2505.14238](https://arxiv.org/abs/2505.14238) and its
[version 4 full text](https://arxiv.org/html/2505.14238v4). At the algebraic
level, both ABBA and PERA turn Hadamard products of factor components into sums
of rank-one outer products. The parameterization and optimization are not shown
to be identical from the available sources: ABBA uses independently learned
factor groups, whereas PERA ties quadratic features to the same (A,B) and
learns one (h_{ij}) per pair. Any novelty claim must therefore compare against
ABBA under matched parameter count and the exact same factorization, rather than
against LoRA alone.

**Blockwise HiRA/BHRA** further shows that “Hadamard high rank” is an active
family rather than a unique idea. Its paper explicitly positions HiRA, ABBA,
blockwise modulation and Kronecker adapters as nearby design choices:
[BHRA, arXiv:2509.21637](https://arxiv.org/abs/2509.21637). Kronecker adapters
are a different tensor factorization and should not be treated as an algebraic
match, but they are a relevant parameter-budget baseline.

## Exact relationship and what is not equivalent

For fixed (A,B,h), PERA is an ordinary rank-factorized update with expanded
width (R=2r+\binom r2): define (\widetilde B=\operatorname{Poly}^2(B)) and
\(\widetilde A=\operatorname{Poly}^2_H(A)\), then
\(\Delta W=\widetilde B\widetilde A\). Thus the *represented matrix* is
equivalent to a LoRA update of rank at most (R), but the trainable degrees of
freedom are not those of an unconstrained rank-(R) LoRA. The columns/rows are
polynomially tied to the base factors, and (h) only scales the quadratic
terms.

The expansion can also be written as a sum of rank-one terms, which makes the
relationship to Hadamard-product methods explicit:
\(BA\) supplies the linear terms and each pair contributes
\(h_{ij}(b_i\odot b_j)(a_i\odot a_j)^T\). This is restricted relative to freely learned expanded factors: there are no
independently learned left/right vectors for each quadratic pair. This observation
does not prove inclusion in a single fixed-rank LoHa or ABBA adapter. Conversely, HiRA’s (W_0\odot BA) cannot be
reduced to PERA without imposing a special relation between (W_0) and the
polynomial factors.

The rank upper bound alone does not establish useful expressivity. It says that
PERA can have a larger *possible* rank than LoRA at the same nominal (r); it
does not say the optimization reaches that rank, that the useful singular
directions are accessible, or that quality improves at a matched parameter and
compute budget.

There is an exact special case: set `h_ii = lambda` and `h_ij = 2 lambda`
for `i < j`. Then `Delta W = BA + lambda * ((BA) odot (BA))`. This follows
by expanding the elementwise square of the sum of rank-one components. The
quadratic term alone is a tied Hadamard product; the complete update also
contains the additive `BA`. Arbitrary pair coefficients do not satisfy this
identity. This is an algebraic relationship, not a new adaptation method or
an equivalence to general independently parameterized LoHa/ABBA.

## Initialization issue exposed by the equations

The stated initialization (A=0,h=0) creates a measurable optimization boundary.
At initialization, the quadratic contribution and its direct gradient with
respect to (h_{ij}) are zero because every (a_i\odot a_j) is zero. The
linear (BA) path can give (A) a nonzero gradient for a nonzero loss signal, so the
quadratic branch can become active after (A) moves, but it is not learning
simultaneously from step one. This is an algebraic consequence of the published
parameterization, not a claim that the method fails. A faithful implementation
must report the first-step gradient norms for (A,B,h) and verify that the
quadratic branch becomes nonzero on the declared schedule.

## Cheapest faithful local check

Before downloading a model or reproducing the paper’s task benchmarks, implement
the four maps (BA), PERA, LoHa/ABBA as available from their source definitions,
and HiRA on small fixed matrices. With a fixed seed, check:

1. exact forward equality between the explicit PERA sum and the concatenated
   \(\operatorname{Poly}^2(B)\operatorname{Poly}^2_H(A)\) implementation;
2. the rank and trainable-parameter counts for (r\in\{1,2,4,8\});
3. first-step gradient norms under the paper’s (A=0,h=0) initialization;
4. the best rank-matched reconstruction of a random full update, comparing PERA,
   standard LoRA and an independently parameterized Hadamard baseline;
5. wall-clock/memory cost for the extra expanded factors.

This test can falsify several attractive interpretations cheaply. If PERA’s
rank advantage disappears under matched trainable parameters, or if the
quadratic branch remains inactive under the stated optimizer schedule, the
paper’s “higher-order” mechanism is not yet a useful composition primitive. A
passing algebraic check still establishes only implementation fidelity; it does
not establish task usefulness, novelty, or superiority to ABBA/LoHa/HiRA.

## Status

Prior art makes a broad claim that “polynomial/Hadamard low-rank adaptation is
new” untenable. The narrower PERA question is whether its *tied, asymmetric
polynomial feature map with zero-initialized pair coefficients* gives a useful
optimization/quality trade-off relative to ABBA, LoHa, HiRA and a parameter-
matched expanded LoRA. The available primary sources do not answer that.

The authors’ [PERA repository](https://github.com/zhangwenhao6/PERA) is public,
labels the work ACL 2026 Findings, and says its implementation is built on
HiRA. Its README reports author-run benchmark numbers, but this note does not
independently reproduce them. The repository contains a `DATA_LICENSE`; that
does not by itself grant permission to redistribute its datasets through this
project. The [local code probe](../../operator-studies/pera-author-probe/README.md) now
records a missing-coefficient failure in an isolated published layer at a pinned
commit. This does not reproduce or refute the authors’ training results.
