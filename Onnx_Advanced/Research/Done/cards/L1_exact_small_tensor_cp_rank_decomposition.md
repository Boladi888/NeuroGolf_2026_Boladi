# Card L1 — EXACT CP/rank decomposition of small {0,1}/integer tensors (selector-bank factorization)

**Deliverable**: a single copyable Markdown research note (no file writes, no code execution). Suggested save name: `l1_exact_cp_decomposition.md`.

**Researcher contract**: internet access, NO repo access; this card is self-contained. Human-readable citations (author-year-title / repo paths). Tag claims `[PROMPT]`/`[PRIMARY]`/`[SECONDARY]`/`[DERIVED]`/`[SPECULATIVE]`.

## Why this matters now (all facts [PROMPT])

- We build tiny static ONNX graphs scored by `max(1.0, 25 − ln(memory + params))`; `params` = initializer bytes. Our newest winning pattern is the **einsum-as-output with selector initializers**: e.g. `Einsum('v,bkhv,kcd,bchw->bdhw', e0, input, BANK, input)` where `BANK` is a small {0,1} tensor (typical shapes: [10,10,10], [19,10,10] factor stacks, [30,30] structured matrices). Output and input reads are free; **the BANK's element count is most of the remaining cost.**
- Live example: a [10,10,10]-semantics bank shipped as a CP-factored construction with **T=19 rank-1 terms** (9 identity + 10 for a δ(d,k) coupling), exactly reconstructing the {0,1} tensor. Matricization gives rank ≥ 10, so **slots T=10..18 are open**: they would need an exact real CP with CANCELLATION (negative coefficients summing to exact 0/1). Each saved term ≈ 30 params ≈ measurable LB.
- Values must reconstruct EXACTLY (we compare bit-level after a `>0.0` threshold; intermediate float arithmetic is fine if final integer values are exact — float32 integers exact to 2^24).
- Tensors of interest are TINY (≤ 30×30×30, mostly ≤ 10×10×10) and highly structured: permutation stacks, δ-tensors/diagonal couplings, circulant/shift families, block patterns, ≤8-class membership masks.

## The questions, in required answer order

1. **Exact CP rank of structured small families** — known results with constructions: δ-tensor/diagonal (the matrix-multiplication-tensor literature's nearest small cases), permutation stacks `T[k,:,:] = P_k`, circulant/shift stacks, outer-product-plus-corrections. For each: tight rank bounds over ℝ, over ℚ/ℤ, and over {0,1} (Boolean semiring) — these can DIFFER; say where.
2. **Cancellation gains** — when does real CP with mixed-sign coefficients beat nonnegative/Boolean rank for {0,1} tensors? Known small examples + magnitude of the gap. This answers our open slots-10..18 question directly.
3. **Exact computation tooling** — practical methods to FIND exact decompositions of a given ≤30³ integer tensor: ILP/MILP encodings (variables per factor entry; products linearized), SAT/SMT for Boolean rank, Groebner/algebraic for tiny cases, alternating least squares does NOT give exactness — say what does. Cite tools (e.g., SageMath/Macaulay2 workflows, MILP via OR-Tools/Gurobi formulations, any dedicated tensor-rank software) with realistic size limits.
4. **Boolean matrix/tensor factorization** — for ≤8-class membership masks consumed by BitwiseAnd (OR-semiring reconstruction is acceptable there): Boolean rank vs real rank, exact BMF algorithms and size limits.
5. **Integer-coefficient constraints** — our factors ride F32 einsum: coefficients must keep every contraction value integer-exact under float32 (≤2^24 intermediate magnitudes). State how to constrain searches accordingly (bounded integer coefficients; power-of-two scaling tricks).
6. **Worked recipe** — for a given 10×10×10 {0,1} tensor with matricization rank 10 and a known T=19 construction: the step-by-step search procedure you'd run (formulation, solver, time budget) to certify either a T<19 exact decomposition or its nonexistence at T≤k.
7. **Overall verdict** — one blunt paragraph: is sub-matricization-gap closing (T → rank) practically computable for our sizes with public tooling, or should we treat T = known-construction as final? Name the single best tool/method.

## Required table schema (answers 1, 3, 4)

| family / method | exact over which semiring | rank bound / size limit | construction or tool | citation |

## Demands

- Counterexample-or-proof discipline: every rank claim gets a construction or a cited proof; every "tool handles size X" gets a benchmark citation or [SPECULATIVE].
- No code beyond ≤15-line formulation sketches (e.g., the MILP linearization).
- Distinguish OUR reconstruction semirings explicitly: real-CP-feeding-F32-einsum (cancellation allowed) vs Boolean-OR (BitwiseAnd banks) — a method exact in one may not serve the other.

## Final verification checklist

- [ ] δ/permutation/circulant families have explicit rank statements with citations.
- [ ] The cancellation question (real < nonnegative rank on {0,1} tensors) answered with examples.
- [ ] At least one CONCRETE exact-search recipe with named solver and size estimate.
- [ ] F32 integer-exactness constraint addressed in the search formulation.
- [ ] Verdict names one tool/method and says compute-it or don't-bother.
