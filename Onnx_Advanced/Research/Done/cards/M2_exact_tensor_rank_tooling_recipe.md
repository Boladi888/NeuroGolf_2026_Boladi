# Research Card M2 — Runnable tooling recipe for exact bounded-integer CP/tensor-rank decomposition (selector banks)

**Deliverable**: a single copyable Markdown research note (no file writes, no code execution). Suggested save name: `m2_exact_rank_tooling_recipe.md`. **This is the EXECUTION follow-through to card L1** (`../Done/l1_exact_cp_decomposition.md`): L1 settled the THEORY (unrestricted-real rank is ∃ℝ-hard; the actionable regime is bounded-integer/dyadic MIQCP + symmetry breaking + a structured dictionary). M2 must produce the **concrete, runnable recipe + a public-solver benchmark** so I can decompose a specific bank.

## Researcher contract

Internet-research assistant, web access, NO repo access. ONE self-contained Markdown note, no questions, assumptions inline. Source-tag every claim. Code snippets ARE wanted here (this is a tooling card) but must be copy-pasteable and target named public tools.

## The concrete problem (all facts inline)

I have small 3-D integer/{0,1} "selector bank" tensors `T[k, i, j]` used inside an ONNX `Einsum` final node, where the cost driver is the bank's element count (params). I want the minimum-element exact factorization. Representative target: a `[10, 10, 10]` integer tensor (also smaller `[5,5,5]`..`[19,*,*]` cases). Concrete live instance: a bank with a known rank-19 construction (`T=19`), and I want to know if an exact `T<=18..10` bounded-coefficient CP decomposition exists.

Two semirings matter:
1. **Real CP feeding an F32 Einsum** — mixed signs / cancellation allowed, provided every contraction stays exactly representable in binary32 (integers up to 2^24 are exact).
2. **Boolean OR/AND** — when the bank is consumed by BitwiseAnd-style logic.

L1 already established (treat as PROMPT): flattening (matricization) ranks are unconditional CP lower bounds; permutation-support banks have Boolean rank n (OR-incompressible); the practical method is bounded-alphabet (S={0,±1}, then {0,±1,±2}, then small dyadic) feasibility search with first-nonzero=+1 normalization + lexicographic support ordering for symmetry breaking, plus a structured dictionary (basis/ones/class-indicator/cyclic-shift/diagonal vectors) before free dense factors.

## What M2 must deliver (the gap L1 left open)

A researcher with these tools should answer: **which public solver actually closes a 10×10×10 bounded-integer CP feasibility problem fastest, and exactly how do I run it?**

## Required answer order

1. **Solver shortlist + verdict**: compare public tools for the bounded-integer CP feasibility model — at minimum Gurobi (academic license), OR-Tools CP-SAT, Z3 (SMT), and a Gröbner/algebraic option (Macaulay2 / Singular / Sage) for fixed-support certificates. For each: can it express the model, expected scaling at 10×10×10×T, license/cost, and a one-line verdict.
2. **The recommended recipe** — a runnable, copy-pasteable formulation for the WINNER: variable definitions, the bilinear/linearized constraints, the exact-arithmetic guard (|partial sums| ≤ 2^24), the symmetry-breaking constraints (first-nonzero normalization + lex support order), and the dictionary-restricted variant.
3. **Lower-bound + certificate procedure**: how to compute the three flattening ranks first (cheap kill-test), and how to interpret an infeasibility result honestly (bounded-alphabet infeasible ≠ real-infeasible — what would a real-rank certificate actually require, and is it worth it).
4. **Time-budget guidance**: realistic wall-clock for T=18→10 at S={0,±1} then {0,±1,±2} on a desktop; a stopping rule.

## Required table schema

`| solver | expresses bounded-int CP? | exact arith? | est. scaling @10x10x10 | license | verdict |`

## Source-tag standard

`[PROMPT]` · `[PRIMARY]` (solver docs / papers / canonical repos) · `[SECONDARY]` · `[DERIVED]` · `[SPECULATIVE]`. Human-readable citations (tool + version/page), not bare links.

## Mandatory caveats

- Exact arithmetic only: float LP relaxations that "round to integer" are NOT acceptable certificates — the decomposition must reproduce the tensor EXACTLY. State how each solver guarantees exactness (integer vars / rational / SMT).
- The F32-Einsum consumer means real (signed) factors are allowed but every materialized contraction must be ≤2^24 in magnitude — the recipe must encode that bound.
- Distinguish "found a shorter decomposition" (great) from "proved none shorter exists in the alphabet" (a bounded certificate) from "proved none over ℝ" (usually infeasible to get).

## Verification checklist (end with this)

- [ ] At least 4 public solvers compared in the table with an exactness column.
- [ ] One fully runnable recommended recipe (copy-pasteable) for the winner.
- [ ] Flattening-rank kill-test included as step 0.
- [ ] Honest infeasibility-interpretation paragraph (alphabet vs real).
- [ ] **One-paragraph blunt verdict**: name the single solver+recipe I should run on a 10×10×10 bank, and the realistic chance it beats a known T=19 construction down toward rank-10.
