# DEAD ROUTE — task380 sub-66 fused-Einsum core reduction

- **Live bar:** cost **66**, profile 66 (static=66, memory 0, params 66), owner Vamshi, manifest 2026-07-02.
  LIVE sha256 `f68be2221a8a3688bd779f2ff146190449943c226439072b19829fa3369b7cf4`.
- **Date:** 2026-07-07 (worker `bo_ng_cld_worker3`, Rot270CeilingAudit).

## The LIVE graph (already the fused direct-output floor)
Single `Einsum` node, equation `ncjk,ia,a,ka,id,de,ke->ncij`, initializers:
`F`(30×2 f32=60) + `d`(2) + `M`(2×2=4) = **66 params, 0 memory** (writes output directly, no
materialized intermediate). Operator reconstructs to `out[n,c,i,j] = in[n,c,j,2-i]` =
transpose + reversal of the 3 active columns = rot270. The charged structure is:
`K[i,k] = (F·diag(d)·Fᵀ) ⊙ (F·M·Fᵀ)` = the 3×3 anti-identity embedded in 30×30.

## Attempted route: reduce the 66 below live
Two independent from-scratch reductions of the initializer element count, both **infeasible**:

1. **Shrink F below 60.** F is 30 (output/contracted canvas axis, forced) × 2 features. 2 features
   is the minimum: 3 active positions need degree-2 separation and inactive rows must vanish
   (drop the constant monomial → 2 features {x,x²}). 1 feature (30×1) makes every bilinear form
   rank-1 → product rank ≤2, but the target anti-identity is **rank 3** → impossible. So |F|=60 forced.
2. **Shrink the cores below 6.** Sub-66 in this family means core-total < 6. Both bilinear forms
   MUST be rank-2: a rank-1 core (stored as a length-2 vector, or a scalar/`Φ Φᵀ`) drops the Hadamard
   product to rank ≤2 ≠ 3. Minimal rank-2 core storage = a length-2 diagonal (2 elems). Reachable
   core totals are therefore only **4** (diag⊙diag → 64) or **6** (diag⊙full → 66); **65 is
   unreachable**. So sub-66 ⟺ an exact two-diagonal solution at 64.
   - `K3 = (Φ diag(a) Φᵀ) ⊙ (Φ diag(b) Φᵀ)` has **no float32-exact solution**:
     - vectorized exact grid search 46,656 Φ candidates (∈{±0.5,±1,±2}) × 256 core combos
       (∈{±1,±2}²): **0 solutions**.
     - `least_squares` (trf), 200+ random restarts: best residual plateaued at ~1e-11 (never
       machine-zero) and drifted to degenerate (b0→0, Φ→large) — the signature of a real solution
       that is irrational/at-a-boundary, which cannot be float32-exact.
   - float32-exactness is mandatory: LIVE output is exact one-hot 0.0/1.0; the `ab` byte-equal gate
     rejects any candidate whose Einsum yields 0.9999997 instead of 1.0. Vamshi's 66 works precisely
     because F/d/M are small integers.
   - Related near-miss forms also non-exact (residual ~1e-6, Φ blows up): `(ΦΦᵀ)⊙(Φ diag(b) Φᵀ)`
     (would be 62), `(ΦΦᵀ)⊙(Φ C Φᵀ)` (would be 64). Confirmed infeasible.

## Verdict
**WALL at 66.** 66 is the exact forced floor of the zero-memory fused-Einsum family. The prompt's
"cost ≤ 1 perfect-25 floor" is **DISPROVEN** for task380: rot270 here is transpose **+ a 3-reversal**,
and the reversal forces a rank-3 anti-identity payload — it is not a no-payload Transpose/Identity.

## Dead siblings (do not re-explore)
- Materializing routes (whole-canvas Transpose / negative-step Slice / Gather / Pad): correct but
  memory blows up (720–36,368) and the negative-step Slice family is the old public-hidden-bad
  guardrail (see task380.md history). Not sub-66.
- Sparse `sparse_initializer` for F (nnz=6): **fiction** — ORT densifies for Einsum and
  `profile_real_cost` credits `max(static,runtime)` = dense 60+. Would fail the profile gate.

## One remaining hypothesis (low confidence)
A non-Einsum zero-memory op with an *implicit* coordinate structure (so the 30-axis embedding isn't a
stored initializer) — none found in ORT: Conv can't rotate (not translation-invariant), Gemm/matmul
materialize intermediates. If a future opset adds a free index-reversal-on-subrange op, revisit.
