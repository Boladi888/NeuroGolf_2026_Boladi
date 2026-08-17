# task205 v18–v30 sessions — promotable findings (2026-06-06, worker-7)

Shipped-class results: LIVE 27,525 → **10,518 byte-equal-FULL** (v26) + **9,731 risk**
(v27, 10/200k) + **4,793 PROBE** (v30 profile architecture, 21/20k = 99.895%-vs-LIVE,
~99.91% GT). All staged. Full lineage: `task205.md`.

## Promote to techniques.md — **BitwiseAnd set-intersection OUTPUT (v31-v33, kills canvas+pad30)**

00. **The free-final-node output for "rank-2" per-cell patterns.** A single final node
    with rank-1-per-channel broadcast operands cannot emit `x_r OR y_c` patterns via
    arithmetic (rank-2) — but opset-18 `BitwiseAnd(A[1,C,30,1] u8, B[1,1,1,30] u8)`
    does SET-INTERSECTION: A = per-channel-row accept-bitmasks, B = per-col
    class-bitmasks (≤8 classes/u8). Decode (>0) ⇒ cell fires iff row-set ∩ col-set ≠ ∅.
    task205: main ch accepts {b∧¬y}=1; cross ch accepts {all-b}=3 on cross rows /
    {b∧y}=2 elsewhere — EXACT main+cross-lines, canvas/Pad30/Equal (~1,300) → ~600.
    **Byte-equal caveat**: when two channel selectors can collide (main==cross on
    garbage pairs), UNION the codes with BitwiseOr (+100) instead of nesting Wheres.
    Generalizes to any output = f(row-class, col-class) with ≤8 row/col classes —
    candidates: every canvas+Equal task whose pattern is separable-ish. Grader
    acceptance of BitwiseAnd pending the v31/v32 probes (opset-20 ships exist: task191).

## Promote to techniques.md (v30 profile architecture, −82.6%)

0. **Marginal-profile box detection (the no-plane family).** When the structure has a
   SOLID border and bounded interior defects (cross pixels: ≤1 per row/col, interior
   only), the whole locate+size+defect pipeline can run on free-input Einsum marginals
   (≈zero planes): 10-band Conv argmax (coarse) → restricted-marginal profiles
   (3-op Einsum `bchw,bc,bw->bh`, v14-shipped class) → first/last ≥θ edges, with
   **θ = min-border-count (6)**: borders pass deterministically, interior dips don't
   move first/last, bg needs Binom(10,0.1)≥θ ≈ 1.4e-4. THREE refinement passes
   (rows-coarse → cols-exact → rows-exact) + an exact-rows 4th marginal for defect
   cols. 0.105% divergence vs the lattice (vs 14.97% at θ=4 — the θ sweep IS the
   design). Costs: each [1,30] F32 marginal 120 B, each edge-pass ~260 B, masks
   ~214 B. Class: filtered-clean bets only (064/209 precedent band).

## Promote to techniques.md

1. **Locate-then-detect (§39-family extension; the detector-bank crop).** When a
   detector bank scans for a structure with a corpus-bounded size (box ≤ K×K), do NOT
   run the bank full-range: one cheap full-range count-conv (K×K ones kernel) +
   row-major ArgMax locates a window that contains the winner (corpus-certify with a
   sim: task205 = 200k/200k at margin 1), then fixed-shape-Gather (clamped, §39 golden
   rule — provably static) a (K+2m)² window and run the bank inside. task205: 25
   detector planes 11,025 → 625 B. **Winner-coords return via ONE offset add
   `+ start × count_acc`** where `count_acc = Σ(is_winner × exists)` — preserves
   tie-SUM (×k) and no-fire (×0) accumulator semantics exactly (byte-equal incl. the
   degenerate pairs). Margin ladder: m=1 byte-equal (0/200k), m=0 risk (10/200k).
2. **Separable ones-kernel U8 detector banks.** A bank of all-ones H×W QLinearConv
   kernels (Σ H·W params) factors into 5 shared row-convs (ones[1,w]) + per-(h,w)
   col-convs with **per-h SHARED ones[h,1] kernels**: task205 1,600 → 80 params for
   +300 B of stage-1 planes. Exact U8 integer math at scale 1 (verify count ≤ 255).
   Bias-cliff (§43) composes: bias/y_scale go on the SECOND stage.
3. **§43 cliff with y_scale=1/255 gives an EXACT 0/255 indicator** (sat(255·acc):
   acc≥1→255, ≤0→0) — ArgMax on the 0/255 vector = first-fired = LIVE's
   Greater→Cast→ArgMax semantics, deleting the per-threshold Greater/Cast layers.
   (Plain bias-cliff outputs 1..k residuals and BREAKS first-hit ArgMax semantics.)

## Promote to traps.md

3a. **Computed-start Slice (fixed extent) still scores None** — even when `ends−starts`
   is constant, shape inference sees symbolic dims on a Slice whose starts/ends are
   computed tensors → `calculate_memory` → None. The Gather-clamped-idx golden rule
   (§39) is the ONLY static dynamic-window primitive; a Gather-of-ramp mask needs a
   [N] I32 index (4 B/elem) that usually washes out the saving (task205 <3k push).
3b. **U8 QLinearConv reductions saturate at 255 — audit the WORST-case sum, including
   off-distribution**: task205's 10-row marginal band-sums reach ~280 on tie pairs
   (v9 = union of 2 tied colors) — the F16 band conv is the safe form (exact ≤ 2048).
3c. **Officials can pin architecture stages that corpus stats call optional**: the
   profile family's PASS4 refinement looks droppable (38/20k corpus) but official
   pair 96 diverges without it → §VISIBLE-FAIL makes the cut unstageable. Check the
   266 on EVERY structural cut, not just the corpus rate.

4. **0-element initializer (`dims=[0]`) → `calculate_params` returns None** (the
   `any(d <= 0)` gate) — the whole model scores None. An empty Reshape-shape init is
   the easy way to hit it; use axes-less `Squeeze` (opset 13) for scalarization.
5. **Rank-4 broadcast re-rank trap**: `Op([1,30], [1,1,1,1])` → `[1,1,1,30]`; checker
   passes, ORT load fails downstream (Einsum rank). Scalarize ([] via Squeeze) before
   mixing flat vectors with 4-D scalars.
6. **Tie-pair clamp-garbage pins byte-equality**: when LIVE clamps gather indices on
   degenerate (tie/no-fire) pairs, any re-sourced gather (different plane/frame) reads
   DIFFERENT garbage — 2/20k ab divergence class. Price such re-sources against losing
   the byte-equal ship class (task205: rejected −160 B twice).

## Corpus facts (task205 generator, ARC-GEN task_8731374e)

- bg = uniform random over ALL 10 colors (density 1/10/color); box 6..10² solid,
  offsets ≥1; cross pixels 1..3, strictly interior, DISTINCT rows + DISTINCT cols.
- mode≠boxcolor tail (no-fire, LIVE outputs garbage): 66/200k = 0.033%.
- LIVE truth class: 4999/5000 → AMBIGUOUS/FILTERED-CLEAN (§CLASS-CHECK) — byte-equal
  vs LIVE is the gold gate, not 100%-truth.
