# task239 — dead routes / floor record (2026-07-06, Claude w2)

**Live bar at record time:** 619, sha `5293b770cb6e...`, owner Boladi (manifest-authoritative,
`ng_current_bar.py`, not stale). This session STAGED **579** (`staged_candidates/task239_foldactive_trim_cost579/`).

## The prompt's "~200 floor via U8 color-index [1,1,12,5]+Equal-tail" is OBSOLETE — do not re-chase it
That premise describes the RETIRED local v2 (cost 1,177), which used a 12×5 color-index canvas →
`Equal` → `Pad` and materialised a `[1,10,12,5] BOOL = 600 B` emit. The **live 619 (Path B)** is a
different architecture: **one polynomial Einsum that renders straight into the FREE `output`
[1,10,30,30]**. The official scorer (`neurogolf_utils.calculate_memory`, line 234) does **NOT charge
the `input`/`output` tensors**, so there is no 600 B emit to collapse. **~200 is unreachable** for the
histogram-by-rank render. Effective family floor ≈ **575–590**.

## Forced components (why sub-575 is a wall)
- `rank_to_col [5,30] = 150 params` — cheapest carrier for the free 30-wide output column axis with
  per-slot placement `δ(w,j)`. A shared `row_basis` can't express it (needs degree-4 Lagrange →
  `[30,5]=150 + [5,5]=25`, worse). Rendering `[1,10,30,5]` then `Pad` materialises a 3000 B pre-pad
  intermediate. FORCED.
- `row_basis [30,3] = 90` — bg band `count≤r<M` is a bounded interval ⇒ its indicator is quadratic in
  `r` (`(q−2count)(2M−q)`, q=2r+1). q² row feature FORCED once `active_rows` is folded.
- `render_core [4,4,3,2] = 96` — channel delta `−32(c−color)²` + the M-fold (`x=[M,1]` operand) fix the
  dims. ~50% structural zeros but **sparse initializers are DEAD** (traps §8e, core.md §39-45,
  task204/337: sanitizer dangles the sparse ref, strict shape-infer/full_check reject → score None).
  So dense 96 is paid.
- memory ~196 — count/sort floor: `counts10f[1,10] F32=40` (ReduceSum dtype follows F32 input; casting
  to F16 first would materialise a full `[1,10,30,30] F16=18000` intermediate — dead), TopK
  `top_idx[1,5] INT64=40` + `sorted[1,5] F32=20` forced by the op.

## Formulations tried this session
- **F-A** folded max-gate via expanded `slot_features[1,5,6]` (M, M·count) + `render_core[4,6,3]=72`:
  verified 8000/8000 numpy, but costs ~590–624 (extra features bloat memory). Confirms the
  max-operand fold is memory-optimal. NOT staged.
- **F-B** sparse `rank_to_col`/`render_core`: DEAD on library evidence (above). Not built.
- **F-C (STAGED)** max-operand fold (589) + `colors_sq` trim → **579**. Byte-equal.

## One remaining hypothesis (unproven)
A sub-575 rung would need an emit that introduces neither a `[5,30]` one-hot column axis nor a TopK
sort — both look structurally forced for "histogram bars in rank-sorted columns". No construction found.
Treat 579 as at-floor unless a novel column/sort mechanism appears.
