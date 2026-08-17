# Transferable insight — when the output is a pure function of distance-from-center (D4-symmetric), render a small fixed **distance-domain table** + **reflect via two 1-D gathers**, instead of full-size position-grid planes. Makes the diagonal/distance LUT a FREE constant and shrinks every render plane.

**Source**: task096 (`bo_ng_cld_worker1`, 2026-07-12 pass 2). Stacked on the INT8-ranking shave:
LIVE **5537** → **4652** byte-equal (−885 total; the render shave alone was −361). Staged
`task096_int8rank_distdom_cost4652` (sha `d0cbe3f9…`). Proven byte-equal on ab --n 2000 (2266/2266)
and the full 200 000-sample corpus (0 divergences).

## The pattern to look for
A graph renders an `[H,W]` `color_grid` (later decoded free-output) where the pixel value depends
**only on the distances** `(a_row,a_col) = (|R−center|, |C−center|)` from a (possibly data-dependent)
center — i.e. the output is **D4-symmetric** (concentric rings/frames/shells, diamonds, plus/cross
shapes, symmetric stamps). On task096 LIVE the render was 8–9 planes of `11×11`:
`cheby=max(a_row,a_col)` → `Gather(color_by_radius)`, `corner_depth=|a_row−a_col|+1` (materialised as
a **169-param `corner_big` LUT** sliced at the runtime center), `in_box`, `stroke`, two nested
`Where`s. All of it is a function of `(a_row,a_col)`.

## The shave — factor through the distance domain
1. Build the fundamental-domain table `dist_T[D,D]` on **fixed** distance indices `a_row,a_col ∈ [0,D)`
   (`D` = max possible distance; here `a11=|coords−maxd_c|`, `maxd_c∈[3,5] ⇒ a11≤7 ⇒ D=8`). Compute it
   exactly like the original render but on the tiny `[D,D]` grid: `Gather` color/length by the 1-D
   distance vectors, `Where(a_col≥a_row,…)` to realise `max`, etc.
2. **`corner_depth` (and any pure-distance LUT) collapses to a compile-time constant** `[D,D]`
   (`|i−j|+1` here) — because on the distance domain there is NO runtime center offset. This **deletes
   the position-grid LUT initializer** (task096: `corner_big[13,13]`=−169 params) and its dynamic
   `Slice`. This is the single biggest win.
3. **Reflect back to full size with two 1-D gathers** (NOT a 2-D index):
   `Trow = Gather(dist_T, a_row_idx, axis=0)` → `[H,D]`; `color_grid = Gather(Trow, a_col_idx, axis=1)`
   → `[H,W]`, where `a_row_idx/a_col_idx = |coords−center|` are the `[H]`/`[W]` distance vectors.

## Why this is cheaper (do the byte math)
- Render planes shrink `H×W → D×D` (task096: 121 B → 64 B each, ×~8 planes).
- The distance/diagonal LUT becomes a free constant (−169 params) with no `Slice`.
- **CRUCIAL: keep gather indices 1-D.** A 2-D distance-index `cheby[D,D]` would be **int32 = 4 B/elem**
  (a `[8,8]` int32 index = 256 B > an `[11,11]` U8 plane = 121 B — a net LOSS). Use the original
  graph's `Gather-by-1D-vector-then-Where(max)` trick on the `[D]` vectors, and 1-D reflect gathers.
  The only added `[H,·]`/`[H,W]` tensors are `Trow[H,D]` and the final `color_grid[H,W]` (U8).
- Net on task096: **−268 memory, −93 params = −361 B**, byte-exact.

## Byte-equality proof obligations
1. Confirm the render output really is a pure function of `(a_row,a_col)` — verify `color_grid[R,C] ==
   color_grid[2·center−R, C]` (and the col mirror) over the corpus, OR just build it and diff.
2. The reflect-gather index range must fit `dist_T`'s size `D` (here `a11≤7 < 8`). Bound it
   analytically (`maxd_c∈[3,5]`) AND confirm on the full corpus (an out-of-range gather silently
   returns wrong values → divergence, so the 200k byte-equal run *is* the boundary check).
3. Match the original's exact shape at the decode boundary (`Unsqueeze` `[H,W]`→`[1,1,H,W]` so the
   downstream `Mul`/`Concat`/`QLinearConv` still type-checks).

## Rule of thumb
Concentric / symmetric renders (rings, frames, shells, diamonds, crosses) built as full-size planes
with a sliced distance LUT are over-charged. Render the tiny `[D,D]` distance table (the diagonal LUT
becomes a free constant), reflect with two 1-D gathers, and prove byte-equal on the full corpus.
Complements [[int8_narrow_ranking_arithmetic]] — they stack (a byte-equal graph has *independent*
shaves in its decision chain **and** its render).
