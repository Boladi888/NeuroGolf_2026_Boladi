# task061 — archived floor-audit history

Verbose confirmatory audit prose moved out of `Task_by_Task/task061.md` on 2026-07-08
(protocol: superseded/confirmatory audits collapse to a ledger row; full prose lives here).
The canonical floor proof + forced-component table stays in `task061.md`; the compact
hard record is `Task_by_Task/task061/hard_records/globalmaxu8collapse_wall.md`.

---

**Floor proof audit (Worker-I, 2026-06-19)**: Both attack fronts exhaustively
analysed.  Front 1 (PAD-LAST reorder): Equal-at-18 then Pad costs
10×18×18=3,240 B intermediate — more than 3× the current 900 B padded canvas.
The crossover law (#81) requires 10×h×w < 900 B, i.e. h×w < 90; our 18×18=324
grid is 3.6× over the crossover threshold.  Current Equal-LAST ordering is
already optimal.  Front 2 (collapse two 324 B formula maps): no standard
opset-16 op computes `(a×b) % n` without materialising `a×b` first; both 18×18
planes are irreducible under the current algorithm.  Additional angles explored:
(a) Transpose-elimination via two initializers — saves 18 B memory but costs
+18 params, net zero; (b) Range-based row_vals — Range output costs 18×8=144 B
memory vs 18 params for the stored init, net worse; (c) lookup table for
row_mod or full residue table — 108 or 1,944 params vs 18+34 current, net
worse; (d) INT16 raw-product path — 648 B per plane vs 324 B UINT8, worse; (e)
staircase-code rank-1 assembly (#62) — does not apply because `(r×c)%n` is
multiplicative, not additively separable; (f) ConstantOfShape for channel_ids —
not a uniform fill, requires 10-element init regardless; (g) detector
alternatives — ReduceMax+ArgMax already at minimum 50 B + 1 param.  Confirmed:
v9 cost 1,668 is the true floor for any ONNX opset-16 static graph that uses
this algorithm and honours the [1,10,30,30] ABI.

Gates rerun 2026-06-19:
- truth_full: 200000/200000 (100.000%)
- profile_real (arcgen 266): static=1668 profile=1668 (memory 1634 + params 34; 267 real inputs)

**Fresh OneHot/free-output tail audit (worker-061a, 2026-06-21)**:
`v15_onehot_pad` tested the prompt's remaining direct-output route by keeping the
v9 detector/formula, converting `mod0 + 1` to INT32, using
`OneHot(axis=0, depth=10)` on the 18x18 color-index grid, then making `Pad` the
graph output.  It is correct on visible officials, but costs 15,347
(`profile_real --arcgen 266`: memory 15,314 + params 33).  The crossover fails
because ORT OneHot cannot consume UINT8 indices or emit BOOL/UINT8 values:
`color_idx_i32` costs 1,296 B, `onehot_small` costs 6,480 B, and the Unsqueeze
view is also charged at 6,480 B.  This is strictly worse than v9's 900 B padded
UINT8 residue canvas plus final BOOL Equal output.  No staging.

**Worker-061 free-output audit (2026-06-22)**: Tested the remaining #109-style
tail dissolve without building a dominated artifact.  The exact tensor
`T[n, channel, row, col]` for `n in {4..9}` has matrix-rank lower bound 28 under
the `(n,channel) x (row,col)` flattening.  Even the most favorable separable
Einsum bank therefore needs at least 28 terms; with n-select, channel-select,
row, and column factors that is already at least `28*(6+10+30+30)=2,128`
params before detector memory, so it cannot beat 1,668.  A signed-logit variant
could in principle use thresholded output values instead of exact 0/1, but the
only dense polynomial basis small enough to beat v9 is 4x4 or smaller
(`60*d*d + 60*d` params; d=5 is already 1,800 before gates).  Linear-program
separation over the full 30x30 output grid was infeasible for `n=4, color=1`
for d=1..7 in the monomial basis, so the cost-eligible signed-polynomial route
is closed.  Current v9 rerun: visible 267/267, truth 200000/200000, A/B
200267/200267 byte-equal, static=profile=1,668.

**Periodic block-tile audit (worker-16, 2026-06-23)**: Built
`prototypes/build_task061_v16_blocktile_static.py` to test the prompt's fresh
n-periodicity angle without dynamic shape tricks.  The graph computes a static
9x9 `block_mod = (i*j) % n`, then tiles it to 18x18 with `row_idx = arange(18) %
n` through `Gather(axis=0)` and `Gather(axis=1)`.  It is correct but dominated:
visible 267/267, truth 200000/200000, A/B 200267/200267 byte-equal,
`profile_real --arcgen 266` static=profile=1,726 (memory 1,683 + params 43),
sha256 `059f8dae900e480ec20d5df9764ae164e43c0001031612250c4b0c83ae907635`.
Charged tensors are `idx30` 900, `mod0` 324, `tile_rows` 162,
`block_prod`/`block_mod` 81 each, `row_idx_i32` 72, detector 50, and
`block_vals_t_u8` 9.  The 9x9 block deletes v9's 324B full-size product plane,
but static ONNX needs the INT32 Gather index and row-tile surface, making the
net cost **+58** vs v9.  A true n-shaped block would require data-dependent
tensor extents, which is the rejected v11/v12/§39 dynamic-shape class for this
task.

**Batch 5405 fresh from-scratch builds (bo_ng_cld_worker2, 2026-07-08)**:
- `v17_tile_to30`: tiled the 9×9 block DIRECTLY into the 30×30 canvas (two Gathers
  with a length-30 modular index padded to sentinel index 9), deleting BOTH v9
  18×18 planes at once. `_score_v2` = **1763** (memory 1717 + params 46). The
  `tile_rows`[30,10]=300 + INT32 `idx30_vec`[30]=120 + `row_idx`[18]=72 + block
  (81+81+100) exceed the 648 saved. Confirms the Gather-tile family caps ≥1,726.
- `v18_equal18_padlast`: Equal at 18×18 then Pad the `[1,10,18,18]` BOOL to
  `[1,10,30,30]`. `_score_v2` = **4012** (memory 3974 + params 38); the BOOL core is
  3,240 B, empirically confirming Front-1's 900-vs-3,240 crossover. Both losers.
