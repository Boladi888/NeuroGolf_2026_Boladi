# hard_record: PositionFormulaFold (formula-to-~700) is a WALL — rank test

- **Live bar**: 2693 (manifest, owner Boladi, sha `dcadc581e3ca873c600fb9c0ad8b993418a3b79749de8fd737edaccc2d6d9664`, `ng_current_bar --task 14`).
- **Attempted route**: prompt's PositionFormulaFold — "turn coordinate/position logic into marginal scalars and a final formula output, avoiding a materialized 2D coordinate plane" → floor_est ~700.
- **Fast probe run**: coordinate-rank test on the rare-color interior mask (the crop = `grid[ys.min:ys.max+1, xs.min:xs.max+1]` restricted to the rare color).
- **Result — DEAD**: over all 266 bundled pairs the rare-color interior mask has
  - real matrix rank: min=1, **max=10, mean=4.8**
  - rank==1 (separable / outer-product / formula-expressible): **1.5%** (only the 4 solid-rectangle cases)
  - rank>=3: **91.7%**
  - full rank == min(h,w): 20.7%
  - interior fill ratio mean 0.89 (262/266 examples have bg holes inside the bbox).
- **Why it's dead**: the output is the rare color's bbox crop *keeping its internal bg holes*. A rank>1 2D mask is NOT an outer product of a row-vector and a col-vector, so it cannot be reconstructed from marginal scalars + a coordinate formula. A pure `rowmask[R] ⊗ colmask[C]` formula produces a SOLID rectangle → wrong on 262/266 (visible-fail, certain hidden 0). The 2D interior pattern must be read cell-by-cell → a spatial carrier (crop tensor + 30×30 canvas) is FORCED.
- **Forced-component floor** (why ~700 is unreachable): profiling the live v18 (2693) shows the two spatial tensors alone are `color_grid` [1,1,30,30] u8 = 900 B (the padded output canvas, forced by the output contract [1,10,30,30] BOOL) + `tgt_crop_f` [1,1,h,w] F32 = 672 B (runtime max area 168 × 4 B, the cheapest way to read the F32 one-hot's rare channel over the bbox). 900 + 672 = 1572 B > 700 before any bbox-discovery cost. The forced structural floor is ≈ 2.4–2.5 KB, not 0.7 KB. The ~700 estimate assumed a formula-expressible (solid/low-rank) interior, which the data falsifies.
- **One remaining hypothesis** (pursued separately, see prototypes/build_task014_v19.py): shave the crop→canvas tail of v18 — drop one of the two redundant [1,1,h,w] crop copies by folding the color mapping into a data-dependent `Equal` compare-vector (mask-canvas {0,1,2} + compare {0,1,3}) instead of `Greater→Where` color-index. Small (~120 B) win, NOT a formula-fold.

VERDICT: PositionFormulaFold to ~700 is infeasible for this task family. Do not re-run the rank probe or rebuild a marginal-only/formula output — the interior is provably high-rank 2D data. Improvements must stay in the crop+canvas family and are byte-shaves, not a 3.85× structural collapse.
