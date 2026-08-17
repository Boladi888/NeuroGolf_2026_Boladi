# task361 dead-route record — GridSample color-read & scalar-pad assembly levers

- **Live bar**: team-best **4839** (Vamshi), manifest.csv `task361,Vamshi,4cb2923eff16...,4839`.
  Team-best ONNX sha256 `4cb2923eff16f8fe64eabf818f7f3a34e2dabb94c534b77d69cffd1318d38524`.
  Confirmed via `profile_real_cost --arcgen 266`: static=4839 profile=4839 (memory 4472 + params 367).
- **Batch**: 42 slot 05. Worker `bo_ng_cld_worker2` (Claude), 2026-07-03.

## What the team-best graph actually does (decoded, not the stale auto-diag)
Input one-hot `[1,10,30,30]` (channel = color, grid lives in top-left 10×10). Algorithm:
1. `Slice` channel-0 (background) plane → `bg10_f [1,1,10,10]` F32; `Flatten` → `[1,100]`.
2. `TopK(k=15, largest=0)` on the background plane → the ≤15 **object cells** (filled cells have bg=0,
   sort-smallest picks them first, padding with low-index empties whose color 0 is a harmless scatter).
   `top_idx` (flat 0-99), `top_rows`/`top_cols` via Div/Mod.
3. **Color read (lever A)**: `Gather(sample_grid_table[100,2], top_idx)` → normalized GridSample coords,
   `GridSample(input, grid)` → `sampled_colors [1,10,15,1]` F32, `Conv(w=[0..9])` → 15 scalar colors.
4. **Center detection**: two `MaxPool` erosions of the empty plane (`empty2 9×9`, `empty3 8×8`) + slices +
   `ArgMin`/`Where` → integer-center (odd object) vs half-center (even object) `center_id`.
5. Compute cr2/cc2/a/b from center; build 3 rotated index sets (90/180/270°) around the center, `Clip[0,99]`.
6. `ScatterElements(zero_flat[1,100], concat(top_idx, rot1, rot2, rot3)[1,60], concat(colors)×4)` →
   4-fold rotationally-symmetric completion.
7. `Reshape`→`[1,1,10,10]`, `Pad`→`[1,1,30,30]` sentinel 255 (**lever B, 900 B**), `Equal(out_colors[1,10,1,1])`
   → one-hot output `[1,10,30,30]`.

## Scoring model (measured, exact)
`neurogolf_utils.calculate_memory` sums max(static, runtime) bytes over all intermediates **excluding the
tensors literally named `input`/`output`**; `calculate_params` = Σ initializer **element counts** (NOT bytes).
Memory 4472 + params 367 = 4839. Biggest charged items:
memory — `completed_scalar30`(900), `sampled_colors`(600), `bg_flat_f`(400), `bg10_f`(400), `scatter_idx_i32`(240);
params — `sample_grid_table[100,2]`(200), `zero_flat_u8[1,100]`(100).

## Lever A1 — replace coord table [100,2] (200 params) with per-axis nx10[10] LUT: **DEAD (measured)**
Candidate `prototypes/cand_nx10lut.onnx`: split the pre-interleaved `[100,2]` table into a length-10 LUT
`nx10[k] = -1+2k/29` (F16, bit-identical values), gather x by col and y by row, then interleave.
- Correctness: `ng_gate visible` **266/266 PASS** (behaviorally identical — same F16 coords into GridSample).
- Cost: `profile_real_cost` = **4891** (memory 4712 + params 179). Params −188 as predicted, but **memory +240**
  from materializing I32 col/row indices (`Gather` **rejects int16 indices** — verified ORT INVALID_GRAPH, so
  the existing I16 rows/cols can't feed Gather; must Cast to I32: +120 B) plus the Unsqueeze/Concat interleave
  (+120 B). **Net +52 → LOSES.** The `[100,2]` layout (one Gather yields interleaved pairs) IS the optimization.

## Lever A2 — eliminate GridSample, read color via static Gather on a scalar map: **DEAD (forced-cost)**
Any static color map requires collapsing the 10 one-hot channels to a scalar:
`Conv/MatMul(input[1,10,30,30], [0..9])` → `[1,1,30,30]` F32 = **3600 B**, or `Slice`→`[1,10,10,10]` = **4000 B**
before collapse. Either single tensor dwarfs the entire GridSample subgraph (~800 B: table 200p + sampled 600).
GridSample is memory-optimal precisely because it materializes only 15 sampled points, never the canvas.
This is exactly the prompt's "without materializing the same canvas" — it cannot be done statically. WALL.

## Lever B — scalar-pad assembly (completed_scalar30 = 900 B): **DEAD (forced by output shape)**
Output is `[1,10,30,30]`; producing it via `Equal` needs a `[1,1,30,30]` scalar = 900 B U8 (minimal for 30×30).
Alternative "Equal on 10×10 then Pad the one-hot" makes the charged tensor `[1,10,10,10]` BOOL = **1000 B** (worse;
one-hot can't be padded per-channel with a single constant anyway). Pad-scalar-then-Equal is already the cheaper
assembly. WALL.

## Verdict: WALL at 4839
All three levers (coord table, GridSample color-read, scalar-pad) fail for quantified reasons above. The graph is
at/near its structural floor for the 4-fold-rotation algorithm. Remaining un-probed surface = center-detection
erosions (~250 B, correctness-critical, not named by the prompt). Not worth a generic siege per prompt guidance.
One remaining hypothesis if revisited: a fundamentally different symmetry primitive (canvas Transpose-rotate) —
but it still needs the GridSample color read AND recenter index-math, so unlikely to beat 4839.
