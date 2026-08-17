# Transferable insight — mode-fill (dominant-color) cost floor: scalar-collapse beats vector-materialize

**Source**: task129 batch 53 (`bo_ng_cld_worker3`, 2026-07-07). LIVE 49 (Vamshi) confirmed at forced floor; no improvement. Perfect-25 rumor disproven.

**Task family**: "paint the output block with the dominant (mode) input color", one-hot `[1,10,30,30]` canvas. 44 similar tasks (task016/023/035/040/054/064/070/077/094/118/120/125/…). Generator invariant worth checking first: here the **dominant color always has count EXACTLY 3, no ties** (`max-count dist = {3:265}`), so the mode is "the color with count==3".

**Pattern (the winning cost-49 architecture)**: detect-then-render, collapsing to a **scalar** as early as possible.
1. **Falling-factorial mode detector**: `count·(count−1)·(count−2) = 6` iff `count==3`, else `0` for `count∈{0,1,2}`. So `Σ_c g_c · FF3(count_c)/6 = g_dominant` — a single **scalar** color code. Built from 3 count-moments (degree 1/2/3) via repeating `input` in the Einsum (`bcij`, `bckl`, `bcmn`) + `Mean`. No ArgMax/OneHot/index tensors.
2. **Rank-2 signed-factor render**: `Einsum('i,ij,k,lk,jc,lc,bdhw->bchw', target_basis, F, target_basis, F, basis, basis, input)` renders `(basis_c − t(code))`-type per-channel indicator, and **reuses `input` (`bdhw`, sum over channels = occupancy) as the spatial stencil** → zero stored spatial mask, output-exempt.

**Why it's the floor (forced-component argument)**:
- The render's per-code channel indicator `v_c = a·1 + b·w_c` needs BOTH a **constant** and a **linear** per-channel row → shared `channel_basis[C,2]`=`2C` params forced. A rank-1 (linear-only) basis is a parabola centred at 0 with no per-code shift → cannot select. + `factor_matrix[2,2]`=4 (bilinear form) + `one`=1 (affine constant). Detection reuses the same basis → no extra params.
- Memory is 5 **scalar/`[2]`** intermediates (≈24 B). This is the whole trick.

**The lever / anti-pattern (measured)**: any route that materializes the `[C]` count OR mask **vector** pays ≥~80 B memory and loses. Measured: zero-param `Einsum(bchw->bc)→ReduceMax→Equal→Cast→Einsum(bc,bdhw->bchw)` is visible-100% but costs **94** (two `[1,10]` F32 vectors = 80). ArgMax/OneHot tile routes cost 101/158. **Collapse the C-vector to a scalar code before rendering; never carry the vector to the render.**

**Guards**:
- Compact `[1,10,3,3]` direct-output probes score cheap (~75) but `ng_gate visible` compares the full `[1,10,30,30]` encoded tensor → 0/265. Always render full-canvas (reuse input as stencil).
- U8 `OneHot(11)` / U8 `Einsum(12)` are NOT_IMPLEMENTED in ORT; F16 is the narrow carrier.
- Perfect-25/cost-1: impossible for any mode/argmax task (output ≠ any permutation of input). One Identity visible gate confirms 0/265; don't grind the sweep.

**Applies to**: the mode/dominant-color-fill family (catalog #119 free-output Einsum micro-floors). Before building, check the generator's count invariant (is the mode a fixed count?) — if so the falling-factorial-at-that-count detector applies directly.
