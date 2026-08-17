# task387 hard record — free-output Einsum render @ 2607 is a tight local optimum

**Batch 42 slot 08 (Claude lane bo_ng_cld_worker1), 2026-07-03.**

## Live bar (refreshed this session)
- Team-best cost **2607** (memory 1906 + params 701), creator **Vamshi**.
- Source: `manifest.csv` line 389 `task387,Vamshi,f5ceadec7d5555c4de729447f7ea848f8cda1c27e4f47b272b0c1f61e0b1b7f2,2607,17.1340,2026-07-02T02:35:15Z`; identical sha on `Team_Best_Onnx/task387.onnx` (5962 bytes on disk). Workbook row agrees (2607).
- `score_v2` = (0,1906,701) → 2607. `profile_real_cost --arcgen 266` = static 2607 == profile 2607 (no credit-probe gap).

## IMPORTANT: prompt's named levers are STALE
The prompt (and local helper / task doc auto-diag) describe a **ScatterElements color-grid + visible-sentinel + Pad** algorithm floored at **3842**. That is NOT the live best. Vamshi's live 2607 is a **completely different algorithm**: a fused rank-6 free-output Einsum render. There is **no ScatterElements, no Pad, no sentinel color-grid** in the 2607 graph. The prompt's shave targets (scatter sentinel / palette / Pad handling) do not exist in the live model.

## The live 2607 algorithm (decoded)
Input `[1,10,30,30]` F32 one-hot → output `[1,10,30,30]` F16 one-hot. 90 nodes, opset 12.
Front-end computes bbox edges (top/bottom/left/right via moment Einsums + Sqrt-of-variance), active height/width, and the two marker colors A,B (ArgMax of a per-channel cross moment; B = colorsum/2 − A). Then it builds 6 row-profiles `R[6,18]` and 6 col-profiles `C[6,18]` (background rectangle, two 3×3-marker HardSigmoid suppressors, marker-color perimeter, two color-5 dashed-connector diagonals). Final render is ONE fused 7-operand Einsum:
`equation=kt,tc,ku,ur,kv,vs,b->bcrs` over `W[6,4], T[4,10], R[6,18], P[18,30], C[6,18], P[18,30], batch[1]`.
`P[18,30]` is a pure identity-embed (P[i,i]=1, 18 nonzeros) = zero-pad of the 18-cell crop basis into the 30-wide canvas. Content is always within rows/cols 0..17, so rows 18..29 are zero.

## Memory/params cost model (confirmed by reading grader `neurogolf_utils.calculate_memory`/`calculate_params`)
- **params = element count** of initializers (dtype-free). Sparse initializers count only nonzero VALUES.
- **memory = Σ max(static,runtime) BYTES** over every intermediate node-output tensor; **excludes initializers and the graph `input`/`output` tensors.**
- Consequence: the fused single Einsum is optimal because its huge `[1,10,30,30]` result IS the graph `output` → **free memory**. And a dense F16 initializer (N params) is strictly cheaper than materializing that constant anywhere else (2N memory bytes).

## Levers attempted and FALSIFIED
1. **Sparse-P initializer** (params 540→18, would be cost 2085). ORT loads it, but grader `calculate_memory` calls `onnx.checker.check_model(full_check=True)` + `shape_inference(strict_mode=True)`, both FAIL: `Einsum input 3 rank 0 does not match equation indices (2)` — onnx shape-inference does not propagate sparse-initializer shapes. score_v2 → None (unscoreable). The only fix (add P to `value_info`) re-charges P's 1080 bytes into memory (+558 net). DEAD.
2. **Drop P, widen render basis coord16 → [1,30]** (`task387_noP.onnx`, numerically equivalent). Measured `score_v2 = (0,2962,173) → cost 3135` (+528). Params −528 but memory +1056: materializing the 540-element constant costs ~2× in memory-bytes vs dense-init params. DEAD.
3. **Rank reduction below k=6.** The 6 layers are distinct spatial patterns (bg rectangle; 3×3 marker suppress ×2; marker-color perimeter; color-5 dashed connector ×2). W rows 2&3 share the same color vector but have distinct R/C outer products → genuinely rank-2, cannot collapse. k=6 is forced. DEAD.
4. **Per-tensor memory audit** (`_membreak.py`): all 90 intermediates are already minimal-dtype (F16 for arithmetic, BOOL for masks, F32 only where forced by the F32 input Einsums) and structurally required. No redundant/over-wide tensor to delete or narrow.

## Other algorithm families (documented worse than 2607)
- ScatterElements color-grid + sentinel: floor 3842. No-sentinel colorgrid: visible 0/266 (dead). ScatterND scalar: 9535+. Bool/UINT8/Conv structural rewrite: 44917.

## Verdict: WALL
2607 is a tight local optimum for the best-known algorithm family (fused free-output Einsum render). The one genuinely promising byte-shave (sparse-P) is blocked by the grader's mandatory strict shape-inference; the structural alternative (no-P) loses by a measured +528; rank<6 is forced; all other families are documented worse. No candidate strictly below 2607 was found; nothing staged (never stage a tie/loss).

## One remaining hypothesis (for a future batch)
If onnx shape-inference ever propagated sparse-initializer shapes (or if the grader dropped `strict_mode`/`full_check`), sparse-P would immediately yield cost ~2085 (−522). Alternatively, a legal way to give the Einsum P's rank-2 shape WITHOUT materializing P as a counted intermediate and WITHOUT a dense 540-param initializer would break the wall — none found under current onnx/ORT.

---

## ADDENDUM — Claude batch (bo_ng_cld_worker3), 2026-07-12: **WIN to 2566** (−41 vs live); P-embed wall now sits at 2566

**NEW STAGED BEST: `task387_drop_coord2_allc_cost2566`** (sha `6a3f8962…`, score `(0,1906,660)=2566`, byte-equal full-battery truth/ab). **−41 vs live manifest 2607, −40 vs prior effective best 2606.** Forced-2nd-pass found two param-only **operand-fold** shaves that the batch-42 per-tensor audit missed (they are NOT P levers — P is untouched):
1. **drop `coord2_f[30]` (−30 params).** It fed ONLY the two second-moment Einsums (`row_s2 'bcrs,c,r->'`, `col_s2 'bcrs,c,s->'`). Since `coord2_f = coord_f²`, form r² by contracting `coord_f` TWICE on the same axis: `'bcrs,c,r,r->'` / `'bcrs,c,s,s->'`. `coord_f` already exists (the `cross` node already doubles it) ⇒ no new init, no new intermediate. Byte-exact (r² exact in F32 for r≤29).
2. **drop `allc_f[10]` (−10 params).** All-ones channel weight is redundant — Einsum sums the `c` axis implicitly when the operand is dropped: `area_f 'bcrs,c->'→'bcrs->'`, `active_rsum_f 'bcrs,c,r->'→'bcrs,r->'`, `active_csum_f 'bcrs,c,s->'→'bcrs,s->'`. Byte-exact (area/active are small-integer sums ≤~13000, exact in F32 regardless of reduction order; single-operand `'bcrs->'` runs in ORT).

Gates (2566): visible 266/266 · profile static==2566==profile · truth --n2000 2000/2000 · ab --n2000 byte_equal=True (2266/2266, 0 div) · ng_safety PASS 0/0. `ng_graph_cost_diff` vs live 2607 = −41 (batch1 −1, coord2_f −30, allc_f −10), memory +0. **No further droppable init** (`coord_f`/`coord16`/`nonbg_f`/`channel_id_f`/`e0`/`e5`/`W`/`oh16` each load-bearing; `nonbg_f` is a genuine ch0-exclusion, NOT all-ones; deriving any materializes a charged intermediate > the param saved). polish no-op on 2566.

**The P[18,30]=540-param embed wall is UNCHANGED — it now sits at 2566** (540 of 660 params = 82%). Everything below still holds; the folds were orthogonal to P.

### Fresh verification this session (onnx 1.21.0 / ort 1.24.4)
- **sparse-P STILL dead.** Re-ran the real grader on `task387_sparseP.onnx`: `check_model(full_check=True)` AND `shape_inference(strict_mode=True)` both raise `[ShapeInferenceError] Rank of input 3 (0) does not match the equation indices (2)` → `score_network_v2 = None`. Confirmed against the grader source `neurogolf_utils.calculate_memory` (runs both) — the crash is structural, not version drift.
- **Grader source read (ground truth).** `calculate_params`: dense init → `prod(dims)` (elements); **sparse init → `prod(values.dims)` = nnz (18)**; Constant `value`/`sparse_value` → prod(dims). `calculate_memory`: charges every node-output + `value_info` + `output` tensor that is in `value_info`; **initializers (dense OR sparse) are NOT charged**; `input`/`output` skipped by name; **any node output missing from `value_info` ⇒ returns None** (so shape-inference MUST cover every node output). Multi-input graph or `io∩init` name overlap ⇒ None (kills the "P as 2nd graph input" idea). ⇒ sparse-P would be −522 IF the Einsum could see its shape; it can't without a `value_info` entry, which then charges P's 1080 B (net +558).
- **polish_candidate.py on the 2606 graph = no-op** (onnxsim, safe_opt, const_to_init, clip_mul_to_min, scalar_broadcast_ones, slice_axis_collapse all no-op). Pipeline is already tight.

### Complete P-avoidance cost space (the output's size-30 axis MUST come from some final-Einsum operand)
| route | how the 30-axis is supplied | measured cost | Δ vs 2606 |
|---|---|---:|---:|
| **dense P[18,30]** (current) | 540-param init, **0 new memory** | **2606** | — |
| sparse P | 18-param sparse init | **None** | dead (check_model crash) |
| no-P wide coord16→[1,30] | whole profile pipeline at width 30 | 3135 | +529 (mem +1056 / par −528) |
| **zero-block Concat** `R30=Concat(R,zeros[6,12])` | two new [6,30] tensors | **2858 (MEASURED today, byte-equal to R@P)** | +252 (mem +720 / par −468) |
| Pad-after-Concat `R30=Pad(R,[0,0,0,12])` | two new [6,30] tensors, pads init 4 par | ~2790 (accounting) | +184 (mem +720 / par −536) |
| P via OneHot / Constant / value_info | materialized [18,30] node/vi | ~3165 | +559 (charges ~1080 B) |

**Mechanism:** dense P is optimal because `params` counts *elements* (dtype-free, 540, and inits cost 0 memory) while any width-30 R/C tensor costs *memory-bytes* (+720 min for the two, or +1056 for the wide pipeline). Trading 540 element-params for ≥720 memory-bytes always loses. rank<6 forced (batch-42 proof stands: W rows 2&3 share color, distinct R/C). Front-end/profile per-tensor memory is minimal (polish no-op).

*(P-avoidance table Δ column is vs the 2606 baseline; the +offsets are unchanged by the 2566 folds — those removed coord2_f/allc_f, not P.)*

**Verdict: floor is now 2566 (staged win this session); the P-EMBED wall stands at 2566.** The render subsystem (P + R/C + rank-6 profiles) is a proven tight local optimum — the only break is still a legal way to feed the Einsum a size-30 zero-pad with neither a 540-param dense init NOR a charged ≥540 B materialization (sparse init blocked by onnx strict shape-inference, re-confirmed 2026-07-12; Einsum INT8 has no ORT kernel so the render can't narrow below F16). The 2566 win came from PARAM-ONLY operand folds ORTHOGONAL to the render (dropped the coord2_f/allc_f moment-helper inits). Do NOT re-run no-P / zero-block / Pad-after / sparse-P / rank<6 / narrow-dtype-render; all measured or proven dead above. Remaining param floor 660 = P 540 + coord_f 30 + coord16 18 + nonbg_f 10 + channel_id_f 10 + e0 10 + e5 10 + W 24 + oh16 2 + scalars 6 — every non-P init is load-bearing (verified this session).
