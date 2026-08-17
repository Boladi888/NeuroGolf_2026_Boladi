# task284 hard record: 2537 is the forced floor (all formulations swept) — 2026-07-12 pass-2

- **Live bar:** manifest `2634`, sha `1721f6c7195abb…` (`ng_current_bar.py --task 284 --json`).
- **Effective best (this session's WIN):** staged `task284_f16coord_cost2537`, sha
  `556fd194324595f9d103007abcaaf28b02f5859cc465c190e9f15563d6741316` — strict win −97 vs 2634.
- **Verdict:** **STRICT WIN at 2537 + WALL below 2537.** No sub-2537 correct formulation exists on any
  route surveyed. This record exists so the next batch does not re-explore documented-dead routes.

## Scoring model (exact, from `neurogolf_utils`)
`cost = memory + params`. `memory = SUM over every node-output/value_info tensor of numel×itemsize`
(input & output EXCLUDED; profile refines each to `max(static, runtime)`; this graph is fully static so
profile==static). `params = SUM of initializer/constant ELEMENT counts` (dtype-blind). Key asymmetries:
- An **initializer** costs `numel` params, **0 memory**. An F16 **intermediate** costs `2·numel` memory.
- ⇒ moving element-count into constants is a win — **but every dominant task284 tensor is data-dependent**
  (components, valid canvas, bands, moments, color one-hots), so none can be a constant.

## Exact 2537 fat map (forced-component accounting)
| tensor(s) | shape/dtype | bytes | forced because |
|---|---|---:|---|
| row/col_components | F16[6,30]×2 | 720 | Einsum operands; U8/I8 Einsum = NO ORT kernel ⇒ F16 is min dtype; width 30 = output width (a [.,24] operand ⇒ pre-pad Einsum output [1,10,.,24] = 14 400 B intermediate) |
| row/col_components_b | BOOL[6,30]×2 | 360 | comparisons yield BOOL; concat-in-BOOL then ONE cast is cheaper than cast-before-concat (+180/axis) or F16-band build |
| valid_row/col_count | F32[1,30]×2 | 240 | active-grid canvas r_valid⊗c_valid; `ReduceSum(input over ch)` output inherits F32 input dtype; can't narrow without an 18 000 B input cast; can't derive H/W as scalars without first materialising this [.,30] reduce |
| 10 bands (d + band) | U8[1,30]+BOOL[1,30] ×10 | 600 | each interval indicator over 30 positions IS 30 output values; `Less(Sub(idx_u8,lo),span)` = 2 tensors is the minimal 1-sided-compare (And(ge,le)=3; Clip+Equal=2 same) |
| moments | F32[1,10]×4 | 160 | per-color row/col moment extracts marker position AND color together (`Equal(row_mom,top_r)` one-hot); Einsum on F32 input ⇒ F32 |
| color machinery | F16[3,10]+F16[1,10]×2+BOOL[1,10]×6 | 160 | 2-color output; channel_groups reuses `present` (already computed) for the right color — no extra one-hot |
| scalars (F16 wall/where/gap, U8 band, 4 F32 coords) | [1,1] | ~130 | band endpoints + coords; already F16-narrowed this session (the −56 win) |
| **params** | initializers | **108** | idx ramps (60) + group_map (18) + color/axis consts; near-minimal |

Sum ≈ 2429 memory + 108 params = **2537** (every byte attributed & forced).

## Formulations swept in pass-2 (all lose; measured or derived)
| formulation | dominant intermediate | cost vs 2537 | evidence |
|---|---|---|---|
| **CumSum/prefix ordinal** (prompt #114 route) | CumSum step + `Equal` index-planes [5,30] | **+2140** | BUILT+MEASURED pass-1 (`probe_cumsum_vs_band`); #114 is for genuine run/ordinal tasks (task184), not bracket geometry |
| **batched [6,30] direct** | broadcast d/band [5,30] | **+40** (2577) | derived; W15 measured the pre-F16 twin at 2613 (+20 over 2593) — `lo_vec/hi_vec` Concats are pure overhead, the [5,30] d/band == 5×[1,30] |
| **2D summed-area / MatMulInteger** | [.,30,30] or [30,30] plane | **≥+1000** | any materialised 2D plane = 1 800 B (F16) / 3 600 B (I32); ≥2 needed; folding color forces ≥3 |
| **coordinate-predicate stencil** (#108) | [1,1,24,24] BOOL planes | **~+5000** | W17 priced ≈7 500 B (10 forced 2D planes) |
| **ConvTranspose stamp** (#132) | render plane | loses | #132 needs STATIC placements; task284 geometry (arm length, spine pos) varies with `half` ⇒ variable-size stamp, not a fixed kernel |
| **U8/I8 Einsum components** | — | dead | ORT NOT_IMPLEMENTED (traps + don't-do) |

## Why K=6 (geometry) is the minimal rank
Single-color bracket matrix is **rank-3**: spine-line ⊗ rows, caps ⊗ 2-row-set, arm ⊗ row — independent.
The 2-color split is COLUMNAR (col ≤ mid = left color) and the output Einsum applies color per-GROUP (index
`g`), not per-column `s`; so each geometry term that straddles `mid` must be pre-split into its left/right
halves ⇒ 3×2 = **6** components. `central_gap` is a distinct rank-1 rectangle (1960-case proof, prior). No
sub-6 decomposition and no per-column color operand cheaper than a [3,10,30] plane (1 800 B).

## Remaining hypothesis (low confidence, for the next batch)
The only untried structural win would be an op that emits an in-range indicator from `(idx, lo, span)` in a
SINGLE charged tensor (halving the 600 B band lane), or a reduce that outputs the active-grid canvas in
BOOL/U8 instead of F32 (halving the 240 B valid lane). Neither exists in opset ≤ 21. Absent a new primitive,
**2537 is the tight floor.**
