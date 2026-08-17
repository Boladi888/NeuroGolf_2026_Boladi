# task040 — archived history

Moved out of `task040.md` on 2026-06-24 (w13, beta_p29_claude_cli) to keep the active file under the 20 KB tripwire. The CURRENT floor is `## FLOOR PROOF (362)` in `task040.md`; the 1785 colorcanvas floor below is RETRACTED (the 362 free-output Einsum family beats it). Kept only as history for rejected Equal-tail/colorcanvas formulations.

## RETRACTED FLOOR PROOF (1785 — forced-component, 2026-06-11 w-A3)

Retracted 2026-06-21: the 374 free-output Einsum family and the 362 `task040_shared_term_basis` candidate beat this colorcanvas floor. Keep the table below only as history for rejected Equal-tail/colorcanvas formulations.

Per-tensor measured (largest_intermediate + score_v2): memory 1724 + params 61.
| tensor | dtype/shape | bytes | forced because |
|---|---|---:|---|
| `ci30` (Pad) | UINT8 [1,1,30,30] | 900 | **architectural floor** for any [1,10,30,30] one-hot output — single-channel U8 color-index canvas fed to the free `Equal(canvas,ids)`-as-output (cost_floor_reference.md:164-168). Eliminating it (BOOL off a smaller intermediate) is proven WORSE (task388 v6 2546>1998). Active region is 10×10 ≳90 cells ⇒ canvas beats cropped-BOOL (10·h·w crosses 900 at 90). |
| `idx_f32` (Conv) | F32 [1,1,10,10] | 400 | crop+channel-collapse of the F32 one-hot input. **ORT forces every channel-collapse op (Conv/MatMul/Einsum/ReduceSum) to F32 output** (same floor class as task400/138/383). ArgMax→INT64 is 800B (worse). QLinearConv-U8 is blocked: it needs a U8 feed and `Cast(input→U8)` on the full one-hot costs 9000B (#137 prerequisite). Crop is exactly 10×10 — col-9 border (c99 read) and all 10 rows are live, cannot shrink. |
| `idx_u8` (Cast) | UINT8 [1,1,10,10] | 100 | the U8 crop; consumed by 3 corner GatherND reads, the probe Greater, and the recolor Where. Only U8 source. |
| `repl` (Where) | UINT8 [1,1,10,10] | 100 | per-cell replacement color = `Where(col_axis, col_repl[1,1,1,10], row_repl[1,1,10,1])`. col_axis is data-dependent (random transpose), so the [1,10,1]×[1,1,10] broadcast materializes 10×10 — cannot stay 1-D. |
| `is_probe` (Greater) | BOOL [1,1,10,10] | 100 | the green mask. ONNX `Where` requires a materialized BOOL condition tensor; no inline-predicate form exists. |
| `ci10` (Where) | UINT8 [1,1,10,10] | 100 | recolored crop = `Where(is_probe, repl, idx_u8)`; differs from idx_u8 at green cells, so cannot alias idx_u8; it IS the Pad input. |
| 1-D vectors + scalars | UINT8 | ~24 | row_repl/col_repl [10] + 3 corner scalars. |
| params | — | 61 | w_color[10], chan_ids[10], 2 halves[10], pad/coord/sentinel inits. |

The recolor is an irreducible per-cell **2-way data-dependent select on a 10×10 crop**: minimum materializations = source crop + condition + replacement + result = 4 planes (400B). No 2-plane/arithmetic-blend form exists (green↔bg color collision blocks Min/Add fusion; nibble-packing fights the column-half boundary which straddles pack p=2, and the transpose orientation, and the canvas must unpack to 30×30 for the output Equal anyway).

REJECTED (w-A3): Where-as-output over input (`Where(probe30, repl_onehot, input)`) needs repl_onehot = `Equal(repl_canvas[1,1,30,30], ids)` = a 900B canvas PLUS a 900B probe mask ⇒ ≥1800B, strictly worse than canvas+Equal. ArgMax channel-collapse (INT64 800B > F32 400B). QLinearConv-U8 collapse (9000B Cast blocker). Nibble-pack (#83) of the recolor planes (column-half boundary + transpose + mandatory unpack make it net-neutral-or-worse on a fixed 10×10). All prior None-scoring routes (rank-0 GatherND, If-branch, dynamic Reshape) remain in the Don't-do log.

Retracted verdict: **1785 is not a forced floor**; the free-output Einsum family bypasses this Equal-tail/colorcanvas materialization.

## Pre-generator-decoded vecmask history (47194 -> 25234)

Live model (old cost 47194 = memory 46773 + params 421): dominated by 4 separable half-grid region masks stored as full [1,1,10,10] float32 arrays (400 elems each = 100 elems each = 1600B params + large area intermediates). Each mask's Mul with a [1,10,1,1] selector broadcasts to [1,10,10,10] = 40000B/tensor × 6 intermediates. Total memory ≈46773B.

The vecmask prototype (cost 25234 = memory 24873 + params 361) replaces the 4 full masks with 1D vectors ([1,1,10,1] for row-varying, [1,1,1,10] for col-varying). Broadcasting still works: top_area[1,10,10,1] + left_area[1,10,1,10] → [1,10,10,10] via Add. Saves 360 params and 6 × 3600B = 21600B memory.
