# task208 — archived attempt history (pre-2026-07 full prose)

Compressed out of `Task_by_Task/task208.md` on 2026-07-12 (bo_ng_cld_worker1) to
keep manual content under the 15 KB budget. These are the historical from-scratch
structural-compression rungs that built the 2-placement conv-stamp algorithm down
from ~28k to the 3,983 that shipped. The CURRENT bar (3752) and its forced-floor
proof live in the main `task208.md` + `hard_records/`.

## Cost ladder (superseded rungs)

| date | approach | result | cost |
|---|---|---|---|
| 2026-05-19 | worker-7 HUGE_RMAX_a group sweep | SKIP — no improvement | (base) |
| 2026-05-22 | radical structural compression (`build_task208_coloridx_tail.py --shape-mode=hw --coord-stamp`) | PASS 266/266 | 28,713 |
| 2026-05-23 | count-shape + direct-output + target-only coord stamp | PASS 266/266 | 14,256 |
| 2026-05-23 | dynamic QLinearConv kernel (`build_task208_dynamic_kernel.py`) | PASS 266/266 | 5,692 |
| 2026-05-23 | moment-height + quantized two-rectangle renderer | PASS 266/266 | 3,983 |
| 2026-05-23 | **SHIPPED Kaggle** 5-stack component 5937.98→5947.66 (+3.15 LB) | shipped | 3,983 |

## Full prose (verbatim)

- **2026-05-19 | worker-7 HUGE_RMAX_a group sweep** | SKIP — no improvements found. Model already internally crops to 21×21 (first two Slice nodes: v5=[0,0,0,0]/v6=[1,1,21,21] and v8=[0,1,0,0]/v9=[1,10,21,21]); the 1 [30,30] intermediate reported by crop advisory is the output tensor (false positive). All polish patterns no-op. Template-matching with 15 kernel sizes (4×5..7×7), AveragePool+ConvTranspose — no obvious cheaper reformulation.

- **2026-05-22 | radical structural compression** (`prototypes/build_task208_coloridx_tail.py --shape-mode=hw --coord-stamp`) | PASS vs LIVE 266/266 | 28,713. Replaced F16 output tail with UINT8 color-index tail + BOOL output; replaced F16 background AveragePool probes with UINT8 MaxPool-on-color-index `==0`; relaxed per-color selector to active height/width only; replaced full BOOL input cast + second ArgMax with a 1x1 Conv color-index plane; derives per-color row/col spans from compact input projections; replaced 15 F16 ConvTranspose stamp renders with coordinate rendering from UINT8 location maps and MatMulInteger moments. Polished with `tools/polish_candidate.py`. Advisory adversarial against live: 3/100 on random off-contract grids.

- **2026-05-23 | count-shape + direct-output + target-only coordinate stamp** (`--shape-mode=hw --coord-stamp --argmin-coord --count-shape --target-only --perimeter-cache --direct-f32-output`) | PASS vs LIVE 266/266 | 14,256. Direct final `Where(mask, paint_onehot, input)` removes full color-index output reconstruction; count-shape selector uses selected color perimeter count plus the "exactly two empty placements" signal; `ArgMin(select_last_index=1)` gets first/last empty placements without reverse copies; `GatherND` samples the first candidate's selected-color top-left pixel to choose target-only rendering instead of drawing both source+target rectangles. Added perimeter-cache selector so the six possible color-count masks are shared across the 15 shape branches, and cleaned the target picker by running `ArgMax` on the 4D color selector directly. Polished with `tools/polish_candidate.py`. Advisory adversarial against live: 62/100 off-contract agreement.

- **2026-05-23 | dynamic QLinearConv kernel** (`prototypes/build_task208_dynamic_kernel.py`) | PASS vs LIVE 266/266 | 5,692. Replaced the 15 static empty-interior `MaxPool` bank with one generator-constrained dynamic 5x5 `QLinearConv`: selected source color by count range `0 < count < 25`, computed height from selected-color row activity, derived width from perimeter (`h+w=count/2+2`), built a UINT8 kernel for the interior, and used channel-0 background sums with QLinearConv padding to find the two empty placements. Later tightened this path with generator margins: 16x16 empty-score map, 19x19 render mask, full-10-channel color selector with gathered selected count, UINT8 ArgMax selector, `Min(UINT8,UINT8)` dynamic kernel construction, and `GreaterOrEqual/Less` inside-range masks. Uses onnxsim-only polish because full `polish_candidate.py` currently fails on a missing local `tools.clip_mul_to_min_v2` import. Advisory adversarial against live: 19/100 on random off-contract grids.

- **2026-05-23 | moment-height + quantized two-rectangle renderer** (`prototypes/build_task208_dynamic_kernel.py`) | PASS vs LIVE 266/266 | 3,983. Removed the selected-color row-activity projection by deriving rectangle height from row central moments using two `Einsum` reductions and perimeter-dependent thresholds. Then drew both source and target rectangles (source repaint is a no-op), shrank the empty-score map to 15x16 because target top rows never exceed 15, quantized the first `QLinearConv` directly to a 0/1 seed with dynamic `y_scale=2*area-1`, and rewrote the 7x7 paint kernel as `Min(row_inside, col_inside, Max(row_border, col_border))` in UINT8. Polished with onnxsim. Advisory adversarial against live: 61/100 off-contract agreement.

- **2026-05-23 | SHIPPED Kaggle** as 5-stack component on 5937.98 → **5947.66 (+3.15 LB clean)** | 3,983. Largest single-task win in the 5-stack. Key techniques: Einsum row-central-moments replace row-activity projection; QLinearConv with dynamic `y_scale=2*area-1` quantizes directly to a 0/1 seed; variadic `Min(row_inside, col_inside, Max(row_border, col_border))` UINT8 paint-kernel algebra. The 61% off-contract adv was advisory only.
