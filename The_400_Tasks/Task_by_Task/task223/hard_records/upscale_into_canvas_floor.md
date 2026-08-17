# DEAD ROUTE — task223 sub-cost-5 upscale-into-canvas

- **Live bar**: cost **5**, sha256 `60fda0ff702e8a5c46707c1c7dfe15dc3098bb09bca8b068aaa7975c7571eaa8`
  (manifest, owner Aditya). Op = single `MaxRoiPool`, rois `[1,5]` = 5 params, mem 0.
- **Rule (verified 265/265 visible)**: ×3 nearest upscale, `output[c,i,j] = input[c, i//3, j//3]`,
  output shape MUST be exactly `[1,10,30,30]` (harness `mode_visible` does full-tensor `array_equal`
  vs `encode_grid(output_9x9)`).

## Routes attempted / ruled out
| route | gate | cost | why dead |
|---|---|---:|---|
| single Resize ×3 → `[1,10,90,90]` output (params 4) | visible **0/265 FAIL** | (4) | shape barrier — 90×90 never byte-matches the required 30×30 |
| Upsample-7 ×3 + Slice-1 crop → `[1,10,30,30]` | visible 265/265 PASS | **324,000** | attr-only params (0) but `[1,10,90,90]` intermediate counts as memory |
| MaxRoiPool (live) | — | 5 | the floor |

## Why cost 5 is forced (see task223.md "Cost analysis" for full proof)
- mem=0 ⇒ single node; the only single ops that resample a sub-window into a fixed-size canvas
  are ROI ops, whose region spec floors at 5 elements (MaxRoiPool rois `[K,5]`, RoiAlign 4+1).
  Uniform resamplers force 90×90 (fail shape); index/matrix maps need ≥30 elements.
- mem>0 ⇒ cheapest carried intermediate (`[1,10,10,10]` ≈ 4,000 B) already ≫ 4.

## One remaining hypothesis (low confidence)
An opset-18+ single op with a ≤4-element param that does a spatially-varying upscale-into-canvas
(e.g. some `Col2Im`/`GridSample` degenerate form). GridSample needs a 30×30×2 grid (1,800) —
already the old expensive version here. Col2Im is a fold (sum-overlap), not an upscaler. No
credible ≤4-param single-node resample-into-subwindow op found. **Treat as WALL.**

_Recorded 2026-07-07 by bo_ng_cld_worker2._
