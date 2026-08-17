# task340 history archive

## Risk portfolio (2026-06-01, archived 2026-06-24)

All sub-5k variable-crop candidates failed visible gate (ARC-GEN corpus has grids up to 20x20).
The 17x12 risky probe (cost 4932) explicitly failed on hidden set per user report.
Treat all crop-risk candidates as dead lanes unless a new hidden-specific reason exists.

| candidate | cost | train+test | visible+shipped ARC-GEN | truth 2000 | staged dir |
|---|---:|---:|---:|---:|---|
| exact 20x20 paired scatter | 8,612 | 4/4 | 266/266 | 2000/2000 | `.hermes/board/experimental_candidates/task340_20x20_exact_pairscatter_t_AF5DC9C8_cost8612/` |
| crop 17x19 paired scatter | 7,165 | 4/4 | 172/266 | 1319/2000 | `.hermes/board/experimental_candidates/task340_17x19_moderate_risk_pairscatter_t_78C4B649_cost7165/` |
| crop 16x18 paired scatter | 6,508 | 4/4 | 129/266 | 1021/2000 | `.hermes/board/experimental_candidates/task340_16x18_middle_risk_pairscatter_t_4BE6D801_cost6508/` |
| crop 14x17 paired scatter | 5,570 | 4/4 | 83/266 | 634/2000 | `.hermes/board/experimental_candidates/task340_14x17_official_test_max_pairscatter_t_DDDEA7D7_cost5570/` |
| crop 17x12 paired scatter | 4,932 | 2/4 | 67/266 | 409/2000 | **FAILED per user report** |
| crop 12x17 paired scatter | 4,932 | 2/4 | 45/266 | 384/2000 | — |
| crop 14x14 paired scatter | 4,784 | 2/4 | 54/266 | 403/2000 | — |
| crop 15x13 paired scatter | 4,765 | 2/4 | 59/266 | 399/2000 | — |
| crop 13x15 paired scatter | 4,765 | 2/4 | 46/266 | 383/2000 | — |
| crop 10x20 paired scatter | 4,852 | 1/4 | 20/266 | 167/2000 | — |
| crop 20x10 paired scatter | 4,852 | 0/4 | 28/266 | 188/2000 | — |
| fixed 14x17 test-only | 4,946 | 1/4 | 2/266 | 15/2000 | — |

Full post-pairscatter sub-5k variable-crop sweep found 53 under-5k candidates; all fail visible.

## 2026-05-20 PR_HUGE worker note

Canonical (5835.30) already exposes graph output as float16. F16 conversion of already-f16 model: onnxconverter_common rejects; direct converter would not help.

## Cost analysis — 2026-06-01 pre-rewrite parent + v1 reduction log (archived 2026-06-30)

2026-06-01 pre-rewrite `calculate_memory`/shape-inferred parent breakdown (extracted from `submissions/submission.zip`, scorer cost 67,493 = mem 67,340 + params 153):

| tensor | producer | dtype | shape | bytes |
|---|---|---:|---|---:|
| `v21` | `Slice` | F32 | `[1,10,20,20]` | 16,000 |
| `v21_f16` | `Cast` | F16 | `[1,10,20,20]` | 8,000 |
| `v110` | `OneHot` | F16 | `[1,10,20,20]` | 8,000 |

Those are the only `[1,10,*,*]` floating intermediates among the 78 parent nodes after the prior F16 ship. The rest of the cost is a spread-out tail: one INT64 color-index chain (`v58`, `v108`, `v109`: 3,200 bytes each), plus twenty-nine F16 `[1,1,20,20]` mask/color canvases at 800 bytes each.

The v1 rewrite removed the 10-channel crop/OneHot path and used a one-channel UINT8 canvas. Follow-up reductions on 2026-06-01 lowered v1 from 11,406 to 9,012:

- Dense crop Conv -> `2x2` dilated Conv (`dilations=[10,10]`): same effective 11x11 crop, Conv params 1,210 -> 40.
- Dynamic bottom/right edge colors: line mask + `ReduceMax` -> scalar `GatherElements` from the known last row/col index.
- Removed unused `row_second`/`col_second` BOOL masks left over from an earlier overlay route.
- `active AND NOT edge` -> `active XOR edge`, because edge rows/cols are subsets of active rows/cols.
- Projection source comparisons tightened to the ARC-GEN source domain rows/cols `2..17` (16x16 at max size), then padded hit lines back to 20.
- Active row/col detection slices the first column/row of `color20` and tests `< 10`, deleting two F32 input `ReduceSum`s and 30-length masks.
- Row/column inside masks use index comparisons (`0 < idx < last`) instead of stored boolean initializer masks, dropping params 231 -> 192.
- Heartbeat follow-up paired top+bottom and left+right line writes into two `ScatterElements` updates instead of four sequential full-canvas overlays (-400 B).

v1 final: 8,612 = memory 8,420 + params 192. **Superseded 2026-06-30 by v3 @ 7,718** (single-plane U8 `Sub`+`ReduceMin` hit lines; see live `task340.md` Floor proof).
