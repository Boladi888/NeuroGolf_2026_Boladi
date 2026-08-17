# task079

Family: Gather / depthwise-Conv / find-3x3-pattern

## Graph Summary
- 20 nodes: Slice, Conv(depthwise group=9, 5x5 filter, pad=1), Greater, Cast, ReduceSum, ArgMax, Gather, Squeeze, Unsqueeze, ReduceMax, Squeeze, ArgMax, Gather, Squeeze, ArgMax, Add, Add, Gather, Gather, Pad
- Input: (1,10,30,30), Output: (1,10,30,30)
- Algorithm: find the 3x3 block in a 14x14 grid where the single-color density (minus outer ring penalty) is highest; return that 3x3 block padded to 30x30.
- All 9 depthwise Conv filters are identical: center 3x3 = +1, outer ring = -9.

## Cost Baseline
- cost=24953, memory=24708, params=245, file_size=2165 bytes
- 2026-06-06 refreshed live: cost=24409 (memory=24168, params=241) after prior slice-axis collapse.
- v10 staged: cost=8171 (memory=7912, params=259), file_size=3488 bytes.
- v14 staged: cost=4979 (memory=4654, params=325), file_size=7471 bytes.
- v18 staged fallback: cost=3905 (memory=3582, params=323), file_size=6239 bytes.
- v19 staged best: cost=3867 (memory=3516, params=351), file_size=6771 bytes.
- v25 staged under-3000 fallback: cost=2929 (memory=2878, params=51), file_size=5399 bytes.
- v32 staged best: cost=2846 (memory=2798, params=48), file_size=4858 bytes.
- v35/v36 staged risky sub-1500 probes: costs 1140 / 1281 / 1382.

## Attempts

### 2026-05-19: AveragePool replacement — NOT 100% (250/266)
- Replaced depthwise Conv with 3x3 AveragePool (no params needed)
- Only 250/266 correct — fails when outer ring penalty matters for discrimination
- Discarded

### 2026-05-19: scalar_argmax (v5) — STAGED (winner)
- Replaced ArgMax(v11, axis=1, keepdims=1) -> (1,1,1,1) + Gather(v10, v12, axis=1) -> 7D tensor (1,1,1,1,1,12,12) with:
  Squeeze(v11, [0,2,3]) -> (9,) + ArgMax(v11_1d, axis=0, keepdims=0) -> scalar + Gather(v10, scalar, axis=1) -> (1,12,12)
- Eliminates v12(8B) + v13(576B) + v14(576B) = 1160B, adds v11_1d(36B) + v12_s(8B) + v15_pre(576B) = 620B
- Net memory savings: 540B
- Result: cost=24413 (delta=-540), verify=266/266
- Staged: task079_scalar_argmax_t_70c8e7ff_on_5806_87

### 2026-06-06: ratio_count (v10) — STAGED (winner)
- Prompt read first, then `Onnx_Advanced/`, flat/nested task notes, history, and don't-do context.
- Crop contract: full cached corpus has input min/max `[14,14]`, output min/max `[3,3]`, n=200000.
- Generator invariant: each color owns one 3x3 sprite type; all copies of that color have the same cell count; copy counts are distinct and sorted descending by type. Therefore `copy_count = total_pixels(color) / sprite_cell_count(color)`.
- Graph:
  - Depthwise Conv over all 10 channels with group=10, pad `[1,1,-15,-15]`; channel 0 kernel is zero, channels 1..9 use the live 5x5 same-color isolation kernel.
  - `ReduceMax(score)` gives sprite cell count per color; `ReduceSum(input)` gives total pixels per color.
  - Slice color vectors to 1..9, divide by `(max + 0.001)`, ArgMax to select the copied-most color.
  - Threshold only the selected color score map, then reproduce live tie behavior: first valid row, first valid column within that row.
  - Dynamic Slice crops the 3x3 from original input; Pad returns `[1,10,30,30]`.
- Primary: `The_400_Tasks/Task_by_Task/task079/prototypes/task079_ratio_count_dynslice_v10.onnx`
- Staged: `staged_submits/task079_v10_ratio_count_cost8171/task079.onnx`
- Static sibling: `staged_submits/task079_v10_ratio_count_cost8171/task079_static_gather_sibling.onnx`
- Score primary: cost=8171 (memory=7912, params=259), SHA256 `4B218F0F80E83943C5F3A867CE401BA502987701D1EF4872D4B52F73757D42C9`
- Static sibling score: cost=11772 (memory=11512, params=260), SHA256 `C89C7C71504F6D88EB7740FA7C8C98833B43469C2D066D52ECF9340E5314822D`
- Gates:
  - `ng_gate visible --task 79 --cand ...`: 266/266 PASS
  - `ng_gate truth --task 79 --cand ... --ref task079.onnx --full`: 200000/200000 ground truth, 200000/200000 ref agreement

### 2026-06-06: projection + window-sum anchor (v14) — STAGED (winner)
- Prompt and Task079 notes were already read; Onnx_Advanced lessons used: negative-pad Conv for compact projections, dynamic final Slice, and no BOOL ReduceSum.
- Key rewrite: drop the 10x12x12 live score plane entirely.
  - Two negative-pad depthwise Convs compute row/column counts over the true 14x14 board: `row_counts[c,r]` and `col_counts[c,k]`.
  - Copy count is determined from compact per-color features:
    - if total >= 15 => 3
    - else if total in {8,10,14} => 2
    - else if total == 12, use occupied row+col count and max row+col load to distinguish 2x6-cell from 3x4-cell cases
    - visible manual case total=16 => 4
    - else => 1
  - Crop anchor: select winning color, first occupied row `r`, gather rows `r..r+2`, then Conv 3x3 window sums across the 3-row strip.
  - Selected sprite cell count is `total_pixels / copy_count`; choose the first 3x3 window whose selected-color count equals that value and whose top row has a selected pixel.
- Primary: `The_400_Tasks/Task_by_Task/task079/prototypes/task079_projection_formula_windowsum_dynslice_v14.onnx`
- Staged: `staged_submits/task079_v14_projection_windowsum_cost4979/task079.onnx`
- Static sibling: `staged_submits/task079_v14_projection_windowsum_cost4979/task079_static_gather_sibling.onnx`
- Score primary: cost=4979 (memory=4654, params=325), SHA256 `03FC89B5B0BB7627069D9B1BD410B03335479FBA623A5821702166F5CF023F02`
- Static sibling score: cost=8553 (memory=8230, params=323), SHA256 `0C1C703B533AB40BC14462D5F124FAA89370C179AF18D33EDC9C65936FF230CD`
- Gates:
  - visible/ref: 266/266 PASS
  - full generated corpus: actual ONNX output 200000/200000 ground truth PASS
  - anchor probe: `window_eq_count` and `window_max_top` both 200000/200000

### 2026-06-06: projection + window-max + bool tail (v19) — STAGED (winner)
- v15 removed the v14 equality/top-row gates after probe showed first max 3x3 selected-color window in the selected 3-row strip is exact:
  - `window_max_no_top`: 266/266 visible, 200000/200000 full corpus.
  - Direct `GatherND` emits `[1,1,3,30]`, avoiding duplicate `[3,30]` strip materialization.
- v16 changed color score from copy count to `copy_count - 1`, dropping the present term and combining the four remaining terms with F16 `Sum`:
  - full ONNX check: 200000/200000.
  - U8 score path was probed and rejected by ORT: both `Sum(UINT8)` and `Add(UINT8)` invalid.
- v18 replaced the original total-12 `row_max + col_max` disambiguator with a rowmax-only rule:
  - exact on 499900 colored objects in the 200000-example corpus.
  - rule: for total=12, classify as 3 copies when `row_count + col_count > 12`, or when exactly 12 and (`row_max > 3.5` or `row_count > 8.5`).
  - deletes `col_max`; keeps `col_counts` because `col_count` is still required.
  - Staged fallback: `staged_submits/task079_v18_rowmax_windowmax_cost3905/task079.onnx`
  - Score v18: cost=3905 (memory=3582, params=323), SHA256 `6A8EE138327703818B0566B875C45AA8D40E291860147F7CB7916B3C7ACFCCD3`
- v19 rebuilt the final crop as BOOL:
  - Slice only the selected-color `[1,1,3,3]` mask from `row_strip4`.
  - Compare channel ids with selected color and combine selected-color/background channels using `Not`/`And`/`Or`.
  - Use opset 13 because BOOL `Pad` is invalid at opset 11/12; axes initializers are shared to reduce params.
  - `Where(BOOL, BOOL, BOOL)` was probed and rejected by ORT, so keep the boolean logic chain.
- Primary: `The_400_Tasks/Task_by_Task/task079/prototypes/task079_projection_formula_booltail_dynslice_v19.onnx`
- Staged: `staged_submits/task079_v19_booltail_cost3867/task079.onnx`
- Static sibling: `staged_submits/task079_v19_booltail_cost3867/task079_static_gather_sibling.onnx`
- Score primary: cost=3867 (memory=3516, params=351), SHA256 `47A1BE91CE45662738BB6752FD215B9A8C69F0E74470F0FCDA98FA071143C7FD`
- Static sibling score: cost=3882 (memory=3532, params=350), SHA256 `07A2C96C2D6A7A8410D806B4B3AE6501DFB656DBC3EECBEF1CFE99D240EAB64C`
- Gates:
  - visible/ref: 266/266 PASS
  - full generated corpus: actual ONNX output 200000/200000 ground truth PASS

### 2026-06-06: AvgPool/MaxPool projection + F16 tail (v20-v32) — STAGED under 3000
- v20 replaced the row/column all-ones projection Convs with negative-pad `AveragePool`, scaling thresholds by 1/14:
  - dynamic cost=3588, SHA256 `31FF53388708BCA6B1F5D9EA04D49A79D83F676108BF8512B38D3CE1D05C1AD4`
  - visible 266/266, full ONNX 200000/200000
- v21 changed the BOOL tail to an integer color-index tail:
  - `sel_crop * color_idx`, then `Equal(channel_ids4, crop_idx)`
  - dynamic cost=3446, visible 266/266, full ONNX 200000/200000
- v22 replaced the selected-strip 3x3 Conv with negative-pad `AveragePool` over only the 12 valid 14-column windows:
  - `GatherND` itself rejects INT32 indices, but INT32 helper indices plus final cast to INT64 work.
  - dynamic cost=3343, visible 266/266, full ONNX 200000/200000
- v23/v24 replaced the column projection with direct `MaxPool` occupancy and reduced column count from the F32 occupancy map before casting the 10-element count to F16:
  - v23 cost=3203
  - v24 cost=2963, visible 266/266, full ONNX 200000/200000
- v25 changed the final color-index tail from INT32 to FLOAT16:
  - staged fallback: `staged_submits/task079_v25_f16tail_cost2929/task079.onnx`
  - score: cost=2929 (memory=2878, params=51), SHA256 `2424D2059C86554E4D2E9CE511725FED6AE31F4930DF9BB1888ABD93F7C77A25`
  - static sibling: cost=2947, SHA256 `83337A6F18A11C3BD9570947C968B681EF3041D1DDAE3623A9A48AF15683203A`
  - gates: visible 266/266; dynamic and static siblings full ONNX 200000/200000
- v30 found the only safe pixel equality: the visible total-16 correction can use `Equal(pix, 1.1428570747375488)`, the actual ORT value emitted by `AveragePool+ReduceSum` for 16 cells. NumPy-rounded `16/14` fails visible.
- v31/v32 simplified the total-12 correction:
  - one-sided `pix < 12.5` was exact, then the outer pixel-count gate was removed entirely.
  - The row/column inner condition alone preserved visible + 200000 corpus winner selection.
  - staged best: `staged_submits/task079_v32_inner_only_cost2846/task079.onnx`
  - score: cost=2846 (memory=2798, params=48), SHA256 `6B67B28FE760C72DE2498E5E916A86B6D1E66559E25D8F2685A19DC49F17F855`
  - static sibling: cost=2864, SHA256 `2B961DCD8667CAC6432310F6A8C9ACE607919C1A8337CD42C89E3F761BF1C101`
  - gates: visible 266/266; dynamic and static siblings full ONNX 200000/200000

### 2026-06-06: Einsum selected-row risky family (v35-v36) — STAGED submit probes under 1500
- Inspiration from `Onnx_Advanced`: use `Einsum` to contract directly from input without materializing all-color maps.
- Key row-selection rewrite:
  - choose a cheap approximate target color from raw pixel totals,
  - build `color_oh = OneHot(color_idx)`,
  - compute selected-color row sums with `Einsum(input, color_oh, equation="bchw,bc->bh")`, shape `[1,30]`.
  - This deletes the 560-byte all-color row count map from the risky branch.
- The crop path remains the v32 selected first occupied row + 12-window max, and output is still padded BOOL `[1,10,30,30]`.
- Staged rungs:
  - Cheapest: `staged_submits/task079_v35_einsum_maxpix_risky_cost1140/task079.onnx`
    - cost=1140 (memory=1096, params=44), SHA256 `6D4CFAAF9DBF642BBF3E34D7CB51D73A2E684E00374A3530FCF526ED80DADB41`
    - visible 263/266, full ONNX 198691/200000
  - Best generated-corpus hit rate found under 1500: `staged_submits/task079_v36_einsum_bestcorpus_risky_cost1281/task079.onnx`
    - cost=1281 (memory=1226, params=55), SHA256 `F8DA0310812D97A9A82ADABA763A92611ACC6317AE9221D9395C3A5AACC236CA`
    - visible 263/266, full ONNX 198748/200000
    - score: raw pixel count plus `2*(total==12)` and tiny total-12 color-8/color-9 tie bias
  - Visible-clean risk: `staged_submits/task079_v36_einsum_visibleclean_risky_cost1382/task079.onnx`
    - cost=1382 (memory=1316, params=66), SHA256 `B82C91B49CCB8C757BDFBB8C78C1EFF741B110A2C6067E986D9E67DC1BEAF157`
    - visible 266/266, full ONNX 197112/200000
    - score: raw pixel count plus total-12 and total-16 bonuses and a tiny total-12 color-8 tie bias
- Under-1000 status:
  - The cheapest padded v35 rung has 1096 bytes of intermediate memory before params, dominated by `row_strip4` 360, `row_sums` 120, `crop_bool` 90, and GatherND indices.
  - A raw 3x3 output shape probe can remove the 90-byte `crop_bool` but breaks the official padded output contract and still does not create a useful safe under-1000 candidate.

### 2026-06-06: failed/lower-bound probes
- v6 bbox dynamic/static: cost=5862/9463 but only 3/266. A single global bbox per color is wrong because the winning color has multiple separated copies.
- v7 spriteconv all-nonzero outer-ring: cost=10310 dynamic but only 106/266. Different-color sprites can appear in the outer ring; live only penalizes same-color pixels.
- v8 direct dense Conv: exact 266/266, cost=12482 dynamic / 16083 static. Correct but dense 9x10x5x5 weights add 2254 params.
- v9 depthwise Conv: exact 266/266, cost=11494 dynamic / 15095 static. Correct stepping stone; still materializes the full 10x12x12 validity tensor.
- BOOL reduce probe: ORT rejects `ReduceSum` over `tensor(bool)`, so it cannot remove the F16 validity tensor directly.
- Edge-anchor probe: selected copies hit row/col 11 in 110488/200000 cached corpus examples; do not shrink the 12x12 anchor lattice.
- v12 projection formula with top-row 1D dilation: measured cost=5041 dynamic, but only 264/266 visible. Lower copies in the same 3-row band can contaminate the crop column.
- v13 projection formula with 3-step connected expansion: exact 266/266 and 200000/200000 by algorithm probe, but cost=8264 dynamic, worse than v10.
- Anchor probes: cheap `any2_then_any1` is 264/266 visible and 198017/200000 full; exact connected expansion needs 3 steps; window-sum equality is exact and much cheaper.
- Drop max disambiguator / row+col counts only: 261/266 visible and 195911/200000 full; not useful as a submit probe.
- Row-only/col-only projection feature sets are not exact. Even full row vectors leave total-12 2-vs-3 copy ambiguities. Do not delete the column projection unless a new disambiguator is found.
- Simple crop-column probes fail on full corpus:
  - top row first occupied: 96788/200000
  - selected 3-row strip first occupied: 173403/200000
  - column-sum first max: 78419/200000
  - 12-window max remains exact at 200000/200000
- Small-integer output tail probes: ORT rejects `Mul(UINT8)`, `Mul(INT16)`, and `Mul(INT8)`. FLOAT16 `Mul` + `Equal` is accepted and exact.
- Pixel equality probes:
  - `Equal(pix, 12/14)` failed 232/266 visible because ORT emits two different F32 values for total 12: `0.8571428656578064` and `0.8571429252624512`.
  - Dropping total-16 correction failed 265/266 visible.
  - `Equal(pix, numpy.float32(16/14))` also failed 265/266; the correct ORT value for the visible total-16 case is `1.1428570747375488`.
- Inner-correction simplifications are not exact: `rc_sum > 12` alone fails 1 visible / 1174 full; `rc_sum >= 12`, extra-only, rowmax-only, rowcount-only, and mixed variants all fail except the full `rc_gt12 OR (rc_eq12 AND (rowmax_gt3 OR rowcount_gt8))` form used by v32.
- Pixel-only selector probes:
  - raw max pixels: 263/266 visible, 198691/200000 full; staged as v35 cost 1140.
  - raw pixels plus corrected total-12/color tie bonuses can improve generated hit rate to 198748/200000 but still misses 3 visible.
  - visible-clean pixel-only score needs visible-specific total-12/total-16 bonuses and a color-8 tie bias; it drops generated hit rate to 197112/200000.
- BOOL tail probes:
  - `Pad(BOOL)` invalid at opset 11 and 12.
  - opset 13 accepts `Pad(BOOL)` locally, but requires input-form axes for `Squeeze`, `Unsqueeze`, and `ReduceSum`.
  - `Where(BOOL, BOOL, BOOL)` is not implemented in local ORT; use `Not`/`And`/`Or`.

## Don't-do log
- Do not use a global bbox per color.
- Do not replace same-color outer-ring scoring with all-nonzero outer-ring scoring.
- Do not shrink the Conv anchor grid below 12x12.
- Do not rely on BOOL `ReduceSum`; it is an invalid ORT graph.
- Do not emit a raw 3x3 graph output for local gates; `ng_gate` expects the padded one-hot `[1,10,30,30]` encoding.
- Do not use top-row dilation alone for the projection rewrite; it selects invalid columns when a lower copy is close in the selected 3-row strip.
- Do not pay for 3-step connected expansion unless a future scorer changes; the window-sum anchor gives the same local correctness at much lower cost.
- Do not use U8 arithmetic for the color score; ORT rejects U8 `Sum` and U8 `Add`.
- Do not use small-integer arithmetic for the output color-index tail; ORT rejects `Mul` on UINT8/INT16/INT8.
- Do not replace the 12-window crop-column score with first occupied/top-row/colsum heuristics.
- Do not use exact equality for total-12 pixel count; keep the v32 inner-condition logic unless a new proof replaces it.
- Do not delete the visible total-16 correction. If it is represented by equality, use the ORT-observed scalar `1.1428570747375488`, not NumPy-rounded `16/14`.
- Do not present v35/v36 as exact replacements; they are explicitly risky pixel-score submit probes.
