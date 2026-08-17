# task011 — archived history (v_w8a / worker-8 era)

Moved out of `task011.md` 2026-06-22 (w-011) to keep the active file under the 15 KB budget.
This is the pre-v3 (canvas-path) ladder. The current canonical Rule / Attempts / Cost analysis /
Don't-do / FLOOR PROOF all live in `task011.md`. This file is provenance only.

## Pre-v3 worker-8 ladder (const-div-pad → v_w8a)

| Date       | Worker  | Approach | Result | Cost | Notes |
|------------|---------|----------|--------|------|-------|
| 2026-05-10 | alpha-a | "missing-8 sub-region key" organic exploration | no-gain | n/a | Concept identified but no shippable ONNX produced |
| 2026-05-16 | worker-8 (prior session) | const-div-pad: fp32, 6D Reshape+Tile, pre-baked right/bottom padding strips, ReduceSum | 267/267 visible; **73.7% adversarial vs LIVE** | 79,445 | Staged as `worker8_const_div_pad`, approved auto-rate-limited. Diverges from LIVE on shift/empty_quadrant/shrink_grid because ReduceSum sums color masks when multiple blocks have no 8 |
| 2026-05-17 | worker-8 | v_w8a step 1: const-div-pad rewrite in fp16, single 4-axis Slice crop, single Pad-to-30 instead of right/bottom strip Concats, **kept ReduceSum** | 267/267 visible; 73.7% adversarial | ~42,985 | Confirms the ReduceSum issue is independent of the rest of the graph |
| 2026-05-17 | worker-8 | v_w8a step 2: switch ReduceSum → **ReduceMax** to match LIVE on multi-key-block inputs | 267/267 visible; 9879/9879 adversarial (100%) | ~42,985 | Adversarial perfect, but spatial-crop Slice still emits a wide [1,10,11,30] fp32 intermediate |
| 2026-05-17 | worker-8 | v_w8a final: fuse the two 1-axis Slices that crop to inner 11x11 into a single 4-axis Slice | 267/267 visible; 9879/9879 adversarial; 267/267 LIVE | **29,791** | Drops the [1,10,11,30] fp32 intermediate (~13k bytes memory). -54,456 vs LIVE, -49,654 vs staged const_div_pad |

## Cost analysis (v_w8a era)

Memory split for v_w8a final (memory=29,525; params=266):
  - `inner_11_f32` [1,10,11,11] fp32 after the 4-axis crop Slice: 4,840 bytes.
  - 14 × [1,10,...] fp16 intermediates (compact_9x9, compact_6d, masked_6d,
    expand_6d, expand_9x9, row/col concat staging, active_11): ~22,000 bytes total.
  - Small intermediates (ch8, count8_3x3, is_key, keysub, ...): ~1k bytes.
  - The fp32 crop intermediate is the single biggest contributor.

Remaining levers (v_w8a era, likely small):
  - Avoid the fp32 Cast intermediate: would require Cast(input) first, but the
    full [1,10,30,30] fp16 result is 18k bytes — strictly worse than the
    current 4,840 bytes path.
  - Drop pre-baked `div_row_9` / `div_col_11` (90+110 = 200 params) and
    reconstruct dividers from `Slice(inner_11, ch=5)` — but the slice+concat
    overhead to merge channel-5 back into 10 channels costs more memory
    than the 200 params save.

## Don't-do log (v_w8a era — superseded by separable-Einsum lineage but kept for provenance)

- **Do NOT** use `ReduceSum` over the 3x3 block grid after masking by
  `is_key`. LIVE uses `ReduceMax`. With ReduceSum, when adversarial inputs
  produce more than one key block (e.g. `empty_quadrant`, `shrink_grid`, or
  enough `shift` to zero out a block's 8s), the per-channel sum exceeds 1
  and `decode`'s `isclose(.., 1.0)` rejects the value, causing the cell to
  decode as bg. The staged `worker8_const_div_pad` predecessor has this
  bug; v_w8a fixes it by switching to ReduceMax.
- **Do NOT** Cast the full [1,10,30,30] input to fp16 before slicing.
  The fp16 intermediate at full input shape is 18k bytes vs 4,840 bytes
  for the fp32-slice-then-cast order — strictly worse.
- **Do NOT** ship pre-baked `right_strip` / `bottom_strip` padding tensors
  (~7,800 params) when a single Pad to 30x30 suffices. The decode rule
  treats "no channel = 1" as bg, so the bg channel does not need to be
  explicitly set in the outer-grid padding region.

- 2026-05-20 [OB_BIG worker]: no candidate. Tried whole_f16, cast_after_first_slice, cast_after_all_input_consumers, drop_final_cast; all delta>=0 or verify failed or ORT score crash. Task already mostly F16/bool internally or has F16-incompatible ops (Conv/MatMul/CumSum).
