# DONE 2026-05-23T031500Z - task387 color-grid ScatterElements rewrite

Task387 moved from the 44,917 structural BOOL/UINT8 family to a 6,549-cost radical algorithmic color-grid family.

Best artifact:

- `The_400_Tasks/Task_by_Task/task387/prototypes/task387_scatter_alg_v5d_colorgrid_2d_equal.onnx`
- `score_network_v2`: memory=5,842, params=705, total=6,547
- `verify_against_live`: 266/266 prototype-vs-LIVE
- `polish_candidate`: no-op, polished cost unchanged
- No Kaggle submit; `submissions/submission.zip` was not edited

Technique notes:

- Replaced paired one-hot `ScatterND` writes from the input tensor with a uint8 color-index canvas.
- Paint only 52 color cells: 4 marker centers, 32 stamp perimeter cells, and 16 dashed connector candidates.
- Use sentinel color 10 outside the variable visible rectangle, then final graph output is `Equal(color_grid, uint8[0..9])`, a BOOL one-hot tensor. Full-tensor equality accepts BOOL values equal to the live float 0/1 tensor.
- `ScatterElements` accepts INT32 indices on ORT for this path, cutting flat paint-index memory from 416 bytes to 208 bytes and halving many scalar index ops.
- Dynamic `Slice` color reads need explicit value_info annotations (`tl_pix`, `tr_pix`, `tl_arg4`, `tr_arg4`) so the official strict shape-inference scorer does not return memory=None.

Milestones:

- v1 safe scalar `ScatterND`: 13,873
- v1 flat index: 10,803
- v2 slice-color annotated: 9,535
- v3b color-grid with visible mask: 8,924
- v3c prefilled visible sentinel: 8,054
- v4 flat base offsets: 7,085
- v5c INT32 flat paint indices: 6,549
- v5d final `Equal` broadcasts from `[30,30]` instead of `[1,1,30,30]`: 6,547

Current blocker for sub-5k/sub-3k:

- Exact full-tensor live equivalence requires padding outside the ARC rectangle to be all-zero, not channel-0 background.
- The color-grid path therefore still materializes four 900-byte tensors: visible/sentinel base, flat base, painted flat grid, and 2D/broadcast color grid.
- The tempting no-sentinel color-grid variant scores 6,823 but fails 0/266 against live because outside padding becomes channel-0 background.
- The input-base scalar scatter path preserves padding for free, but exact one-hot changes require 96 scalar writes and an unavoidable INT64 `[96,4]` index tensor, leaving a higher 9,535 floor after slice-color optimization.
- Whole-row paint is a possible way to trade the flat base for row updates, but preliminary accounting shows row-update tensors (`[11,30]` base/update plus row indices) are likely to give back most of the 900-byte saving.

Next possible experiments:

- Row-paint color-grid variant: paint 10-11 whole rows into the visible/sentinel base to try to remove one 900-byte flat/reshape tensor. Needs careful sentinel handling for invalid dashed-line candidates.
- Look for a way to construct the visible/sentinel flat base directly as `[900]` without materializing `[1,1,30,30] -> [900]`.
- Probe whether `Slice` and remaining coordinate plumbing can safely move to INT32; expected savings are modest compared with the 900-byte tensor floor.
