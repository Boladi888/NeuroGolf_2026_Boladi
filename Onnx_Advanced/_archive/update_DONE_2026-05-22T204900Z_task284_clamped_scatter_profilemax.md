# DONE 2026-05-22T204900Z - task284 clamped scatter + compact profile max

Task284 dynamic scatter was improved from 7,569 to 6,652 cost without changing the visible/live gate result.

## Candidate

- Builder: `The_400_Tasks/Task_by_Task/task284/prototypes/build_task284_scatter_dynamic.py`
- Candidate: `The_400_Tasks/Task_by_Task/task284/prototypes/task284_scatter_gather_f32_clamped_profilemax.onnx`
- Cost: `memory=5976`, `params=676`, `cost=6652`
- Gates: `verify_against_live 266/266`
- Advisory: generic adversarial-against-live `141/150`, no runtime failures
- Targeted two-marker probes: `1332/1332` fixed 23x24-background cases and `600/600` random variable-background cases

## Techniques

1. Clamp dynamic `GatherND` color-sampling coordinates to the active grid before sampling. This preserves the stated two-marker contract and prevents broad off-contract inputs from crashing the candidate.
2. Push `ReduceMax` before the marker-presence clamp/compare:

   - old: full `[1,1,23,24]` `nonbg_u8`, then row/col reductions
   - intermediate: row/col `ReduceMax(color_idx_u8)` first, then compact `Min(..., 1)`; cost 7,064
   - best: row/col `ReduceMax(color_crop_f32)` first, compact `Greater(..., 0.5)`, then cast compact profiles to F16; cost 6,652

3. For dynamic scatter color sampling, skip the full `UINT8` color-index sheet. Sample the four scalar endpoint colors from the F32 Conv output with clamped `GatherND`, then cast only those four scalar samples to `UINT8`.

The compact-profile step removed the full nonbg U8 sheet and saved 505 bytes. The F32-scalar gather step removed the full color-index U8 sheet and saved another 412 bytes.

## Caution

This is still a dynamic-scatter tail replacement, not a fully structural v2-family compression. The earlier `task284_scatter_dynamic` submission contributed 0.00 LB despite local cost 7,569, so this candidate should be treated as evidence and a possible reviewed follow-up only. Do not submit it automatically.

## Follow-up 2026-05-22T2100Z

`task284_scatter_gather_f32_clamped_sharedweighted.onnx` reuses the existing compact `row_weighted` and `col_weighted` tensors for endpoint `ReduceMax` instead of rebuilding duplicate weighted profiles. It also pre-clamps the four scalar endpoint color coordinates against shared active-grid highs.

- Static local memory: `5976 -> 5882`
- Non-scalar params: `676`
- Static local cost: `6652 -> 6558`
- Gates: `verify_against_live 266/266`
- Advisory: generic adversarial-against-live `141/150`, no runtime failures
- Submission status: no Kaggle submit; `submissions/submission.zip` untouched

`task284_scatter_gather_f32_clamped_sharedweighted_activemax.onnx` then replaces active-size counting with a rectangular one-hot invariant:

- `ReduceMax(active_cols_full_f32)` is the active height
- `ReduceMax(active_rows_full_f32)` is the active width

This removes the active-row/col Slice -> Greater -> Cast -> ReduceSum chains and six slice initializers.

- Static local memory: `5882 -> 5557`
- Non-scalar params: `676 -> 670`
- Static local cost: `6558 -> 6227`
- Gates: `verify_against_live 266/266`
- Targeted two-marker probes: `6529/6529` fixed 23x24-background cases and `600/600` random rectangular cases
- Advisory: generic adversarial-against-live `130/150`; the candidate had no runtime failures, while LIVE failed on 11 off-contract random masks
- Submission status: no Kaggle submit; `submissions/submission.zip` untouched

`task284_scatter_gather_f32_clamped_dynamiccanvas.onnx` goes further by creating a dynamic active canvas:

- `ConstantOfShape([active_h, active_w])` makes the active zero background
- one `ScatterND` paints colored segments
- dynamic `Pad` extends the active canvas back to 30x30 with sentinel color 10

This removes the fixed 23x24 sentinel initializer, active-base scatter, fixed crop canvas, and fixed full-color Pad intermediate from static local accounting.

- Static local memory: `5557 -> 3609`
- Non-scalar params: `670 -> 113`
- Static local cost: `6227 -> 3722`
- Gates: `verify_against_live 266/266`
- Targeted two-marker probes: `6529/6529` fixed 23x24-background cases and `600/600` random rectangular cases
- Advisory: generic adversarial-against-live `130/150`; candidate had no runtime failures, LIVE failed on 11 off-contract random masks
- Caution: this is the most dynamic / least structural tail replacement in the family, so the 0.00-LB scatter warning applies even more strongly
- Submission status: no Kaggle submit; `submissions/submission.zip` untouched
