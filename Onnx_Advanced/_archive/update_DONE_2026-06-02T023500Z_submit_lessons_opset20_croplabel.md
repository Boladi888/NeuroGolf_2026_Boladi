# DONE 2026-06-02T02:35Z — submit-confirmed ONNX lessons

Integrated the reusable lessons from the recent task213/191/383/238/286 submits.

## Added / corrected

- **opset20 BOOL reduction**: task191 v6 proved Kaggle accepts opset 20 and direct
  `ReduceMax(BOOL)` with axes input. Updated `Ort_Advanced/` dtype/opset docs and the task191
  positional-hash membership note.
- **OneHot dtype correction**: INT32 indices work when the `depth` scalar is also INT32; the older
  "INT64 only" rule was a dtype-mismatch artifact. Also documented `index == depth` as an all-zero
  sentinel.
- **task238 scalar color-index map**: keep a small `[1,1,N,N]` selected-color map plus sentinel 10,
  then expand once at the output with `Equal`.
- **task286 cropped-label Conv**: spend initializer params to emit the fixed cropped label plane
  directly, deleting larger runtime label/Cast/Slice tensors. Parent-equivalent candidate held hidden.
- **ship-decision traps**: task383 OOD shift misses can be advisory when the generator fixes placement;
  task286 parent-equivalent compression must gate against the parent, not a stronger oracle.
- **cost calibration**: added submit datapoints through task286; single-task and same-task follow-on
  reductions continued to track raw `ln(old_cost/new_cost)`.

## Files touched

- `techniques.md`
- `traps.md`
- `cost_floor_reference.md`
- `task_techniques_index.md`
- `CHANGELOG.md`
- `Ort_Advanced/ort_op_dtype_matrix.md`
- `Ort_Advanced/ort_errors.md`
- `Ort_Advanced/ort_quirks.md`
- `Ort_Advanced/ort_opset_support.md`
