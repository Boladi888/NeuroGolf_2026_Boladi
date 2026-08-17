# DONE: task284 sparse crop Conv + moment colors + BOOL Pad-13

Task284 dropped from 118,850 to 34,965 while preserving the current v2/live-even midpoint-bracket family.

Reusable techniques:
- **Dilated sparse crop Conv**: a `kernel_shape=[2,2]`, `dilations=[7,6]` Conv on `[1,10,30,30]` emits exactly `[1,1,23,24]`; put the real 1x1 weights at kernel `[0,0]`. This keeps the crop effect without a full 30x30 projection and costs only 40 params per 10-channel projection.
- **F16 Equal + BOOL Pad-13 tail**: after sentinel-masking inactive cells to index 10 in F16, opset 13 `Equal(color_idx_f16, channel_indices_f16)` gives a BOOL one-hot plane. ORT accepts BOOL `Pad` at opset 13, so the output can stay BOOL under NeuroGolf's `>0` decoder.
- **Compact profile center/coordinate recovery**: once row/col counts exist, compute row/col sums and min/max marker coordinates on `[H,1]` / `[1,W]` profiles instead of full `[H,W]` masks.
- **Moment-based marker colors**: for exactly two aligned singleton markers, recover ordered colors from `total_color`, `sum(color*col)`, `sum(color*row)`, and compact marker coordinates. Use safe denominators gated by `same_row`/`same_col`.
- **Compact orientation gating**: apply `same_row` / `same_col` to compact row/column masks before broadcast, avoiding full-grid orientation gate tensors.

Verification:
- `verify_against_live`: 266/266.
- Contract synthetic-LIVE: 790/790 over odd/even row+col distances, distance 2, edge placements, and random contract cases.
- Generic adversarial-against-live: 148/150 (98.7%, advisory).

Files:
- `The_400_Tasks/Task_by_Task/task284/prototypes/build_task284_colorconv_sentinel.py`
- `The_400_Tasks/Task_by_Task/task284/prototypes/task284_sparsecropconv_moment_equal_bool_pad13.onnx`
- `Logs/task284_sparsecropconv_moment_synthetic_live_report.json`
- `Logs/task284_sparsecropconv_moment_adversarial_against_live.txt`
