# task361 dead-route record - direct GatherND sparse color reader

- **Live bar**: manifest/team-best **4839**, sha256
  `4cb2923eff16f8fe64eabf818f7f3a34e2dabb94c534b77d69cffd1318d38524`, owner Vamshi
  (`python tools/ng_current_bar.py --task 361 --json`).
- **Current task-local staged rung**: `task361_f16_scalartrim_cost4805`, profile cost **4805**,
  sha256 `0fc3a3955a834c7351ae1a4c5653119c303722e0ad0aa7ba946b970640045a9d`.
- **Attempted route**: replace GridSample color read with a direct sparse one-hot read from
  `input` using `GatherND(batch_dims=2)`: TopK foreground positions -> row/col indices broadcast
  over 10 color channels -> `GatherND(input, [row,col])` -> `ArgMax(axis=1)` color scalar.
- **Probe artifact**:
  `The_400_Tasks/Task_by_Task/task361/prototypes/task361_gathernd_color_probe.onnx`,
  built by `prototypes/build_gathernd_color_probe.py`.
- **Failed gate**: fast-probe cost gate, before any full candidate build.
  `python -c "from tools._score_v2 import score_network_v2; ..."` returned
  `(0, 9380, 16) 9396`.
- **Why dead**: the final `GatherND` index tensor alone is
  `gather_idx_i64 [1,10,15,2]` = **2400 B** because ONNX `GatherND` only accepts INT64 indices.
  The required row/col broadcast plumbing adds another **4800 B**
  (`rows_1x10x15`, `cols_1x10x15`, and their unsqueezed forms), before the unavoidable
  `sampled_channels [1,10,15]` = **600 B**. This exceeds the entire live GridSample color-read
  structure (`sample_grid_table` 200 params + `sampled_colors` 600 B + small coord tensors) and
  violates the prompt's 2400-B kill rule.
- **One remaining hypothesis**: an ORT-level op trick that gathers channel vectors at dynamic
  row/col without expanding indices over the color axis. Plain ONNX `Gather`, `GatherElements`,
  and `GatherND` do not expose that layout on `[1,10,30,30]` without either a huge channel-last
  transpose/reshape or the INT64 broadcast/index tensors measured here.
