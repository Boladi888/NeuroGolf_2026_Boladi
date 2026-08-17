# axes_len1_rank2_starts

- Live/team-best bar: 495, SHA-256 `9dfe8199296cae425e71d69d161a347cb618832acee6faca0c396ef46a280291` (manifest, 2026-06-14T17:47:49Z); local staged reviewer bar before this probe was 490.
- Attempted route: shrink bool `ReduceMax` axes to length-1 (`[3]` and `[2]`) and keep the singleton batch/channel dimensions through `ArgMax`, then feed the resulting rank-2 starts into the compact `Slice`.
- Failed gate: `python -c "from tools._score_v2 import score_network_v2; ... task384_worker1_axes_len1_probe.onnx ..."` -> `None`; ORT runtime error from `Slice`: `Starts must be a 1-D array`.
- Why dead: ONNX shape inference allowed the rank-2 starts annotation, but ORT's Slice kernel enforces 1-D starts at runtime. Adding reshapes costs more memory than the saved axes params.
- Remaining hypothesis: keep only the singleton batch dimension, not channel, using axes `[1,3]` and `[1,2]`; this preserves 1-D `ArgMax` outputs and produced the staged 488 candidate.
