# no_pad_output dead route

- Live/team-best bar: 417, sha256 `60d682272b6e1fa5205a380cf5534a623a9636e5fa883cb2cca5739b20c7bb16` (OnnxlyFans workbook + manifest).
- Attempted route: remove final `Pad` and emit the logical `[1,10,1,1]` one-hot as output; static estimate 401 after preserving output name.
- Failed gate: `visible` on `task355_bo_ng_worker1_no_pad.onnx` -> `[visible] 0/267 (0.00%)  FAIL`.
- Why dead: NeuroGolf comparison for this task requires the full padded `[1,10,30,30]` output tensor, not only the logical 1x1 cell.
- Remaining hypothesis: only a cheaper full-canvas final emitter than `Pad(best_oh, axes=[2,3], pads=[0,0,29,29])` can beat this tail.
