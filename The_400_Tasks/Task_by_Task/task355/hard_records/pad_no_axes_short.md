# pad_no_axes_short dead route

- Live/team-best bar: 417, sha256 `60d682272b6e1fa5205a380cf5534a623a9636e5fa883cb2cca5739b20c7bb16` (`python tools/ng_current_bar.py --task 355 --json`, authoritative source `manifest`).
- Attempted route: remove the `pad_axes` initializer and rely on the existing short spatial pads `[0,0,29,29]` as a two-input `Pad`, which would have priced at about 415 if accepted.
- Candidate: `The_400_Tasks/Task_by_Task/task355/prototypes/task355_bo_ng_worker1_pad_no_axes_short.onnx`, built by `build_bo_ng_worker1_tiny_probes.py`.
- Failed gate: strict shape inference before visible -> `Pad`: `Pads has incorrect number of values. Expected 2 * 4 values. Got 4 values.`
- Why dead: without the axes input, opset-18 `Pad` requires full-rank pads for the 4-D tensor; the short spatial pad vector is legal only with `pad_axes`.
- Remaining hypothesis: only an alternate full-canvas emitter cheaper than `Pad(best_oh, pad_hw, axes)` could save these params, but mask/Einsum/Constant routes are currently more expensive.
