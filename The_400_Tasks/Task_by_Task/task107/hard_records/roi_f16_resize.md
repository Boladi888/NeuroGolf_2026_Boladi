# task107 hard record: F16 Resize ROI

- Live bar: 2491, SHA-256 `1d5d4467df91cf15adb1e7e77ff1389e328a35bcfb406c04a2f72e9d629c05f6` from `python tools/ng_current_bar.py --task 107 --json`.
- Attempted route: narrow only the `Resize` ROI path to F16 while keeping the F32 zero for `probe_bg` equality (`prototypes/build_task107_roi_f16.py`, `prototypes/task107_roi_f16.onnx`).
- Failed gate: `score_network_v2` returned `None`; ORT failed running `Resize` with `Tensor type mismatch. T != ... MLFloat16` even though static shape inference priced the model at memory 2333.
- Why dead: ORT 1.24.4 accepts the graph statically but the `Resize` kernel does not run with F16 ROI for this U8 input path, so the apparent 10-byte static memory shave is not scoreable.
- One remaining hypothesis: keep ROI F32 and look for scalar-level rewrites around U8 map assembly; do not retry F16 ROI unless ORT/kernel behavior changes.
