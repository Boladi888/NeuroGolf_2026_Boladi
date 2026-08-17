# task372 hard record: grouped Conv factor

- Live bar: 200 bytes, sha256 `b35beb50d83f1b9a90dbfed5a977cc6c9af9ac8d1b314310950baebbdc0da02e` from `python tools/ng_current_bar.py --task 372 --json`.
- Attempted route: terminal `Conv(group=2)` with weight shape `[10,5,2,1]`, cost/profile cost 100, sha256 `f7ea0cc25de33ac282d1666525015120997ea8cb7a417bdd5da34b763b3e6b6d`.
- Exact failed gate: `python tools/ng_gate.py visible --task 372 --cand The_400_Tasks/Task_by_Task/task372/prototypes/task372_probe_grouped_conv2.onnx` -> `[visible] 137/266 (51.50%)  FAIL`.
- Why dead: grouped Conv partitions input channels `[0..4]` and `[5..9]`; output colors 6-9 lose access to shared background channel 0 and lower-panel colors 1-4, so the lower-nonbackground-priority overlay cannot be represented for random two-color ARC-GEN pairs. Any input-channel permutation or duplicated background feed would require a charged pre-output tensor and lose the bar.
- One remaining hypothesis: a different single terminal operator with attribute-level spatial shift and shared channel-0 access might beat dense Conv, but grouped Conv cannot.
