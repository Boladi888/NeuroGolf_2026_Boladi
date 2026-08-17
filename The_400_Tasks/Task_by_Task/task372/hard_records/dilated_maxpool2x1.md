# task372 hard record: dilated MaxPool terminal overlay

- Live bar: 200 bytes, sha256 `b35beb50d83f1b9a90dbfed5a977cc6c9af9ac8d1b314310950baebbdc0da02e` from `python tools/ng_current_bar.py --task 372 --json`.
- Attempted route: one-node terminal `MaxPool(kernel_shape=[2,1], dilations=[6,1], pads=[0,0,6,0])`, intended as a zero-param same-channel replacement for the live dilated `Conv`.
- Prototype: `The_400_Tasks/Task_by_Task/task372/prototypes/task372_probe_dilated_maxpool2x1.onnx`, sha256 `f24a9f8dc372b79551a99f0ca69793eec662993bcbda18b04e42981e4f6c8cdd`.
- Exact failed gate: `python tools/ng_gate.py visible --task 372 --cand The_400_Tasks/Task_by_Task/task372/prototypes/task372_probe_dilated_maxpool2x1.onnx` -> `[visible] ERROR: Fail: [ONNXRuntimeError] ... PoolAttributes ... pads[dim] < kernel_shape[dim] ... Pad should be smaller than kernel.`
- Why dead: ORT rejects the bottom pad of 6 required to keep the 30-row output shape with a two-tap dilation-6 pool. Smaller legal pads shrink the output; adding a Pad/Concat after the pool would make the pool result a charged full-canvas intermediate and lose the 200-byte bar.
- One remaining hypothesis: only terminal ops with Conv-like spatial offset attributes can express this row-pair rule below a full row-selector matrix; MaxPool cannot be that operator under ORT's pad constraint.
