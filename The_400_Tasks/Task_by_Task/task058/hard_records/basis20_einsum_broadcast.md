# task058 active-max basis truncation dead route

Live bar: cost/profile 128, sha256 `795ce7543fc7358b026acfaf9e6db16a21813c32d9f43c5dd9b7b1afd232eff9`.

Attempted route: truncate the current analytic `basis` initializer from `[30,3]` to `[20,3]`, relying on the ARC-GEN max active size of 20 and the existing `input` active-mask gate to zero rows/cols outside the active square.

Prototype: `The_400_Tasks/Task_by_Task/task058/prototypes/task058_basis20_probe.onnx`, built by `build_task058_basis20_probe_20260715.py`. Static scorer returned `None`; graph inspection would price the intended tensors at memory 16 + params 82 = 98 if the runtime accepted the shape.

Failed gate: `python tools/ng_gate.py visible --task 58 --cand The_400_Tasks/Task_by_Task/task058/prototypes/task058_basis20_probe.onnx` returned `[visible] ERROR: InvalidArgument ... Einsum operands could not be broadcast together ... Input shape of operand 1 ... {20,3} ... Another operand has a dim value of 30 in the same dimension`.

Why dead: ONNX checker/shape inference lets the model through, but ORT's `Einsum` runtime requires the shared spatial label dimensions to match exactly. A 20-row coordinate basis cannot feed the current 30-row output equation.

One remaining hypothesis: any active-max crop must also crop the input/output spatial labels and then re-pad, but the required F32 crop/pad intermediates are far above the 128-byte bar.
