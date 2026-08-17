# task249 U8 Pad-10 attribute dead route

Live/team bar: cost 246, sha256
`1e0ed5ec19fb8d0caefc1f3c3425cf88d8f8a242a11667806c4caae1782cb611`,
source `python tools/ng_current_bar.py --task 249 --json` / team manifest.

Attempted route: convert `idx_table10` to UINT8, use opset-10 Pad attributes
`pads=[0,20]`, `value=29`, then cast the padded index to INT32 for the final
Gather. Candidate:
`The_400_Tasks/Task_by_Task/task249/prototypes/task249_worker2_u8_attrpad_cast_probe.onnx`.

Exact failed gate: `score_network_v2` returned `None`; `onnx.checker` failed
with `[ShapeInferenceError] (op_type:Pad): data typestr: T, has unsupported type:
tensor(uint8)`, and ORT load failed with `Type 'tensor(uint8)' ... of operator
(Pad) ... is invalid`.

Why dead: Pad-10 attribute form does not accept narrow integer carriers, matching
the prior INT32 attr-Pad wall. This route cannot legally delete Pad input params.

One remaining hypothesis: keep Pad in opset 14 input form and move the saving to
the index generator itself, as in `task249_worker2_u8_mod_index.onnx`.

