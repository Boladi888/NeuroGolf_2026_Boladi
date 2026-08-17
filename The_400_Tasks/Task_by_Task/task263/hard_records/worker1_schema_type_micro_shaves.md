# task263 worker1 schema/type micro-shaves

Live/team bar: 420 cost/profile cost, SHA-256
`4f5d915c42a7b80d7efabc4fac36932557866ea125136b6e5ef3b082ac54d0fd`,
source `python tools/ng_current_bar.py --task 263 --json`
(`Team_Submission/Backend/manifest.csv`, owner Sho, updated
2026-07-09T01:09:17Z).

Attempted route 1: cast `eq_max` to UINT8 instead of F16 and compare the
`ReduceSum` result against the existing `one_u8` scalar.

Exact failed gate:
`python tools/ng_gate.py visible --task 263 --cand The_400_Tasks/Task_by_Task/task263/prototypes/worker1_u8_tie_count.onnx`
-> `[visible] ERROR: InvalidGraph: [ONNXRuntimeError] : 10 : INVALID_GRAPH : This is an invalid model. Type Error: Type 'tensor(uint8)' of input parameter (safe_name_22) of operator (ReduceSum) in node (safe_name_23) is invalid.`

Why dead: ORT rejects `ReduceSum(UINT8)` for this opset/runtime. The current
F16 tie-count vector appears to be the cheapest legal reduction dtype.

Attempted route 2: omit QLinearConv input 5 (`q_zero_i8`) and rely on a default
zero weight zero-point.

Exact failed build/check:
`python The_400_Tasks/Task_by_Task/task263/prototypes/build_worker1_u8_tie_probe.py`
-> `worker1_qconv_default_wzp.onnx not written: Node (output)'s input 5 is marked single but has an empty string in the graph ... Name: output OpType: QLinearConv`.

Why dead: the ONNX checker marks this QLinearConv input as required/single in
the pinned toolchain; optional-input omission is not legal here.

Attempted route 3: lower the model to opset 12 and move `ReduceSum` axes from
the 3-param `axes_nhw` initializer to an attribute.

Exact failed gate:
`python tools/ng_gate.py visible --task 263 --cand The_400_Tasks/Task_by_Task/task263/prototypes/worker1_opset12_axes_attr.onnx`
-> `[visible] ERROR: InvalidGraph: [ONNXRuntimeError] : 10 : INVALID_GRAPH : This is an invalid model. Type Error: Type 'tensor(uint8)' of input parameter (safe_name_49) of operator (Mul) in node (safe_name_50) is invalid.`

Why dead: opset 12 would save the axes initializer, but it also removes the
UINT8 `Mul` support needed by the current QLinearConv polynomial tail. Replacing
that tail to make opset 12 legal costs more than the 3-param axes saving.

One remaining hypothesis: sub-416 likely requires another scalar-lifetime
rewrite around color/starts, or a fundamentally different coordinate primitive
that removes part of `idx30_f` without materializing a selected full-canvas
mask.
