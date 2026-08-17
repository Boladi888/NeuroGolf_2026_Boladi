# task279 hard record: pad_attr_opset10

Live/team bar: 5,057, sha256 `4f9a0651cbb5bf227feb6d794dab4e84df585bd83cdfb7fa1a7e2928c70574f9`.

Attempted route: mutate Vamshi QLinear graph from opset 18 to opset 10 so the final BOOL Pad can use
attribute pads instead of `pad_spatial` + `pad_axes`, saving 6 params if legal.

Exact failed gate: build/checker before score.

Failure line:
`ShapeInferenceError] (op_type:Min, node name: flood_gate_0): data_0 typestr: T, has unsupported type: tensor(uint8)`.

Why dead: the old Pad attribute schema requires lowering the whole ai.onnx opset, but the graph relies on U8 `Min`
for the gated flood. U8 `Min` is not accepted at opset 10, so this cannot be a legal scorer candidate.

One remaining hypothesis: only revisit if a per-node opset split becomes scorer-legal; otherwise Pad-18 axes is already the
lowest-param legal Pad form for the current U8-Min graph.
