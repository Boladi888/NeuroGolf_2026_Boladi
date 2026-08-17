# qlinear_zero_defaults

Live bar: 354 bytes from `python tools/ng_current_bar.py --task 269 --json`, manifest sha256 `e4e97dc5fbf5b763f6365cdecaf821ddd9bc5b867f2f060d5a00f3d8d83ab6ec`.

Attempted route: remove the scalar `x_zero`/`w_zero` QLinearConv zero-point initializers and rely on default zero-points by passing empty optional inputs.

Failed gate: local builder/checker, before ONNX was emitted:

`onnx.checker.ValidationError: Node (source_u8)'s input 2 is marked single but has an empty string in the graph`

Why dead: in this opset/schema path the QLinearConv zero-point inputs are not accepted as omitted optional inputs by the checker, so the 2-param scalar-zero shave is invalid before runtime.

Remaining hypothesis: only a different quantized op schema or a non-QLinearConv source decoder could remove these scalar zeros; do not retry blank QLinearConv zero-point inputs under ORT/ONNX 1.24.4/1.21.0.
