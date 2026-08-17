# task102 default-zero-point optional-input dead route (2026-07-12)

- Live bar: **1,751** (`memory=1,552`, `params=199`), SHA-256 `5bf47a937a2707dfd0e6d1a6ad772a36be4e8d997e1a7447698b5c2f6e84998a`, from `python tools/ng_current_bar.py --task 102 --json` / backend manifest.
- Attempted route: delete the scalar UINT8 zero initializer by omitting zero-valued optional zero-point inputs from `QuantizeLinear` / `QLinearConv`.
- Exact failed gate: build-time `onnx.shape_inference.infer_shapes(..., strict_mode=True)` before any visible gate.
- Why dead: strict shape inference rejects omitted QLinearConv zero-points in this graph: `input and zero_point pair is expected to have be same type` and downstream type inference becomes null. The official scorer also runs strict shape inference, so this cannot be staged as a 1-param shave.
- One remaining hypothesis: a larger re-quantization that keeps false `red_u8`/`ch5_u8` values unequal to the final `ConvInteger` zero-point might remove the zero scalar, but the current final background channel relies on false feature values being `0` while final `x_zero_point=1`.
