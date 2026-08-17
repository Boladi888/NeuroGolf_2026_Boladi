# task030 QLinearConv optional zero-point dead route

- Live bar used: 1054 profile cost from `python tools/ng_current_bar.py --task 30 --json`; authoritative source `manifest`; live SHA-256 `88cbb94de8c671d33eef5aefee30ea2224755632b487d15dc5f68528598835e8`.
- Attempted route: remove the 1-param `zero_u8` initializer by omitting `x_zero_point` and/or `y_zero_point` from the final `QLinearConv`, relying on default zero semantics.
- Exact failed gate: pre-build ONNX schema check:

```text
onnx.defs.get_schema("QLinearConv", 18).inputs
x_zero_point: FormalParameterOption.Single
w_zero_point: FormalParameterOption.Single
y_zero_point: FormalParameterOption.Single
```

- Why dead: in opset 18 these zero-point inputs are required, not optional. Building blank optional inputs would be checker/schema invalid, and replacing the initializer with a generated zero would trade the 1-param initializer for an equal or larger runtime tensor.
- One remaining hypothesis: keep `zero_u8` and look for runtime-tensor removals instead; the successful `bottom_valid Where` route removed `height5_u8` without touching QLinearConv inputs.
