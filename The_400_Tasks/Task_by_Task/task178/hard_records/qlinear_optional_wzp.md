# qlinear_optional_wzp

- Live bar: manifest cost 462, sha256 `989609792c9127d59ff06536a7e05dad639eb207951a4d405b43b3bbfd17243d`.
- Attempted route: omit the final `QLinearConv` weight zero-point input to delete the scalar INT8 zero initializer.
- Failed gate: `onnx.checker.check_model(full_check=True)` while building `task178_v38_omit_qlinear_wzp.onnx` and `task178_v39_cast_top_bool_omit_qlinear_wzp.onnx`.
- Exact failure: `Node (v_n27)'s input 5 is marked single but has an empty string in the graph`.
- Why dead: this QLinearConv schema/input position is not optional in the checker path, so the model cannot reach score/visible gates.
- Remaining hypothesis: a separate `ConvInteger` or redesigned final decoder might remove scale/zero/bias params, but do not retry blanking QLinearConv input 5.
