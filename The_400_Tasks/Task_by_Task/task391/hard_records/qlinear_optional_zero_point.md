# qlinear_optional_zero_point

- Live bar: 126 profile cost, parent sha `c0ffda1b24ce17155e25d83681ae0c4f99bfc98eb7248d0bae122658525221a2` (`ng_current_bar.py`, manifest source).
- Attempted route: remove the shared scalar uint8 zero initializer by setting `QLinearConv` zero-point inputs 2, 5, and 7 to empty optional inputs (`build_v7_omit_qlinear_zp.py`).
- Exact failed gate: `python The_400_Tasks/Task_by_Task/task391/prototypes/build_v7_omit_qlinear_zp.py` failed in `onnx.checker.check_model`: `Node (v_n3)'s input 2 is marked single but has an empty string in the graph`.
- Why dead: this ONNX schema/checker path treats those `QLinearConv` zero-point inputs as required, so the representation cannot reach scoring or visible gates.
- Remaining hypothesis: only reuse an existing nonzero zero-point value when it preserves the arithmetic; v8 uses `x_zero_point = 2` and keeps the required zero-point inputs populated.
