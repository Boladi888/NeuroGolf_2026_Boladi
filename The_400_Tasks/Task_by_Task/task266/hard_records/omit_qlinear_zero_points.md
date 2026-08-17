# task266 hard record: omit_qlinear_zero_points

- Live bar used: 144 bytes, manifest source, sha256 `ba939e8f4d4f6ae913e8bdee81490293374ea48e61814e5b8bbf80b6c6b13e9a` (`python tools/ng_current_bar.py --task 266 --json`).
- Attempted route: derive from `Team_Best_Onnx/task266.onnx` and omit the explicit zero `QLinearConv` zero-point initializer `v_w3`, setting QLinearConv inputs 2 and 5 to empty optional inputs.
- Prototype builder: `The_400_Tasks/Task_by_Task/task266/prototypes/build_task266_omit_zero_points.py`.
- Exact failed gate/check: `python The_400_Tasks/Task_by_Task/task266/prototypes/build_task266_omit_zero_points.py`.
- Failure: `onnx.checker.ValidationError: Node (v_n2)'s input 2 is marked single but has an empty string in the graph`.
- Why dead: this ORT/ONNX checker path requires QLinearConv input 2 (`x_zero_point`), so the explicit zero initializer cannot be removed by optional-input elision. Keeping `v_w3` and omitting only weight zero-points does not save params.
- Remaining hypothesis: a one-scalar shared-zero-point rewrite could still reach 143 if a two-feature codebook is found with `x_zero_point == w_zero_point == y_zero_point <= 0`; bounded random/small-codebook search did not find one before timeout, so continue only with a real solver/MILP or a new formulation.
