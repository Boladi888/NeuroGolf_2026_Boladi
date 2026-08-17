# task369 - per-channel final input zero-point refit is dead

- Live bar: cost **982**, sha256 `06b03bcda0ed2309c93c2dee084bb9032a057483332ace744d1ff58b10695b01`, source `python tools/ng_current_bar.py --task 369 --json` / manifest.
- Attempted route: remove the second QLinearConv bias vector `[0,0,-12]`, use unbiased second-state features, and recover the final direct-output classifier with a per-channel final input zero-point `[4,5,21]`. Target static cost was **981** (`memory 900 + params 81`) if ORT accepted it.
- Probe builders:
  - `The_400_Tasks/Task_by_Task/task369/prototypes/build_task369_no_second_bias_refit.py`
  - `The_400_Tasks/Task_by_Task/task369/prototypes/build_task369_no_second_bias_convinteger.py`
- Result artifacts:
  - `The_400_Tasks/Task_by_Task/task369/prototypes/no_second_bias_refit/task369_no_second_bias_refit.onnx`, sha256 `570bfb659de4167143a67eda95925da95a69b4ef36c500e234d7caa2c9a9bd3f`
  - `The_400_Tasks/Task_by_Task/task369/prototypes/no_second_bias_convinteger/task369_no_second_bias_convinteger.onnx`, sha256 `3526b5ea6c11a7040b529bd317e9309c4138a6ef19c46c81bdefa0c894b71e0c`
- Failed gates:
  - QLinearConv final head -> `ng_gate visible` runtime error: `QLinearConv : input zero point must be a scalar or 1D tensor of size 1`.
  - ConvInteger final head -> `ng_gate visible` runtime error: `ConvInteger ... X_Zero_Point ... Must be a scalar or 1D tensor or size 1`.
  - LP screen with a scalar/effective-scalar final zero-point found no feasible separator for the no-second-bias features.
- Why dead: the only found 981 refit requires per-channel **input** zero-points, but ORT 1.24.4's ConvInteger/QLinearConv CPU kernels reject them. A scalar final zero-point cannot separate the unbiased second-state features, so this exact one-byte route is not legal.
- One remaining hypothesis: search for a different second-state feature basis that is scalar-zero-point separable after deleting the second bias, or find a non-Conv final primitive that emits direct 30x30 output without padding-bias leakage.

