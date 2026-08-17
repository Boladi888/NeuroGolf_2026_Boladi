# task316 hard record - QLinearConv zero-point omission

- Live bar: 307 profile cost, SHA `c6d2f15d567f1b7243d8cd23ada653d32dedc2dcc4ab548c9946e5c502b567e2` (manifest, Sho, 2026-07-09).
- Attempted route: omit explicit zero-point optional inputs from the final `QLinearConv` and the rank `ConvInteger`, then delete the shared scalar `zero_i8` initializer for a 1-byte param shave.
- Failed gate: build/check. `python The_400_Tasks/Task_by_Task/task316/prototypes/build_task316_worker3_probes.py` failed `task316_no_explicit_zero_points.onnx` during strict shape inference with `[ShapeInferenceError] ... (op_type:QLinearConv): [TypeInferenceError] input and zero_point pair is expected to have be same type in node QLinearConv.`
- Why dead: in this ORT/ONNX stack, keeping the bias in input slot 8 while omitting the middle zero-point inputs is not shape-inference legal. A runtime zero scalar would only trade the 1 param for at least 1 byte of scalar memory, so it cannot beat the bar.
- Remaining hypothesis: only a different final decoder that removes the quadratic QLinearConv bias/scale family entirely can save more than this scalar; placeholder omission is closed.
