# task145 opset-10 attribute param-shave route

- Live bar used: cost **12855**, SHA-256 `2dbfb6f9d5ea294d2e0be0fbfd5fbb2460a1e249de794bd3bb31f95a48b4f883`, source `python tools/ng_current_bar.py --task 145 --json` authoritative_source=`manifest`.
- Attempted route: downgrade Sho/Vamshi's graph to opset 10 and move `ReduceMax`/`ReduceMin` axes plus `Pad` pads/value from initializers into node attributes, aiming to delete `axes_hw` and `core_pads` params. Builder: `The_400_Tasks/Task_by_Task/task145/prototypes/build_task145_opset10_attr_params.py`.
- Exact failed gate: builder checker/shape-inference failed before a candidate ONNX was saved:
  `ShapeInferenceError (op_type:MaxPool, node name: left_marker): X typestr: T, has unsupported type: tensor(uint8)`.
- Why dead: the current floor depends on U8 `MaxPool` marker scans, and that kernel/schema is not accepted at opset 10. Newer opsets accept the U8 scans but require input-form `Pad`/reduction axes, so the intended initializer deletion is unavailable.
- One remaining hypothesis: a different accepted opset/signature combination that supports U8 `MaxPool`, attribute-form reductions, and attribute-form U8 pad simultaneously; current ONNX schemas do not expose that combination.
