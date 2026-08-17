# task247 micro-shave dead routes (2026-07-14)

- Live bar used: manifest `249`, owner Sho, sha256 `c72ac4a0e0786d477e28c370dfa0ba0ff7789da2ac2fb2d8d9b939b5c71bed7f` from `python tools/ng_current_bar.py --task 247 --json`.
- Parent graph: `Team_Best_Onnx/task247.onnx`, profile line `task247 task247.onnx: static=249 profile=249 (memory 164 + params 85; 269 real inputs)`.

## Route A: sparse initializers for tiny dense vectors

- Attempted route: convert dense `v_w9` and/or `v_w8` Einsum weight vectors to `graph.sparse_initializer`; separately tested sparse `v_w3` QLinearConv bias.
- Prototype builder: `The_400_Tasks/Task_by_Task/task247/prototypes/build_task247_v10_sparse_einsum_weights.py`.
- Exact failed gate/check:
  - Sparse `v_w9`: `onnx.checker.check_model(model, full_check=True)` fails with `ShapeInferenceError` on `Einsum`: `Rank of input 2 (0) does not match the equation indices (1)`.
  - Sparse `v_w3`: `onnx.checker.check_model(model, full_check=True)` fails with `QLinearConv`: `B typestr: T4, has unsupported type: sparse_tensor(int32)`.
- Why dead: sparse initializer params would save bytes only if shape inference and ORT accepted them as normal dense operands. They do not for the two profitable one-byte-plus targets in this graph, so the candidate cannot reach `score_v2` or `ng_gate`.
- Remaining hypothesis: only a non-sparse algebraic rewrite of the final QLinearConv decoder or the single-Einsum score can beat 249; sparse params are closed here.

## Route B: signed carrier to reuse the INT8 zero scalar

- Attempted route: switch the small UINT8 carrier, row mask basis, and graph output to INT8 so the existing `v_w5` INT8 zero can replace the separate `v_w1` UINT8 zero.
- Prototype: `The_400_Tasks/Task_by_Task/task247/prototypes/task247_v11_int8_zero_reuse_608050F7.onnx`.
- Exact failed gate/check:
  - `python tools/ng_gate.py visible --task 247 --cand The_400_Tasks/Task_by_Task/task247/prototypes/task247_v11_int8_zero_reuse_608050F7.onnx` fails before examples run: `[visible] ERROR: NotImplemented: [ONNXRuntimeError] : 9 : NOT_IMPLEMENTED : Could not find an implementation for Where(16) node with name 'safe_name_16'`.
  - `python -c "from tools._score_v2 import score_network_v2; ...; print(score_network_v2(p))"` returned `None`.
  - `python tools/agent_helpers/profile_real_cost.py --task 247 --cand The_400_Tasks/Task_by_Task/task247/prototypes/task247_v11_int8_zero_reuse_608050F7.onnx --arcgen 266` fails in ORT load: `NOT_IMPLEMENTED : Could not find an implementation for Where(16)`.
- Why dead: ORT CPU lacks `Where(bool,int8,int8)`. Replacing those `Where` nodes with `Cast(mask)->Mul` adds at least 12 B memory to save 1 param, so the family cannot beat 249.
- Remaining hypothesis: a UINT8-zero-point reformulation of the final QLinearConv might remove the INT8 `w_zero_point`, but scalar row fallback becomes unsafe unless a two-value `[z,z^2]` fallback is added, which prices at no win in the current layout.
