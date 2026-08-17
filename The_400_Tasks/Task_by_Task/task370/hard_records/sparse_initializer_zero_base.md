# task370 hard record — sparse initializer / zero-base scatter target

- Live bar: cost `3607`, profile `3607`, sha256 `c5a03117601f462483d58aa02b9006f792e53b9c77502fd501481d1ad84d89c0`, source `python tools/ng_current_bar.py --task 370 --json` (`authoritative_source=manifest`, owner Vamshi, updated `2026-07-12T15:15:54Z`).
- Attempted route: replace dense BOOL `zeros_base[400]` ScatterElements base with sparse initializer to drop params from 400 to 1; fallback `ConstantOfShape` zero base; also checked sparse `c_onehot0[10]`.
- Artifacts:
  - `The_400_Tasks/Task_by_Task/task370/prototypes/build_task370_zero_base_probes.py`
  - `The_400_Tasks/Task_by_Task/task370/prototypes/task370_sparse_zero_base.onnx`
  - `The_400_Tasks/Task_by_Task/task370/prototypes/task370_constofshape_zero_base.onnx`
  - `The_400_Tasks/Task_by_Task/task370/prototypes/task370_sparse_c_onehot0_probe.onnx`
- Exact failed gates/results:
  - sparse `zeros_base`: `onnx.checker.check_model(..., full_check=True)` -> `[ShapeInferenceError] (op_type:ScatterElements): data typestr: T, has unsupported type: sparse_tensor(bool)`; `_score_v2.score_network_v2` -> `None`.
  - sparse `zeros_base` plus explicit value_info: checker -> `[TypeInferenceError] type case mismatch. existing=tensor_type inferred=sparse_tensor_type`; score -> `None`.
  - sparse `c_onehot0`: checker -> `(op_type:Einsum): [ShapeInferenceError] Rank of input 1 (0) does not match the equation indices (1)`; with explicit value_info -> same sparse/tensor type-case mismatch; score -> `None`.
  - `ConstantOfShape` zero base: profile -> `task370 task370_constofshape_zero_base.onnx: static=3608 profile=3608 (memory 3508 + params 100; 266 real inputs)`; cost-diff vs static live -> `NET COST DELTA +1`.
- Why dead: ONNX strict shape inference will not treat sparse initializers as ordinary dense tensor inputs for `ScatterElements` or `Einsum`; adding value_info creates a sparse/tensor type conflict. `ConstantOfShape` is legal but merely trades `-399` params for `+400` runtime memory, so it is a +1 regression against the 3607 live bar.
- One remaining hypothesis: only a different render that avoids the dense scatter base entirely could remove this 400-param floor; sparse initializer storage is not a valid route for this graph.
