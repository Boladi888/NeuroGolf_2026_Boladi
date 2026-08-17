# sparse_einsum_banks

- live bar: 5,145 from `python tools/ng_current_bar.py --task 110 --json`, sha `d84c238027170cd8bfa1c7e6a56c7eb87a352aba8a5d3c8aa92f5c127811fc6d`; current task-local reviewer bar before this pass is 3,931 from `staged_candidates/task110_raggedbank_f16det_cost3931/manifest.json`, sha `6aaee80304bd51b28118a56cd9eaddfba439972abdb6359cdb6a394ad9148a55`.
- attempted route: convert sparse 0/1 render banks `v_Brow` (`174` nonzeros in `[39,30]`) and `v_Bcol` (`58` nonzeros in `[38,30]`) from dense initializers to ONNX sparse initializers, preserving the same final `Einsum`.
- exact failed gate: `python The_400_Tasks/Task_by_Task/task110/prototypes/bo_ng_cdx_worker2/build_sparse_banks.py` -> `onnx.onnx_cpp2py_export.shape_inference.InferenceError: [ShapeInferenceError] Inference error(s): (op_type:Einsum, node name: v_n69): [ShapeInferenceError] Rank of input 0 (0) does not match the equation indices (2).`
- why dead: official full-check/strict shape inference does not expose sparse initializer rank to `Einsum`, so no scorer-valid candidate is produced. Adding dense `value_info` to force the rank would charge the bank as memory and lose the param saving.
- one remaining hypothesis: only revisit if ONNX/ORT scorer support for sparse `Einsum` initializers changes, or if a non-`Einsum` consumer accepts sparse banks without adding dense value_info.
