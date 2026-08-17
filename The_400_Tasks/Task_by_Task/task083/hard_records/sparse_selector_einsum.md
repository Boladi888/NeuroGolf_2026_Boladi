# task083 hard record: sparse selector Einsum

- live bar used: 210, Team_Best/manifest owner Sho, sha256 `292a56e292cb99f0b6376e103ef994a21e8aa2940d6f444f56e613b793bc8a24`
- attempted route: keep Sho's one-node free-output `Einsum('nkij,ip,ap,jq,bq->nkab')`, but store the two selector operands as sparse initializers (`R[30,3]` nnz=6, `C[30,4]` nnz=8) to test whether task083 could price at 14 params instead of 210 dense params.
- prototype: `The_400_Tasks/Task_by_Task/task083/prototypes/build_task083_sparse_selector_probe.py`
- exact failed gate:
  - `python The_400_Tasks/Task_by_Task/task083/prototypes/build_task083_sparse_selector_probe.py`
  - failure: `onnx.onnx_cpp2py_export.shape_inference.InferenceError: [ShapeInferenceError] Inference error(s): (op_type:Einsum): [ShapeInferenceError] Rank of input 1 (0) does not match the equation indices (2).`
  - `python The_400_Tasks/Task_by_Task/task083/prototypes/build_task083_sparse_selector_probe.py --value-info --out The_400_Tasks/Task_by_Task/task083/prototypes/task083_sparse_selector_probe_value_info.onnx`
  - failure: `onnx.onnx_cpp2py_export.shape_inference.InferenceError: [TypeInferenceError] type case mismatch. existing=tensor_type inferred=sparse_tensor_type`
- why dead: sparse-fed `Einsum` does not clear `onnx.checker.check_model(..., full_check=True)`, so it cannot reach `score_v2`, visible, truth, or staging. Adding dense `value_info` does not rescue it; it creates a tensor-vs-sparse type conflict. This matches traps §8e / scoring_internals sparse-initializer guidance.
- one remaining hypothesis: a sub-210 route would need a non-sparse final primitive that encodes both 30-axis symmetric mirror selectors with fewer than the dense `R[30,3] + C[30,4]` params and no charged intermediate; known `Pad`/`Gather`/`GridSample`/crop-then-reshape families are already above 210 or have wrong symmetric-edge semantics.
