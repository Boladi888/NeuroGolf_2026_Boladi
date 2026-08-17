live bar: 449, sha256 2aa3f29cf7de7a38e49b1e99670f26e20791f085646772e1bf0b8d460cfa8c5f (manifest/current Team_Best).

attempted route: convert task256_shave433 Einsum initializers to sparse tensors: v_red4 only, v_w5+v_red4, and v_w4+v_w5+v_red4.

exact failed gate: builder-side ONNX strict shape inference before scoring/visible. `python The_400_Tasks/Task_by_Task/task256/prototypes/bo_ng_worker1_task256_sparse_probe/build_sparse_probes.py` failed all sparse variants with `ShapeInferenceError` on v_n0/v_n1: sparse Einsum operands are inferred as rank 0, e.g. `Rank of input 1 (0) does not match the equation indices (2)` and `Rank of input 2 (0) does not match the equation indices (1)`.

why dead: the NeuroGolf scorer also runs strict shape inference, so these sparse-initializer Einsum variants cannot reach score/visible; not a viable param-count shave.

one remaining hypothesis: a non-sparse algebraic reduction of the six-feature polynomial basis or the `[4,6,3,3]` coefficient tensor may still lower the floor, but it is a fitted/rewrite problem rather than a legal sparse accounting move.
