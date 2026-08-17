# sparse_einsum dead route

Live bar used: cost 190, sha256 `b3dda820c5b93cb43f60e655a297068590e5a584d36466d5b0201784d0002861` from `python tools/ng_current_bar.py --task 287 --json` / manifest.

Attempted route: convert the current one-node dense bit-factor `Einsum` to sparse initializers. Expected stored-value cost was 57 = `ch_mask` 9 nnz + three bit maps at 16 nnz each.

Prototype: `The_400_Tasks/Task_by_Task/task287/prototypes/task287_sparse_einsum_v1.onnx`; builder: `The_400_Tasks/Task_by_Task/task287/prototypes/build_task287_worker3_sparse_einsum.py`.

Exact gates:
- builder/checker: `saved ... task287_sparse_einsum_v1.onnx bytes=1663`, then `ShapeInferenceError ... (op_type:Einsum): Rank of input 1 (0) does not match the equation indices (1).`
- visible: `[visible] 267/267 (100.00%)  PASS`
- score: `score_network_v2 -> None None`
- profile_real_cost: fails in `calculate_memory` / `onnx.checker.check_model(full_check=True)` with `Rank of input 1 (0) does not match the equation indices (1).`

Why dead: ORT can execute the sparse `Einsum`, but the official scoring/profile path runs ONNX full-check shape inference, which treats sparse `Einsum` operands as rank 0. No valid cost/profile cost exists, so it cannot be staged.

One remaining hypothesis: adding dense `value_info` for the sparse operands might rescue shape inference, but it would charge at least 760 dense bytes plus 57 params, far above the 190 live bar.
