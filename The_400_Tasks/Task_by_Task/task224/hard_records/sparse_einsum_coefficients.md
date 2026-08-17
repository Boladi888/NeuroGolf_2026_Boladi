# sparse_einsum_coefficients

- Live bar: 311, sha256 `86c81ff4da82187c4fa41897c91857043c0f6be14eddc85efbfbf24ac3ea8c37` from `python tools/ng_current_bar.py --task 224 --json`; effective task-local staged bar is 276, sha256 `e92f777cd20a70d69a854424e8008f0765655c482e7b7e756c6c1f9151aaa8cd`.
- Attempted route: `prototypes/probe_sparse_coefficients.py` converted the mostly-zero `branch_coeff`, `p_factor`, `q_factor`, `g_coeff`, and `h_coeff` initializers to ONNX `sparse_initializer` entries while keeping the 276 graph's dense `Einsum` equations.
- Failed gate: builder/checker path before any score/visible gate: `python The_400_Tasks/Task_by_Task/task224/prototypes/probe_sparse_coefficients.py` -> `ShapeInferenceError` at node `output`: `Rank of input 2 (0) does not match the equation indices (3)`.
- Why dead: ONNX shape inference does not expose sparse `Einsum` initializers with their dense rank, so the official scorer path cannot accept this representation family.
- Remaining hypothesis: only dense exact coefficient rewrites are worth testing; any sparse route would need a different consumer op with proven `check_model(full_check=True)` and scorer support.
