# task353 hard record: INT8 ScatterND floor is invalid

- Live bar used: cost 238, SHA256 `df5d0635f518a326b888a7b6fa109fceefaa1309f96d384fde208a25d84c3a32`, source `python tools/ng_current_bar.py --task 353 --json` / team manifest, owner Sho, updated 2026-07-09T01:09:17Z.
- Parent profile: `python tools/agent_helpers/profile_real_cost.py --task 353 --cand Team_Best_Onnx/task353.onnx --arcgen 266` -> `static=238 profile=238 (memory 176 + params 62; 271 real inputs)`.
- Attempted route: delete the final `Cast(indices_i -> INT64)` from Sho's graph and feed the INT8 `[4,1,1,4]` index bank directly to `ScatterND`; this would remove the 128 B dominant index tensor and price near the prompt's ~114 floor if legal.
- Prototype: `The_400_Tasks/Task_by_Task/task353/prototypes/task353_int8_scatter_indices.onnx`, builder `build_task353_int8_scatter_probe.py`.
- Exact failed gate: ONNX checker / scorer pre-gate. `onnx.checker.check_model(full_check=True)` -> `[ShapeInferenceError] (op_type:ScatterND, node name: output): indices typestr: tensor(int64), has unsupported type: tensor(int8)`; `score_network_v2` -> `None`.
- Why dead: `ScatterND` indices must infer as INT64 in this scorer stack. INT8/INT32 index floors bypass the required final 128 B INT64 tensor and are not valid candidates.
- Sparse-init side probe: `task353_sparse_updates.onnx`, `task353_sparse_scatter_consts.onnx`, `task353_sparse_selectors.onnx`, and `task353_sparse_all_consts.onnx` all scored `None`; checker rejects sparse `ScatterND` updates, sparse `Concat` operands, and sparse `Einsum` selector operands.
- One remaining hypothesis: only a representation that avoids `ScatterND` entirely while also avoiding a full output-shaped patch/mask can beat 238; rank-shift/free-output routes must first price the dynamic one-step shift basis, because dense row/col shift bases exceed this bar immediately.
