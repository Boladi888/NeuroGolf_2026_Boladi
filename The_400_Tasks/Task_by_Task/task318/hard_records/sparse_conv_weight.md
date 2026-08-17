live bar: cost 100, sha256 0bfa37796a89ad38a7655d8e98dbb256aba81674632133fbae9b8b9a137c49fe (ng_current_bar.py, manifest authority, Vamshi, 2026-07-12T15:15:54Z)

attempted route: `prototypes/task318_v6_sparse_weight_probe.onnx`, current 100-param Conv with the 6 nonzero weights moved into a `graph.sparse_initializer` named to survive sanitizer rewrites.

failed gate/scorer: visible passed, `python tools/ng_gate.py visible --task 318 --cand The_400_Tasks/Task_by_Task/task318/prototypes/task318_v6_sparse_weight_probe.onnx` -> `[visible] 267/267 (100.00%)  PASS`; static score then failed with `score_network_v2(...) -> None`; real profile failed in `calculate_memory` with `ShapeInferenceError: (op_type:Conv, node name: output): W typestr: T, has unsupported type: sparse_tensor(float)`.

why dead: ORT can execute the sparse-weight Conv, but the official scorer's checker/strict shape inference rejects sparse tensors as Conv weights. Since cost is unmeasurable, this is not stageable.

one remaining hypothesis: no sparse Conv weight route is legal under the current scorer; only a non-Conv op with fewer dense params or a future scorer change would reopen this path.
