# DEAD ROUTE - task152 sparse-B sub-60 probe

- **Live bar at test:** cost **60**, sha256 `4f3a8da74d042fc39dfb17b8ae9d4c2ec629ff1aa50b1f7c2f9ce280c3d0e88f` (manifest/team-best, owner Vamshi, updated 2026-07-12T15:15:54Z).
- **Attempted route:** clone the current single free-output Einsum and replace dense shared `B[30,2]` (60 params, 12 nonzeros) with a sparse initializer so params would charge only the nonzero values.
- **Prototype paths:**
  - `The_400_Tasks/Task_by_Task/task152/prototypes/task152_sparse_b_probe.onnx`
  - `The_400_Tasks/Task_by_Task/task152/prototypes/task152_sparse_b_with_vi_probe.onnx`
  - builder: `The_400_Tasks/Task_by_Task/task152/prototypes/build_task152_sparse_b_probe.py`
- **Exact failed gate/check:**
  - `python tools/ng_gate.py visible --task 152 --cand The_400_Tasks/Task_by_Task/task152/prototypes/task152_sparse_b_probe.onnx` fails before evaluation:
    `Invalid model. Node input 'safe_name_0' is not a graph input, initializer, or output of a previous node.`
  - `score_network_v2('.../task152_sparse_b_probe.onnx') -> None`
  - `onnx.checker.check_model(..., full_check=True)` and strict shape inference fail:
    `Rank of input 1 (0) does not match the equation indices (2).`
  - Adding dense `value_info` for `B` does not salvage it:
    `score_network_v2('.../task152_sparse_b_with_vi_probe.onnx') -> None`;
    checker/shape inference fail with
    `type case mismatch. existing=tensor_type inferred=sparse_tensor_type`.
- **Why it is dead:** ORT can load the raw sparse-initializer model, but `ng_gate`/sanitizer does not preserve the sparse initializer reference, and the NeuroGolf scorer uses ONNX checker/strict shape inference before scoring. Sparse tensors feeding `Einsum` are not inferred as dense rank-2 inputs, and annotating them as dense causes a type-case conflict. Dense `B[30,2]` remains the valid scored representation; sparse initializer accounting cannot be used here.
- **One remaining hypothesis:** a true sub-60 route would need a different single-output op family or a way to express the 3-class mirror equality with less than one dense `[30,2]` term axis without materializing a map/output predecessor. A scalar `[30,1]` basis gives only one monomial inside a single Einsum; any multi-term polynomial needs a term axis or charged generated intermediates, which returns to at least the current 60-param shape.
