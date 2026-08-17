live bar: 360, sha256 dc7be94781b9f121753f5873a123425927af1b776e722e6092ceb6e2b228839a

Attempted route: convert Sho's sparse free-output Einsum initializer bases (`VH`, `VL`, `G`, `H`, `M`) from dense initializers to ONNX sparse initializers; builder `prototypes/build_task041_v17_sparse_init_probe.py`, candidate `prototypes/task041_v17_sparse_init_probe.onnx`, sha256 `d7f19382fa5ac3e193da0d99cb5856e87f58af56efcacfa140bf6ca4b353f5be`.

Failed gate: static scorer/checker before visible. `score_network_v2: None`; `onnx.checker.check_model(full_check=True)` failed with `ShapeInferenceError: Rank of input 1 (0) does not match the equation indices (2)` for `Einsum`.

Why dead: NeuroGolf strict shape inference treats sparse initializer operands to `Einsum` as rank-0, so the apparent params drop to nnz values is not scorer-valid.

One remaining hypothesis: only a dense-param tensor-network rewrite of the mixed-radix comparator or color map can beat 360 without adding charged intermediates.
