# task045 hard record: sparse basis clone under cost-90 bar

- Live bar: 90 bytes, sha256 `151314a926f7a39e4ab3bef82481b135f2e5382cef065996d7d59c1a6115db84`, source `python tools/ng_current_bar.py --task 45 --json` / manifest owner Vamshi, updated 2026-07-12T15:15:54Z.
- Attempted route: clone `Team_Best_Onnx/task045.onnx` and replace the dense `[3,30]` F32 `basis` initializer with a sparse initializer containing only the 30 nonzero values.
- Prototype builder: `The_400_Tasks/Task_by_Task/task045/prototypes/build_task045_sparse_basis_clone.py`.
- Exact failed gate: build-time `onnx.checker.check_model(model, full_check=True)` failed at `Einsum` shape inference with `Rank of input 3 (0) does not match the equation indices (2)`.
- Why dead: ORT/ONNX shape inference does not provide usable rank info from sparse initializers to `Einsum`; a `Constant(sparse_value)` fallback would materialize a charged `[3,30]` F32 intermediate (360 B) and lose to the 90-byte dense initializer.
- Remaining hypothesis: only a scorer change that accepts sparse initializers as typed dense constants for `Einsum` would revive this route.
