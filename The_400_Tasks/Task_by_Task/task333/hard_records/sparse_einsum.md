# sparse_einsum hard record

- Live/team-best bar: cost/profile 635, sha256 `bb14c2aac494c18167f7eab7cc1ac124e94634011c4d2e81da68a8acb1321c61`, source `Team_Submission/Backend/manifest.csv` and workbook row `task333`.
- Attempted route: converted all dense initializers in the one-Einsum team-best graph to sparse initializers to exploit sparse params accounting while preserving the same equation.
- Probe artifact: `The_400_Tasks/Task_by_Task/task333/prototypes/task333_sparse_inits_probe.onnx`.
- Failed gate: static score/checker. `onnx.checker.check_model(full_check=True)` raised `ShapeInferenceError: Rank of input 0 (0) does not match the equation indices (2)` for the Einsum; `score_network_v2` returned `None`.
- Why dead: strict shape inference treats sparse Einsum operands as rank-0/unknown, matching core sparse-initializer warnings. This cannot be made reviewer-ready without leaving the scorer-valid path.
- One remaining hypothesis: only a non-sparse factorization that keeps dense initializers but reduces element count can beat 635; sparse initializer variants should not be retried.
