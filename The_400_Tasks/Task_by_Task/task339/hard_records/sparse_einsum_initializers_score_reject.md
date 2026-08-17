# task339 hard_record — sparse Einsum initializers are scorer-dead

- **Live bar:** cost **70**, sha256 `bf0fcabec2cba20c26099aab42d337856d787074ce508cc891b9686c9bbcf04e`
  (source: `python tools/ng_current_bar.py --task 339 --json`, manifest owner Sho,
  updated 2026-07-09T01:09:17Z).
- **Attempted route:** keep Sho's one-node free-output `Einsum` exactly, but convert the three
  dense F32 operands (`[10]`, `[30]`, `[30]`) to ONNX `graph.sparse_initializer`.
  Hypothetical params would drop from 70 dense elements to 19 stored nonzeros.
- **Prototype:** `The_400_Tasks/Task_by_Task/task339/prototypes/build_sparse_einsum_init_probe.py`
  produced `task339_sparse_einsum_safe2d_probe.onnx`, sha256
  `511e9ba0cf7d78e6ad26442edab3b41452513597cbc790599f41ef7d025d2ae3`.
- **Exact gates/results:**
  - `python tools/ng_gate.py visible --task 339 --cand The_400_Tasks/Task_by_Task/task339/prototypes/task339_sparse_einsum_safe2d_probe.onnx`
    -> `[visible] 266/266 (100.00%)  PASS`
  - `score_network_v2(...)` -> `None` / no cost.
  - `python tools/agent_helpers/profile_real_cost.py --task 339 --cand ... --arcgen 266`
    -> `ShapeInferenceError: (op_type:Einsum, node name: output): Rank of input 1 (0) does not match the equation indices (1).`
  - `python tools/ng_graph_cost_diff.py --task 339 --live Team_Best_Onnx/task339.onnx --candidate ...`
    -> candidate cost_predictor error with the same sparse-Einsum rank inference failure.
- **Why dead:** raw ORT can execute the pre-sanitized sparse operand graph, but the official
  scorer calls `onnx.checker.check_model(full_check=True)` / strict shape inference before
  scoring. Sparse `Einsum` operands are treated as rank-0/unknown there, so the model scores
  `None` and is unstageable. Adding dense `value_info` would repair rank knowledge only by
  making the dense operand shapes charged as memory, which loses to the 70-param live graph.
- **One remaining hypothesis:** a real sub-70 win would need a different single output-writing
  op or a dense `Einsum` factor reuse that removes one of the three forced dense operands:
  channel suppressor `[10]`, output-row selector `[30]`, and count-threshold column vector `[30]`.
  Any multi-node generated operand is likely worse because replacing params with F32 memory
  costs 4x per element.
