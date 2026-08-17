# task040 sparse-initializers route dead

- Live bar: cost 80, SHA256 `01ba84e2bca7236e35b1b674ebe44e86b809ffb65de9c8e29e2f3c6d39f44127`, source `python tools/ng_current_bar.py --task 40 --json` / manifest owner Vamshi.
- Attempted route: copy the live single-Einsum graph and replace dense `half`, `gate`, `sign` initializers with sparse initializers. First raw sparse names failed visible due sanitizer-dangling names; pre-sanitized names (`safe_name_0/1/2`) fixed the visible runner.
- Exact failed gate: `python tools/agent_helpers/profile_real_cost.py --task 40 --cand The_400_Tasks/Task_by_Task/task040/prototypes/task040_sparse_safe_names.onnx --arcgen 266` -> `ShapeInferenceError: Rank of input 1 (0) does not match the equation indices (2)`. `_score_v2.score_network_v2(...)` returned `None`.
- Why dead: sparse `Einsum` operands can run under ORT and visible passes 266/266, but official full-check/shape-inference treats sparse operands as rank-0, so the candidate is unscoreable. Adding value_info would make the sparse tensors dense charged intermediates, losing to the 80-byte dense live graph.
- One remaining hypothesis: a non-sparse container that gives `Einsum` a rank-2 operand while charging fewer than dense element count would reopen this, but current scorer docs say params are element-count and dtype-blind.
