# task180 sparse Conv bias dead route

- Live/team-best bar used: 210 bytes, SHA256 `40910c472a459cc9c5bb649650df59db214116c5b2f43029040bd721b00c8ac3` from `python tools/ng_current_bar.py --task 180 --json` on 2026-07-13.
- Attempted route: keep the exact parent `Conv(group=2)` weight tensor `W[10,5,2,2]` (200 params) and replace dense `B[10]` with `graph.sparse_initializer` values at indices `[0,4]`, theoretical cost 202 if accepted.
- Probe artifact: `The_400_Tasks/Task_by_Task/task180/prototypes/task180_sparse_bias_probe.onnx`, built by `prototypes/probe_task180_sparse_bias.py`.
- Exact failed gate: `onnx.checker.check_model(model, full_check=True)` fails with `[ShapeInferenceError] (op_type:Conv): B typestr: T, has unsupported type: sparse_tensor(float)`.
- Scorer/safety confirmation: `score_network_v2` returns `None`; `ng_safety.py model --task 180 --cand ... --json` returns `verdict: FAIL` with `FAIL_ORT_LOAD_AFTER_SANITIZE` because the sanitized node input `safe_name_1` is not a graph input, initializer, or previous output.
- Why dead: sparse Conv bias cannot be consumed as a legal tensor input by the official checker/scorer, and sanitizer does not preserve the sparse initializer reference. Dense full-length bias remains mandatory for the accepted grouped-Conv formulation.
- One remaining hypothesis: only a non-Conv final op with a native constant/bias mechanism below 210 params could beat the bar; every sparse-initializer-as-op-input variant should be treated as closed for this task.
