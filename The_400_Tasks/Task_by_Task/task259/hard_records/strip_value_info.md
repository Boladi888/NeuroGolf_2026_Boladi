# task259 strip_value_info dead route

- Live bar: 907 from `python tools/ng_current_bar.py --task 259 --json`, sha256 `6f3f1ec1a5937f38712575db8f21f6cdbc08d3df0cb2701c221bc986d39d925f`.
- Effective staged bar: 898 from `staged_candidates/task259_bool_tail_pad_axes_cost898/manifest.json`, sha256 `d5f8ce990629e58a08c5a7379dac297debaf864afc14e73a6006a100943c127e`.
- Attempted route: `prototypes/task259_v14_strip_value_info.onnx`, built by `prototypes/build_task259_v14_strip_value_info.py`; exact v11 computation with all `graph.value_info` removed.
- Raw ONNX bytes: 3036; sha256 `ab3f703db2241f456a36e55438c8169f134d24f263a885acd736c959fb333696`.
- Failed gate: `score_network_v2 returned None`; direct official score call returned `score (None, 37)` after `sanitize_model=True` and `ORT_LOAD_OK`.
- Why dead: the official scorer needs value-info/profiler size reconciliation for intermediate memory; stripping annotations makes memory unpriceable instead of free.
- One remaining hypothesis: do not pursue metadata-only cost cuts here; any valid improvement must reduce real profiled tensor bytes or charged initializer params.
