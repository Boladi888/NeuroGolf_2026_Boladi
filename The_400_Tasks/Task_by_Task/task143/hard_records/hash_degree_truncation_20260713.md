# hash_degree_truncation_20260713

- Live bar: manifest cost 502, SHA `363cf49fdad3b06148b7d66276dede93e9db72e873e532bb5e94fc3724dd9abc`, owner Boladi, refreshed with `python tools/ng_current_bar.py --task 143 --json` on 2026-07-13.
- Attempted route: truncate the current combined hash basis from degree-4 columns (`hpows[:,0:4]` with 4x4 coefficient matrices) to degree-3 columns and 3x3 coefficient matrices.
- Exact failed gate: `python tools/ng_gate.py visible --task 143 --cand The_400_Tasks/Task_by_Task/task143/prototypes/task143_worker1_hash_degree3.onnx` returned `[visible] 0/266 (0.00%)  FAIL`.
- Why dead: the degree-3 hash no longer preserves the shape equality invariant; the lower degree-2 sibling was not gated after the degree-3 visible zero.
- One remaining hypothesis: a different invariant, not a prefix truncation of the current polynomial basis, might still lower `hpows`/coefficient params, but it needs an offline collision proof before visible.
