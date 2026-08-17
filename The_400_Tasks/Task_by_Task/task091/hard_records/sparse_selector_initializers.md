# sparse_selector_initializers

- live bar: 837 profile cost, sha256 `12a877bf1f3ce4e023155663a1a7fc05447fa132135b8344097d220b8a8d0f29` (`ng_current_bar.py --task 91 --json`, manifest source, Sho, 2026-07-10T09:55:00Z).
- route attempted: convert Sho's sparse binary selector initializers (`E0..E3`, then `E0..E3/e5/Amid/oh0`) to `graph.sparse_initializer`, pre-renamed to sanitizer `safe_name_*`; builder `prototypes/build_task091_sparse_selector_probe.py`.
- failed gate: `score_network_v2` returned `None`; `profile_real_cost.py --task 91 --arcgen 266` fails in `onnx.checker.check_model(full_check=True)` with `ShapeInferenceError`: sparse `Gather`/`Einsum` inputs have unsupported type `sparse_tensor(float)`.
- why dead: raw ORT visible can run sparse initializers (`visible 266/266 PASS`), but the official scorer rejects them before memory/params accounting, so the apparent param drop is uncreditable.
- one remaining hypothesis: a dense non-sparse algebraic rewrite would need to delete a named F32 `[30]` intermediate (`cmask`, `row_any`, or `col_any`) without adding more than 120 params; sparse params are not available.
