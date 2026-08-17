# task356 signed INT8 carrier dead route

- Live/team-best bar: 1220 profile_cost, SHA `e81463c0cf23b583f80f00c8224100f082a46eeab21f8ea5460ef26e28c3aedc` from `python tools/ng_current_bar.py --task 356 --json` / OnnxlyFans manifest.
- Attempted route: quantize the 10x10 channel-8 crop to signed INT8 values {-1, +1}, use signed `Min`/`Max` as the both-sides span logic, and omit the final `ConvInteger` x-zero-point (`task356_signed_int8_probe_525BB7BD.onnx`, SHA `525bb7bdd699dac7f16e5f70cf50e291e821bf91a56d3fda97341a91b81a6f39`).
- Correctness gate: `python tools/ng_gate.py visible --task 356 --cand The_400_Tasks/Task_by_Task/task356/prototypes/task356_signed_int8_probe_525BB7BD.onnx` -> `[visible] 266/266 (100.00%)  PASS`.
- Cost/profile gate: `score_v2` -> `(0, 1200, 20) 1220`; `profile_real_cost --arcgen 266` -> `task356 task356_signed_int8_probe_525BB7BD.onnx: static=1220 profile=1220 (memory 1200 + params 20; 266 real inputs)`.
- Why dead: omitting the final `conv_x_zero_point` saves one param, but the signed `QuantizeLinear` needs an INT8 zero-point initializer, adding the same one param back. The memory ledger is unchanged: F32 crop 400 B plus eight 100 B INT8 direction/span/union planes.
- One remaining hypothesis: beat 1220 only by deleting/fusing one required 100 B direction/span/union tensor, or by finding a legal 30x30 final writer with fewer than the current 12 non-crop writer params while preserving channel-0 background completeness.
