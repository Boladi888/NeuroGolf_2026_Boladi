# DEAD ROUTE - task151 QLinear/grouped carrier probes (worker2, 2026-07-13)

Worker: `bo_ng_cdx_worker2`. Verdict: DROP / no staged candidate.

## Live bar
- Authoritative bar: cost 900 from `python tools/ng_current_bar.py --task 151 --json`.
- Source: team manifest, owner Vamshi, updated 2026-07-12T15:15:54Z.
- Parent/live SHA256: `923f0ee2c55298f44b9f8669305997a4277a7143c4a847c20d536b039d0e163f`.
- Parent graph: single `Conv(input, W[10,10,3,3]) -> output`, memory 0, params 900.

## Attempted routes

1. `#84` grouped Conv halving probe:
   - Builder/report: `The_400_Tasks/Task_by_Task/task151/prototypes/build_task151_worker2_probes.py`
   - Candidate: `The_400_Tasks/Task_by_Task/task151/prototypes/task151_group2_conv_bias_probe.onnx`
   - Candidate SHA256: `1a4cc61624787ff7f2c2e9321976ee0812e9ce24de8159a7aa7920087ffac6b2`
   - Static/profile sanity: 460 (`profile_real_cost --arcgen 0`; run after visible fail, not a proof gate)
   - Exact failed gate: `ng_gate visible --task 151 --cand ...task151_group2_conv_bias_probe.onnx` -> `[visible] 0/266 (0.00%)  FAIL`
   - Why dead: the contiguous ONNX `group=2` partition prevents output channels 0-4, especially target color 4, from seeing line colors in input channels 5-9. Patch fitting confirms non-separability: only 1318/1660 unique patches exact, with channel 4 at 1357/1660.

2. Direct QLinearConv dtype patch:
   - Builder/report: `The_400_Tasks/Task_by_Task/task151/prototypes/build_task151_worker2_probes.py`
   - Candidate: `The_400_Tasks/Task_by_Task/task151/prototypes/task151_qlinear_cast_probe.onnx`
   - Candidate SHA256: `e0c2aa5246f96f023619472509d855da20ce1218b571bc321de98c77a7f203f8`
   - Exact gates:
     - `ng_gate visible --task 151 --cand ...task151_qlinear_cast_probe.onnx` -> `[visible] 266/266 (100.00%)  PASS`
     - `profile_real_cost --task 151 --cand ...task151_qlinear_cast_probe.onnx --arcgen 266` -> `static=9916 profile=9916 (memory 9000 + params 916; 266 real inputs)`
     - `_score_v2.score_network_v2(...)` -> `(0, 9000, 916)`
   - Named charged structure: `ng_graph_cost_diff` identifies the new full-frame `input_u8` cast `[1,10,30,30]` as +9000 B, plus `Wq` 900 and `Bq`/scale/zp scalar params. The tool's displayed net is +9010; authoritative score/profile cost is 9916, so the live delta is +9016.
   - Safety: `ng_safety.py` returns WARN with `WARN_X86_QUANTIZED_OP_FAMILY` for `QLinearConv`.
   - Why dead: QLinearConv cannot consume the official F32 input. The required `Cast(input -> UINT8)` materializes the full one-hot frame (9000 B), erasing any dtype benefit and making the probe 11.0x the 900 bar.

## One remaining hypothesis

Only revisit QLinearConv if task151 has an already-paid U8/INT8 carrier before the final output op, or if a future scorer/ORT path accepts F32 input to QLinearConv without a full-frame cast. The current team-best has no such carrier: it is already the one-node, zero-memory Conv floor.

No truth/AB/full gates were run on either losing route.
