# task120 hard record: QLinearConv carrier via input Cast

- Live bar: 910, SHA-256 `13dafeb021b56dac928f38d32b4a0b841fcf481296ab752c9ac203fa425e1acc` (`ng_current_bar.py --task 120 --json`, manifest source).
- Attempted route: exact UINT8 `QLinearConv` rewrite of the live free-output Conv, with `Cast(input F32 -> UINT8)` feeding `QLinearConv`; prototype `The_400_Tasks/Task_by_Task/task120/prototypes/task120_qlinear_cast_probe.onnx`, builder `build_task120_qlinear_cast_probe.py`.
- Gate result: visible passed, `[visible] 266/266 (100.00%)  PASS`; cost/profile failed the bar: `score_network_v2=(0, 9000, 913)`, `profile_real_cost static=9913 profile=9913 (memory 9000 + params 913; 266 real inputs)`.
- Why dead: QLinearConv cannot read the official F32 input directly; the required `input_u8` cast materializes a charged `[1,10,30,30]` UINT8 tensor (9000 bytes). The int8 weights and shared QLinear scalars do not offset that, so the best exact QLinear carrier is `+9003` over the 910 free-output Conv.
- Remaining hypothesis: only a route that keeps the single free-output node and avoids materializing any carrier can compete; known grouped-Conv variants are blocked because output channel 8 cannot read input channels 1..3 under ONNX grouping.
