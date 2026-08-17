live bar: cost 736, sha ecc524227040ec2d9903fa3f0a51ac7aa87d8c86020122493935b5e42ff7b83b (ng_current_bar manifest, 2026-06-22T13:23:24Z).

Attempted route: replace final `encoded10 = bit_hits10 - 1` + `ConvInteger` sign carrier with direct `QLinearConv(bit_hits10, dynamic_w)` using `w_zp=2`, `x_zp=0`, `y_zp=0`, and int32 background bias.

Exact failed gate: `python tools/ng_gate.py visible --task 62 --cand The_400_Tasks/Task_by_Task/task062/prototypes/task062_qlinear_tail.onnx` -> `[visible] 0/267 (0.00%)  FAIL`.

Why dead: the direct QLinearConv output-tail representation does not preserve the live decoded one-hot output even on visible officials. The required background int32 bias violates the free-output QLinearConv bias-leak law: padded output cells also receive the positive background bias. Removing the leak requires materializing a 10x10 QLC result and padding it afterward, which costs more than the 100-byte `encoded10` carrier the route was meant to delete. Per task protocol, visible-zero means no truth/ab/profile follow-up and no variants in this family.

One remaining hypothesis: a non-QLC tail shave may still be possible by reducing the packed-row reflection lookup or scalar descriptor path while keeping the proven `encoded10` + `ConvInteger` sign carrier.
