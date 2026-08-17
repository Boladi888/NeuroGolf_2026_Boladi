# inclusive_qtail_yzp

- live bar: 1162, sha `8ded3453e806a64e32691d9de41bea2422d11c25054df8edc6ac528cca81377c` (`ng_current_bar`, manifest, 2026-07-09T01:09:17Z)
- attempted route: delete the `not_red` plane and classify `[red, inclusive_span]` directly in the final `QLinearConv` using per-output `y_zero_point`.
- failed gate: `ng_gate visible` load/runtime error on `task381_inclusive_qtail.onnx`.
- exact failure: `QLinearConv : result zero point must be a scalar or 1D tensor of size 1`.
- why dead: ORT 1.24.4 does not accept per-output result zero-points for this kernel, so the needed per-channel bias cannot be encoded in the free tail.
- one remaining hypothesis: adding a charged bias/constant feature might make the inclusive state classifiable, but the extra 10x10 feature channel costs about the same as the deleted plane and is unlikely to beat live.
