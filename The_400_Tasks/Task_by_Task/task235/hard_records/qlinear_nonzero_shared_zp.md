# qlinear_nonzero_shared_zp

- Live bar: cost 112, sha256 `427fc827b730eba035e3f0b38dca9124c26e9b3b4e730f8c4d2d703bd0d41cbe` from `python tools/ng_current_bar.py --task 235 --json` / manifest.
- Attempted route: convert final `q_w` to UINT8, use one nonzero UINT8 zero-point (`3`) for QLinearConv x/w/y, and shift `q_b` to preserve the parent polynomial.
- Failed gate: `python tools/ng_gate.py visible --task 235 --cand The_400_Tasks/Task_by_Task/task235/prototypes/task235_w2_shared_u8_zp.onnx` -> `[visible] 0/69 (0.00%)  FAIL`.
- Why dead: final `QLinearConv` uses numeric zero in padded windows; with nonzero x/y zero-points those padded windows produce positive off-answer logits across the 30x30 output, so the family leaks outside the 3x3 answer.
- Remaining hypothesis: zero-valued shared zero-points are still viable if transport, weights, and output have the same signed dtype; this is the staged INT8 transport candidate.
