# qscale_one1_decode dead route

- Live/team-best bar: 417, sha256 `60d682272b6e1fa5205a380cf5534a623a9636e5fa883cb2cca5739b20c7bb16` (`python tools/ng_current_bar.py --task 355 --json`, authoritative source `manifest`).
- Attempted route: change the position ramp from `exp(2.5 + 4*i)` to a base-e ramp and reuse `one1` as the `QuantizeLinear` scale, deleting `qscale4`; static estimate 416.
- Failed gate: `visible` on `task355_bo_ng_cdx_worker2_qscale_one1.onnx` -> `[visible] 58/267 (21.72%)  FAIL`.
- Why dead: with scale 1, the log-count contribution from row/column population is too large; the quantized position bins no longer recover rectangle split/max coordinates.
- Remaining hypothesis: a scale-sharing log decode would need an independent way to cancel `log(population)`, which costs more than the one scalar initializer it tries to delete.
