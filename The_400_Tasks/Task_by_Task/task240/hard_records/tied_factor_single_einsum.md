# task240 — DEAD ROUTE: tied source/output latent factor in single Einsum

- **Live bar at audit:** cost **246**, authoritative source = manifest/current bar.
  sha256 `f74284bfc0e2bdeafe0a843f06c6fbe7e9e4ed0e36b0c7f8a3dfd24b05c1f39c`.
- **Attempted route:** keep the live zero-intermediate single `Einsum` but tie the 30-param output
  latent factor to the transpose view of the 30-param source factor:
  `bohw,hx,wy,xk,yl,pk,kl,ql,rp,sq->bors`. This would reduce params from
  `150 + 30 + 30 + 36 = 246` to `150 + 30 + 36 = 216`.
- **Prototype:** `The_400_Tasks/Task_by_Task/task240/prototypes/task240_tied_src_factor.onnx`,
  sha256 `78d543502a2217f3c7377dcac887c285671d1edc7eb3403a3dcf38a624aafcb0`.
- **Failed gate:** `python tools/ng_gate.py visible --task 240 --cand The_400_Tasks/Task_by_Task/task240/prototypes/task240_tied_src_factor.onnx`
  -> `[visible] 0/266 (0.00%)  FAIL`.

## Why it is dead

The live graph's source and output latent factors are not interchangeable:
`max(abs(src5x6 - out6x5.T)) = 1.8858190774917603`. Tying them destroys the
operator on every visible example even though the static cost would be 216.
Per the visible-fail axiom, this representation family is not a candidate and
must not receive truth/AB/profile/battery gates.

## One remaining hypothesis

Only a genuinely different single-output-node factorization below 246 could beat
the current bar. The already-recorded rank-5 latent refit and this tied-factor
shave both fail before staging, so the remaining credible space is a new algebraic
decomposition that preserves zero intermediates while reducing total initializer
elements without tying `src5x6` and `out6x5`.
