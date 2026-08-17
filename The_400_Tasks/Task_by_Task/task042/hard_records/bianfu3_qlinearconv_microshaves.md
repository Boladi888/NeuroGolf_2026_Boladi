# bianfu3 QLinearConv micro-shave probes

- **Live bar when recorded:** manifest/profile cost **1813**, SHA `cdfd71e0fbef2fde3f7b44be3668817c16aae450cb19219e0cf6ee02497c5b15`, source `python tools/ng_current_bar.py --task 42 --json` / `Team_Best_Onnx/task042.onnx`, owner Sho, updated `2026-07-09T01:09:17Z`.
- **Attempted route:** small parent-preserving shaves on Sho's `bianfu3` graph: optional QLinearConv zero-points, signed orientation collapse for n=1/n=2, and trimming apparent all-zero detector rows.
- **Exact failed gates / rejects:**
  - `build_task042_omit_zero_zp.py`: build-time checker reject, `Node (v_n2)'s input 2 is marked single but has an empty string in the graph`; QLinearConv zero-points are required in opset 18, so the shared uint8(0) scalar cannot be omitted.
  - `task042_signed_n1.onnx`: profile probe `static=1710 profile=1710 (memory 1304 + params 406; 266 real inputs)`, then visible `[visible] 192/266 (72.18%)  FAIL`.
  - `task042_signed_n2.onnx`: profile probe `static=1680 profile=1680 (memory 1336 + params 344; 266 real inputs)`, then visible `[visible] 247/266 (92.86%)  FAIL`.
  - `task042_signed_n1_n2.onnx`: built but not gated because signed n=1 failed visible; per visible-fail rule, no combined-family gate was run.
  - `task042_trim_detector_rows.onnx`: visible `[visible] 196/266 (73.68%)  FAIL`.
  - `task042_polish_team_best.onnx`: polish no-op, `original cost: 1813`, `polished cost: 1813  (-0 bytes)`.
- **Why dead:** the n=1/n=2 two-channel detector planes are load-bearing. Sho's n=3 signed plane works because its signed orientation encoding does not create visible cancellation cases, but the same collapse on n=1/n=2 does. The apparent leading zero rows in `v_w6`/`v_w7` are also not neutral: with QLinearConv weight zero-point 64, zero-valued weights are negative absence checks.
- **One remaining hypothesis:** a hand-derived alternative first-stage n=1 detector that shrinks one spatial axis while preserving edge cases might save a few bytes, but the visible failures show the negative context mask is delicate. Do not pursue without first enumerating generator edge cases and proving the pair at the clipped border is impossible.
