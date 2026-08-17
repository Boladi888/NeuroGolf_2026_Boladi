# task386 ch0+bias detector no-gain route

- live bar: cost 78, sha256 `9c54eca017da2e3ce23451448e72ac42cbf846b2b13e96e2bad3781978a6cf9e`, source `python tools/ng_current_bar.py --task 386 --json` / team manifest.
- attempted route: make the detector color-agnostic by using only the two channel-0 taps plus a bias of `-1.5`, then reuse the current final Conv output tail.
- probe path: `The_400_Tasks/Task_by_Task/task386/prototypes/task386_ch0_bias_probe.onnx`, sha256 `20cd62b791580200f91b77efc0fc663bbbcca2005fb55db0965978ad158e3680`.
- exact gate/profile lines:
  - `ng_gate visible`: `[visible] 268/268 (100.00%)  PASS`
  - `score_network_v2`: `(0, 48, 31)`
  - `profile_real_cost.py --arcgen 266`: `task386 task386_ch0_bias_probe.onnx: static=79 profile=79 (memory 48 + params 31; 268 real inputs)`
- why dead: it is one byte worse than the 78-byte live graph. The current Sho graph avoids the bias byte by exploiting fixed foreground colors 7 and 5 in `w_hit`.
- one remaining hypothesis: a no-bias color-agnostic detector would need a different threshold carrier in the final free output without adding a 10-element bias vector; no legal operator path found here.
