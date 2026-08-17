# task374 hard record - 2-bit packed code + Equal expander loses on cost

- **Live bar**: 1366, sha256 `8b6fdcc9538df7001e3b861cdae1861f27b94ba5dcabfbee31a94584cb90426a`, source `python tools/ng_current_bar.py --task 374 --json` -> authoritative `Team_Submission/Backend/manifest.csv`, owner Vamshi, updated 2026-07-02T02:35:15Z.
- **Task-local reviewer bar**: `staged_candidates/task374_qscale_share_cost1365/task374.onnx`, cost/profile 1365, sha256 `a4b47de87e8f317148931ead2f15f21c0b0479e1824b25f665dd674a230e5e35`.
- **Attempted route**: replace the current two feature planes with one 2-bit `UINT8 [1,1,10,10]` code plane. Bits are `not is_min` and `not is_max`, packed by `BitShift(..., LEFT)` + `BitwiseOr`; classes are bg=0, max=1, min=2, mid=3. A padded U8 identity `QLinearConv` expands the code to `[1,1,30,30]` with no-color sentinel 10, then final `Equal(code30, chan_code)` emits the one-hot output.
- **Prototype**: `The_400_Tasks/Task_by_Task/task374/prototypes/task374_twobit_equal_probe.onnx`, sha256 `ef21fa7cb38a165fbe17f2b2d36d84c074ee393c0a5360ca18de9ef809fc6739`.
- **Builder**: `The_400_Tasks/Task_by_Task/task374/prototypes/build_task374_twobit_equal_probe.py`.
- **Gate results**:
  - `python tools/ng_gate.py visible --task 374 --cand The_400_Tasks/Task_by_Task/task374/prototypes/task374_twobit_equal_probe.onnx` -> `[visible] 267/267 (100.00%)  PASS`
  - `python -c "from tools._score_v2 import score_network_v2; ..."` -> `(0, 2010, 49) 2059`
  - `python tools/agent_helpers/profile_real_cost.py --task 374 --cand The_400_Tasks/Task_by_Task/task374/prototypes/task374_twobit_equal_probe.onnx --arcgen 266` -> `static=2059 profile=2059 (memory 2010 + params 49; 267 real inputs)`
  - `python tools/ng_graph_cost_diff.py --task 374 --live <TEAM_REPO>/Team_Submission/Backend/Team_Best_Onnx/task374.onnx --candidate The_400_Tasks/Task_by_Task/task374/prototypes/task374_twobit_equal_probe.onnx` -> named `+900 B code30`, `-200 B selected_features`, `-200 B masked_features`, net `+694` by the diff tool's predictor.
- **Failed gate**: cost/profile, not correctness. It is +693 vs live 1366 and +694 vs staged 1365 by `score_network_v2`/profile, so no truth/AB/full gate was run.
- **Why dead**: the single-channel code saves only 200 B across `selected_features`/`masked_features`, but any nonlinear one-hot decode needs a charged full 30x30 code canvas. The QLinearConv sentinel expander makes `code30 UINT8 [1,1,30,30]` = 900 B, so the net route loses before hidden gates.
- **One remaining hypothesis**: only a final-output convention that lets a packed 10x10 code expand to `[1,10,30,30]` without a charged 30x30 canvas, or a verified scorer loophole equivalent to that, can beat the current two-channel padded `ConvInteger` family.
