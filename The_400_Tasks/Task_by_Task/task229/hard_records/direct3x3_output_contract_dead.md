# task229 hard record: direct 3x3 output contract

- Live bar: cost/profile cost 231, sha256 `61c31d0bc6b36f45b6806bf23c12f7a6a9b6cf6cabd441504c609fb6f0cd45ad`, source `python tools/ng_current_bar.py --task 229 --json` / team manifest.
- Attempted route: delete the final Pad from Vamshi's count -> ArgMax -> dynamic Slice -> Where -> Equal-tail graph, expand channel ids to `[0..9]`, and make the Equal-tail `[1,10,3,3]` the graph output.
- Probe artifact: `The_400_Tasks/Task_by_Task/task229/prototypes/bo_ng_worker1_direct3x3_tail.onnx`, static cost 143, sha256 `bbb0f762b4b8746e25b434fcfb7922cf1d2482ee6c5c7e669452541f3fcaf609`.
- Failed gate: `python tools/ng_gate.py visible --task 229 --cand The_400_Tasks/Task_by_Task/task229/prototypes/bo_ng_worker1_direct3x3_tail.onnx` -> `[visible] 0/267 (0.00%)  FAIL`.
- Why dead: task229 gates require a full `[1,10,30,30]` output tensor with only the top-left 3x3 populated; direct `[1,10,3,3]` logical-output models decode/compare as visible-zero.
- Do not rerun: no-Pad / no-canvas direct 3x3 graph-output variants.
- Remaining hypothesis: only full-output-contract shaves remain, likely tiny index/scalar reductions around the current 231 graph; worker3 already found axes-attr, Pad-axes, channel-0-inclusion, and OneHot-tail families do not beat 231.
