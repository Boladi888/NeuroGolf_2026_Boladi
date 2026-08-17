# task144 direct final writer

- live bar: 94 (Team_Best_Onnx/task144.onnx; sha256 a2d4398721869b677b4ef0f035b719726d6ba5621e7cd25ccfab9f96e30f8bd8)
- attempted route: replace Sho's two-Conv `hit -> render` graph with one direct final Conv that writes `[1,10,30,30]` output, avoiding the charged `[1,1,4,4]` hit tensor.
- prototype: `The_400_Tasks/Task_by_Task/task144/prototypes/task144_direct_final_conv_probe.onnx` (sha256 60fd0cf5a505e7bee039bbbf241ea5ed692fd6088b81e8e6fceb11ae0bd303c2)
- exact failed gate: cost/profile gate after visible pass.
  - `python tools/ng_gate.py visible --task 144 --cand The_400_Tasks/Task_by_Task/task144/prototypes/task144_direct_final_conv_probe.onnx` -> `[visible] 267/267 (100.00%)  PASS`
  - `score_network_v2(...)` -> `(0, 0, 210)`
  - `python tools/agent_helpers/profile_real_cost.py --task 144 --cand The_400_Tasks/Task_by_Task/task144/prototypes/task144_direct_final_conv_probe.onnx --arcgen 266` -> `static=210 profile=210 (memory 0 + params 210; 267 real inputs)`
  - `python tools/ng_graph_cost_diff.py --task 144 --live Team_Best_Onnx/task144.onnx --candidate The_400_Tasks/Task_by_Task/task144/prototypes/task144_direct_final_conv_probe.onnx` -> memory -64, params +180, net cost +116.
- why dead: one final Conv removes the charged 64-byte `hit`, but direct writing needs dense `[10,10,2,1]` weights plus bias (210 params) to suppress padding/outside-result rows. It is correct but far above the 94-byte live bar.
- nearby non-route: sparse `W_render` would price at 86 if accepted, but sparse initializers are documented scorer-None under `traps.md` §8e / `core.md` because sanitizer and strict shape inference do not support sparse-fed op inputs.
- one remaining hypothesis: a legal non-sparse way to make the final render kernel param-count by nonzeros, or to produce the 4x4 detector in a sub-F32 dtype without first materializing larger slices, would be the only visible path below 94.
