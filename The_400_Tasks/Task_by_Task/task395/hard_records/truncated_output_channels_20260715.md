# task395 hard record: truncated output channels

- live bar: cost/profile 66, sha256 `5535b42173135a7faf900988a0ed66dedf6ad9ffb75349e5b88fed623144943f`, source `python tools/ng_current_bar.py --task 395 --json` / team manifest.
- attempted route: keep Sho's two-Conv signed-code graph but shrink terminal `w_out` from `[10,1,1,1]` to `[3,1,1,1]` and declare graph output `[1,3,30,30]`, since only colors 0 and 2 are emitted.
- candidate: `The_400_Tasks/Task_by_Task/task395/prototypes/build_task395_microopfuse_output3/task395_output3.onnx`, sha256 `3ddef7e3851399f5a404863d29c688d38b01d2809966700c71812193013789ab`.
- cost proof: `spread_score_isolated` cost 59 = memory 36 + params 23; `profile_real_cost --arcgen 266` profile 59; `ng_graph_cost_diff` names the saving as `w_out` shape `[10,1,1,1] -> [3,1,1,1]`, delta -7 params / -7 cost.
- exact failed gate: `python tools/ng_gate.py visible --task 395 --cand The_400_Tasks/Task_by_Task/task395/prototypes/build_task395_microopfuse_output3/task395_output3.onnx` -> `[visible] 0/268 (0.00%)  FAIL`.
- why dead: NeuroGolf visible verification does not accept the 3-channel output representation even though channels 3..9 are semantically zero for this rule. Do not build more `<10 output channel` variants for task395.
- one remaining hypothesis: the only non-rejected path below 66 would need to reduce the 36-byte F32 `signed_code` plane itself, but Conv from F32 input forces F32 output and the current 2-row dilated Conv is already the minimum-param signed-code producer found.
