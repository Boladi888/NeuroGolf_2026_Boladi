# task356 final Equal 10x10 writer dead route

- Live/team-best bar: 1220 profile_cost, SHA `e81463c0cf23b583f80f00c8224100f082a46eeab21f8ea5460ef26e28c3aedc` from `python tools/ng_current_bar.py --task 356 --json` / OnnxlyFans manifest.
- Attempted route: keep the U8 directional-pool mask pipeline, but replace the 30x30 `ConvInteger` final writer with a graph-output `Equal(mask2, channel_values)` of shape `[1,10,10,10]` (`task356_equal10_probe_6D0A929E.onnx`, SHA `6d0a929e9198e7608f72ca92050534142500acd7b61c30ca479826d0b4c0b686`).
- Static cost: `(0, 1200, 19) 1219`, saving 1 byte if the smaller one-hot output contract were accepted.
- Failed gate: `python tools/ng_gate.py visible --task 356 --cand The_400_Tasks/Task_by_Task/task356/prototypes/task356_equal10_probe_6D0A929E.onnx` -> `[visible] 0/266 (0.00%)  FAIL`.
- Why dead: the gate requires the NeuroGolf embedded `[1,10,30,30]` one-hot output contract; a logical 10x10 one-hot graph output is not accepted even though the task grid is 10x10. This is a certain hidden zero, so no truth/ab/profile gates were run.
- One remaining hypothesis: only a legal 30x30 final writer with fewer than the current 12 non-crop params, or a real deletion of a required 100 B U8 span/union tensor, can beat 1220 without changing correctness.
