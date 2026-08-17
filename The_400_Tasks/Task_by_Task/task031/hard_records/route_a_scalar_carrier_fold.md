# task031 route-A scalar carrier fold

- Live bar used: 1168, sha256 `98c2f1ef1375125702fe7e62a4c84df22910337a30087ef540767c45ef3cd9a0`, source `python tools/ng_current_bar.py --task 31 --json` / OnnxlyFans manifest.
- Attempted route: fold the dynamic 10-channel `crop_hot` carrier into a scalar UINT8 crop, pad the scalar crop, and make final `output` with `Equal(channel_ids, scalar_padded)`.
- Candidate: `The_400_Tasks/Task_by_Task/task031/prototypes/worker2_scalar_carrier/task031_worker2_scalar_pad_equal.onnx`, sha256 `daffc400919e8ce31af78c5653bead8e9d4a52f7dc2789c9aad92e1a221a6c34`.
- Exact failed gate: `python tools/ng_gate.py visible --task 31 --cand The_400_Tasks/Task_by_Task/task031/prototypes/worker2_scalar_carrier/task031_worker2_scalar_pad_equal.onnx` -> `[visible] 0/266 (0.00%)  FAIL`.
- Why dead: zero scalar padding becomes channel-0 background after final `Equal`, so off-grid output padding is wrong. Per the visible-fail rule, no truth/profile/AB was run. The current team-best largest charged tensors are the forced 10x10 ch0 support read (`bg`, 400 B), `fg_b`/`fg_u8` support masks (100 B each), and small row/col/pad carriers; worker3 already showed bool-reduce only reaches profile 1410 and support-window shrink variants visible-fail.
- One remaining hypothesis: a sentinel-10 scalar pad would fix this specific failure, but it adds a charged `[1,1,30,30]` scalar carrier (900 B) and is unlikely to beat the 1168 bar unless paired with a separate real support-carrier fold; otherwise the remaining lever is a route-B rewrite.
