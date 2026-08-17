# task357 hard record — live279 no-mask ConvInteger

- Live bar used: manifest/current_bar cost 279, sha256 `42ebe5e0ea08962858f8bd082f711a22b5996eea1510d58386b7f0e5e5f19d5c` (Vamshi, 2026-07-12T15:15:54Z).
- Attempted route: remove the live graph's `Less(valid_col)` and `Where(valid_col, delta, one)` so the final `ConvInteger` consumes `delta` directly. Prototype: `prototypes/work_task357_probe_nomask_live279.onnx`, sha256 `9e1d844ff98174ef7b2a4d65841fe5071004e3da5db8a328a9d7fdda11f685ed`.
- Static cost if it worked: 169 by `_score_v2.score_network_v2` (memory 137 + params 32), saving 110 bytes vs the 279 live profile/static bar; named saving is deletion of `code` U8 `[1,1,10,10]` and `valid_col` BOOL `[1,1,1,10]`.
- Failed gate: `python tools/ng_gate.py visible --task 357 --cand The_400_Tasks/Task_by_Task/task357/prototypes/work_task357_probe_nomask_live279.onnx` -> `[visible] 1/13 (7.69%)  FAIL`.
- Why dead: padding columns `c >= W` keep positive `delta` values and the final `ConvInteger` decodes them as color 8, but official padding cells must emit all-zero channels. No deeper truth/ab/battery run per visible-fail rule.
- One remaining hypothesis: only a different output family that can test ball/background/padding in one free output op without materializing a second 10x10 plane could beat 279; simple ConvInteger thresholding cannot.
