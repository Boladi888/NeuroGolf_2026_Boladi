# task372 hard record: channel-drop direct

- Live bar: 200 bytes, sha256 `b35beb50d83f1b9a90dbfed5a977cc6c9af9ac8d1b314310950baebbdc0da02e` from `python tools/ng_current_bar.py --task 372 --json`.
- Attempted route: single terminal Conv with output channel 5 removed, weight shape `[9,10,2,1]`, cost/profile cost 180, sha256 `d084b5c7ea6515ce8a41fd727b4c641eee2b0c8a02d8824df0b4a327ee5e9d9b`.
- Exact failed gate: `python tools/ng_gate.py visible --task 372 --cand The_400_Tasks/Task_by_Task/task372/prototypes/task372_probe_drop_channel5_direct.onnx` -> `[visible] 0/266 (0.00%)  FAIL`.
- Why dead: even though ARC-GEN never emits color 5 as output, the gate compares full `[1,10,30,30]` one-hot tensors; a `[1,9,30,30]` graph output is shape-wrong. Restoring the missing middle channel with Split/Concat/Pad would make the 9-channel Conv output a charged full-canvas intermediate, far above the 200-byte bar.
- One remaining hypothesis: only a terminal op that preserves the 10-channel ABI with fewer than 200 charged elements could win; channel truncation itself is not viable.
