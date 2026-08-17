# task350 shared-z tail fold dead route

- Live bar: 533, SHA-256 `4303e186062a423a999d6323dcc90aa58de1249380ab8f1db3fff82b7795babc` (`python tools/ng_current_bar.py --task 350 --json`, team manifest / Team_Best).
- Attempted route: in `task350_dense_channel_fold.onnx`, fold `G(t,z) * A2(z,k)` into one `H(t,k)` inside the terminal free-output `Einsum`, combined with an input-channel selector fold.
- Exact failed gate: `python tools/ng_gate.py visible --task 350 --cand The_400_Tasks/Task_by_Task/task350/prototypes/task350_dense_channel_fold.onnx` returned `[visible] 0/267 (0.00%)  FAIL`.
- Why dead: the fold was algebraically invalid because `z` is also used by the live input selector operands `zJ,Jb`; it is not a constant-only tail axis.
- One remaining hypothesis: channel-support folding without touching the shared `z` tail may still be valid; do not retry `G*A2 -> H` unless the `z` dependency is first eliminated another way.
