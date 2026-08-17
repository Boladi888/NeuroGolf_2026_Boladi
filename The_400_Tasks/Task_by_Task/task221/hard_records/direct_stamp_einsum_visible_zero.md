# task221 direct-stamp Einsum visible-zero record

- Live bar used: 635 from `python tools/ng_current_bar.py --task 221 --json`; manifest SHA `f8029feb04334d79723b2e61c000e0a95c829941f0d956532c3ded628fea7fb2`, owner Vamshi, updated 2026-07-02T02:35:15Z.
- Attempted route: task-local direct-stamp free-output Einsum prototype `prototypes/task221_direct_stamp_einsum.onnx` built by `prototypes/build_direct_stamp_einsum.py`; static score `(macs=0, memory=29, params=305)`, cost 334, SHA `7cf98854b06cd347ff84cc7f8e137dd0fe2e8aa9c6c7c5b40a111fcda5a86435`.
- Failed gate: `python tools/ng_gate.py visible --task 221 --cand The_400_Tasks/Task_by_Task/task221/prototypes/task221_direct_stamp_einsum.onnx` -> `[visible] 0/267 (0.00%)  FAIL`.
- Why dead: this representation does not reproduce any official visible pair, so it is a certain hidden zero and should not receive truth/AB/battery spend.
- One remaining hypothesis: the live-derived sign-channel coefficient route keeps the proven selector geometry and only removes the z-case dimension from `channel_coeff`; continue that route after full-gate capacity returns.
