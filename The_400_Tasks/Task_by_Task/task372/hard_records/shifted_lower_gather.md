# task372 hard record: shifted lower-row Gather

- Live bar: 200 bytes, sha256 `b35beb50d83f1b9a90dbfed5a977cc6c9af9ac8d1b314310950baebbdc0da02e` from `python tools/ng_current_bar.py --task 372 --json`.
- Attempted route: one-node terminal `Gather(axis=2)` with row index `[6,7,...,29,29,29,29,29,29,29]`, cost 30, intended to test whether the lower panel alone was sufficient.
- Prototype: `The_400_Tasks/Task_by_Task/task372/prototypes/task372_probe_shifted_lower_gather.onnx`, sha256 `ba7b295e6dc2eb7a5f3cc06e5e4080dc160e58fb1fc0f48f798ec5bc6160292b`.
- Exact failed gate: `python tools/ng_gate.py visible --task 372 --cand The_400_Tasks/Task_by_Task/task372/prototypes/task372_probe_shifted_lower_gather.onnx` -> `[visible] 0/266 (0.00%)  FAIL`.
- Why dead: fresh ARC-GEN confirms the true rule is not "take lower"; it is `output[r,c] = lower[r+6,c]` only when that lower cell is non-background/non-5, otherwise it preserves the upper cell. The fallback to upper rows is load-bearing on every visible pair, so a single shifted Gather is a visible-zero representation.
- One remaining hypothesis: a sub-200 win would need to preserve the upper-on-background fallback inside one terminal op; any two-node Gather/Where construction charges a full-canvas intermediate and loses.
