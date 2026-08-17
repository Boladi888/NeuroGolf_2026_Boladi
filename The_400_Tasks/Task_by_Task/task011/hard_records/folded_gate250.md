# task011 folded_gate250

- **Live bar used:** 318, sha256 `df7ddffdefd8d1f168424610c67e41dcbc66e7251fc7d35e61f2aa883f311ad9`, from `python tools/ng_current_bar.py --task 11 --json` (manifest, 2026-07-09T01:09:17Z).
- **Attempted route:** Fold both weighted row/column gate reductions into the terminal free-output `Einsum`, repeating the free graph input three times so `wr4`/`wc4` are no longer charged intermediates. Static/profile cost was 250 (`memory=0`, `params=250`).
- **Exact failed gate:** `python tools/ng_gate.py visible --task 11 --cand The_400_Tasks/Task_by_Task/task011/prototypes/task011_folded_gate250.onnx` -> `[visible] 0/267 (0.00%)  FAIL`.
- **Why dead:** Per `traps.md VISIBLE-FAIL`, a visible miss is a certain hidden zero. The single-node folded-gate representation as built changes the decoded output, despite pricing well.
- **One remaining hypothesis:** Orthogonal dense-basis compression below the safe 282 weighted-gate rung would need to reduce `EXP4`/`OFF4` element count without `sparse_initializer` or a charged generated basis; no such route was found in this pass.
