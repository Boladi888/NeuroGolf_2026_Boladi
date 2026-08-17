# log_descriptor_no_floor

- Live bar: 311, sha256 `86c81ff4da82187c4fa41897c91857043c0f6be14eddc85efbfbf24ac3ea8c37` from `python tools/ng_current_bar.py --task 224 --json`; effective task-local staged bar is 276, sha256 `e92f777cd20a70d69a854424e8008f0765655c482e7b7e756c6c1f9151aaa8cd`.
- Attempted route: `prototypes/probe_log_descriptor_fold.py` removed the two `Floor` nodes and fed `row_log_tail`/`col_log_tail` directly into the final free-output descriptor basis, expected local cost 260 if valid.
- Failed gate: `python tools/ng_gate.py visible --task 224 --cand The_400_Tasks/Task_by_Task/task224/prototypes/task224_log_descriptor_fold.onnx` -> `[visible] 0/266 (0.00%)  FAIL`.
- Why dead: the final polynomial depends on integer bucketed descriptors; continuous log offsets flip every visible output, so this representation family is a certain hidden zero.
- Remaining hypothesis: only exact integer-descriptor folds are worth testing, e.g. factor/constant-row reductions that preserve `Floor(Log(moment))` semantics without adding descriptor-split memory.
