# task206 target_nt_shave Path-A dead record

- live bar: manifest cost 976, SHA `c934bd261eb4ec9d8830f53f24d6405eca11df8b7f14632f806cc9ffd5885c4b` (`python tools/ng_current_bar.py --task 206 --json`)
- attempted route: make `target_row_code` / `target_col_code` Einsums emit `[1,1]` directly, delete `target_row_col` and `target_col_col` Unsqueeze nodes, save 2 credited bytes
- candidate: `The_400_Tasks/Task_by_Task/task206/prototypes/task206_target_nt_shave.onnx`, SHA `c1602a11ded6fa4bc5a142360b195a61c27fd90b55390e8726a3a7a4be0a5bb8`, score/profile cost 974
- exact failed gate: `python tools/ng_gate.py ab --task 206 --cand The_400_Tasks/Task_by_Task/task206/prototypes/task206_target_nt_shave.onnx --full` -> `[ab] vs submission.zip: match 200136/200266 | divergences=130 | byte_equal=False (first_div=7954)`
- why dead as local gold: full A/B disproves byte-equality; candidate and parent both score `[truth] vs ARC-GEN ground truth: 199870/200000 (99.935%)`, so this is only a same-truth ambiguous probe, not a Path-A inherited compression
- one remaining hypothesis: find a shape-only way to remove the two Unsqueeze outputs without changing the floating Einsum reduction order, or attack the `[14,4]` INT64 ScatterND index tensor/zero-color duplicate-row floor directly
