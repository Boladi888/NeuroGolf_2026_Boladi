# split_basis_9x9_output_contract

- Live bar used: cost 118, SHA-256 `34571a7d5ea37c04e7562fe7ab21d293de3197cf9746aa02dec81bacda25d84c`, source `python tools/ng_current_bar.py --task 104 --json` / team manifest.
- Attempted route: split current coordinate basis into `cin[30,2]` for the 3x3 input detector plus `cout[9,3]` for active output rows/cols; prototype `../prototypes/task104_split_basis_113.onnx`, static cost 113, SHA-256 `1a37e0e266ac378896bdc580290f2c758aa4179ee7651531a3c0e85d9156ae03`.
- Exact failed gate: `python tools/ng_gate.py visible --task 104 --cand The_400_Tasks/Task_by_Task/task104/prototypes/task104_split_basis_113.onnx` -> `[visible] 0/7 (0.00%)  FAIL`.
- Why dead: raw logits match the 118 model on the active 9x9 region, but the harness requires the full `[1,10,30,30]` output contract; producing `[1,10,9,9]` is visible-zero, and padding the 9x9 result would charge an 810-element prepad tensor.
- One remaining hypothesis: a true shared full-output `coord[30,2]` sign basis would still beat 118, but a bounded fit did not find a positive visible-pattern margin.
