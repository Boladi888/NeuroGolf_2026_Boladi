# task108 shifted-free-gate AB8000 timeout

- Live bar: cost 150, sha256 `8dfa6aeacd8d07fa91a5ea79895e7246064d03a81ddf591b1c455b068b85a82c`, source `python tools/ng_current_bar.py --task 108 --json` / team manifest, owner Vamshi, updated 2026-07-12T15:15:54Z.
- Attempted route: shift the current 150 one-node Einsum codes (`input_features[0]-=0.5`, `output_features[1]+=0.5`) so the nonselected input code is `-1`, then replace the 2-param `input_gate` operand with free `sum(input_features[:,h])` and `sum(input_features[:,w])`.
- Candidate: `The_400_Tasks/Task_by_Task/task108/prototypes/task108_shifted_free_gate.onnx`, sha256 `a91653dad782a9a0b9aaf9a1f6e4b1ac8f5e1857e2002bb153cc08f37f87e169`, static/profile cost 148.
- Passed gates/tools before timeout: `ng_gate visible` = `[visible] 266/266 (100.00%)  PASS`; `profile_real_cost.py --arcgen 266` = `static=148 profile=148`; `ng_safety.py` = `PASS` with 0 warnings/failures.
- Exact failed gate: `python tools/ng_gate.py ab --task 108 --cand The_400_Tasks/Task_by_Task/task108/prototypes/task108_shifted_free_gate.onnx --ref Team_Best_Onnx/task108.onnx --n 8000` -> `[ng_gate] wall budget 900s exceeded for ab task108; exiting to avoid orphaned gate work`.
- Why this is not reviewer-ready: Path-A byte-equality was intended algebraically but the required `ab --n 8000` proof did not complete under the shared gate wall budget; do not relaunch the same command with larger `ng_gate --max-seconds`.
- One remaining hypothesis: a reviewer can validate the algebraic sign-flip proof or run an approved equivalent proof path; the named cost delta is only removal of `input_gate` (-2 params), with no memory/profile gap.
