# task202 poly_where_s gate-timeout record

- live bar: cost 128, sha256 `6a39458b1e2f3ab6c9b567c9d0a3d43d11b337af097a1dd2f3ed3adb442bb2d9`, source `python tools/ng_current_bar.py --task 202 --json` / team manifest.
- attempted route: reuse task-local `prototypes/task202_poly_where_s.onnx`, replacing live `Cast(rows_hom_b)->Sub(one-s)->Concat` selector with `Where(rows_hom_b, S_hom, S_het)`.
- local price: static/profile-officials cost 126 (`memory 25 + params 101`), named delta vs live `-8 memory +6 params = -2`.
- exact failed gate: `python tools/ng_gate.py ab --task 202 --cand "The_400_Tasks/Task_by_Task/task202/prototypes/task202_poly_where_s.onnx" --ref Team_Best_Onnx/task202.onnx --n 2000 --json` returned `[ng_gate] wall budget 900s exceeded for ab task202; exiting to avoid orphaned gate work`.
- why dead for this worker pass: task rules say a per-candidate gate wall-budget exceed stops the candidate as gate-timeout/pathological and must not be relaunched with a larger `--max-seconds`; without `ab --n 8000`, this is not reviewer-ready despite visible 230/230 and safety PASS.
- one remaining hypothesis: a reviewer could prove the selector rewrite algebraically or add a cheaper sanctioned A/B proof mode for this unusually slow final-Einsum parent; do not rerun the same `ng_gate ab` ladder as a worker.
