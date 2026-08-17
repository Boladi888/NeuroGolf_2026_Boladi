# DEAD ROUTE - task174 Where(eqz,Ns,0) without background suppression

**Recorded:** 2026-07-13 by `bo_ng_worker1`.

## Live Bar
- Bar used: 867 profile cost from `python tools/ng_current_bar.py --task 174 --json`.
- Parent/current-best: `Team_Best_Onnx/task174.onnx`
- Parent SHA-256: `64745efcf4d40c5adbe5ad6801204f6466cc835c535593eaa375f5e68834c6fa`

## Attempted Route
Replace `Greater(Ns,thr) -> And(eqz,gt) -> Cast(sel)` with `Where(eqz,Ns,0)`, then change the row/col log locator from base 8 to base 64 so a count-weighted selector still finds the same top-left.

Prototype: `The_400_Tasks/Task_by_Task/task174/prototypes/task174_where_count_base64.onnx`

## Failed Gate
- Passed visible: `[visible] 266/266 (100.00%) PASS`
- Passed profile: `profile=837 (memory 564 + params 273)`, apparent -30 vs live
- Passed capped truth/AB: truth `2000/2000`; AB `2266/2266 byte_equal=True`
- **Failed formal full AB:** `python tools/ng_gate.py ab --task 174 --cand ...task174_where_count_base64.onnx --full`
- Exact failure: ORT `InvalidArgument` in final `Einsum`: operand 4 `E5[5,30]` incompatible with another operand dimension 0.

## Why It Is Dead
Cached corpus index 52163 has a background channel with `D==0` and `Ns=84`. The unsuppressed `Where(eqz,Ns,0)` selector chooses channel 0 by `ArgMax`, causing invalid dynamic Slice starts and a zero-sized crop. This is not a base-size issue; larger bases still select channel 0.

## Remaining Hypothesis
The fixed route is to suppress channel 0 inside `Ns` with a non-background mask and keep the count-weighted base-64 locator. That version is staged as `task174_where_count_base64_nb_cost847`.
