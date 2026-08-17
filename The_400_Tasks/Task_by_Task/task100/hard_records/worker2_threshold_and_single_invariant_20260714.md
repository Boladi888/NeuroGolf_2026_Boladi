# task100 hard record - threshold and single-invariant routes

- Worker: bo_ng_cdx_worker2 task100
- Live bar: 183 static/profile cost, source `python tools/ng_current_bar.py --task 100 --json`, owner Vamshi, sha256 `ca221eee6f25754161b6d2d14f91d15def8fc33898f32e4babd189bb766cd56f`, current graph `Team_Best_Onnx/task100.onnx`.
- Parent checks:
  - `python tools/profile_real_cost.py --task 100 --cand Team_Best_Onnx/task100.onnx --arcgen 266` -> `task100 task100.onnx: static=183 profile=183 (memory 174 + params 9; 266 real inputs)`
  - `python tools/ng_gate.py visible --task 100 --cand Team_Best_Onnx/task100.onnx --json --no-watch-queue` -> `266/266 (100.0%) PASS`
  - `python tools/ng_gate.py truth --task 100 --cand Team_Best_Onnx/task100.onnx --n 2000 --json --no-watch-queue` -> `1992/2000 (99.6%), first_fail_idx=433`
- Attempted route A: replace `ReduceMax+Equal` with fixed-threshold `Greater(score,t)` to save the 4-byte `mx` tensor and possibly params. Failed pre-build finite generator feasibility before ONNX staging: over valid `task_445eab21` dimensions/placements no `(p,t)` separates winner from loser/background for `score = N*p - Q*p^2`. At current `p=1/48`, winner scores overlap loser/background (`winner min 0.171875`, `loser max 0.3229166667`, `background max 0.3107638889`).
- Attempted route B: count-only/four-only standalone selectors. Count-only ties official-style cases such as 9x5 vs 10x4 (areas 45 vs 40 but both perimeter count 24), and finite checks show count/four/negative-four all misorder valid rectangle pairs.
- Attempted route C: one self-correlation `Einsum` invariant. Exhaustive equality-pattern search for k <= 4 operands found no single invariant that orders bounding-box area over valid generator dimensions.
- Exact failed gate: no ONNX candidate was gated for these routes. The finite generator feasibility checks killed them before `visible`; running `visible` on a known-infeasible family would be burn, and one visible miss would kill the family anyway.
- Why dead: non-byte-equal threshold/single-feature sub-183 families are either finite-infeasible or weaker than the current live parent. The 183 graph already uses the minimal two-feature score plus a compact `Expand`/axis-`Pad` write tail.
- Remaining hypothesis: search k >= 5 self-correlation patterns or find a true byte-equal algebraic compression of the 183 graph that eliminates one charged tensor (`score`, `mx`, or `block`) without adding more params than it saves.
