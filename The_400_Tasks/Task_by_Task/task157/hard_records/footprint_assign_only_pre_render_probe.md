# Hard record - footprint matcher without Scatter pre-render probe (task157, bo_ng_cdx_worker2, 2026-07-07)

## Live bar / state
- Authoritative manifest bar: **6925**, owner Vamshi, sha `17e73ca4e6d94039e27a8da4f0da8dcffadae4fc32a863c000d160b43b0596c6`, updated 2026-07-02T02:35:15Z (`python tools/ng_current_bar.py --task 157 --json`).
- Effective task-local bar to beat before staging: **6470**, `staged_candidates/task157_vamshi_bool3_pad_cost6470/task157.onnx`, sha `165ddccd6190ed6ca963043c0a21beffa103fde2122321be3e24c57170966d4b`.

## Attempted route
Prompt route: build only key extraction plus assignment scorer, no output render, and stop if it already needs the Scatter rank-1 `[64]` index family or costs above the ~3200 target before render.

Probe builder: `prototypes/build_task157_assign_only_probe.py`.
Probe ONNX: `prototypes/task157_assign_only_probe.onnx`, sha `0cb90fff576c8144703966c8abf18969debfa11994aca18f7bb3da30679bfa9a`.

## Exact failed gate
- `_score_v2`: `(0, 4391, 228) 4619`
- `profile_real_cost --arcgen 266`: `task157 task157_assign_only_probe.onnx: static=4619 profile=4619 (memory 4391 + params 228; 265 real inputs)`
- top charged pre-render tensors: `bottom_f` 240B F32, `top_window_cols` 192B INT32, `top_combined_masked` 128B UINT16, `candidate_masks` 128B UINT16, `candidate_masks_flat` 128B UINT16.

## Why it is dead
The matcher-only graph is already **4619** before placement-index expansion, `ScatterElements`, state render, 3-channel one-hot, or output pad. That exceeds the prompt's ~3200 pre-render kill threshold, so a correct rendered solver in this representation cannot reach the requested floor.

## One remaining hypothesis
Only a different assignment formulation that avoids the current 4x2x8 candidate mask grid and `top_window_cols` fanout could reopen the floor. It would need to prove byte-equality to the Vamshi anchor because task157 is ambiguous-class; otherwise it is a learning artifact.
