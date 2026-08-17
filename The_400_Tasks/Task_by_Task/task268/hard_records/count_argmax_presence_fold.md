# task268 - DEAD ROUTE: count ArgMax presence fold

- **Live bar**: cost/profile cost **2455** (memory 2399 + params 56), owner Aditya.
  - Source: `python tools/ng_current_bar.py --task 268 --json`
  - Manifest SHA-256: `ffc4f310eeff4185490461d25e5859ca8740fa03e4159650a36c3728dd99577f`
  - Updated UTC: `2026-06-14T17:43:29Z`

## Attempted route

Delete the four 30-byte bbox presence carrier tensors:

- `row_has` BOOL `[1,30]`
- `row_has_u8` UINT8 `[1,30]`
- `col_has` BOOL `[1,30]`
- `col_has_u8` UINT8 `[1,30]`

The probe rewires the four `ArgMax` bbox nodes directly to the raw F32 count vectors
`row_sum` and `col_sum`, leaving the rest of Aditya's graph unchanged.

Candidate:
`The_400_Tasks/Task_by_Task/task268/prototypes/task268_probe_count_argmax_presence_fold.onnx`

Builder:
`The_400_Tasks/Task_by_Task/task268/prototypes/probe_task268_count_argmax_presence_fold.py`

## Exact failed gate

- `python tools/ng_gate.py visible --task 268 --cand The_400_Tasks/Task_by_Task/task268/prototypes/task268_probe_count_argmax_presence_fold.onnx`
- Result: `[visible] 0/266 (0.00%)  FAIL`

Cheap diagnostics after the visible fail:

- `score_network_v2: (0, 2279, 56) total=2335`
- `ng_graph_cost_diff: NET COST DELTA -120`, entirely from removing the four 30-byte presence tensors.
- Candidate SHA-256: `e7eaaddbee68965c13871312b69811f7e9e9c996c7a5c25fe0c64593fef5829a`

No `profile_real_cost`, `truth`, `ab`, `battery`, or `ng_safety` was run because the visible
gate failed.

## Why it is dead

`row_sum` and `col_sum` are object-pixel count vectors. `ArgMax(row_sum)` returns the first or
last row with the **maximum count**, not the first or last occupied row. For this task, the open
side of the hollow rectangle can have fewer object pixels than the opposite closed side, while
interior side-wall rows can tie the open side count. The bbox endpoints and side-open comparison
therefore become wrong before the geometry stage runs, giving 0/266 visible.

The BOOL presence vector is not just a dtype bridge for `ArgMax`; it is the semantic conversion
from "count" to "occupied row/column". ORT still needs the U8 carrier because BOOL `ArgMax`
is rejected in this environment, as recorded in `orientation_selector_prune.md`.

## One remaining hypothesis

A sub-2455 win still needs either a new final-output family that avoids the 900-byte full-canvas
overlay condition without an F32 crop, or a legal boolean selector identity that removes one
10x10 plane without replacing it with a U8 plane.
