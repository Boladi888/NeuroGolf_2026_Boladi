# task364 - DEAD ROUTE: EdgeBgOnlySentinel

**Worker:** bo_ng_cdx_worker2, 2026-07-15
**Live bar used:** 12,319 from `python tools/ng_current_bar.py --task 364 --json`.
Authoritative source: team manifest, owner Sho, updated 2026-07-09T01:09:17Z.
Live SHA-256: `224d157edef9bbb1f98e83345ca1ed6a44610c97d0b18edbd0e58b45a3119808`.

## Attempted Route

The previous hard record left one weak hypothesis: maybe the edge-validity part of the
background sentinel could be repriced. I tested the smallest clean rewrite:

- keep the outside-grid sentinel machinery and final `Equal` tail unchanged;
- infer row validity from `bg_col0_b` only, not `Or(bg_col0_b, fg_col0_b)`;
- infer column validity from `bg_row0_b` only, not `Or(bg_row0_b, fg_row0_b)`;
- prune the now-dead foreground edge slices and `Or` nodes.

Builder:
`The_400_Tasks/Task_by_Task/task364/prototypes/build_edge_bgonly_variant.py`

Candidate:
`The_400_Tasks/Task_by_Task/task364/prototypes/task364_edge_bgonly.onnx`

Candidate SHA-256:
`e81a83244b3ba01d6f7f05fe59cbfb873c94b914d8a4fd8433ef05f78348e105`

## Priced Delta

`score_network_v2`:

- live: `(0, 12214, 105)`, cost 12,319
- candidate: `(0, 12130, 105)`, cost 12,235
- delta: -84 bytes

`python tools/ng_graph_cost_diff.py --task 364 --live Team_Best_Onnx/task364.onnx --candidate The_400_Tasks/Task_by_Task/task364/prototypes/task364_edge_bgonly.onnx`
named the same -84 memory delta:

- removed `fg_col0_b` BOOL [1,1,20,1]: -20 B
- removed `row_valid20` BOOL [1,1,20,1]: -20 B
- removed `fg_row0_b` BOOL [1,1,1,22]: -22 B
- removed `col_valid22` BOOL [1,1,1,22]: -22 B

Note: `ng_graph_cost_diff` printed stale absolute params/costs (live 12,311,
candidate 12,227), but its structural delta reconciled to -84 and matches
`score_network_v2`/`diff_graphs`.

## Failed Gate

Exact gate:

`python tools/ng_gate.py visible --task 364 --cand The_400_Tasks/Task_by_Task/task364/prototypes/task364_edge_bgonly.onnx`

Result:

`[visible] 35/266 (13.16%)  FAIL`

## Why It Is Dead

The foreground edge slices are load-bearing. The active-grid row/column test cannot use
channel-0 edge pixels alone because a valid top-row or left-column active cell can be
occupied by the color-3 sprite. In that case channel 0 is false, and the live graph's
`Or(..., fg_edge)` is required to distinguish active-grid foreground from outside-grid
padding. Removing the foreground edge half paints some in-grid cells as outside-grid
sentinel, so visible correctness collapses.

Per the visible-fail rule, no truth, AB, profile, battery, or safety gates were run.
This is a `REJECT_LOCAL` probe, not a staged win.

## Remaining Hypothesis

Only a different edge-sentinel formulation could remain, and expected upside is now below
one small edge tensor while risking the same grid-boundary failure. Do not retry the
channel-0-only edge family or adjacent variants without a new proof that preserves
foreground-on-top/left active cells.
