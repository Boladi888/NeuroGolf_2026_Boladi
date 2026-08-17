# col2_anchor_dummy_static dead route

Live/team-best bar: 488 cost/profile cost, sha256
`aef210154e34d5e7ce9b57a6c3f9e2f79aef31777ae0455bd43141aea7ee95a5`,
from `python tools/ng_current_bar.py --task 126 --json`
(`authoritative_source=manifest`, owner Sho, updated 2026-07-09T01:09:17Z).

Attempted route: replace `ReduceMax(col_bg_count) - 1` with
`Gather(col_bg_count, [2], axis=1)`, relying on the ARC-GEN invariant that every
task126 input has the left anchor U-shape centered at column 2. On real inputs,
that column's background count is exactly the bottom row index.

Exact failed gate: score/static dummy path after visible pass.

Failure lines:
`[visible] 266/266 (100.00%)  PASS`
`task126 task126_col2_anchor.onnx: static=None profile=484 (memory 466 + params 18; 266 real inputs)`
`score_network_v2(...) -> None`
`ScatterND ... invalid indice found, indice = 30`

Why dead: `_score_v2` runs a valid all-background dummy with ch0=1 over the full
30x30 canvas. In that dummy, `col_bg_count[:,2] == 30`, so the gathered row index
is out of bounds for ScatterND. A legal cap such as `Min(gathered, 29)` adds back
the scalar/table cost and loses the strict-below-488 profile bar.

One remaining hypothesis: a future win must either find a dummy-safe way to use a
fixed anchor without adding a cap scalar, or remove/narrow the current `[2,30]`
F32 ScatterND updates / INT64 index path directly.
