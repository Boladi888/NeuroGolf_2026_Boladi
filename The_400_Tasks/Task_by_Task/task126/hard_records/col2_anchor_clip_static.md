# col2_anchor_clip_static dead route

Live/team-best bar: 488 cost/profile cost, sha256
`aef210154e34d5e7ce9b57a6c3f9e2f79aef31777ae0455bd43141aea7ee95a5`,
from `python tools/ng_current_bar.py --task 126 --json`
(`authoritative_source=manifest`, owner Sho, updated 2026-07-09T01:09:17Z).

Attempted route: rescue the prior fixed-column anchor shave by replacing
`ReduceMax(col_bg_count) - 1` with `Gather(col_bg_count, [2], axis=1)` plus
a dummy-safe `Clip(max=29)` before the existing ScatterND tail.

Exact failed gate: cost/profile after visible pass.

Failure lines:
`score_network_v2(...) -> (0, 470, 19), cost 489`
`[visible] 266/266 (100.00%)  PASS`
`task126 task126_col2_anchor_clip.onnx: static=489 profile=489 (memory 470 + params 19; 266 real inputs)`
`NET COST DELTA +1  -> candidate is MORE EXPENSIVE (regression, sign positive).`

Why dead: the cap fixes the scorer dummy out-of-bounds failure from
`col2_anchor_dummy_static`, but the required `max_row_f` scalar adds one counted
parameter. Memory returns to the same 470-byte live bill (`bottom_raw` replaces
`row_count`), so the static/profile cost is 489 and does not beat the 488 bar.

One remaining hypothesis: a future win must remove or narrow the `[2,30]` F32
ScatterND `updates` tensor or the INT64 index path; fixed-anchor row derivation
cannot beat 488 once made scorer-dummy-safe under opset-17 Clip semantics.
