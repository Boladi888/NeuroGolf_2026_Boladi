# where_output_passthrough dead route

Live/team-best bar: 488 cost, sha256
`10a94404614387089c8f0c6f2a4ea399c3ee0b8f857cc9d178be70d0c3313e85`
from `python tools/ng_current_bar.py --task 126 --json` (`manifest`, owner Vamshi,
updated 2026-07-02T02:35:15Z).

Attempted route: build the prompt-requested output passthrough tail:
`output = Where(bottom_row & center_column, color4_onehot, input)`.

Exact failed gate: cost/profile gate after visible pass.

Failure lines:
`[visible] 266/266 (100.00%)  PASS`
`(0, 1148, 55) 1203`
`task126 task126_where_output_probe.onnx: static=1203 profile=1203 (memory 1148 + params 55; 266 real inputs)`

Why dead: the full broadcast condition `cond_4d [1,1,30,30]` costs 900 B by
itself. The current 488 team-best already deleted the old `[1,10,30,30]` delta
and uses ScatterND row-slice patching, so Where-as-output is now dominated.

One remaining hypothesis: a future win must shrink the current ScatterND row-slice
graph itself, especially the `[2,30]` F32 update row or INT64 index path, rather
than materializing a full condition mask.
