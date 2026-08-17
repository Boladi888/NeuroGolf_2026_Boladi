# task268 - DEAD ROUTE: geometry U8 index share

- **Live bar**: cost/profile cost **2455** (memory 2399 + params 56), owner Aditya.
  - Source: `python tools/ng_current_bar.py --task 268 --json`
  - Manifest SHA-256: `ffc4f310eeff4185490461d25e5859ca8740fa03e4159650a36c3728dd99577f`
  - Updated UTC: `2026-06-14T17:43:29Z`

## Attempted route

Prompt lever: share the nine 10x10 BOOL geometry/selector planes as a U8 index and recover the
selected fill mask with `Equal`/U8 selection, while also checking whether the F32 marginals can be
narrowed.

## Exact failed gate

The only correctness-preserving concrete U8-index selector probe is:
`The_400_Tasks/Task_by_Task/task268/prototypes/task268_probe_u8_where_selector.onnx`

Gate lines from this pass:
- `score_network_v2: (0, 2399, 56) total=2455`
- `profile_real_cost --task 268 --arcgen 266: static=2455 profile=2455 (memory 2399 + params 56; 266 real inputs)`
- `ng_gate visible: [visible] 266/266 (100.00%)  PASS`
- `ng_graph_cost_diff: NET COST DELTA 0 -> tie (no charged-cost change)`

This fails the staging gate because profile cost is not strictly below the authoritative live bar.
No `truth`, `ab`, `battery`, or `--full` gate was run on the tie.

## Why the route is dead

The U8 selector tail deletes `fill_c_t`, `fill_xor`, and `fill_sel_delta` as 10x10 BOOL planes,
but adds `fill_c_u8`, `fill_c_t_u8`, and `fill_any_u8` as 10x10 U8 planes before casting back to
BOOL for `Pad` and the final output `Where`. Net charged memory is unchanged.

A broader U8 label/index is not free to construct. The formula
`fill = interior OR (outside AND (central OR diag))`, then orientation-select, still requires
distinct 10x10 binary decisions:

- `interior_c` is the row-inside and column-inside product.
- `diag` is a row-by-column equality against the distance-dependent diagonal rail.
- `beam/proj_c` must keep the central band and rails separated; the single filled-wedge merge
  overfills and was already visible-failed at 53/266.
- The orientation mux cannot use `Where(BOOL, BOOL, BOOL)` in this ORT/scorer environment.
  The legal U8 `Where` replacement ties the XOR mux byte-for-byte.

The F32 marginals also do not narrow profitably: `row_sum` and `col_sum` are `Einsum` results from
the F32 graph input. Casting them to U8 would add two 30-byte carriers while only shrinking four
F32 scalar gathers by at most 12 bytes, so it is strictly worse.

## One remaining hypothesis

A sub-2455 win would need a genuinely new final-output family that avoids the 900-byte full-canvas
output condition without materializing an F32 crop, or a legal boolean selector identity that removes
one 10x10 plane without replacing it with a U8 plane. No such route was found here.
