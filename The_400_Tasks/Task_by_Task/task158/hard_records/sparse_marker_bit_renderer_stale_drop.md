# task158 SparseMarkerBitRenderer — refreshed DROP (2026-07-10)

```json
{
  "verdict": "DROP",
  "candidate_class": "NO_CANDIDATE_STALE_DIRECTION",
  "cost": null,
  "live_cost": 23201,
  "status": "REJECT_LOCAL"
}
```

## Bars refreshed before building

- Authoritative deployed LIVE: **23,201**, owner Sho, SHA-256
  `cd08cc20468a7a8d5fe3cd69a2896a26340f72bdfa1ec50b1aee4e3cf498e4ce`.
  Source: `python tools/ng_current_bar.py --task 158 --json` (manifest,
  updated `2026-07-09T01:09:17Z`). The bar tool's lightweight team-best source
  advertises 23,194 and flags that source stale, but a direct current
  `score_network_v2` plus 266-input real profile both return **23,201**.
- Lower documented task-local ceiling: existing v128 **22,550** static/profile,
  SHA-256 `5fcb99b35151eefbb62abae199b7f1a01a20376a2354085009eb799ef4857ec0`,
  from `staged_candidates/task158_where_gated_locator_cost22550_sha5FCB99B3/manifest.json`.
- This prompt adds a stricter stage threshold: **cost < 11,000**.

## Attempted route and exact failed gate

Attempted route: the prompt's fast-probe premise, replacing three dense
`s1/s2/s3` valid-score planes with a sparse-coordinate or row-bit selected
marker-pair renderer.

Failed gate: **refresh / charged-component-presence gate, before prototype build**.
`python tools/agent_helpers/inspect_graph.py Team_Best_Onnx/task158.onnx`
reports Sho's current 62-node graph with no `s1`, `s2`, or `s3` tensors. The
current detector is already one cropped `DET=QLinearConv` tensor
`UINT8 [1,3,24,23]` (1,656 B), followed by one cropped
`CP=ConvInteger` tensor `INT32 [1,1,26,25]` (2,600 B) and a runtime
`Gather(TF,CP)` render. The prompt explicitly says to DROP when its named
charged component is already gone, so no stale-family ONNX variant was built
and no truth/AB/full gate was burned.

The independent representation check is also already recorded in
`floor_and_dead_routes_22550.md`: the legend paint stamp is runtime-dependent
(18 distinct paint patterns in 4,000 corpus legends), so a static sparse stamp
kernel cannot replace `Gather(TF,CP)`; Gather forces an INT32/INT64 index. A
dynamic coordinate/bit renderer materializes enough shift/gate/index state to
reprice upward rather than below 11,000.

## Refreshed ceiling evidence (not a new candidate)

The existing v128 ceiling was rechecked only to anchor the handoff:

- LIVE score/profile: `live_score_v2=(0, 22014, 1187) cost=23201` and
  `static=23201 profile=23201 (memory 22014 + params 1187; 266 real inputs)`
- visible: `[visible] 266/266 (100.00%)  PASS`
- score: `score_v2=(0, 21364, 1186) cost=22550`
- real-input profile: `static=22550 profile=22550 (memory 21364 + params 1186; 266 real inputs)`
- safety: `WARN`, zero failures, one `WARN_X86_QUANTIZED_OP_FAMILY`
  warning for `ConvInteger` and `QLinearConv`.
- named static graph delta versus Sho bytes: remove materialized `g16`
  `UINT8 [1,1,26,25]` = **-650 B**, plus dead scalar initializer `c16`.
  The mechanical diff's lightweight totals omit seven scalar params in both
  graphs and display net -650; the official scorer sees params 1,187→1,186,
  so the credited total is **-651**. This is the already-documented v128
  locator shave, not a SparseMarkerBitRenderer win.

## Why this route is dead

The proposed target attacks a superseded graph family, while the current graph
already pays the smaller fused 3-scale `DET` plus data-dependent `CP/Gather`
floor. Reconstructing sparse coordinates or row words does not remove the
runtime-selected paint pattern; it merely moves the charge into dynamic
bit-shifts, gates, indices, and unpack state. With the current effective ceiling
22,550 and the prompt's stage threshold 11,000, this is a priced wall rather
than a candidate family.

## One remaining hypothesis

A genuinely different route would have to contract packed per-row marker-pair
selection **and** the runtime 3x3 legend paint pattern directly into the final
free BOOL output, avoiding both dense `DET` and INT32 `CP` intermediates. The
first kill test is a static/profile price below 11,000 for the selected-pair
kernel alone; no such legal ONNX contraction is currently known.
