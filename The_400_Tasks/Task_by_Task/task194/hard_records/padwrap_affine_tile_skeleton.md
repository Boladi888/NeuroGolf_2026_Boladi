# task194 padwrap_affine_tile_skeleton

- live bar: cost 234, sha256 `5b633082bb0cd78d1eded256094a1dddf9241aa56bae15676868013a5d500a4e`, source `python tools/ng_current_bar.py --task 194 --json` / team manifest.
- attempted route: reviewer-locked Path-B PadWrapAffineTile coordinate skeleton. Use wrapped period coordinates `phase = arange(6) % 3`, `anti = 2 - phase`, and quadrant masks to derive the 6x6 D4 source-index map without Sho's stored `flat_idx`.
- exact proof command: `python The_400_Tasks/Task_by_Task/task194/prototypes/build_task194_padwrap_affine_skeleton.py`.
- exact proof result: `index_map_ok=True`; `fresh_cases=5/5`; `generated_int32_idx_optimistic_cost_floor=342`; `direct_quadrant_feature_optimistic_cost_floor=330`; `grid_sample_6x6_floor=1512`.
- why it is dead below 234:
  - Generated-index route: deleting Sho's 36-param `flat_idx` but materializing even an ideal `INT32[6,6]` computed index costs `234 - 36 + 144 = 342` before any coordinate constants or arithmetic intermediates. `INT64[6,6]` is `486`.
  - Direct feature-transform route: avoiding `flat_idx` by rotating/concatenating the two-channel `[code, code^2]` feature grid has an optimistic memory floor of 288 plus at least 42 params for color read and final classifier, so floor `>=330`.
  - GridSample route: `[1,10,6,6]` F32 remap costs 1440 memory plus 72 grid params (`1512`) or 1800 params for direct 30x30, both above live.
- one remaining hypothesis: a future sub-234 route must be a genuinely fused spatial/color classifier that avoids both a materialized coordinate/index tensor and a materialized 6x6 feature tensor, or a new free-output factorization below the prior 283-Einsum family.
