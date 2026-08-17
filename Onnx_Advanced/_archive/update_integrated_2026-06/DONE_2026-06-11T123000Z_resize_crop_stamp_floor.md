# Resize crop-and-resize free-output stamp (the aditya task021 171-cost construction)

## What I discovered
A previously-undocumented free-output stamping technique that drives the output tensor entirely from a tiny `Resize` op, producing a filled rectangle in a chosen channel at cost ~171. Found as the **team-best** for task021 (aditya), 77× cheaper than the personal/canonical net (13,142) that the library and task021.md were tracking. The library had NO record of it (searched `techniques.md`, `cost_floor_reference.md`, `task_techniques_index.md` for Resize/crop_and_resize/source_const — only the stale 13k net was indexed).

## The construction (16 nodes, opset 18)
Output is a solid `(h×w)` rectangle of one color stamped into one channel of `[1,10,30,30]` UINT8, where h/w/channel are all computed from the input.

1. `source_const` = UINT8 `[1,2,2,2]` with a single `1` at `[0,0,0,0]` (8 params). The 2×2×2 "octant" is the stamp seed.
2. `Resize(source_const, roi, , sizes=[10,30,30])` with `axes=[1,2,3]`, `coordinate_transformation_mode=tf_crop_and_resize`, `mode=nearest`, `nearest_mode=floor`, `extrapolation_value=0`. Output is named `output` ⇒ **free** (I/O tensors skipped by `calculate_memory`).
3. The `roi` (a `Concat` of 6 F32 scalars, `[c1,0,0,c2,row_end,col_end]` for the 3 axes) crops the source so that, after nearest/floor upsampling to 30×30, the single `1` octant expands into exactly the desired rectangle in exactly the desired channel. `c1/c2` select the channel (axis 1) via `c1 = center − argmax_channel`, `c2 = c1 + span`; `row_end/col_end = end_num / separator_spacing` set the rectangle extent.
4. Per-channel statistics that drive the roi come from `AveragePool` with a **stride-collapse trick**: kernel `[1,30]` stride `[30,30]` over `[1,10,30,30]` emits `[1,10,1,1]` directly (40 B) = the per-channel fraction in grid **row 0** (kernel `[30,1]` → col 0). `ReduceMax`/`ReduceSum`/`ArgMax` then reduce that 10-vector to the divider color and separator spacing.

## Why it matters
- **Free output via Resize**: like OneHot/ScatterND free-output tricks (#67 family), but the *whole* output is synthesized by upsampling a 1-pixel seed. No canvas tensor is ever materialized and charged. For any "fill a computed rectangle in a computed channel/color" rule, this is dramatically cheaper than building a `[1,10,30,30]` mask (which the 13k net did).
- **tf_crop_and_resize ROI as a computed-geometry channel**: the 2*len(axes) ROI floats let you place the stamp by arithmetic on input-derived scalars — no Slice/Pad/Gather placement ops.
- **AveragePool stride-collapse** as a 40-byte per-channel row/col projector: `kernel=[1,W] stride=[H,W]` collapses straight to `[1,C,1,1]` with NO large intermediate, beating `Slice(row0)→ReduceSum` (which materializes `[1,C,1,W]`).

## Floor proof (task021 = 171 is forced)
Verified static `_score_v2` 171 (memory 156 + params 15), `profile_real_cost --task 21` 171/171 no drift, `ng_gate visible` 77/77, `ng_gate truth --full` 200000/200000, `ng_gate ab --full` byte-equal vs LIVE (200077/200077, 0 div). Attacks rejected: source_const `[1,2,2,2]`→any of `[1,1,1,1]/[1,2,1,1]/[1,1,2,2]/[1,2,2,1]` all 0/2000 truth (octant geometry forced); F16 on pool vectors impossible (AveragePool inherits F32; casting input materializes 18 KB F16 `[1,10,30,30]`); 2-axis ROI saves 8 B but needs a >8 B channel-placement op; all scalars 4 B irreducible. **NO sub-171 candidate.** Files: `task021/prototypes/EXTRACT_TEAM_task021.onnx` (sha256 `973e1edb62ba96f50e4db8d2224032e019ff98c9d85ed565f39666ae67e3cac9`), proof prose in `The_400_Tasks/Task_by_Task/task021.md` Cost analysis.

## Note for captain
The personal stack's `live_costs.json`/task_state still says 13,142 for task021, but the shipped team net is 171 and byte-equal to LIVE. This is a sync/bookkeeping gap, not a worker submit.

## Sorter note: techniques.md — add as new technique "resize_crop_and_resize_free_output_stamp": drive the full output from `Resize(tiny_seed, computed_roi, sizes)` with tf_crop_and_resize/nearest/floor; output named `output` is free; ROI floats place a computed rectangle in a computed channel; pair with AveragePool stride-collapse (`kernel=[1,W] stride=[H,W]` → `[1,C,1,1]`, 40 B) for per-channel row/col stats. Exemplar: task021 aditya 171 (vs 13,142 mask build).
## Sorter note: cost_floor_reference.md — add Resize-stamp free-output floor: a "fill computed rectangle in computed channel" rule floors near ~171 (156 mem: 2× `[1,C,1,1]` pool vectors 80 + scalars 52 + ArgMax INT64 8 + 6-float ROI 24; 15 params: 8 seed + 3 sizes + 4 scalar consts). Beats canvas/mask builds by ~13k.
## Sorter note: by_symptom.md — route "output is a solid rectangle whose size/position/color is computed from the input" → resize_crop_and_resize_free_output_stamp (task021 exemplar).
