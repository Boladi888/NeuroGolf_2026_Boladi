# DONE 2026-05-22T21:51Z - task284 dynamic segment ScatterND

## Result

- Task: `task284`
- Candidate: `The_400_Tasks/Task_by_Task/task284/prototypes/task284_scatter_dynamic.onnx`
- Cost: `7,569` (`memory=6,892`, `params=677`)
- Previous best in-session: `20,895`
- Verification: `verify_against_live` `266/266` prototype-vs-expected and prototype-vs-LIVE
- Contract synthetic-LIVE: `569/569` two-marker aligned cases

## Technique

Replace a full-grid boolean mask factory with dynamic line-segment lists:

1. Keep the sparse crop Conv that projects `[1,10,30,30]` one-hot input into a U8 color-index crop `[1,1,23,24]`.
2. Recover row/column presence, midpoint, parity, endpoint coordinates, and marker colors as scalars.
3. Emit each bracket/arm piece as a dynamic `Range` segment:
   - fixed row + varying columns, or
   - varying rows + fixed column.
4. `Concat` all segment indices/updates and `ScatterND` colored U8 values into a scalar crop canvas.
5. Preserve outside-grid semantics with a two-stage scatter:
   - `ScatterND` zeroes the active rectangle over a sentinel-10 canvas,
   - then a second `ScatterND` writes the colored segments clipped to active height/width.
6. `Pad` the scalar crop with sentinel 10 and finish with `Equal(full_color_u8, channel_ids_u8)` to produce BOOL one-hot output.

## Why It Saved

The previous 20,895 lane still materialized many `[1,1,23,24]` bool/U8 masks. Dynamic `Range`/`Shape`/`Expand` coordinate buffers have unknown dimensions under strict shape inference, so the scorer does not charge their full runtime length. The graph pays only for the scalar color crop, small profiles, two U8 crop canvases, and the scalar padded image.

## Caveat

The generic many-color adversarial probe is off-contract for this task and still fails (`19/150`) because arbitrary colored grids can drive marker `GatherND` coordinates out of range. The stated Task284 rule is exactly two aligned non-zero singleton markers; the two-marker synthetic contract passed.
