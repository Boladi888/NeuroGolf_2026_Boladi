# DONE: task284 U8 tail + GatherND marker colors + Equal masks

Task284 dropped from 34,965 to 20,895 after the sparse-crop/moment baseline, while preserving the stated two-aligned-singleton-marker contract.

Reusable techniques:
- **Full-output Equal tail with pre-padded scalar ids**: pad the scalar color-id image to `[1,1,30,30]`, then compare directly against channel ids so the `[1,10,30,30]` BOOL tensor is the graph output, not an intermediate.
- **U8 class-id tail**: once colors are integer class ids, cast final color image lanes to `UINT8`, use `Pad` with a `UINT8` sentinel value, then `Equal` against `UINT8` channel ids. This halves the last cropped color planes and full-frame pad plane versus F16.
- **GatherND marker colors**: when exactly two marker coordinates are already known, sample marker colors from the scalar color-index plane with four scalar `GatherND`s instead of full-plane color moments.
- **U8 presence profile**: `Cast(color_index, UINT8)` plus `Min(color_index_u8, 1)` creates a binary marker-presence plane accepted by ORT `ReduceMax`, avoiding float count planes for row/col profile recovery.
- **Exact mask collapse**: replace exact-position masks encoded as `Less -> Not -> Greater -> Not -> And` with `Equal(grid, scalar)` where F16 grid/scalar values are small exact integers.

Verification:
- `verify_against_live`: 266/266.
- Balanced contract synthetic-LIVE: 790/790 across targeted/random row/col odd/even cases.
- Generic adversarial-against-live: 19/150 advisory/off-contract. The generic generator creates many colored cells, which can drive GatherND indices outside the cropped plane; Task284's documented contract is exactly two aligned singleton markers.

Files:
- `The_400_Tasks/Task_by_Task/task284/prototypes/build_task284_colorconv_sentinel.py`
- `The_400_Tasks/Task_by_Task/task284/prototypes/task284_u8tail_u8coloridx_minprofile_equalmask_gather.onnx`
- `Logs/task284_u8tail_u8coloridx_equalmask_synthetic_live_report.json`
- `Logs/task284_u8tail_u8coloridx_equalmask_adversarial_against_live.txt`
