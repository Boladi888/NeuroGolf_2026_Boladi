# 03_crop_and_f16 — worked case studies

Spatial crop-to-active-grid and F16 placement. The bread-and-butter of Pattern_Recognition optimization. Often stack with bool-first techniques.

## Files

- `task117_crop15_f16.md` — ★ THE cornerstone. Crop-to-active-grid + F16-output trick stacked. -100,525 bytes, +1.1249 LB. Read this first.
- `task055_per_channel.md` — F16 placement rule: cast per-channel when N < 10.
- `task384_f16_drop.md` — F16-output trick alone (drop final Cast).
- `task128_crop15.md` — Crop variant at H=15.
- `task281_crop13.md` — Crop variant at H=13 (smaller active grid = larger relative savings).
- `task364_crop22.md` — Crop variant at H=22 (when crop is still worth doing despite smaller savings).

## Decision flow

For a Pattern_Recognition task with cost > 50,000:

1. Check `data/competitions/neurogolf-2026/taskNNN.json` for max active grid (H, W from input/output shapes).
2. If H < 30 (or W < 30):
   a. Apply crop-to-active-grid (`task117_crop15_f16.md` template).
   b. If graph ends with `Concat → F16 → Cast(F32) → output`, also apply F16-output trick.
   c. Stack: total savings can exceed 100k bytes.
3. If H == 30 (no crop possible):
   a. Apply F16 placement rules (`task055_per_channel.md` for Gather-channel patterns, `task384_f16_drop.md` for Concat-end patterns).
4. After crop + F16, return to `02_bool_first/` for remaining mask intermediates.

## Critical: read the false-positive guard

See `traps.md §21`: if the model already Slices to a small region on its first node, pre-crop adds overhead. Verify by reading the graph before applying crop.
