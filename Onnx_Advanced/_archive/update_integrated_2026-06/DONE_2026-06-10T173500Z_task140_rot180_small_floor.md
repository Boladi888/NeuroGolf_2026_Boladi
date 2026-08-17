# task140 — rot180 of a small (3×3) one-hot region: floor = 210, crossover-law confirmation

## What I found

task140 rule = **rot180 of a fixed 3×3 one-hot grid**, embedded top-left of `[1,10,30,30]`.
LIVE did it the expensive way: `Slice(steps=[-1,-1])` on the 10-channel F32 one-hot → `rev[1,10,3,3]`
**F32 = 360 B** → Pad. cost 368.

Drove it to **210** (byte-equal, 200265/200265 ab-full; truth 200000/200000; adv 5633/5633) via #8b:
dilated crop-collapse `Conv` → `[1,1,3,3]` F32 color canvas (36) → `Cast U8` (9) → `Slice` rot180 in U8 (9)
→ `Equal(rot, chan_ids)` → `onehot_b[1,10,3,3]` BOOL (90) → `Pad` → output (free).
memory 36+9+9+90=144, params 40(conv)+8(rot)+10(ids)+8(pad)=66, **cost 210**, profile==static.

Staged: `staged_submits/task140_canvas_rot180_u8_cost210/`. Builder:
`The_400_Tasks/Task_by_Task/task140/prototypes/build_task140_v3_u8rot.py`.

## Why it matters — a clean small-region rotation/reflection floor + crossover datapoint

The existing `cost_floor_reference.md` "single-channel canvas = 900 B floor" + the crossover note (L163,
`10·h·w` cropped-BOOL vs `900` full-canvas-U8) are validated here at the **small end (h·w=9)** with hard
measurements, which the current exemplars (task400 5×5, task388 12×12) don't pin down:

- **cropped-BOOL Equal output `[1,10,3,3]` = 90 B** is the emit floor. Measured the alternative
  (full-canvas Equal-as-output: pad the color canvas to `[1,1,30,30]`, Equal = free output) at **3739** —
  the padded `[1,1,30,30]` F32 (3600) dominates. So at 3×3 the cropped path wins by ~18×. Crossover law holds.
- **The crop-collapse Conv output is ORT-forced to F32 (36 B)** even when you only want a U8 color index;
  the `Cast→U8` then rotating in U8 (9+9) is cheaper than rotating the F32 canvas (36). V1 (F32 rotate) = 229,
  V3 (U8 rotate) = 210 — an 18 B win just from rotating the color-index in U8 before the Equal-expand.
- **QLinearConv to emit a U8 canvas directly is a trap here**: it needs `QuantizeLinear(input)` first, which
  materializes the full `[1,10,30,30]` U8 (9000 B). Measured **9180**. Do not reach for QLinearConv to dodge
  the forced-F32 Conv output on whole-input collapses.
- **Rotation floor generalization**: "rotate/reflect a small K×K one-hot REGION and re-embed" has floor
  `≈ 10·K² (Equal) + 36 (F32 crop-collapse conv) + 18 (U8 cast+rotate) + ~66 params`. For K=3 that's 210.
  The 360→90 win is the **channel collapse**, not a same-dtype crop (a same-dtype Slice-crop stays F32 360).

## Dead-ends (don't retry)
- opset-10 attribute Pad to save the 8 pads-input params: **Pad(BOOL) won't load at opset 10**
  (`score_network_v2 → None`); opset 11+ needs the pads input tensor.
- 1×1 Conv collapse (10 params) then Slice-crop: outputs `[1,1,30,30]` F32 = 3600 before the crop.
- Slice-crop the one-hot to `[1,10,3,3]` first then collapse: the `[1,10,3,3]` F32 crop = 360 (= LIVE) dominates.

## What's missing from the library
`cost_floor_reference.md` has the canvas/crossover laws but no compact **"rotate/reflect a small one-hot
region"** floor entry. The numbers above (210 floor; F32-conv-output-forces-cast; U8-rotate beats F32-rotate
by the cast delta; full-canvas-output=3739 vs cropped=210 at 3×3) are a ready calibration row.

## Sorter note: cost_floor_reference.md — extend the "single-channel canvas / crossover" section (around L159-163) with a small-region rotation/reflection floor row: task140 rot180 3×3, floor 210, cropped-BOOL [1,10,K²]=10·K² beats full-canvas (measured 90 vs 3739 at K=3); ORT-forced-F32 conv canvas + U8-rotate (210) beats F32-rotate (229); QLinearConv whole-input dodge = 9180.
