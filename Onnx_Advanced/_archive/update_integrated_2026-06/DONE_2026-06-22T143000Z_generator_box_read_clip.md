# Generator-box read clip: clip the input READ to the ARC-GEN spawn box on BOTH axes, not just rows

## What I discovered

When a charged F32 input read (a `Slice` of a channel) feeds a shape mask, the read footprint
should be clipped to the **generator's spawn box on every axis**, not just the obvious one. The
saving is `(full - clipped) * elem_bytes` directly off the dominant charged tensor, and the clip
is byte-equal because the clipped-away cells are corpus-guaranteed background.

The trap: it is easy to clip ONE axis (the visually obvious one) and declare the result "the
read floor", missing the symmetric clip on the other axis.

## Where (task267, marker-recolor)

- Generator `external/ARC-GEN/tasks/task_aabf363d.py`: the shape is built by
  `common.continuous_creature(randint(12,15), 5, 5)` — i.e. inside a **5×5 box** — then every
  pixel is `+1`-shifted into the 7×7 grid. So the shape ALWAYS lives in **rows 1-5 AND cols 1-5**.
  Cols 0 and 6 (rows 1-5) are always background; the marker is the sole non-bg cell in the last
  row / col 0.
- Prior task267.md had clipped the ch0 F32 read from 7 rows to 5 rows (`bg7`=[1,1,5,7]=140B) and
  explicitly called 5×7 "the F32-read floor (rows 0/6 carry no shape)". It missed that the
  COLUMN span is also exactly 5 (cols 1-5), so the true box is **5×5 = 100B**.
- Fix: `bg7` slice `cols 1-5` instead of `cols 0-6`; Pad the mask back to 7×7 on BOTH H and W
  axes (`pads=[1,1,1,1]`, `axes=[2,3]`) before the carrier Concat. Mask planes `sm`/`sm8_5`
  shrink [1,1,5,7]→[1,1,5,5] too (35→25 each).
- Cost: 525 → **468** (memory 381 + params 87). −40 (bg7) −10 (sm) −10 (sm8_5) = −60 memory,
  +3 params (pad arrays grow from [2]+[1] to [4]+[2]). Net −57.
- Builder: `The_400_Tasks/Task_by_Task/task267/prototypes/build_task267_marker9_convinteger_5x5.py`.
  Staged: `staged_submits/task267_marker9_convinteger_5x5_cost468/`.
- Gates: visible 264/264, truth 200000/200000, ab byte_equal=True (0 div), profile==static==468.
  Adversarial: all 72 M≠C color pairs, shape cells at every box corner (rows 1/5, cols 1/5),
  output(6,0)=0, pad region 0.

## Why it matters

This is the team-best beat for task267: team manifest had QLinearConv @ 546 (Boladi); the local
staged ladder was 525 (byte-equal). The 5×5 clip lands a byte-equal **468** — a −78 vs the shipped
team-best. Beat came purely from reading the generator's spawn-box geometry, no new op family.

## What's missing from the library

The "shrink the read to the active footprint" lever exists implicitly (task267.md used it for
rows). But there's no general statement of: **decode the generator's spawn box and clip the read
to it on ALL axes — and don't trust a single-axis clip as the floor.** A one-liner in
`techniques.md` (#82 marginals / free-read family) or `cost_floor_reference.md` (read-plane
taxonomy) saying "read footprint floor = the generator's spawn box, clipped per-axis; verify with
a corpus argwhere min/max bbox scan, not just the visually obvious axis" would have caught this
40B earlier. Cheap recipe to add: a corpus `np.argwhere(grid>0)` bbox scan over ~20k generated
inputs, exclude known fixed cells (the marker), take per-axis min/max → that IS the read box.
