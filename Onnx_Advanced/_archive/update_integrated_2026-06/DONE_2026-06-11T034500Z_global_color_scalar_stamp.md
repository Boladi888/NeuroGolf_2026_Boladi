# Global-color-scalar read: dodge the 3600-B F32 30×30 channel-collapse on stamp/expansion tasks

## What I discovered (task012, worker-5, 2026-06-11)

For a task whose rule "places/expands a fixed motif", if the foreground palette is **grid-global**
(few colors, constant across the grid — e.g. task012: one (B=center, A=arm) pair per grid), you do
NOT need a per-cell color-index plane. Resolving the colors as **global scalars** from cheap channel
histograms avoids the single biggest cost on these tasks: the F32 30×30 channel-collapse.

The trap I was about to fall into: to get a per-cell color-index plane from the F32 one-hot input
`[1,10,30,30]`, the cheapest single intermediate is `Conv(input, w[1,10,1,1]=[0..9]) -> [1,1,30,30]`
**F32 = 3600 B** (you can't crop inside a 1×1 conv; slicing the input first keeps all channels =
5760 B; a 19×19 corner-crop kernel trades it for 3610 params). 3600 alone blows past tight floors.

**The dodge (cost 4469, down from LIVE 15220, byte-equal 200265/200265 + truth 200000/200000):**
- Spatial **presence** comes from a single-channel slice `Slice(input, ch0, :12,:12) -> [1,1,12,12]
  F32 = 576 B`, then `presence = Less(ch0, 0.5)` (valid ONLY when the active grid is fixed-size so
  there are no padding cells inside the slice — otherwise the 1-ch0 trap bites).
- **Global colors** from `ReduceSum(input, axes=[2,3]) -> [1,10,1,1]` (40 B). The color with the most
  cells / fewest cells etc. is picked with `ArgMax` / `ArgMin` (set zero-count channels to +big before
  ArgMin). These give scalar `[1,1,1,1]` color values you broadcast-Mul onto binary stamp masks.
- Stamps stay single-channel F16 12×12 (288 B each); final canvas is a single-channel **INT8**
  `[1,1,30,30]` color-index (900 B, the doctrine floor) + `Equal([0..9]) -> BOOL output` (free).

This is the same family as task388-v5 (which slices ch0 to a 6×6 F32 = 144 B and uses ReduceMax/ArgMax
for its single global color) — the principle generalizes: **when foreground color is global, read it
as a channel-histogram scalar, not a spatial plane.**

## Two concrete traps confirmed
1. **INT8 canvas Pad value must be an out-of-palette sentinel (−1), not 0.** With pad 0, outside-grid
   cells get `Equal(0, ch0_id)=True` → channel-0 fires everywhere outside the active region → the
   visible gate (raw one-hot `np.array_equal`) returns 0/N. Sentinel −1 → all-zero one-hot outside. Correct.
2. **ORT rejects `Mul` on tensor(int8)** (InvalidGraph at load) — an int8 colorize to shrink the stamp
   planes from 288→144 won't load; keep colorize in F16.

## Where
- Builder: `The_400_Tasks/Task_by_Task/task012/prototypes/build_v3.py`
- Staged: `staged_submits/task012_global_color_stamp_cost4469/` (manifest + inspected-weight note)
- task notes: `The_400_Tasks/Task_by_Task/task012.md`

## Sorter note
- `cost_floor_reference.md` already has the "900-B single-channel canvas is the floor" block and the
  task140 small-region floor + task388 single-color ArgMax read. **Extend** it (or `techniques.md` /
  `by_symptom.md`) with the generalization: **"foreground palette is grid-global ⇒ resolve colors as
  channel-histogram scalars (ReduceSum/ReduceMax + ArgMax/ArgMin), NOT a per-cell color plane — avoids
  the 3600-B F32 [1,1,30,30] channel-collapse that a color-index Conv forces."** Cite task012 (15220→4469,
  byte-equal) alongside the existing task388 example.
- Add to `traps.md` (or the INT8-canvas note in cost_floor_reference): **INT8/U8 canvas Pad value must be
  an out-of-palette sentinel** when the tail is `Equal(canvas, [0..9])`, else outside-grid channel-0 fires.
- Add to `ort_reality.md` error→fix table: **`Mul`/elementwise on tensor(int8) → InvalidGraph at load**
  (ORT has no int8 elementwise-arith kernel; cast to F16 first).
