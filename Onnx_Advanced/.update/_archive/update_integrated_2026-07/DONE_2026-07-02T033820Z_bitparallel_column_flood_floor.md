# Bit-parallel column-packed flood breaks the geodesic-per-step floor (task286)

**Where:** task286 (Filling Regions / maze 4-conn reachability). Team-best is
now **21558** (Vamshi), a bit-parallel column-packed flood — well below the
previously-documented "flood depth × geodesic per-step + I/O ≈ 23,210, sub-20k
unsupported" floor in `cost_floor_reference.md:28`. Analysis:
`The_400_Tasks/Task_by_Task/task286/WALL_2026-07-02_bo_ng_cld_worker2.md`.

## What it is
Pack each grid **column into a UINT32 scalar** (bit r = row r), so a 25×25 maze
becomes 25 four-byte scalars. Flood = repeated passes of:
- **vertical segmented log-fill** per column: `r |= (r<<s) & runmask_s` for
  s∈{1,2,4}, both directions. The `runmask_s` (`o&(o<<1)`, `that&(that<<2)`, …)
  are pure-open-derived → **computed once and reused across all passes**
  (hoisted). NB: the run-mask is REQUIRED — a naive `(r<<s)&o` leaks across
  walls (jumps the gap), a classic bug.
- **horizontal merge** with neighbor columns, serpentine (Gauss-Seidel L→R then
  R→L) so a wavefront crosses all columns in a few passes.

Because each op works on `[1,1]` (or `[1,W]`) scalars instead of full
`[1,1,25,25]` canvases, the credited **Σ-activation-bytes** cost collapses from
~50k (MaxPool/Min grid flood) to ~21.5k. This is the lever that beat the old
geodesic floor: the floor argument assumed per-step tensors sized to the grid;
bit-packing makes per-step tensors O(W) scalars.

## Falsified sub-lever (don't re-try): full-log vertical fill
Deepening the vertical shifts to {1,2,4,8,16} to "flood whole columns per pass
and need fewer passes" does **not** help. Measured on the corpus
(`task286/prototypes/flood_core_sim2.py`, all schedules correct/never-fail):
pass count is bounded by the maze's **horizontal direction-reversal count**
(~20 worst-case on 500 grids), which vertical depth barely changes (20→17),
while per-pass cost rises 21→33 ops/col. Net: {1,2,4}=42,800 B beats
{1,2,4,8,16}=57,700 B for the flood core. Vamshi's {1,2,4} choice is correct.
(His ~9 real passes vs a naive serpentine's ~20 implies a tighter horizontal
wavefront — likely horizontal log-doubling or bidirectional — still unrecovered.)

## Cost-model fact worth banking
For the ORT-profiled Σ-bytes score: **graph inputs and the final graph OUTPUT
tensor are NOT charged** — only interior intermediates (value_info) are. task286
outputs a `[1,10,30,30]` UINT32 (36,000 B) for free; the last broadcast/scatter
to output is uncharged. Build the expensive assembly to land ON the output node.

## Sorter note:
- Add to `Onnx_Advanced/techniques.md` (flood/region-fill family) + `by_symptom.md`
  under "flood-fill / maze reachability": **bit-parallel column-packed flood**
  (UINT32 bit-columns, hoisted segmented run-masks, shifts {1,2,4}, serpentine
  horizontal) as the current cheapest known maze-reachability family; note it
  beats the grid-tensor geodesic floor.
- Update `cost_floor_reference.md:28` task286: floor moved from ~23,210 to
  **21558** (bit-parallel), old "sub-20k unsupported" note retracted; full-log
  vertical-fill lever falsified (bounded by horizontal reversals).
- Add/extend the scoring-internals note: **graph inputs and final output tensor
  are not charged** in the Σ-activation-bytes profile cost (only intermediates).
