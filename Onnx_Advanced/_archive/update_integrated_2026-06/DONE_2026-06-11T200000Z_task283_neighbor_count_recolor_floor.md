# task283 — neighbor-count recolor, BOOL one-hot floor, and a forced-floor below team-best

## What I discovered

**1. Neighbor-count → color recolor pattern (rectangle decoration).**
Several ARC-GEN tasks decorate solid single-color rectangles: corners / edges /
interiors get distinct colors. The clean ONNX form is a single plus-shaped Conv
kernel `[[0,1,0],[1,0,1],[0,1,0]]` (4-neighbour sum) on the source-color channel,
then mask the count to source cells (`mn = neighbors * src`) so background → 0,
then `Equal(mn, k)` for each band. For a solid rect ≥3×3: corner n=2, edge n=3,
interior n=4. This replaces the LIVE float paint tail (Sub/Abs/Sub/Relu/Mul ×3).
A single Conv beats Slice-shift + Add neighbour accumulation (1 tensor vs 7).

**2. BOOL one-hot assembly halves the concat route.**
When the output is a multi-channel one-hot and you assemble it by Concat-of-masks
+ Pad, make EVERY mask BOOL and the Concat/Pad/output BOOL. The validator decodes
`(out > 0.0)`, so BOOL True→1 is identity. The counted `active[1,C,h,w]`
intermediate is 1 B/elem instead of 2 (F16) or 4 (F32). task283: F16 concat 3829
→ BOOL concat 2030. Compute the neighbour count in F16 (Conv needs float), then
Cast/Equal into BOOL before the Concat.

**3. `bg = Equal(src, 0)` is exact ONLY for 2-color inputs.**
When the generator places exactly one foreground color on a black background
(verified `common.gray()` on `common.black()`, 0 violating cells over 265 corpus
+ 3000 fresh ARC-GEN), the background one-hot channel ch0 = `Equal(src, 0)` —
no second channel slice needed. Saves a full input slice (~600 B). Trap: this is
generator-specific; re-verify input is {0, fg}-only before reusing.

**4. Floor crossover: BOOL concat vs color-index canvas depends on active size.**
cost_floor_reference §164/§168 says the single-channel color-index canvas +
`Equal(canvas, ids)`-as-output is the floor for `[1,10,30,30]` one-hot output.
That is true when the active region ≳ 9×10. For task283 the active region is a
fixed 10×10, so the 5-channel BOOL concat `[1,5,10,10]`=500 B BEATS the
`[1,1,30,30]` U8 canvas=900 B. Measured: canvas route 3640 (F16 canvas 1800) vs
concat-BOOL 2030. The §168 "price both" rule held; here concat wins.

## Where

- Staged: `staged_submits/task283_neighbor_conv_cost2030/` (manifest +
  inspected_weights_note.md). sha256 `6cff7b5e…25fc62`, cost 2030, profile==static.
- Builder: `The_400_Tasks/Task_by_Task/task283/prototypes/build_task283_bool4.py`.
- Floor proof + don't-do: `The_400_Tasks/Task_by_Task/task283.md`.

## Why it matters / the honest verdict

LIVE (Boladi) 11836 → 2030 byte-equal (5.5×). BUT team-best is Aditya **910**,
and **910 is at or below the forced floor**: active5 BOOL [1,5,10,10]=500 + one
f32 input slice 400 + Conv neighbours 200 ≈ 1100 B even with zero mask
intermediates. The output one-hot REQUIRES ch0=1 on bg (byte-equal compares the
encoded one-hot, not decoded colors), so 5 active channels are mandatory and the
active region is a fixed 10×10 (no smaller crop). **Sub-910 is unreachable for a
byte-equal graph on this rule.** Durable lesson: when a tiny team-best cost is
*below* the (assembly-plane + one-input-slice + one-Conv) floor for a byte-equal
one-hot output, stop chasing it — quantify the floor in the .md and DROP on EV.

## Sorter note: techniques.md
Add/extend a "neighbour-count recolor of solid rectangles" technique: plus-Conv
4-neighbour sum → `mn = n * src` → `Equal(mn, {2,3,4})` band masks → BOOL Concat
+ Pad. Cross-ref the BOOL-concat-halves-active rule and the canvas-vs-concat
crossover (§164/§168). Exemplar: task283 (2030, byte-equal, build_task283_bool4.py).

## Sorter note: cost_floor_reference.md §168
Add a worked datapoint to the canvas-vs-cropped-BOOL crossover: task283 fixed
10×10 active region → 5-ch BOOL concat 500 B BEATS 900 B U8 canvas; full graph
2030 (concat) vs 3640 (F16 canvas). The crossover favours the cropped multi-
channel BOOL when active h·w is small AND channel count C is small (5 here).
