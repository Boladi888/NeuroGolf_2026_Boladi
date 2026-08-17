# Accumulator-plane folds for the single-plane class-LUT output (#67/#81 family)

**Where:** task349 (ARC-GEN db93a21d "death stars"). Builders
`The_400_Tasks/Task_by_Task/task349/prototypes/w3/build_w3_v{2,3}.py`. Staged
`staged_submits/task349_w3_accumfold_cost15710/`. Beat team-best 17509 → **15710**
(-1799), all gates pass incl. ab FULL byte_equal vs LIVE + profile==static.

## Pattern: one-plane integer accumulator → free `Equal(acc, class_values)` BOOL output
The cheapest multi-color renderer here is NOT a Concat of 10 channel planes. Build a
SINGLE UINT8 [1,1,30,30] "accumulator" whose value at each cell is a small integer
class code (bg=0, blue=1, green=2, maroon=3, pad=255), then:
`output = Equal(accumulator, class_values[1,10,1,1])` where
`class_values[k]` = the code that lights output channel k. The BOOL [1,10,30,30]
output is FREE (scorer skips `output`), so the whole cost is the accumulator build.
This is the #67/#81 free-output-Equal idea but with a class-LUT broadcast over the
channel axis — collapses N channel planes into ONE plane + one Equal. (Aditya's
team-best already used this; the two folds below are new on top of it.)

## Fold 1 (-1 plane): make a QLinearConv emit code "2" directly via INT8 kernel + bias
To put green code **2** (not 1) into the plane straight from the run-detector,
DON'T try a linear `y_scale` — the requant constraint to map a full-W run→2 and a
(W-1) run→0 is `2(W-1) < y_scale ≤ W/1.5`, which is **empty for W≥2**. Instead use
INT8 kernel `[2]*W`, x_scale=w_scale=y_scale=1, y_zp=0, INT32 bias `-(2W-2)`:
full W-run conv_sum=2W → y=sat_u8(2W-(2W-2))=2; any shorter run ≤2(W-1) → sat to 0.
This drops the separate `Add(halo,halo)` doubling plane (-900B). Generalizes: a
QLinearConv can emit ANY small target code c at an exact-W match via kernel `[c]*W`
+ bias `-(cW-c)` (= -c(W-1)), y_scale=1. (The existing per-W bias note in task349
history used kernel-1 + bias to emit 1; this extends it to arbitrary code values.)

## Fold 2 (-1 plane): Max-then-Add to layer a "+1" class without a separate Max plane
When one class strictly contains another (here maroon ⊂ green halo always), you can
skip the intermediate `Max(...)`→`Add(ch9)` two-plane chain. Instead Max ALL the
lower-priority planes together first: `t = Max(beam[1], halo_2..halo_10[2], sents[255])`
then `acc = Add(t, mask_inner)` where mask_inner∈{0,1} is the contained class. The
Add promotes contained cells (2→3) while leaving others untouched (the contained
mask is 0 off its region). Saves the dedicated accumulate plane (-900B). Works
because the containment guarantee means every inner cell already sits on the right
base code.

## Trap re-confirmed
- The F32 per-channel slice of a maroon plane (3600B for [1,1,30,30]) is the hard
  floor on F32 one-hot input — casting all 10 channels is 9000B (worse). Matches
  techniques.md L437. The single biggest tensor; not removable in this family.
- Row+col 255-sentinels (from `Where(Cast(ReduceMax(input,[1,3])), 0, 255)`) are
  REQUIRED to keep beams/halos out of the N×N→30×30 padding and to stay byte-equal
  vs LIVE — dropping them passes argmax-decode but fails `ng_gate visible` and AB.
  Grids square → BOTH row and col sentinels needed (L-shaped padding corners).
