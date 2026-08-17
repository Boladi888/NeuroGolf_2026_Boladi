# task381 — byte-equal conflict fusion + uint8-Mul rejection + crop-floor

## What I discovered (3 transferable items)

### 1. Two exact byte-equal fusion identities for 0/1 mask graphs
When two boolean conflict flags `cu, cd ∈ {0,1}` are combined as `(1-cu)*(1-cd)` (the
classic "no conflict above AND no conflict below" pattern), this is byte-identical to
`1 - Max(cu, cd)`. Lets you replace two `Sub(1,·)` + one `Mul` (3 ops, 3 planes) with
one `Max` + one `Sub` (2 ops, 2 planes).

And `ReduceMax_w(mask · cu) ` combined with `ReduceMax_w(mask · cd)` via outer `Max`
equals `ReduceMax_w(mask · Max(cu, cd))` — ReduceMax distributes over elementwise Max.
So the two separate `Mul→ReduceMax` conflict legs collapse into one
`Max(up,down) → Mul(mask,·) → ReduceMax`.

Net on task381: the vertical-conflict path dropped from 10 intermediates to ~6,
fully byte-equal (verified `ng_gate ab --full` 200265/200265, div=0). General use:
any graph that ANDs two independent boolean reductions of the same mask.

### 2. ORT Mul REJECTS uint8 (hard load failure)
Trying to run the whole pipeline in U8 to halve plane bytes (100B vs F16 200B) fails:
`INVALID_GRAPH : Type 'tensor(uint8)' of input parameter (ch0) of operator (Mul) is invalid`.
MaxPool(uint8) IS supported (tested OK), but Mul/Sub on uint8 are not in ORT's CPU EP.
So "collapse all mask planes to U8" is not a viable byte-shrink when the graph contains
elementwise Mul — stay F16 for the arithmetic planes. (Separately, Pad(BOOL) needs
opset 13; rejected at opset 10 — so BOOL output_active forces an opset-13 Pad with
pads-as-input.) This belongs in the ORT dtype×opset matrix (`ort_reality.md`) and the
trap row about U8/BOOL plane-collapse.

### 3. Crop-safe ≠ profitable: structural floor can sit above team-best
task381's 10×10 crop is provably hidden-safe (generator `task_ef135b50.generate(size=10)`,
size hardcoded, `task_list` binds no args, `common.grids(size,size)` always 10×10, 5000
fresh draws all (10,10)). But even the maximally-collapsed byte-equal graph floors at
~5044 (forced: 800B 2-channel F32 input slices + 1000B BOOL 10-channel output_active +
≥1600B F16 working planes), which is ABOVE the team-best 2067. A crop being safe does
not imply a profitable ship — always price the forced-component floor before chasing.

## Where
- Build: `The_400_Tasks/Task_by_Task/task381/prototypes/build_task381_min.py`
  (builds `task381_min.onnx` F16/opset10 = 5736, and `task381_minbool.onnx` BOOL/opset13 = 5044).
- Floor proof + crop-safe proof: `The_400_Tasks/Task_by_Task/task381.md` (Floor proof section).

## Why it matters
Identities (1) are reusable byte-equal shrinks for any "AND of two boolean reductions"
graph. (2) prevents a worker wasting a build cycle on a U8 plane-collapse that ORT rejects.
(3) is a DROP-discipline data point: byte-equal-to-LIVE + crop-safe still produced 0 EV
because the floor (~2600B forced minimum) exceeds team-best 2067.

## What's missing from the library
- `ort_reality.md` dtype matrix: I did not find an explicit "Mul/Sub reject uint8" row
  (MaxPool-U8-OK vs Mul-U8-FAIL is the useful contrast).
- No general note on the `(1-a)(1-b) == 1-max(a,b)` and `ReduceMax`-distributes-over-`Max`
  byte-equal fusions for 0/1 graphs — worth a techniques.md §.
