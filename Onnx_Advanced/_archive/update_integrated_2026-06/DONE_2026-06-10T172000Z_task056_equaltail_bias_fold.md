# task056 — fixed-silhouette→color classifier collapsed to cost 55 (from LIVE 538)

**Where:** task056 (ARC-GEN uuid 27a28665). Staged `staged_submits/task056_cropch0_equaltail_cost55/`.
Builder `The_400_Tasks/Task_by_Task/task056/prototypes/build_candidates.py` (build_E2_minimal).
Byte-equal vs LIVE 200046/200046; truth 200000/200000; static==profile==55.

## Transferable insights

### 1. BIAS-FOLD INTO THE Equal-TAIL THRESHOLD (the main reusable trick)
When you compute a scalar `color = linear_read + b` and then build a one-hot via
`Equal(color, channel_ids[1,10,1,1])`, you do NOT need the `Add` (bias) node, the `color`
intermediate, OR the bias init. Fold the bias into the comparison vector:
`color == k  <=>  linear_read == k + (k_threshold - b)`. Shift the ids:
`Equal(linear_read, ids_shifted)` where `ids_shifted[k] = k - b` (here ids+2 because color=s-2).
Saved one Add op + one F32 scalar (4 B) + the bias param. General whenever an Equal-tail
follows an affine scalar read. (This is a small companion to #79's Equal(iota, code) — the
library shows the iota/code form but not the "absorb the additive constant into the iota" move.)

### 2. TINY-OUTPUT FLOOR: Pad-a-[1,10,1,1]-one-hot beats the canvas (extends #81/#82)
When the OUTPUT grid is 1×1 (single answer cell) but must be emitted as the full [1,10,30,30]
one-hot, the cheapest output is `Equal(color, ids[1,10,1,1]) -> BOOL[1,10,1,1] (10 B) -> Pad`.
This is the extreme case of #81's "small-grid inversion": the [1,1,30,30] color-index canvas +
Equal (the #82/054/387 pattern) costs 900 B because the active region is 1×1, NOT ≳10×10.
Measured: canvas route 900+, OneHot-opset9 route 120, Pad-of-1×1-onehot route 26 B memory.
Rule of thumb the library already half-states: the 900-B canvas only wins when active ≳10×10;
for a literal 1×1 output, pad the tiny one-hot.

### 3. ORT Einsum/Conv CANNOT broadcast a small spatial weight against a 30×30 input
`Einsum('bchw,hw->b', input, W[3,3])` PASSES onnx shape-inference but FAILS at ORT runtime
("operands could not be broadcast together ... dim 0 is {3,3} vs 30"). You MUST Slice the
input to the exact window first. For reading a few fixed cells, crop to the minimal box
(here ch0/row0/cols0-2 = [1,1,1,3] = 12 B) then Einsum over the small slice. This is a
silent build trap: checker + shape_inference both green, only `sess.run` errors.
(I did not find this stated in ort_reality.md's Einsum notes — worth a row.)

### 4. Fixed-pattern→color tasks: read the generator, the silhouette IS the class
task056's generator paints ONE of 4 fixed 3×3 silhouettes (idx∈{1,2,3,6}) with a RANDOM fill
color; output = idx. So the fill color is a decoy — the 9-bit FOREGROUND mask is the only
signal, and there are exactly 4 of them. An exhaustive search found a 3-cell linear map
(row-0 cells only) that lands directly on {1,2,3,6} with NO LUT/MatMul/ArgMax:
`color = -5·m0 -4·m1 -3·m2 + 10`. The LIVE graph's MatMul[9,4]-template + ArgMax +
40-param output_bank was entirely replaceable. Lesson: for "classify a small fixed silhouette
into a color", brute-force a tiny linear-or-place-value discriminator over the (few) actual
masks before building any template/bank — the floor is formulation-relative to the mask count,
not the 9-bit space. ch0-read (`1 - input[0]`) is exact for foreground when the grid is
generator-guaranteed full (size fixed = no interior padding, so the 1-ch0 trap does not apply).
