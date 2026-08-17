# Durable lesson: scattered-point GatherND beats Slice/separable-Gather for sparse reads over the F32 [1,10,30,30] input

## What I discovered (task033 w-C1, floor-confirmation)

When you need to read N scattered points from the F32 one-hot input `[1,10,30,30]`,
a single **GatherND with a flat `[...,3]` `[ch,r,c]` index** is the cost floor — it
reads exactly N values (N×4 B F32 output) and never materializes a plane. Every
"obvious" attempt to shrink the GatherND index table is strictly worse because it
forces a plane/row intermediate:

- **Slice ch0 → `[1,1,30,30]` then two 1-D Gathers** (to use an 18-element index
  instead of the N×3 GatherND index): materializes the full **3,600 B** F32 ch0 plane
  + a `[1,9,30]`=1,080 B row plane. task033 went 2,256 → **6,707**.
- **Separable row-then-col read** (GatherND `[0,r]` row index, 18 elems, then Gather
  cols): the row read is full-width `[1,9,30]` = **1,080 B F32**. task033 → **3,066**.
- **Pre-slice channel to drop the channel column from the index** (`[r,c]` = 162 elems
  vs 243): needs a squeezed `[1,30,30]` F32 plane = **3,600 B**. Net loss.
- **Runtime-generate the index with Range/Tile/Concat (0 params)**: the assembled
  `[1,9,9,3]` INT64 *intermediate* = 243×8 = **1,944 B** ≫ the 243-element initializer
  (params count ELEMENTS not bytes, so an INT64 index init costs its element count).

Rule of thumb: **the GatherND index initializer (N×3 INT64 = 3N params) is cheaper than
any intermediate plane that a separable read would create**, because a `[*,30]`-wide F32
intermediate already costs 30×rows×4 bytes. The crossover strongly favors the index for
any N where 3N < (materialized plane bytes), which is essentially always for sparse reads.

## Second lesson: "separable row/col bases" cannot represent a rank>1 2D shape

task033's prompt suggested deriving the 9×9 mask from "separable row/col bases" instead of
the template AND-chain. This is ruled out by the RULE, not just cost: the template is an
arbitrary 3×3 shape (the train[0] example is a plus `020/222/020`, rank>1). A
row-base × col-base outer product can only express axis-aligned rectangles (rank-1). Using
it would silently miss non-separable shapes — the same hidden-fail class as the rejected
`gencontract` shortcut. Before reaching for separable/outer-product reconstruction of a 2D
mask, check the shape is actually rank-1.

## Where
- Task: task033 (ARC-GEN 1e32b0e9, hollywood-squares copy). Floor confirmed at **2,256**
  (`staged_submits/task033_bool_sparse9_rank3_cost2256/`).
- Negative-result builders (kept, .onnx deleted):
  `The_400_Tasks/Task_by_Task/task033/prototypes/build_task033_bool_sparse9_slicegather.py`,
  `..._rowgather.py`.
- Full per-tensor floor proof: `The_400_Tasks/Task_by_Task/task033.md` "FLOOR PROOF" section.

## Why it matters / what's missing from the library
`op_cost_patterns.md` has GatherND/ScatterND thresholds; suggest adding the explicit
rule "for sparse scattered reads over a float input, a flat GatherND index beats Slice +
separable Gather because the separable path materializes a full-width plane; the N×3 INT64
index init (3N params) is the floor." I did not find this stated as a crossover rule.
The rank-1-only limitation of separable/outer-product mask reconstruction also belongs near
the FAMILY-LABEL / gencontract hidden-fail discussion as a "decode the shape rank first" note.
