# Recovering a grid dimension from a SINGLE full-reduction scalar via a squared ramp

## What I discovered

When a generator FORCES a relationship between grid dimensions (e.g. `W = 3*H`),
you can recover a dimension-derived threshold for a per-column/row mask from a
**single full-graph ReduceSum scalar** — you do NOT need the per-column `[1,1,1,30]`
reduction intermediate at all.

NeuroGolf encoding fact that makes this work: every active cell is one-hot on the
channel axis, so `ReduceSum(input, axes=[1,2,3])` = the exact count of active cells
= `H*W` (a scalar, 4 B). Background color 0 still sets ch0=1, so it counts; padding
cells are all-zero, so they don't. This holds regardless of the colors used.

Then a comparison that is *quadratic* in the column index linearizes the constraint:

```
keep column c  <=>  c < H = W/3
               <=>  c^2 < H^2
               <=>  3*c^2 < 3*H^2 = H*W = total_cells
```

So with a static ramp `ramp_sq[c] = 3*c^2` ([1,1,1,30] initializer, 30 params) and
`keep_b = Less(ramp_sq, total_cells_scalar)`, you get the full per-column mask with
NO `[1,1,1,30]` F32 reduction intermediate (the usual `col_occ`, 120 B). The output
is the free-output `Where(keep_b, input, 0)`.

The general move: if the rule needs "first f(H,W) columns/rows" and the generator
ties dims so the boundary is monotone in a scalar S you can full-reduce, pick a ramp
g(c) such that `keep <=> g(c) < S` and bake g(c) as the initializer. Linear ramp
needs the per-axis reduction (to compare index vs total); a ramp matched to the
dimensional relationship (here squared, because total = H*W = 3*H^2 and you compare
against H) collapses the per-axis reduction into one scalar.

## Where I encountered it

- task067 (ConvWhereMicro, generator `external/ARC-GEN/tasks/task_2dee498d.py`,
  `grid = common.grid(3*size, size, 0)` → W=3H forced; output = leftmost W/3 cols).
- Builder: `task067/prototypes/build_v2_sq.py`
- Staged: `staged_submits/task067_freeout_colcrop_sqramp_cost65/`

## Why it matters

Cost 315 (aditya team-best) → **65**, byte-equal to LIVE on 200,266 corpus pairs +
100% vs ARC-GEN ground truth. Δ-LB estimate ln(315/65)=1.58. The intermediate linear
-ramp version (`Less(3*c, total)`, still keeping `col_occ[30]`) was 185; the squared
ramp removed the last [30] intermediate to reach 65.

Two reusable sub-lessons:
1. A *full* `ReduceSum(axes=[1,2,3])` to a scalar (4 B) can replace a per-axis
   reduction `[1,1,1,30]` (120 B) when a generator invariant lets you back out the
   needed dimension arithmetically. The encoding's one-hot exactness makes the cell
   count an exact `H*W`.
2. Choosing the ramp polynomial to match the dimensional relationship (squared here)
   turns a 2-reduction mask into a 1-scalar mask.

## What's missing from the library

`zero_param_generators.md` covers runtime mask/index generation (Range/Trilu/
ConstantOfShape) but I did not find the "recover a dimension from a scalar full-
reduction + non-linear static ramp, exploiting a forced generator dim-invariant"
pattern. The free-output Where part is covered (#67/#81); the *scalar-dimension-
recovery* part is the new bit.

## Sorter note:
ADD to `techniques.md` a new technique (next free §): "Scalar-dimension recovery via
squared ramp" — when a generator forces a dim relation (e.g. W=kH), recover the crop
boundary from a single full ReduceSum scalar (= H*W via one-hot exactness) compared to
a static polynomial ramp matched to the relation, eliminating the per-axis `[1,1,1,30]`
reduction intermediate. Exemplar: task067 315→65 (`build_v2_sq.py`). Cross-ref the
free-output #67/#81 entry and `zero_param_generators.md`. Tag: free-output, scalar-
reduction, generator-invariant, mask-construction.
