# task374 worker3 scratch plan

## Decoded rule
Color 5 cells are non-overlapping horizontal or vertical line segments on a 10x10 grid. Rank the distinct positive segment lengths globally: longest -> color 1, second-longest -> color 4, remaining positive lengths -> color 2. Horizontal cells use their row rank; vertical cells use their column rank. Background stays color 0.

## Output-size relation
Input and visible output grids are 10x10, embedded in the NeuroGolf 30x30 one-hot frame. A cheap exact graph must avoid materializing a padded label or one-hot plane as an intermediate; the final op should write `[1,10,30,30]` directly.

## Falsification cases
- Three segments with lengths 6, 5, and 3, split across both orientations: verifies max/middle/min coloring, not per-row raw count.
- A row containing a perpendicular singleton from a vertical line: rejects raw row/column occupancy count as a segment-length proxy.
- A horizontal endpoint versus a vertical segment cell: verifies orientation selection is based on same-orientation neighbors, not just color-5 presence.

## ONNX families considered
- v8-style label tail: exact but pays `label30` 900 B plus `ch5_f32` 400 B before ranking, so it cannot beat the 1366 team-best bar.
- Direct two-bit output via `ConvInteger`: current team-best family; hard tensors are `gray_crop_f` 400 B, two 200 B selected/masked feature planes, and small row/column rank tensors.
- One-scalar class code: not credible for 4 mutually exclusive decoded classes under `>0` without a bias or extra threshold plane.

## Chosen probe
Keep the current two-bit `ConvInteger` output family, but delete the extra `h_scale` scalar. Set the shared QLinearConv scale to 0.2 and multiply `row_kernel`/`col_kernel` entries by 5. This preserves:

- horizontal mask: `(1,2,1)` sums 2/3/4 become round(0.2 * sum) = 0/1/1;
- row/column counts: kernels of 5 make round(0.2 * 5 * count) = count.

Expected cost delta: `1366 -> 1365` by removing one initializer element, with unchanged memory profile.
