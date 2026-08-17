# Task374 Rule And Facts

## Task Summary

Family: color and logical transformation.

Visible grid: 10x10. NeuroGolf output frame: 10 channels by 30x30, with the
visible 10x10 result embedded in the top-left region.

Input signal: color 5 line segments on background 0.

Output signal: recolor only the color-5 segment cells. Background remains 0.

## Rule

1. Build a mask for cells equal to color 5.
2. For every row, compute the maximum contiguous horizontal run of color-5
   cells in that row.
3. For every column, compute the maximum contiguous vertical run of color-5
   cells in that column.
4. Rank all distinct positive row and column run lengths globally.
5. Longest length maps to color 1.
6. Second-longest length maps to color 4.
7. Any other positive length maps to color 2.
8. For a color-5 cell with a vertical color-5 neighbor, use the column run rank.
9. Otherwise use the row run rank.

## Falsification Cases To Think About

- A row may contain a cell from a perpendicular vertical segment. Raw row counts
  are not enough.
- A column may contain a cell from a perpendicular horizontal segment. Raw column
  counts are not enough.
- The second-longest distinct segment length must map to color 4, not color 2.
- The output has four classes to represent in the final visible grid:
  background, longest, second-longest, and remaining.

## Bundled Raw Inputs

- `task374.json` contains the original task examples.
- `task374_preview.png` gives a visual-first overview.

The worker should reason from these before anchoring on existing ONNX graphs.
