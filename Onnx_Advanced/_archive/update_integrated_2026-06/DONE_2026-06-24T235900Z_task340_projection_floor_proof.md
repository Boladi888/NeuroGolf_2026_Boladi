# task340 border-projection floor proof (2026-06-24)

**Task**: ARC-GEN d687bc17 — border-color projection to interior rows/cols.
**Current cost**: 8,612 (mem 8,420 + params 192). Floor confirmed.

## Key traps discovered

### 1. Source-slice border inclusion trap
When slicing source_color from a fixed [2..H_max-2, 2..W_max-2] window in a fixed-size
H_max×W_max canvas, SMALLER grids (H < H_max or W < W_max) have their BORDER CELLS
(row H-1 and/or col W-1) INSIDE the fixed slice window.

Example: W=15 grid in a 20×20 canvas. Col W-1=14 (right border) falls inside the fixed
[2..18] column slice. The right border color appears in source_color, falsely matching
right_color and corrupting the hit lines.

**Fix required**: `interior_color = Where(interior, color20, sentinel10)` BEFORE slicing
source. Cannot be eliminated. Attempts to remove it produce 0/266 visible accuracy.

### 2. Max operations NOT redundant for projection rows
`canvas_border = Where(interior, 0, color20)` keeps color20 at NON-INTERIOR cells.
At row 1 (an interior row), cols 0 and W-1 are NOT interior (col_inside is False).
Therefore `canvas_border[row1, col0] = left_edge_color` and `canvas_border[row1, colW-1] = right_edge_color`.

The hit_line pad adds 2 zeros at positions 0,1 and W-2,W-1. So `hit_line[col0] = 0`.
`Max(hit_line, canvas_border_row1)` at col 0 = max(0, left_color) = left_color.
This preserves the border color at the intersection of projection row × border column.
REMOVING the Max breaks border colors at projection rows.

### 3. Architecture floor for this class
This "4-edge-color border projection" pattern costs approximately:
- Conv argmax (1600B F32, unavoidable)
- 3 canvas copies at 400B each (border → row-scatter → col-scatter)
- 8 × 256B source match/hit planes (4 edges × BOOL match + U8 hit)
- 900B canvas30 for 30×30 output Pad
- F16 counting chain (minimum dtype for ReduceSum)

Total ~8,500-8,600B is a hard floor for exact full-size formulations.
Sub-5k requires partial crops, which fail ARC-GEN visible test (grids up to 20×20).

## Where it matters
Task: task340 (`The_400_Tasks/Task_by_Task/task340.md`)
Build: `The_400_Tasks/Task_by_Task/task340/prototypes/build_task340_v1.py`
SHA256 of live best: af5dc9c8623206d04fd752854fa0a508af822e560b677b6b9b208c83ee2 0d59d

## Library gap
No existing techniques.md section covers the "fixed-window border-cell inclusion" trap
for sliced source windows. Should be added as a dont-do in any task using fixed-slice
source windows on variable-size grids.
