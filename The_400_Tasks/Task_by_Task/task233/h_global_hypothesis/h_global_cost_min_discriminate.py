"""Discriminators separating H-Global cost-min ('b') from BOTH greedy AND count-max ('a').

Existing disc2 (in h_global_discriminate2.py) already separates a vs b. We construct 2 more
disc inputs to satisfy the prompt's "≥3 discriminating inputs" gate.

Methodology: scenarios where score-descending order (count-max) and slot-index-ascending order
(cost-min) pick DIFFERENT slots for the same tile.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "The_400_Tasks/Task_by_Task/task233/prototypes"))
sys.path.insert(0, str(REPO / ".tmp_h3"))

from task233_numpy_reference import predict_task233
from h_global_numpy import predict_task233_h_global
from h_global_cost_min import predict_task233_h_global_cost_min


def render(g):
    g = np.asarray(g)
    return '\n'.join(''.join('.' if v == 0 else str(int(v)) for v in row) for row in g)


def construct_disc_b1():
    """2 same-shape tiles below + 3 same-shape slots (interior + 2 edge).
    count-max: tile 1 → interior slot c=4 (higher margin), tile 2 → edge c=0 (higher score).
    cost-min: tile 1 → smallest-index slot c=0, tile 2 → c=4."""
    grid = np.zeros((25, 25), dtype=np.int64)
    # Main crop 11x11 at rows 2-12, cols 2-12
    grid[2:13, 2:13] = 2
    # 3 vertical-line hole-clusters at crop cols 1, 5, 9
    for slot_col in [1, 5, 9]:
        for dr in range(3):
            grid[3 + dr, 2 + slot_col] = 0
    # 2 source tiles below (rows 15-17), same shape
    for col_start, color in [(2, 5), (7, 7)]:
        for ri in range(3):
            grid[15 + ri, col_start] = color
            grid[15 + ri, col_start + 1] = 2
            grid[15 + ri, col_start + 2] = color
    return grid


def construct_disc_b2():
    """3 same-shape tiles below + 3 same-shape slots arranged so margin order differs from index order.
    Slots at cols 1, 5, 9 of an 11-wide crop:
      c=1 (edge, margin=0), c=5 (interior, margin=1), c=9 (edge, margin=0).
    Score-desc order: c=5 (margin=1, beats both edges); within edges score key uses -c so c=1 beats c=9.
    Slot-index-asc order: c=1, c=5, c=9.
    With 3 tiles iterating sequentially (numpy uses comp-discovery = source row-major order):
      count-max: t0→c=5 (best), t1→c=1 (next), t2→c=9 (last)
      cost-min:  t0→c=1 (smallest idx), t1→c=5 (next), t2→c=9 (last)
    Tile-to-slot assignments differ (e.g., tile color 4 ends up at c=5 in count-max vs c=1 in cost-min)."""
    return None  # = disc2 essentially; reuse via test loop


def construct_disc_b3():
    """LEFT-side variant: 2 same-shape tiles to the LEFT of main crop, 3 same-shape slots in crop.
    Side = 'left', score key = (c, margin, -r, rot0) → larger c first.
    Cost-min: smaller (r, c) first.
    Two methods pick different slots."""
    grid = np.zeros((25, 25), dtype=np.int64)
    # Main crop 11x11 at rows 6-17, cols 8-19
    grid[6:17, 8:19] = 2
    # 3 horizontal-line hole-clusters in same column, at crop rows 1, 5, 9, col 1 of crop
    # Hole pattern: 3 cells in a row at crop col 1
    for slot_row in [1, 5, 9]:
        for dc in range(3):
            grid[6 + slot_row, 8 + 1 + dc] = 0
    # Wait — that's vertical position. Let me make it horizontal slot pattern.
    # Actually make 3 vertical-line hole-clusters at crop cols 1, 5, 9 row 1:
    grid[6:17, 8:19] = 2
    for slot_col in [1, 5, 9]:
        for dr in range(3):
            grid[7 + dr, 8 + slot_col] = 0
    # 2 source tiles to LEFT of crop (cols 2-4 and 2-4? need different rows)
    # Tiles at rows 6-8 (left of crop rows) and 12-14, both at cols 2-4
    for row_start, color in [(8, 3), (12, 6)]:
        for rj in range(3):
            grid[row_start + rj, 2] = color
            grid[row_start + rj, 3] = 2
            grid[row_start + rj, 4] = color
    return grid


def construct_disc_b5():
    """Inverted-color variant of disc_b1: tile color 7 at left, tile color 5 at right.
    Differentiates ONNX count-max (color-order iteration → tile 7 takes interior c=4)
    from ONNX cost-min (row-major iteration → tile 7 at col 2 takes smallest c=0)."""
    grid = np.zeros((25, 25), dtype=np.int64)
    grid[2:13, 2:13] = 2
    for slot_col in [1, 5, 9]:
        for dr in range(3):
            grid[3 + dr, 2 + slot_col] = 0
    # Swapped colors: tile color 7 at LEFT (col 2-4), tile color 5 at RIGHT (col 7-9)
    for col_start, color in [(2, 7), (7, 5)]:
        for ri in range(3):
            grid[15 + ri, col_start] = color
            grid[15 + ri, col_start + 1] = 2
            grid[15 + ri, col_start + 2] = color
    return grid


def construct_disc_b4():
    """ABOVE side: 3 tiles above the crop, 3 same-shape slots inside crop.
    side = 'above', score key = (r, margin, -c, rot0) → larger r first = bottom of crop preferred.
    cost-min: smallest (r, c).
    Distinct slot positions → different assignment under each tiebreak."""
    grid = np.zeros((25, 25), dtype=np.int64)
    # Main crop 11x11 at rows 8-19, cols 2-13
    grid[8:19, 2:13] = 2
    # 3 vertical-line hole-clusters spread along crop in different rows.
    # Slot positions: (r=1, c=4), (r=5, c=4), (r=7, c=4)
    # Add interior preference variation: slots at (r=1, c=4) edge-top, (r=5, c=4) interior, (r=7, c=4) edge-bottom
    for slot_r, slot_c in [(1, 4), (5, 4), (7, 4)]:
        for dr in range(3):
            grid[8 + slot_r + dr, 2 + slot_c] = 0
    # 3 source tiles above the crop (rows 2-4), same shape
    for col_start, color in [(2, 4), (7, 6), (12, 9)]:
        for ri in range(3):
            grid[2 + ri, col_start] = color
            grid[2 + ri, col_start + 1] = 2
            grid[2 + ri, col_start + 2] = color
    return grid


def main():
    # Use disc_b1 and disc_b3 and disc_b4 (skip b2 which is just disc2)
    disc_cases = [
        ("Disc_b1 (2 tiles below + 3 same-shape slots)", construct_disc_b1()),
        ("Disc_b3 (2 tiles left + 3 same-shape slots)", construct_disc_b3()),
        ("Disc_b4 (3 tiles above + 3 same-shape slots)", construct_disc_b4()),
    ]
    for label, grid in disc_cases:
        print(label)
        print("=" * 70)
        g = np.asarray(predict_task233(grid.tolist()))            # greedy
        a = np.asarray(predict_task233_h_global(grid.tolist()))   # count-max
        b = np.asarray(predict_task233_h_global_cost_min(grid.tolist()))  # cost-min
        print(f"  greedy:   {g.shape}")
        print(f"  count-max:{a.shape}")
        print(f"  cost-min: {b.shape}")
        print(f"  cost-min != greedy:    {not np.array_equal(b, g)}")
        print(f"  cost-min != count-max: {not np.array_equal(b, a)}")
        if not np.array_equal(b, a) or not np.array_equal(b, g):
            print(f"  greedy out:");   print(g)
            print(f"  count-max out:"); print(a)
            print(f"  cost-min out:");  print(b)
        print()


if __name__ == "__main__":
    main()
