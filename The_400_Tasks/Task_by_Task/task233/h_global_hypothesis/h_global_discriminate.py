"""Construct DISCRIMINATING INPUTS where H-Global and structural-slots-greedy produce different outputs.

This is the differentiating-evidence required by the pivot prompt before shipping a new hypothesis.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))  # h_global_hypothesis/ (was .tmp_h3/)
sys.path.insert(0, str(REPO / "The_400_Tasks/Task_by_Task/task233/prototypes"))

from task233_numpy_reference import predict_task233
from h_global_numpy import predict_task233_h_global


def render(g):
    g = np.asarray(g)
    return '\n'.join(''.join('.' if v == 0 else str(int(v)) for v in row) for row in g)


def construct_discriminator_1():
    """Two source tiles with SAME shape pattern (different colors), and two same-shape slots.
    Greedy: both tiles want same slot, later iteration overwrites earlier. Only one tile visible.
    H-Global: tiles assigned to distinct slots, both visible."""
    grid = np.zeros((20, 20), dtype=np.int64)
    # Main crop: rows 2-11, cols 2-11 (10x10), filled with 2s
    grid[2:12, 2:12] = 2
    # Two identical hole-clusters (cross-pattern centered):
    # Cluster A at crop (1-3, 1-3) = grid (3-5, 3-5)
    grid[3, 4] = 0
    grid[4, 3] = 0
    grid[4, 4] = 0
    grid[4, 5] = 0
    grid[5, 4] = 0
    # Cluster B at crop (5-7, 5-7) = grid (7-9, 7-9)
    grid[7, 8] = 0
    grid[8, 7] = 0
    grid[8, 8] = 0
    grid[8, 9] = 0
    grid[9, 8] = 0
    # Two source tiles below main (rows 14-16), same shape (4 corners decoration + cross 2s).
    # Tile 1 (color 5) at cols 2-4:
    grid[14, 2] = 5; grid[14, 3] = 2; grid[14, 4] = 5
    grid[15, 2] = 2; grid[15, 3] = 2; grid[15, 4] = 2
    grid[16, 2] = 5; grid[16, 3] = 2; grid[16, 4] = 5
    # Tile 2 (color 7) at cols 7-9:
    grid[14, 7] = 7; grid[14, 8] = 2; grid[14, 9] = 7
    grid[15, 7] = 2; grid[15, 8] = 2; grid[15, 9] = 2
    grid[16, 7] = 7; grid[16, 8] = 2; grid[16, 9] = 7
    return grid


def run_discriminator(grid):
    greedy = np.asarray(predict_task233(grid.tolist()))
    global_h = np.asarray(predict_task233_h_global(grid.tolist()))
    return greedy, global_h


def main():
    grid = construct_discriminator_1()
    print("DISCRIMINATOR 1: Two same-shape tiles + two same-shape slots")
    print("-" * 60)
    print("INPUT (20x20):")
    print(render(grid))
    print()

    greedy, global_h = run_discriminator(grid)
    print(f"GREEDY structural-slots output ({greedy.shape}):")
    print(render(greedy))
    print()
    print(f"H-GLOBAL output ({global_h.shape}):")
    print(render(global_h))
    print()
    print(f"Outputs differ: {not np.array_equal(greedy, global_h)}")
    if not np.array_equal(greedy, global_h):
        diff = (greedy != global_h)
        print(f"Number of differing cells: {int(diff.sum())}")
        print(f"Greedy has 5s at: {[tuple(p) for p in np.argwhere(greedy == 5).tolist()]}")
        print(f"Greedy has 7s at: {[tuple(p) for p in np.argwhere(greedy == 7).tolist()]}")
        print(f"H-Global has 5s at: {[tuple(p) for p in np.argwhere(global_h == 5).tolist()]}")
        print(f"H-Global has 7s at: {[tuple(p) for p in np.argwhere(global_h == 7).tolist()]}")


if __name__ == "__main__":
    main()
