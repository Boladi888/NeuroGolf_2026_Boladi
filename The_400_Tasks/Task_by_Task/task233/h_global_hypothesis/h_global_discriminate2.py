"""Additional discriminating inputs for H-Global vs greedy."""
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


def construct_discriminator_2():
    """3 same-shape tiles + 3 same-shape slots. Greedy: all 3 want same slot, only 1 visible.
    H-Global: 3 distinct placements."""
    grid = np.zeros((25, 25), dtype=np.int64)
    grid[2:13, 2:13] = 2  # 11x11 crop
    # 3 identical hole-clusters (vertical line of 3 holes):
    grid[3, 3] = 0; grid[4, 3] = 0; grid[5, 3] = 0  # Cluster A
    grid[3, 7] = 0; grid[4, 7] = 0; grid[5, 7] = 0  # Cluster B
    grid[3, 11] = 0; grid[4, 11] = 0; grid[5, 11] = 0  # Cluster C
    # 3 tiles with vertical-line pattern (column 1 = 2, rest = decoration):
    # Tile shape: 3 cells of 2 in middle column, 6 cells of decoration
    for col_start, color in [(2, 4), (7, 6), (12, 9)]:
        for ri in range(3):
            grid[15 + ri, col_start] = color
            grid[15 + ri, col_start + 1] = 2
            grid[15 + ri, col_start + 2] = color
    return grid


def construct_discriminator_3():
    """2 source tiles whose ONLY valid match is the same slot.
    Greedy: one is lost. H-Global: only 1 placed (since only 1 slot fits both)."""
    grid = np.zeros((20, 20), dtype=np.int64)
    grid[2:11, 2:11] = 2  # 9x9 crop
    # ONE hole-cluster (4 holes at L-shape):
    grid[3, 3] = 0; grid[3, 4] = 0; grid[4, 3] = 0; grid[4, 4] = 0
    # 2 tiles, both match the L-shape:
    # Tile shape: 4 cells of 2 at top-left 2x2, 5 cells of decoration:
    # 2 2 D
    # 2 2 D
    # D D D
    for col_start, color in [(2, 5), (7, 7)]:
        # Top-left 2x2 = 2s, rest = decoration
        grid[14, col_start] = 2; grid[14, col_start+1] = 2; grid[14, col_start+2] = color
        grid[15, col_start] = 2; grid[15, col_start+1] = 2; grid[15, col_start+2] = color
        grid[16, col_start] = color; grid[16, col_start+1] = color; grid[16, col_start+2] = color
    return grid


def main():
    for label, grid in [("DISCRIMINATOR 2 (3 same-shape tiles, 3 slots)", construct_discriminator_2()),
                         ("DISCRIMINATOR 3 (2 same-shape tiles, 1 slot)", construct_discriminator_3())]:
        print(label)
        print("=" * 70)
        print("INPUT:")
        print(render(grid))
        greedy = np.asarray(predict_task233(grid.tolist()))
        global_h = np.asarray(predict_task233_h_global(grid.tolist()))
        print(f"\nGREEDY ({greedy.shape}):")
        print(render(greedy))
        print(f"\nH-GLOBAL ({global_h.shape}):")
        print(render(global_h))
        differ = not np.array_equal(greedy, global_h)
        print(f"\nDIFFER: {differ}")
        if differ:
            for color in range(1, 10):
                g_cells = [(r, c) for r, c in zip(*np.where(greedy == color))]
                h_cells = [(r, c) for r, c in zip(*np.where(global_h == color))]
                if g_cells != h_cells:
                    print(f"  Color {color}: greedy={g_cells}, h_global={h_cells}")
        print()
        print()


if __name__ == "__main__":
    main()
