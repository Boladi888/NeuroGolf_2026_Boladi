"""Canonical grid <-> [1,10,30,30] one-hot encoding for the harness.

Matches neurogolf_utils.convert_to_numpy exactly (one-hot, >30x30 -> None), but
also provides VECTORIZED batch encode from compact uint8 color-index grids so the
ARC-GEN corpus can be stored tiny and expanded on load.
"""
from __future__ import annotations

import numpy as np

H = W = 30
NCH = 10


def encode_grid(grid):
    """Single grid (list[list[int]]) -> [1,10,30,30] float32, or None if >30x30.
    Identical semantics to neurogolf_utils.convert_to_numpy for one mode."""
    if not grid or not grid[0]:
        return None
    h, w = len(grid), len(grid[0])
    if max(h, w) > 30:
        return None
    a = np.asarray(grid, dtype=np.int64)
    t = np.zeros((1, NCH, H, W), dtype=np.float32)
    for c in range(NCH):
        t[0, c, :h, :w] = (a == c)
    return t


def grid_to_index(grid):
    """Variable-size grid -> ([30,30] uint8 color-index, 0-padded), (h, w).
    For compact corpus storage. Returns (None, None) if >30x30."""
    h, w = len(grid), len(grid[0])
    if max(h, w) > 30:
        return None, None
    idx = np.zeros((H, W), dtype=np.uint8)
    idx[:h, :w] = np.asarray(grid, dtype=np.uint8)
    return idx, (h, w)


def encode_batch_index(idx, shapes):
    """idx [N,30,30] uint8 color-index + shapes [N,2] (h,w) -> [N,10,30,30] f32.
    Vectorized; padding cells (r>=h or c>=w) are all-zero (off-grid), so a real
    color-0 active cell and a padding cell don't collide."""
    idx = np.asarray(idx)
    shapes = np.asarray(shapes)
    n = idx.shape[0]
    rr = np.arange(H)[None, :, None]
    cc = np.arange(W)[None, None, :]
    active = (rr < shapes[:, 0, None, None]) & (cc < shapes[:, 1, None, None])  # [N,30,30]
    oh = np.zeros((n, NCH, H, W), dtype=np.float32)
    for c in range(NCH):
        oh[:, c, :, :] = ((idx == c) & active)
    return oh


def decode_to_grid(out_tensor, h, w):
    """[1,10,30,30] (already >0-normalized or one-hot) -> h x w color grid (list)."""
    o = np.asarray(out_tensor)[0]
    cells = (o > 0.0)
    grid = []
    for r in range(h):
        row = []
        for c in range(w):
            idxs = np.where(cells[:, r, c])[0]
            row.append(int(idxs[0]) if len(idxs) == 1 else (-1 if len(idxs) == 0 else 99))
        grid.append(row)
    return grid
