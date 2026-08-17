"""H-Global cost-min v2: GREEDY (no brute force), iterate sources in (num_valid_candidates, comp_idx) asc order.

This is the variant we can faithfully implement in ONNX without permutation search. Test whether
this greedy approximation still passes 266/266 visible.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "The_400_Tasks/Task_by_Task/task233/prototypes"))

from task233_numpy_reference import _components, _inside, _transforms, _side, predict_task233


def _slot_box(cand):
    r, c, _, _, transformed = cand
    th, tw = transformed.shape
    return (r, c, r + th, c + tw)


def _boxes_overlap(b1, b2):
    r0a, c0a, r1a, c1a = b1
    r0b, c0b, r1b, c1b = b2
    return r0a < r1b and r0b < r1a and c0a < c1b and c0b < c1a


def predict_task233_cost_min_v2(grid_input):
    grid = np.asarray(grid_input, dtype=np.int64)
    comps = _components(grid != 0)
    main = max(comps, key=lambda c: int(c["area"]))
    main_bbox = main["bbox"]
    r0, c0, r1, c1 = main_bbox
    crop = grid[r0:r1, c0:c1]
    out = np.where(crop == 0, 2, crop).copy()
    h_crop, w_crop = crop.shape

    source_info = []
    for comp_idx, comp in enumerate(comps):
        tile_bbox = comp["bbox"]
        if _inside(tile_bbox, main_bbox):
            continue
        tr0, tc0, tr1, tc1 = tile_bbox
        tile = grid[tr0:tr1, tc0:tc1]
        if tile.shape != (3, 3):
            continue
        colors = sorted(int(x) for x in set(tile[tile != 0].tolist()))
        decoration = [c for c in colors if c != 2]
        if len(decoration) != 1:
            continue
        side = _side(main_bbox, tile_bbox)

        candidates = []
        for name, transformed in _transforms(tile):
            th, tw = transformed.shape
            for r in range(h_crop - th + 1):
                for c in range(w_crop - tw + 1):
                    window = crop[r : r + th, c : c + tw]
                    if np.array_equal(window == 0, transformed == 2) and np.all(window[transformed != 2] == 2):
                        margin = min(r, c, h_crop - r - th, w_crop - c - tw)
                        candidates.append((r, c, margin, name, transformed))

        # Pool hierarchy: rot0 > non-rot0 interior > all
        rot0_pool = [c for c in candidates if c[3] == "rot0"]
        if rot0_pool:
            pool = rot0_pool
        else:
            interior_pool = [c for c in candidates if c[2] > 0]
            pool = interior_pool if interior_pool else candidates
        # COST-MIN: sort by slot index asc
        pool_sorted = sorted(pool, key=lambda c: (c[0], c[1]))
        source_info.append((decoration[0], side, pool_sorted, comp_idx))

    if not source_info:
        return out.tolist()

    # Iterate sources in (num_valid_candidates, comp_idx) ASCENDING order.
    # Sources with fewer valid candidates are processed first ("most constrained first").
    iter_order = sorted(range(len(source_info)), key=lambda i: (len(source_info[i][2]), source_info[i][3]))

    used_boxes = []
    placements = []
    for src_idx in iter_order:
        _, _, pool_sorted, _ = source_info[src_idx]
        placed = None
        for cand in pool_sorted:
            box = _slot_box(cand)
            if not any(_boxes_overlap(box, b) for b in used_boxes):
                placed = cand
                used_boxes.append(box)
                break
        placements.append((src_idx, placed))

    # Apply placements in comp_idx order (row-major) so scatter is deterministic
    placements_by_comp = sorted(placements, key=lambda x: source_info[x[0]][3])
    for src_idx, cand in placements_by_comp:
        if cand is None:
            continue
        r, c, _, _, transformed = cand
        th, tw = transformed.shape
        out[r : r + th, c : c + tw] = transformed
    return out.tolist()


def verify():
    import json
    REPO_LOCAL = Path(__file__).resolve().parents[1]
    task = json.loads((REPO_LOCAL / "data/competitions/neurogolf-2026/task233.json").read_text())
    pairs = task["train"] + task["test"] + task.get("arc-gen", [])
    ok = 0; fails = []
    for i, pair in enumerate(pairs):
        try:
            pred = predict_task233_cost_min_v2(pair["input"])
            if pred == pair["output"]:
                ok += 1
            else:
                fails.append(i)
        except Exception as e:
            fails.append((i, str(e)))
    print(f"cost-min v2 (constraint-asc greedy): {ok}/{len(pairs)}")
    if fails:
        print(f"First fails: {fails[:5]}")
    return ok, len(pairs), fails


if __name__ == "__main__":
    verify()
