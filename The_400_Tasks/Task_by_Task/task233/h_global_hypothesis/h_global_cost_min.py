"""H-Global variant 'b': COST-MINIMIZING tiebreak.

Definition: among count-max non-overlapping assignments, pick the one with the LOWEST sum of
slot-position indices (r*T + c). Within source, prefer LOWEST-INDEX valid candidate; pool
hierarchy (rot0 > interior > all) preserved.

Differs from variant 'a' (count-max + rank-min): 'a' sorts candidates within pool by side score
descending (each tile's "best slot"). 'b' sorts within pool by slot-index ascending (each tile's
"smallest slot index").
"""
from __future__ import annotations

import sys
import itertools
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "The_400_Tasks/Task_by_Task/task233/prototypes"))

from task233_numpy_reference import (
    _components,
    _inside,
    _transforms,
    _side,
    predict_task233,
)


def _candidate_cost_key(cand):
    """Cost-min key: prefer smaller slot index (r*T + c)."""
    r, c, _margin, _name, _ = cand
    return (r, c)  # lex by (r, c) = same as r*T + c since lex on tuples is identical


def _slot_box(cand):
    r, c, _, _, transformed = cand
    th, tw = transformed.shape
    return (r, c, r + th, c + tw)


def _boxes_overlap(b1, b2):
    r0a, c0a, r1a, c1a = b1
    r0b, c0b, r1b, c1b = b2
    return r0a < r1b and r0b < r1a and c0a < c1b and c0b < c1a


def _slot_index(cand):
    r, c, _, _, _ = cand
    return r * 18 + c  # T=18; for sum tiebreak


def predict_task233_h_global_cost_min(grid_input):
    """H-Global hypothesis with cost-min tiebreak."""
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
        dec = decoration[0]
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

        # Preserve 3-tier pool hierarchy
        rot0_pool = [c for c in candidates if c[3] == "rot0"]
        if rot0_pool:
            pool = rot0_pool
        else:
            interior_pool = [c for c in candidates if c[2] > 0]
            pool = interior_pool if interior_pool else candidates
        # COST-MIN: sort by slot-index ASCENDING within pool
        pool_sorted = sorted(pool, key=_candidate_cost_key)
        source_info.append((dec, side, pool_sorted, comp_idx))

    if not source_info:
        return out.tolist()

    best_assignment = None
    best_score = None  # tuple: (-n_placed, slot_index_sum, lex_tuple)
    n_sources = len(source_info)

    if n_sources > 5:
        return predict_task233(grid_input)

    for perm in itertools.permutations(range(n_sources)):
        used_boxes = []
        assignment = []
        for src_idx in perm:
            dec, side, pool_sorted, comp_idx = source_info[src_idx]
            placed = None
            for cand in pool_sorted:
                box = _slot_box(cand)
                if not any(_boxes_overlap(box, b) for b in used_boxes):
                    placed = cand
                    used_boxes.append(box)
                    break
            assignment.append((src_idx, placed))

        n_placed = sum(1 for _, c in assignment if c is not None)
        # Tiebreak: sum of slot indices (lower better)
        slot_index_sum = sum(_slot_index(c) for _, c in assignment if c is not None)
        # Final tiebreak: lex-smallest tuple of slot indices in source-comp-discovery order
        sorted_by_comp = sorted(assignment, key=lambda x: source_info[x[0]][3])
        lex_tuple = tuple(_slot_index(c) if c is not None else 10**9 for _, c in sorted_by_comp)
        # Score: lower is better. Use negative n_placed so count-max stays primary.
        score = (-n_placed, slot_index_sum, lex_tuple)
        if best_score is None or score < best_score:
            best_score = score
            best_assignment = assignment

    if best_assignment is None:
        return predict_task233(grid_input)

    assignment_by_comp = sorted(best_assignment, key=lambda x: source_info[x[0]][3])
    for src_idx, cand in assignment_by_comp:
        if cand is None:
            continue
        r, c, _, _, transformed = cand
        th, tw = transformed.shape
        out[r : r + th, c : c + tw] = transformed

    return out.tolist()


def verify_visible():
    import json
    REPO_LOCAL = Path(__file__).resolve().parents[1]
    task = json.loads((REPO_LOCAL / "data/competitions/neurogolf-2026/task233.json").read_text())
    pairs = task["train"] + task["test"] + task.get("arc-gen", [])
    ok = 0
    fails = []
    for i, pair in enumerate(pairs):
        try:
            pred = predict_task233_h_global_cost_min(pair["input"])
            if pred == pair["output"]:
                ok += 1
            else:
                fails.append(i)
        except Exception as e:
            fails.append((i, str(e)))
    print(f"H-Global cost-min vs visible: {ok}/{len(pairs)}")
    if fails:
        print(f"First few fails: {fails[:5]}")
    return ok, len(pairs), fails


if __name__ == "__main__":
    verify_visible()
