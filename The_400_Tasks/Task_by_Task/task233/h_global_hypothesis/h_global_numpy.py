"""H-Global hypothesis: task233 uses GLOBAL non-overlapping assignment.

Per arc_rule_taxonomy.md open family "general packing, assignment, and collision-free layout":
multiple tiles must be assigned to UNIQUE slots so no overlaps occur. The greedy structural-slots
rule (current wrong baseline) picks per-tile independently, which can OVERWRITE one tile with
another when they share a matching slot pattern.

Algorithm:
1. For each source tile, enumerate ALL valid (slot, transform) candidates.
2. Find an assignment (source -> slot) such that:
   a. Each source is assigned to a valid candidate (or skipped if no valid).
   b. No two sources share the same slot (no overlap).
3. Among valid assignments, prefer the one with the most-preferred score under the existing
   numpy ref tiebreak (per source, score is the lex (position, margin, tertiary, rot0_flag)).

For ONNX simplicity we'll use a brute-force search over permutations (MAX_TILES = 5, so up to
5! = 120 permutations to try).
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
    _choose_candidate,
    predict_task233,
)


def _candidate_score_key(cand, side):
    """Returns the numpy-ref scoring key for a candidate (used to compare alternatives)."""
    r, c, margin, name, _ = cand
    rot0 = (name == "rot0")
    if side == "above":
        return (r, margin, -c, rot0)
    if side == "below":
        return (-r, margin, -c, rot0)
    if side == "left":
        return (c, margin, -r, rot0)
    if side == "right":
        return (-c, margin, -r, rot0)
    return (margin, rot0)


def _slot_box(cand):
    r, c, _, _, transformed = cand
    th, tw = transformed.shape
    return (r, c, r + th, c + tw)


def _boxes_overlap(b1, b2):
    r0a, c0a, r1a, c1a = b1
    r0b, c0b, r1b, c1b = b2
    return r0a < r1b and r0b < r1a and c0a < c1b and c0b < c1a


def predict_task233_h_global(grid_input):
    """H-Global hypothesis: assign each source to a unique non-overlapping slot."""
    grid = np.asarray(grid_input, dtype=np.int64)
    comps = _components(grid != 0)
    main = max(comps, key=lambda c: int(c["area"]))
    main_bbox = main["bbox"]
    r0, c0, r1, c1 = main_bbox
    crop = grid[r0:r1, c0:c1]
    out = np.where(crop == 0, 2, crop).copy()
    h_crop, w_crop = crop.shape

    # Gather source tiles + their candidate lists.
    source_info = []  # list of (decoration, side, candidates_sorted_by_score_desc, comp_idx)
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

        # Use the SAME 3-tier preference as numpy ref: prefer rot0 if any, else interior if any, else all.
        # Within preferred pool, sort by score key descending.
        rot0_pool = [c for c in candidates if c[3] == "rot0"]
        if rot0_pool:
            pool = rot0_pool
        else:
            interior_pool = [c for c in candidates if c[2] > 0]
            pool = interior_pool if interior_pool else candidates
        pool_sorted = sorted(pool, key=lambda c: _candidate_score_key(c, side), reverse=True)
        source_info.append((dec, side, pool_sorted, comp_idx))

    if not source_info:
        return out.tolist()

    # Brute-force search: try all orderings of sources, greedy-pick best non-conflicting candidate per source.
    # Track the assignment that places the MOST tiles (and within those, maximizes per-source preferred-rank sum).
    best_assignment = None  # list of (source_idx, candidate)
    best_score = -1
    n_sources = len(source_info)

    # If too many sources for brute-force, fall back to row-major iteration.
    if n_sources > 5:
        # Fallback: row-major greedy as current numpy ref.
        return predict_task233(grid_input)

    for perm in itertools.permutations(range(n_sources)):
        used_boxes = []
        assignment = []  # (source_idx, candidate_or_None)
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
        # Tie-break: prefer assignment where each source got its TOP-RANKED candidate (lower rank = better).
        rank_score = 0
        for src_idx, cand in assignment:
            if cand is None:
                continue
            _, _, pool_sorted, _ = source_info[src_idx]
            rank_score += pool_sorted.index(cand)
        # Higher n_placed is better; lower rank_score is better.
        score = n_placed * 10000 - rank_score
        if score > best_score:
            best_score = score
            best_assignment = assignment

    # Apply best assignment in row-major source order (so later-iteration overwrites in case of true overlap, but we minimized conflicts).
    if best_assignment is None:
        return predict_task233(grid_input)

    # Sort by source comp_idx (row-major component discovery order, matching numpy ref iteration).
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
            pred = predict_task233_h_global(pair["input"])
            if pred == pair["output"]:
                ok += 1
            else:
                fails.append(i)
        except Exception as e:
            fails.append((i, str(e)))
    print(f"H-Global vs visible: {ok}/{len(pairs)}")
    if fails:
        print(f"First few fails: {fails[:5]}")


if __name__ == "__main__":
    verify_visible()
