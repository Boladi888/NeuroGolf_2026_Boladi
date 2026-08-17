"""Karnak-informed mining angles vs the truly-unsolved 150 set.

Three angles informed by karnak transformation labels (`tools/karnak_xref.py`):

  - foreground-bbox-crop : output == input cropped to the bounding box of non-zero cells
  - color-filter-bbox-crop : output == input cropped to bbox of cells with one specific color
                              (the "kept" color); other colors are masked out
  - symmetry-completion-h : input has a horizontal axis of symmetry; one half is
                              partly zeroed; output fills the zeros via reflection

Each angle is a numpy detector that takes pairs and returns either None (no match)
or a small dict (params for downstream Alpha implementation).

Cross-references hits against the 150 truly-unsolved set so we focus on pure-LB-gain
candidates.

Usage:
    python tools/mine_karnak_targeted.py
    python tools/mine_karnak_targeted.py --truly-unsolved-only
    python tools/mine_karnak_targeted.py --angle foreground-bbox-crop

Cross-worktree note: set NEUROGOLF_DATA_DIR if running outside main worktree.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mining_lib import iter_unsolved_tasks, load_pairs, print_report, run_sweep, SOLVED_TASKS  # noqa: E402

NG_ROOT = Path(__file__).resolve().parents[1]


def _truly_unsolved_set():
    """Tasks not solved organically AND not covered by external (per latest blend csv)."""
    candidates = sorted((NG_ROOT / "Logs").glob("external_blend_*.csv"))
    if not candidates:
        return None
    external_covered = set()
    with open(candidates[-1]) as f:
        for row in csv.DictReader(f):
            tid = int(row["task"])
            if row.get("winner_src", "").strip():
                external_covered.add(tid)
    truly_unsolved = set()
    for tid in range(1, 401):
        if tid not in SOLVED_TASKS and tid not in external_covered:
            truly_unsolved.add(tid)
    return truly_unsolved


def _foreground_bbox(arr):
    """Return (r0, r1, c0, c1) for non-zero bbox, or None."""
    mask = arr != 0
    if not mask.any():
        return None
    rs = np.where(mask.any(axis=1))[0]
    cs = np.where(mask.any(axis=0))[0]
    return int(rs[0]), int(rs[-1] + 1), int(cs[0]), int(cs[-1] + 1)


def detect_foreground_bbox_crop(pairs):
    """output == input cropped to non-zero bbox."""
    if not pairs:
        return None
    for p in pairs:
        inp = np.array(p["input"])
        out = np.array(p["output"])
        bbox = _foreground_bbox(inp)
        if bbox is None:
            return None
        r0, r1, c0, c1 = bbox
        cropped = inp[r0:r1, c0:c1]
        if cropped.shape != out.shape or not np.array_equal(cropped, out):
            return None
    return {"rule": "output = input[non_zero_bbox]"}


def detect_color_filter_bbox_crop(pairs):
    """For some color C consistent across pairs, output == bbox-crop of cells where input == C.

    Note: C may differ per pair (each pair finds its own C); but the rule is "find one color
    such that bbox-crop-of-that-color matches output".
    """
    if not pairs:
        return None
    # Find a candidate color C for each pair; if pair-specific C is consistent across pairs
    # by some heuristic (e.g. always the "most common non-zero", "the unique non-bg color",
    # or "a specific named color"), this matches.
    chosen_colors = []
    for p in pairs:
        inp = np.array(p["input"])
        out = np.array(p["output"])
        candidates = sorted(set(int(c) for c in inp.flatten()) - {0})
        match_color = None
        for c in candidates:
            mask = inp == c
            if not mask.any():
                continue
            rs = np.where(mask.any(axis=1))[0]
            cs = np.where(mask.any(axis=0))[0]
            crop = inp[rs[0]:rs[-1]+1, cs[0]:cs[-1]+1]
            if crop.shape == out.shape and np.array_equal(crop, out):
                match_color = c
                break
        if match_color is None:
            return None
        chosen_colors.append(match_color)
    return {"rule": "output = input[bbox(color C)] where C varies per pair", "sample_colors": chosen_colors[:5]}


def _is_h_symmetric(arr):
    """Returns True if arr is mirror-symmetric across the horizontal axis (top<->bottom)."""
    return np.array_equal(arr, arr[::-1, :])


def _is_v_symmetric(arr):
    """Returns True if arr is mirror-symmetric across the vertical axis (left<->right)."""
    return np.array_equal(arr, arr[:, ::-1])


def detect_symmetry_completion_h(pairs):
    """Output is the input with zeros filled in by reflecting across horizontal axis.

    More precisely: output[r, c] = input[r, c] if input[r, c] != 0 else input[H-1-r, c].
    Output must be h-symmetric (verifies the rule).
    """
    if not pairs:
        return None
    matched = 0
    for p in pairs:
        inp = np.array(p["input"])
        out = np.array(p["output"])
        if inp.shape != out.shape:
            return None
        H = inp.shape[0]
        # Reflect along axis 0 (h-flip means top<->bottom for visual; rows reversed)
        reflected = inp[::-1, :]
        # Fill zero cells with reflected
        filled = np.where(inp != 0, inp, reflected)
        if not np.array_equal(filled, out):
            return None
        if not _is_h_symmetric(out):
            return None
        matched += 1
    return {"rule": "output = input fill-zeros-from-h-mirror; output is h-symmetric"}


def detect_symmetry_completion_v(pairs):
    """Same but for vertical-axis reflection."""
    if not pairs:
        return None
    for p in pairs:
        inp = np.array(p["input"])
        out = np.array(p["output"])
        if inp.shape != out.shape:
            return None
        reflected = inp[:, ::-1]
        filled = np.where(inp != 0, inp, reflected)
        if not np.array_equal(filled, out):
            return None
        if not _is_v_symmetric(out):
            return None
    return {"rule": "output = input fill-zeros-from-v-mirror; output is v-symmetric"}


def detect_symmetry_completion_both(pairs):
    """Fill zeros via reflection across BOTH horizontal AND vertical axes.

    For each zero cell, take the first non-zero among: input[H-1-r, c], input[r, W-1-c],
    input[H-1-r, W-1-c]. Output should be symmetric in both axes.
    """
    if not pairs:
        return None
    for p in pairs:
        inp = np.array(p["input"])
        out = np.array(p["output"])
        if inp.shape != out.shape:
            return None
        H, W = inp.shape
        filled = inp.copy()
        for r in range(H):
            for c in range(W):
                if filled[r, c] != 0:
                    continue
                for rr, cc in [(H-1-r, c), (r, W-1-c), (H-1-r, W-1-c)]:
                    if inp[rr, cc] != 0:
                        filled[r, c] = inp[rr, cc]
                        break
        if not np.array_equal(filled, out):
            return None
    return {"rule": "output = input fill-zeros-from-h+v-mirrors"}


CANDIDATES = {
    "foreground-bbox-crop": detect_foreground_bbox_crop,
    "color-filter-bbox-crop": detect_color_filter_bbox_crop,
    "symmetry-completion-h": detect_symmetry_completion_h,
    "symmetry-completion-v": detect_symmetry_completion_v,
    "symmetry-completion-both": detect_symmetry_completion_both,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truly-unsolved-only", action="store_true",
                    help="Filter targets to truly-unsolved set (excludes external-covered).")
    ap.add_argument("--angle", choices=list(CANDIDATES.keys()),
                    help="Run only one angle.")
    args = ap.parse_args()

    if args.truly_unsolved_only:
        targets = sorted(_truly_unsolved_set())
        print(f"=== Mining target: truly-unsolved set ({len(targets)} tasks) ===\n")
    else:
        targets = list(iter_unsolved_tasks())
        print(f"=== Mining target: all unsolved ({len(targets)} tasks) ===\n")

    candidates = {args.angle: CANDIDATES[args.angle]} if args.angle else CANDIDATES
    results = run_sweep(candidates, targets)
    print_report(results, target_n=len(targets))


if __name__ == "__main__":
    main()
