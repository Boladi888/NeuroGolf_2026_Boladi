"""Mining sweep using detectors ported from rule-based-onnx-solver notebook.

Mines truly-unsolved 150 set with detectors NOT yet in our build pipeline:

  - bbox-fill : bbox of non-zero cells gets filled with fixed color
  - hole-fill : flood-fill from boundary; interior holes get fixed color
  - outline-extract : keep only boundary cells of each non-zero region
  - keep-largest-color / keep-smallest-color : keep cells of dominant color
  - period-row-tiling / period-col-tiling : input has periodic rows/cols
  - bbox-then-tile : output = bbox crop tiled
  - frame-border : output = input + 1-pixel border of fixed color
  - border-add : output = input padded with fixed-color border (configurable thickness)
  - nonzero-recolor / bg-recolor
  - fixed-largest-bbox-crop : crop to LARGEST non-zero CC's bbox
  - diag-symmetry-complete : fill zeros from transpose
  - translation : non-zero shift of cells (shape-preserving)
  - color-filter-remove : remove specific colors (set to 0)
  - row-broadcast / col-broadcast : output = single row/col broadcasted
  - concat-rot180-w : input concatenated with 180-rotation horizontally
  - mirror-h-concat / mirror-v-concat : input concat with h/v flip
  - quad-mirror : 2x2 mirror tile

Cross-references hits against the truly-unsolved set.

Usage:
    python tools/mine_rule_based_detectors.py
    python tools/mine_rule_based_detectors.py --truly-unsolved-only
    python tools/mine_rule_based_detectors.py --angle bbox-fill
"""

import argparse
import csv
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mining_lib import iter_unsolved_tasks, print_report, run_sweep, SOLVED_TASKS  # noqa: E402

NG_ROOT = Path(__file__).resolve().parents[1]


def _truly_unsolved_set():
    candidates = sorted((NG_ROOT / "Logs").glob("external_blend_*.csv"))
    if not candidates:
        return None
    external_covered = set()
    with open(candidates[-1]) as f:
        for row in csv.DictReader(f):
            tid = int(row["task"])
            if row.get("winner_src", "").strip():
                external_covered.add(tid)
    return {tid for tid in range(1, 401) if tid not in SOLVED_TASKS and tid not in external_covered}


def _shapes_match(p):
    return np.array(p["input"]).shape == np.array(p["output"]).shape


def detect_bbox_fill(pairs):
    """Output = input but the bbox of non-zero cells gets filled with fixed color C
    (excluding the original non-zero cells which are preserved)."""
    if not all(_shapes_match(p) for p in pairs):
        return None
    fill = None
    for p in pairs:
        ig = np.array(p["input"])
        og = np.array(p["output"])
        mask = ig != 0
        if not mask.any():
            if not np.array_equal(ig, og):
                return None
            continue
        rs, cs = np.where(mask)
        r0, r1, c0, c1 = rs.min(), rs.max(), cs.min(), cs.max()
        inside = np.zeros_like(ig, dtype=bool)
        inside[r0:r1+1, c0:c1+1] = True
        if not np.array_equal(ig[~inside], og[~inside]):
            return None
        changed = (ig != og) & inside
        if np.any(ig[changed] != 0):
            return None
        new_vals = og[changed]
        if new_vals.size == 0:
            continue
        u = np.unique(new_vals)
        if len(u) != 1:
            return None
        if fill is None:
            fill = int(u[0])
        elif fill != int(u[0]):
            return None
    return {"fill_color": fill} if fill is not None else None


def detect_hole_fill(pairs):
    """Flood-fill 0-cells from boundary; remaining 0-cells (interior holes) get filled with fixed color."""
    if not all(_shapes_match(p) for p in pairs):
        return None
    fill = None
    for p in pairs:
        ig = np.array(p["input"])
        og = np.array(p["output"])
        h, w = ig.shape
        outside = np.zeros_like(ig, dtype=bool)
        dq = deque()
        for c in range(w):
            if ig[0, c] == 0:
                dq.append((0, c))
                outside[0, c] = True
            if ig[h-1, c] == 0:
                dq.append((h-1, c))
                outside[h-1, c] = True
        for r in range(h):
            if ig[r, 0] == 0:
                dq.append((r, 0))
                outside[r, 0] = True
            if ig[r, w-1] == 0:
                dq.append((r, w-1))
                outside[r, w-1] = True
        while dq:
            r, c = dq.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                rr, cc = r+dr, c+dc
                if 0 <= rr < h and 0 <= cc < w and not outside[rr, cc] and ig[rr, cc] == 0:
                    outside[rr, cc] = True
                    dq.append((rr, cc))
        interior = (ig == 0) & ~outside
        if not np.array_equal(ig[~interior], og[~interior]):
            return None
        if not interior.any():
            continue
        u = np.unique(og[interior])
        if len(u) != 1:
            return None
        if fill is None:
            fill = int(u[0])
        elif fill != int(u[0]):
            return None
    return {"fill_color": fill} if fill is not None else None


def detect_outline_extract(pairs):
    if not all(_shapes_match(p) for p in pairs):
        return None
    for p in pairs:
        ig = np.array(p["input"])
        og = np.array(p["output"])
        h, w = ig.shape
        exp = ig.copy()
        for r in range(h):
            for c in range(w):
                if ig[r, c] == 0:
                    continue
                if r == 0 or r == h-1 or c == 0 or c == w-1:
                    continue
                neighbors = [
                    ig[r+dr, c+dc]
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
                    if 0 <= r+dr < h and 0 <= c+dc < w
                ]
                if not any(n == 0 for n in neighbors):
                    exp[r, c] = 0
        if not np.array_equal(exp, og):
            return None
    return {"rule": "keep_only_boundary_cells"}


def detect_keep_largest_color(pairs):
    if not all(_shapes_match(p) for p in pairs):
        return None
    for p in pairs:
        ig = np.array(p["input"])
        og = np.array(p["output"])
        counts = {c: int(np.sum(ig == c)) for c in range(1, 10)}
        if all(v == 0 for v in counts.values()):
            return None
        max_c = max(counts, key=counts.get)
        if not np.array_equal(np.where(ig == max_c, ig, 0), og):
            return None
    return {"rule": "keep_largest_color"}


def detect_keep_smallest_color(pairs):
    if not all(_shapes_match(p) for p in pairs):
        return None
    for p in pairs:
        ig = np.array(p["input"])
        og = np.array(p["output"])
        counts = {c: int(np.sum(ig == c)) for c in range(1, 10) if int(np.sum(ig == c)) > 0}
        if not counts:
            return None
        min_c = min(counts, key=counts.get)
        if not np.array_equal(np.where(ig == min_c, ig, 0), og):
            return None
    return {"rule": "keep_smallest_color"}


def detect_period_row_tiling(pairs):
    for p in pairs:
        ig = np.array(p["input"])
        og = np.array(p["output"])
        ih, iw = ig.shape
        oh_, ow_ = og.shape
        if ow_ != iw:
            return None
        for period in range(1, ih + 1):
            if ih % period != 0 or oh_ % period != 0:
                continue
            base = ig[:period, :]
            if not np.array_equal(np.tile(base, (ih // period, 1)), ig):
                continue
            if np.array_equal(np.tile(base, (oh_ // period, 1)), og):
                return {"period": period, "out_h": oh_}
    return None


def detect_period_col_tiling(pairs):
    for p in pairs:
        ig = np.array(p["input"])
        og = np.array(p["output"])
        ih, iw = ig.shape
        oh_, ow_ = og.shape
        if ih != oh_:
            return None
        for period in range(1, iw + 1):
            if iw % period != 0 or ow_ % period != 0:
                continue
            base = ig[:, :period]
            if not np.array_equal(np.tile(base, (1, iw // period)), ig):
                continue
            if np.array_equal(np.tile(base, (1, ow_ // period)), og):
                return {"period": period, "out_w": ow_}
    return None


def detect_bbox_then_tile(pairs):
    results = []
    for p in pairs:
        ig = np.array(p["input"])
        og = np.array(p["output"])
        mask = ig != 0
        if not mask.any():
            return None
        rs, cs = np.where(mask)
        r0, r1, c0, c1 = int(rs.min()), int(rs.max()), int(cs.min()), int(cs.max())
        crop_h, crop_w = r1 - r0 + 1, c1 - c0 + 1
        crop = ig[r0:r1+1, c0:c1+1]
        oh_, ow_ = og.shape
        found = False
        for tr in range(1, oh_ // max(crop_h, 1) + 1):
            for tc in range(1, ow_ // max(crop_w, 1) + 1):
                if crop_h * tr == oh_ and crop_w * tc == ow_:
                    if np.array_equal(np.tile(crop, (tr, tc)), og):
                        results.append((r0, c0, crop_h, crop_w, tr, tc))
                        found = True
                        break
            if found:
                break
        if not found:
            return None
    if len(set(results)) == 1:
        return {"params": results[0]}
    return None


def detect_frame_border(pairs):
    if not all(_shapes_match(p) for p in pairs):
        return None
    color = None
    for p in pairs:
        ig = np.array(p["input"])
        og = np.array(p["output"])
        if not np.array_equal(ig[1:-1, 1:-1], og[1:-1, 1:-1]):
            return None
        edge_vals = (set(og[0, :].tolist()) | set(og[-1, :].tolist()) |
                     set(og[:, 0].tolist()) | set(og[:, -1].tolist()))
        if len(edge_vals) != 1:
            return None
        c_ = int(edge_vals.pop())
        if color is None:
            color = c_
        elif color != c_:
            return None
    return {"frame_color": color} if color is not None else None


def detect_nonzero_recolor(pairs):
    if not all(_shapes_match(p) for p in pairs):
        return None
    tc = None
    for p in pairs:
        ig = np.array(p["input"])
        og = np.array(p["output"])
        if not np.all(og[ig == 0] == 0):
            return None
        nb = og[ig != 0]
        if nb.size == 0:
            continue
        u = np.unique(nb)
        if len(u) != 1:
            return None
        if tc is None:
            tc = u[0]
        elif tc != u[0]:
            return None
    return {"target_color": int(tc)} if tc is not None else None


def detect_bg_recolor(pairs):
    if not all(_shapes_match(p) for p in pairs):
        return None
    tc = None
    for p in pairs:
        ig = np.array(p["input"])
        og = np.array(p["output"])
        if not np.array_equal(og[ig != 0], ig[ig != 0]):
            return None
        bo = og[ig == 0]
        if bo.size == 0:
            continue
        u = np.unique(bo)
        if len(u) != 1:
            return None
        if tc is None:
            tc = u[0]
        elif tc != u[0]:
            return None
    return {"target_color": int(tc)} if tc is not None else None


def detect_diag_symmetry_complete(pairs):
    if not all(_shapes_match(p) for p in pairs):
        return None
    for p in pairs:
        ig = np.array(p["input"])
        og = np.array(p["output"])
        if ig.shape[0] != ig.shape[1]:
            return None
        if not np.array_equal(np.where(ig != 0, ig, ig.T), og):
            return None
    return {"rule": "diag_symmetry_complete"}


def detect_h_symmetry_complete(pairs):
    if not all(_shapes_match(p) for p in pairs):
        return None
    for p in pairs:
        ig = np.array(p["input"])
        og = np.array(p["output"])
        if not np.array_equal(np.where(ig != 0, ig, np.flip(ig, 1)), og):
            return None
    return {"rule": "h_symmetry_complete"}


def detect_v_symmetry_complete(pairs):
    if not all(_shapes_match(p) for p in pairs):
        return None
    for p in pairs:
        ig = np.array(p["input"])
        og = np.array(p["output"])
        if not np.array_equal(np.where(ig != 0, ig, np.flip(ig, 0)), og):
            return None
    return {"rule": "v_symmetry_complete"}


def detect_both_symmetry_complete(pairs):
    if not all(_shapes_match(p) for p in pairs):
        return None
    for p in pairs:
        ig = np.array(p["input"])
        og = np.array(p["output"])
        stack = np.stack([ig, np.flip(ig, 1), np.flip(ig, 0), np.flip(np.flip(ig, 0), 1)], 0)
        if not np.array_equal(np.where(ig != 0, ig, stack.max(0)), og):
            return None
    return {"rule": "both_symmetry_complete"}


def detect_translation(pairs):
    if not all(_shapes_match(p) for p in pairs):
        return None
    ih, iw = np.array(pairs[0]["input"]).shape
    for dr in range(-min(ih, 11) + 1, min(ih, 11)):
        for dc in range(-min(iw, 11) + 1, min(iw, 11)):
            if dr == 0 and dc == 0:
                continue
            ok = True
            for p in pairs:
                ig = np.array(p["input"])
                og = np.array(p["output"])
                hh, ww = ig.shape
                s = np.zeros_like(ig)
                sr0 = max(0, -dr); sr1 = min(hh, hh - dr)
                sc0 = max(0, -dc); sc1 = min(ww, ww - dc)
                dr0 = max(0, dr); dc0 = max(0, dc)
                rh = sr1 - sr0; rw = sc1 - sc0
                if rh > 0 and rw > 0:
                    s[dr0:dr0+rh, dc0:dc0+rw] = ig[sr0:sr1, sc0:sc1]
                if not np.array_equal(s, og):
                    ok = False
                    break
            if ok:
                return {"shift": (dr, dc)}
    return None


def _connected_components_4conn(g):
    h, w = g.shape
    visited = np.zeros_like(g, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if visited[r, c] or g[r, c] == 0:
                continue
            stack = [(r, c)]
            cells = []
            while stack:
                rr, cc = stack.pop()
                if not (0 <= rr < h and 0 <= cc < w):
                    continue
                if visited[rr, cc] or g[rr, cc] == 0:
                    continue
                visited[rr, cc] = True
                cells.append((rr, cc))
                stack.extend([(rr+1, cc), (rr-1, cc), (rr, cc+1), (rr, cc-1)])
            comps.append(cells)
    return comps


def detect_largest_cc_bbox_crop(pairs):
    boxes = []
    for p in pairs:
        ig = np.array(p["input"])
        og = np.array(p["output"])
        comps = _connected_components_4conn(ig)
        if not comps:
            return None
        biggest = max(comps, key=len)
        rs = [r for r, _ in biggest]
        cs = [c for _, c in biggest]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        if og.shape != (r1 - r0 + 1, c1 - c0 + 1):
            return None
        if not np.array_equal(ig[r0:r1+1, c0:c1+1], og):
            return None
        boxes.append((r0, c0, r1 - r0 + 1, c1 - c0 + 1))
    if len(set(boxes)) <= 5:  # allow some variation
        return {"sample_boxes": boxes[:5]}
    return None


def detect_concat_rot180_w(pairs):
    for p in pairs:
        ig = np.array(p["input"])
        og = np.array(p["output"])
        if og.shape != (ig.shape[0], ig.shape[1] * 2):
            return None
        if not np.array_equal(np.concatenate([ig, np.rot90(ig, 2)], 1), og):
            return None
    return {"rule": "concat_with_rot180"}


def detect_color_filter_remove(pairs):
    """Output = input with specific color(s) zeroed out."""
    if not all(_shapes_match(p) for p in pairs):
        return None
    cand = None
    for p in pairs:
        ig = np.array(p["input"])
        og = np.array(p["output"])
        diff = (ig != og)
        if not np.any(diff):
            continue
        if not np.all(og[diff] == 0):
            return None
        local = set(int(x) for x in ig[diff])
        cand = local if cand is None else cand | local
    if not cand:
        return None
    for p in pairs:
        ig = np.array(p["input"]).copy()
        for c in cand:
            ig[ig == c] = 0
        if not np.array_equal(ig, np.array(p["output"])):
            return None
    return {"removed_colors": sorted(cand)}


CANDIDATES = {
    "bbox-fill": detect_bbox_fill,
    "hole-fill": detect_hole_fill,
    "outline-extract": detect_outline_extract,
    "keep-largest-color": detect_keep_largest_color,
    "keep-smallest-color": detect_keep_smallest_color,
    "period-row-tiling": detect_period_row_tiling,
    "period-col-tiling": detect_period_col_tiling,
    "bbox-then-tile": detect_bbox_then_tile,
    "frame-border": detect_frame_border,
    "nonzero-recolor": detect_nonzero_recolor,
    "bg-recolor": detect_bg_recolor,
    "diag-symmetry-complete": detect_diag_symmetry_complete,
    "h-symmetry-complete": detect_h_symmetry_complete,
    "v-symmetry-complete": detect_v_symmetry_complete,
    "both-symmetry-complete": detect_both_symmetry_complete,
    "translation": detect_translation,
    "largest-cc-bbox-crop": detect_largest_cc_bbox_crop,
    "concat-rot180-w": detect_concat_rot180_w,
    "color-filter-remove": detect_color_filter_remove,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truly-unsolved-only", action="store_true")
    ap.add_argument("--angle", choices=list(CANDIDATES.keys()))
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
