"""Diagnostic for NeuroGolf task data.

Usage:
    python tools/inspect_tasks.py N                # visualize one task (train/test/arc-gen pairs)
    python tools/inspect_tasks.py --summary        # aggregate stats across all 400
    python tools/inspect_tasks.py --shape-changes  # tasks where output shape differs from input
    python tools/inspect_tasks.py --unsolved [N]   # visualize a sample of unsolved tasks (default 5)

Reads the bundled `data/competitions/neurogolf-2026/taskNNN.json` files via
`neurogolf_utils.load_examples`. Stubs the notebook display deps so we don't
need IPython/matplotlib in the venv.
"""

import collections
import os
import pathlib
import random
import sys
import types

for _name in ("matplotlib", "matplotlib.pyplot", "IPython", "IPython.display"):
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules["IPython.display"].display = lambda *a, **kw: None
sys.modules["IPython.display"].FileLink = lambda *a, **kw: None
sys.modules["IPython"].display = sys.modules["IPython.display"]
sys.modules["matplotlib"].pyplot = sys.modules["matplotlib.pyplot"]

NG_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = pathlib.Path(os.environ.get("NEUROGOLF_DATA_DIR", str(NG_ROOT / "data" / "competitions" / "neurogolf-2026")))
sys.path.insert(0, str(DATA_DIR / "neurogolf_utils"))

import neurogolf_utils  # noqa: E402

neurogolf_utils._NEUROGOLF_DIR = f"{DATA_DIR.as_posix()}/"

# Tasks we already score on (from runs.jsonl + leaderboard); excluded from --unsolved sampling.
# 16, 276, 309, 337: color-perm (PR #35 extended detector to route unmapped output channels
# to a guaranteed-zero input channel; 276/309 reattributed from color-replace at cost
# 90,100 each → cost 10 each via Gather). 179, 241: transpose. 53: fixed-shift_v_1.
# 87, 140: fixed-rot180. 380: fixed-rot270. 129: const-dominant-color. 135, 326:
# slice-extract. 266: marker-corners. 272: isolated-marker-recolor. 81: l-shape-missing-corner.
# 50, 350: aligned-marker-fill. 55: cross-section-grid-label. 59: grid-max-count-fill.
# 19: tile-diagonal-neighbor-mark (2x2 tile plus color-8 diagonal neighbors).
# 71: bilateral-symmetry-denoise (solid rectangular noise removed by horizontal symmetry axis).
# 137: diagonal-marker-square-rings (Chebyshev square rings from three diagonal markers).
# 175: euclidean-wedge-fill (21x21 symmetric color lattice hole fill).
# 184: grid-object-summary (nonzero row/column band compact color summary).
# 126: pi-project-bottom. 258: sandwich-fill.
# 73, 95, 262, 282, 283, 294, 331: bin-nbr-lut (3x3 binary-neighborhood LUT).
# 317: marker-dilate (PR #35 — MaxPool-3x3 dilate-then-recolor; reattributed from
# bin-nbr-lut at cost 168,938 → marker-dilate at cost 78,340).
# 56, 103: grid-pattern-to-color (3x3 input, 1x1 output, binary-pattern LUT, one-hot float LUT).
# 104, 186: grid-pattern-to-grid (binary input pattern -> fixed-shape output via int LUT + OneHot).
# 1, 211: grid-pattern-color-to-grid (binary input pattern AND nonzero color -> output via LUT).
# 116, 164, 172, 210, 311: mirror-concat (output = concat(input, transformed_input) along H or W;
# transform in {flip_w, flip_h, flip_hw, identity}). Pure shape ops -> 0 MACs.
# 83, 142, 152: mirror-grid-2x2 (output is 2x2 layout [[inp, fw],[fh, fhw]]). Pure shape ops.
# 214, 373, 385: pixel-perm (Slice + Reshape + Gather + Reshape + Pad; per-cell spatial
# permutation via length-(h*w) int64 table). Cost varies with grid size; ~16-17 pts.
# 108, 194, 223: shape-changing-pixel-perm (same pipeline but indices length is out_h*out_w
# and Slice/Pad use different shapes). Tile-and-rearrange tasks. ~14-17 pts each.
# 2, 251: enclosed-region-fill (BFS from canvas boundary blocked by walls; fill enclosed
# color-0 cells with a single color). 30 dilation iters via Conv 3x3 + Clip. Cost ~742k -> ~11.5 pts.
# 304, 315: outer-product-marker-tile (3x3 -> 9x9). 304 uses mode="most_common" (per-pair
# argmax marker); 315 uses mode="fixed" (marker = color 2 across all pairs). Cost ~13k -> ~15.5 pts.
# 327: diagonal-shear (3x3 -> 6x6 sum-of-shifts; each output anti-diagonal r-c=k holds
# input cells (i, j) with i-j == k. ~22k -> ~15.0 pts.
# 15, 220, 352: per-color-stamp (any same-shape; rule list of (in_color, stamp_color, kernel)).
# task015 = (1, 7, ortho)+(2, 4, diag) (LB-5126 truly-unsolved); task220 = three "all8"
# rules with a 2026-05-08 single-Conv fast path; task352 = (2, 1, all8). Generic cost
# ~67-128k -> ~13.2-13.9 pts; task220 fast path is ~900 cost. Pattern also picks up task095
# at cost ~67k vs bin-nbr-lut ~169k -> reclassified by cost-aware dispatcher.
# 113: symmetry-completion on v-axis (h=10, w=10). Fill input's zero cells with reflected
# non-zero values across H/2; preserve non-zeros. Slice + Gather + Greater + Where. Cost
# ~41k -> ~14.4 pts.
# 166: bbox-fill (target=2). Fill the tight bounding box of input non-zero cells with
# target_color, preserving non-zeros and cells outside bbox. Cumulative-OR via MaxPool
# kernel-30 simulates cumsum (opset-11+ CumSum unavailable). Cost ~78k -> ~13.7 pts.
# 98: outline-extract. Keep boundary cells of each connected non-zero region; interior
# cells (cell + all 8 neighbors same color) become 0. Grouped Conv 3x3 + Greater + Cast
# + Mul + Sub + ReduceSum + Add. Cost ~314k -> ~12.3 pts.
# 171: active-grid-frame (border=8). Input all-zero; output = 1-pixel border of fixed
# color around the active grid; interior color 0. is_active = ReduceSum + Conv 3x3
# 8-neighbor kernel + Less + Cast + Mul + Sub + Concat. Cost ~43k -> ~14.3 pts.
# 31: foreground-bbox-crop. Output = input cropped to bbox of non-zero cells, placed
# at top-left of canvas. ReduceSum + Slice + Sub + ReduceMax x2 + ArgMax x4 (with
# descending/ascending weight to pick first/last) + Cast + Add + Min + Cast + Gather x2
# + Sub/Add + Less + Reshape + Mul. Variable in/out shape (single-example only).
# Cost ~134k -> ~13.2 pts.
# 123: fractal-ring-expansion (5x5 -> 10x10). Diagonal colors define K=4 or K=5
# fg sequence; output[i][j] = fg[max(i,j) mod K]. 5 Slice + Concat (one-hot diag) +
# Slice + Greater (is_K4) + Where(is_K4, idx_k4, idx_k5) + Reshape + Gather +
# Reshape + Pad. Cost ~10k -> ~15.8 pts.
# 36: color-filter-bbox-crop with picker='smallest_bbox'. Picks the color whose
# nonzero cells span the smallest bbox area. Per-color bbox area via 9-channel
# ReduceMax + ArgMax-trick + Sub/Add/Mul. Cost ~159k -> ~13.0 pts.
# 243: seed-flood-fill (fill_color=1). 4-conn flood from each color-1 seed
# through 0-cells; reached 0-cells become color 1. Slice ch0 + Slice ch1 (seed)
# + Add (not_blocking) + 30x(Conv 3x3 cross + Clip + Mul) + Mul + Mul + Add.
# Cost ~718k -> ~11.5 pts.
# 14, 49, 310: color-filter-bbox-crop with the LEAST-COMMON nonzero color picker.
# Output = input cropped to bbox of cells of the least-common color (preserving
# original colors inside bbox). Slice + ReduceSum + Less + Mul + Where + ArgMax
# + Cast + Add + Cast + Gather (channel) + foreground-bbox-crop pipeline applied
# to the selected channel. Cost ~134k -> ~13.2 pts.
# 307: kron-block-up. Output = input with each cell scaled to a (scale_h x
# scale_w) block (Kronecker product with ones). Constant-axes Slice to top-left
# (30/scale_h x 30/scale_w) + Resize-nearest with scale (1, 1, scale_h, scale_w).
# Both ops are 0 MACs in onnx-tool 1.0.1; cost dominated by the FLOAT slice
# intermediate. Cost ~9k -> ~15.9 pts.
# 369: zero-adjacency-label. Fixed 10x10 {0,5} input; preserve color 5 and
# recolor color-0 cells by their 4-neighbor zero count plus heavy-neighbor
# context. Conv cross-kernel + small arithmetic indicators.
# 100: const-largest-bbox. Output = uniform fixed-shape block of the channel
# (1..9) whose bounding box has the largest area in the input. Per-channel
# bbox extraction via ReduceMax + ArgMax-trick + Sub + Add + Mul, then
# ArgMax(area) + OneHot + Tile + Pad. Cost ~37k -> ~14.5 pts.
# 346: const-framed-cell. Output = uniform fixed-shape block of the color of
# the **core** of a 3x3 frame anywhere in the input (8 frame cells of one
# non-zero color X, center cell of a different non-zero color Y; output Y).
# Grouped Conv with 3x3 frame kernel + Pad + Greater + Mul + ReduceMax +
# ArgMax + OneHot + Tile + Pad. Cost ~394k -> ~12.1 pts.
# 153: two-blob-tile-fit. Normalize two compact connected blobs into 3x3
# bboxes, enumerate placements, and select the unique non-overlap full cover.
# 106: fourfold-rotation-tile. Output is a 2x2 tile of square input rotations:
# input, rot90, rot270, rot180.
# 348: diagonal-rays-from-line. Color-7 cells emit upward diagonals; odd
# distances become 8 and even distances become 7, clipped to active background.
# 231: period-extend-row-2x. Each active row is extended to double width using
# the smallest period in {1,2,3}; zero rows stay active background.
# 145: rectangular-zero-rank-fill. Color-2 walls partition zero rectangles;
# largest zero rectangle(s) become 1 and smallest non-largest rectangle(s)
# become 8.
# 102: framed-zero-square-fill. Closed color-5 one-cell frames fill only square
# zero interiors with color 2; non-square framed rectangles stay zero.
# 248: bouncing-diagonal-ray. A bottom-left color-1 seed emits a 10-row diagonal
# ray that reflects off the left/right walls; active width is inferred from row 0.
# 176: trirow-periodic-four-fill. Fixed 3-row color-2 scaffold gains color-4
# marks on a 12-column phase cycle.
# 58: square-spiral-three. All-zero square input becomes a color-3 inward
# square spiral.
# 105: gridline-adjacent-fill. Fill bbox edge/count>=4 row/col lines, plus
# count-3 row/col lines when the three anchors include an adjacent pair.
# 017: periodic-hole-fill. Infer the 21x21 pattern period from non-zero
# horizontal agreement, then fill zero holes from the matching 2D residue class.
# 396: largest-rectangle-recolor-crop. Most-common nonzero color forms hollow
# 4..8 rectangles; crop max enclosed-area frame and recolor border to marker.
# 025: line-projection-to-full-line. Colors with a full active row/column keep
# that line; off-line same-color cells project to the adjacent side of the line.
# 370: diagonal-zero-mask-projection. A lone marker color points diagonally at
# the farthest zero hole; translate the zero mask repeatedly by that vector.
# 77: color2-bbox-fill. Compact color-2 components (height <=3, width <=7)
# recolor nonzero/non-2 cells inside each component bbox to 4.
# 009: block-grid-line-fill. On 2x2-cell separator grids, each color fills
# spans between every same-color pair sharing a coarse row or column.
# 040: border-distance-probe-recolor. Color-3 probe cells in a 10x10 grid take
# the nearest opposite border-strip color along the active row/column axis.
# 038: count-2x2-one-blocks. Count 2x2 blocks of color 1 in a 9x9 grid and
# emit a 1x5 unary row with that many leading color-1 cells.
# 097: same-color-neighbor-keep. Preserve nonzero cells with at least one
# same-color 8-neighbor; erase singleton 8-connected components.
# 120: rectangle-interior-fill8. Same-color 3x3 interior cells in solid
# rectangles are recolored to 8; borders stay original.
# 125: frame6-border3-fill4. Color-6 rectangular frames get a color-3 outer
# ring; hollow color-8 interiors are filled with color 4.
# 089: marker-shape-copy. Source markers are adjacent to a non-marker shape
# color; isolated same-color markers receive that shape (2 = hflip, 3 = id).
# 306: grid-section-copy. Color-4 divider grid with one 9x9 source section;
# copy the source section into every existing section.
# 138: frame-raycast-crop. Crop to two full nonzero rows/cols, then ray-fill
# the lone interior marker color toward its matching border side.
# 119: edge-seed-bounce-ray. Color-8 cells mark the start of a diagonal ray from
# an outer edge; continue through marked cells, bounce off the color-2 wall, and
# paint later zero cells color 3.
# 339: foreground-count-row. 3x3 input with one foreground color becomes one
# row of that color, width equal to the foreground cell count.
# 064: marker-to-rectangle-line. Most-common color is background; the solid
# rectangle is the non-background component whose cells have same-color row and
# column neighbors; isolated marker-color cells draw straight aligned spans to it.
# 062: marker-adjacent-bbox-reflection. Repaint the active background color 3,
# recolor marker 2 to the object color, and mirror the object across that bbox side.
# 228: frame-inner-markers-to-outer-opposite. Four inner-corner markers inside a
# hollow frame are moved to the diagonally opposite corners just outside the frame.
# 333: anchor-block-aligned-marker-line. Non-3 markers aligned with either row
# or column of the 2x2 color-3 anchor draw the segment back to that anchor.
# 362: marker-count-cross-shift. Count color-5 marker cells, then redraw the
# full same-color cross shifted down-left by that count.
# 354: top-marker-run-recolor. Color-5 horizontal runs inherit the color of the
# top-row marker whose column intersects the run.
# 303: empty-row-col-recolor2. Active-grid zero cells in rows or columns with no
# nonzero cells become color 2.
# 165: arch-marker-fall. In the fixed 20x20 arch pattern, columns with marker
# cells below the arch fall downward through background in the marker color.
# 037: diagonal-pair-connect. Each nonzero color appears exactly twice; if the
# pair is diagonal, paint the same-color segment between the endpoints.
# 261: downshift-recolor8-to2. Color-8 cells shift down one row, become color 2,
# and leave background behind.
SOLVED_TASKS = {1, 2, 3, 6, 9, 10, 14, 15, 16, 17, 19, 20, 21, 25, 26, 27, 29, 31, 36, 37, 38, 40, 42, 43, 47, 49, 50, 52, 53, 55, 56, 58, 59, 62, 64, 65, 71, 72, 73, 77, 81, 83,
                87, 89, 95, 97, 98, 100, 102, 103, 104, 105, 106, 108, 113, 116, 119, 120, 123, 125, 126, 129, 135, 137, 138, 140, 142, 143, 144, 145, 149, 152, 153, 164,
                165, 166, 167, 171, 172, 174, 175, 176, 179, 184, 186, 194, 206, 210, 211, 214, 220, 223, 227, 228, 229, 231, 236, 241, 243, 248, 249, 251, 258, 261,
                262, 263, 265, 266, 267, 272, 276, 282, 283, 294, 299, 300, 303, 304, 306, 307, 309, 310, 311, 315, 316, 317, 318, 320, 322,
                326, 327, 331, 333, 334, 337, 338, 339, 346, 347, 348, 350, 352, 354, 362, 369, 370, 373, 380, 385, 386, 388, 389, 395, 396}


# ---------------------------------------------------------------- visualization

def _shape(g):
    return (len(g), len(g[0]) if g else 0)


def _render_grid(grid):
    return [" ".join(str(c) for c in row) for row in grid]


def _show_pair(pair, label):
    in_lines = _render_grid(pair["input"])
    out_lines = _render_grid(pair["output"])
    in_h, out_h = len(in_lines), len(out_lines)
    in_w = max((len(line) for line in in_lines), default=0)
    in_shape = _shape(pair["input"])
    out_shape = _shape(pair["output"])
    print(f"  {label}: input {in_shape}  ->  output {out_shape}")
    for i in range(max(in_h, out_h)):
        l_in = in_lines[i] if i < in_h else " " * in_w
        l_out = out_lines[i] if i < out_h else ""
        print(f"    {l_in:<{in_w}}   |   {l_out}")
    print()


def visualize_task(task_num, max_train=3, max_test=1, max_argen=1):
    ex = neurogolf_utils.load_examples(task_num)
    train = ex.get("train", [])
    test = ex.get("test", [])
    argen = ex.get("arc-gen", [])
    solved = " [SOLVED]" if task_num in SOLVED_TASKS else ""
    print(f"=== task{task_num:03d}{solved}: {len(train)} train / {len(test)} test / {len(argen)} arc-gen pairs ===\n")
    for i, p in enumerate(train[:max_train]):
        _show_pair(p, f"train[{i}]")
    for i, p in enumerate(test[:max_test]):
        _show_pair(p, f"test[{i}]")
    for i, p in enumerate(argen[:max_argen]):
        _show_pair(p, f"arc-gen[{i}]")


# ----------------------------------------------------------------------- stats

def task_stats(task_num):
    """Compact summary of one task's structural properties (no grid rendering)."""
    ex = neurogolf_utils.load_examples(task_num)
    pairs = ex.get("train", []) + ex.get("test", []) + ex.get("arc-gen", [])
    if not pairs:
        return None
    in_shapes = {_shape(p["input"]) for p in pairs}
    out_shapes = {_shape(p["output"]) for p in pairs}
    shape_eq_per_pair = all(_shape(p["input"]) == _shape(p["output"]) for p in pairs)
    in_colors = set()
    out_colors = set()
    for p in pairs:
        for row in p["input"]:
            in_colors.update(row)
        for row in p["output"]:
            out_colors.update(row)
    return {
        "task": task_num,
        "n_pairs": len(pairs),
        "in_shape_unique": in_shapes,
        "out_shape_unique": out_shapes,
        "shape_preserved_per_pair": shape_eq_per_pair,
        "in_colors": in_colors,
        "out_colors": out_colors,
    }


def summary():
    n_pairs_total = 0
    shape_preserved = 0
    shape_changes = 0
    in_size_varies = 0
    out_size_varies = 0
    color_count_in_dist = collections.Counter()
    color_count_out_dist = collections.Counter()
    new_colors_in_output = 0
    fewer_colors_in_output = 0
    all_30x30 = 0
    smallest_grid = (99, 99)
    largest_grid = (0, 0)
    for n in range(1, 401):
        s = task_stats(n)
        if s is None:
            continue
        n_pairs_total += s["n_pairs"]
        if s["shape_preserved_per_pair"]:
            shape_preserved += 1
        else:
            shape_changes += 1
        if len(s["in_shape_unique"]) > 1:
            in_size_varies += 1
        if len(s["out_shape_unique"]) > 1:
            out_size_varies += 1
        color_count_in_dist[len(s["in_colors"])] += 1
        color_count_out_dist[len(s["out_colors"])] += 1
        if s["out_colors"] - s["in_colors"]:
            new_colors_in_output += 1
        if len(s["out_colors"]) < len(s["in_colors"]):
            fewer_colors_in_output += 1
        # Combined shape stats
        for sh in s["in_shape_unique"] | s["out_shape_unique"]:
            if sh < smallest_grid:
                smallest_grid = sh
            if sh > largest_grid:
                largest_grid = sh
        if all(sh == (30, 30) for sh in s["in_shape_unique"] | s["out_shape_unique"]):
            all_30x30 += 1

    print(f"Tasks scanned: 400")
    print(f"Total visible pairs: {n_pairs_total} (~{n_pairs_total // 400} per task on average)")
    print()
    print("=== Shape relationships ===")
    print(f"  shape preserved per pair (input shape == output shape): {shape_preserved}/400")
    print(f"  shape changes (input shape != output shape):           {shape_changes}/400")
    print(f"  input shape varies across pairs (within a task):       {in_size_varies}/400")
    print(f"  output shape varies across pairs (within a task):      {out_size_varies}/400")
    print(f"  all-30x30 tasks (all pairs are 30x30):                 {all_30x30}/400")
    print(f"  smallest grid seen: {smallest_grid}   largest grid seen: {largest_grid}")
    print()
    print("=== Color counts ===")
    print(f"  distinct input colors (count -> # tasks):  {sorted(color_count_in_dist.items())}")
    print(f"  distinct output colors (count -> # tasks): {sorted(color_count_out_dist.items())}")
    print(f"  tasks introducing colors in output that weren't in input: {new_colors_in_output}/400")
    print(f"  tasks losing colors in output (fewer than input):         {fewer_colors_in_output}/400")


def shape_changes(limit=20):
    print(f"Tasks where input.shape != output.shape (per pair) — first {limit}:\n")
    print(f"  {'task':<7} {'in_shapes':<28} {'out_shapes':<28}")
    count = 0
    for n in range(1, 401):
        s = task_stats(n)
        if s is None or s["shape_preserved_per_pair"]:
            continue
        if count >= limit:
            count += 1
            continue
        in_shapes = sorted(s["in_shape_unique"])
        out_shapes = sorted(s["out_shape_unique"])
        print(f"  task{n:03d} {str(in_shapes):<28} {str(out_shapes):<28}")
        count += 1
    if count > limit:
        print(f"\n  ... ({count - limit} more not shown)")


# --------------------------------------------------------------------- entry

def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(0)
    if args[0] == "--summary":
        summary()
        return
    if args[0] == "--shape-changes":
        limit = int(args[1]) if len(args) > 1 else 20
        shape_changes(limit=limit)
        return
    if args[0] == "--unsolved":
        n = int(args[1]) if len(args) > 1 else 5
        candidates = [t for t in range(1, 401) if t not in SOLVED_TASKS]
        random.seed(0)
        sample = sorted(random.sample(candidates, n))
        for task_num in sample:
            visualize_task(task_num, max_train=2, max_test=0, max_argen=0)
            print("---")
        return
    visualize_task(int(args[0]))


if __name__ == "__main__":
    main()
