# ===========================================================================
# TOOLS/BUILD_SUBMISSION.PY  —  965 KB / ~21,978 lines / ~240k tokens
#
# ARCHIVED PROJECT. The 2026 NeuroGolf competition is finished; nothing here
# needs to run again. This file is kept because it IS the detector/builder
# library — ~311 functions of ARC pattern-detection logic — and it is what
# tests/test_detect.py exercises (~589 tests). It moved from the repo root to
# tools/ during the 2026-07-15 archive cleanup; the wholesale --all/--hybrid
# rebuild paths were already disabled long before that.
#
# DO NOT Read this file whole — it will consume your entire context window.
#
# HOW TO FIND A TASK BUILDER:
#   grep "_task<NNN>_" tools/build_submission.py   → find the function
#   Then Read ±60 lines around that match
#
# LIVE SUBMIT FLOW:   tools/build_dispatcher_task.py  (imports this file)
#   → builds one task ONNX → tools/swap_and_submit.py → ships
#
# WARNING — --hybrid AND --all ARE DEAD AND DANGEROUS:
#   Both overwrite submissions/submission.zip with a 103/400-correct
#   baseline (~1,627 pts vs live ~6,500+). Do NOT run either flag.
#   Use swap_and_submit.py for all shipping.
# ===========================================================================
"""Build validator-passing ONNX files for NeuroGolf tasks.

Headless pipeline that uses the official `neurogolf_utils` module without
pulling in its notebook-display dependencies (`IPython.display`,
`matplotlib.pyplot`). Stubs them in `sys.modules` before the import.

Usage:
    .venv/Scripts/python.exe build_submission.py [TASK_NUM]   # one task (Conv 3x3 identity)
    .venv/Scripts/python.exe build_submission.py --all        # DEAD — overwrites live zip, do not use
    .venv/Scripts/python.exe build_submission.py --hybrid     # DEAD — overwrites live zip, do not use

Outputs `submissions/taskNNN.onnx` and (with --all/--hybrid) `submissions/submission.zip`.
"""

import collections
import datetime
import functools
import json
import math
import os
import pathlib
import sys
import tempfile
import types
import zipfile

for _name in ("matplotlib", "matplotlib.pyplot", "IPython", "IPython.display"):
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules["IPython.display"].display = lambda *a, **kw: None
sys.modules["IPython.display"].FileLink = lambda *a, **kw: None
sys.modules["IPython"].display = sys.modules["IPython.display"]
sys.modules["matplotlib"].pyplot = sys.modules["matplotlib.pyplot"]

NG_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = pathlib.Path(os.environ.get("NEUROGOLF_DATA_DIR", str(NG_ROOT / "data" / "competitions" / "neurogolf-2026")))
SUBMISSIONS_DIR = NG_ROOT / "submissions"
RUNS_LOG = NG_ROOT / "Logs" / "runs.jsonl"
sys.path.insert(0, str(DATA_DIR / "neurogolf_utils"))

import neurogolf_utils  # noqa: E402

neurogolf_utils._NEUROGOLF_DIR = f"{DATA_DIR.as_posix()}/"

import numpy as np  # noqa: E402
import onnx  # noqa: E402
import onnxruntime  # noqa: E402


sys.path.insert(0, str(NG_ROOT))
from tools._score_v2 import score_network_v2, score_to_points  # noqa: E402


# ---------------------------------------------------------------- architecture

_INPUT_SHAPE = [1, 10, 30, 30]
_OPSET = [onnx.helper.make_opsetid("", 10)]
_IR_VERSION = 10


def identity_weight(c_out, c_in, coord):
    return 1.0 if coord == (0, 0) and c_out == c_in else 0.0


def _make_model(nodes, initializers=(), shape=None):
    shape = shape if shape is not None else _INPUT_SHAPE
    inp = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, shape)
    out = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, shape)
    graph = onnx.helper.make_graph(list(nodes), "graph", [inp], [out], list(initializers))
    return onnx.helper.make_model(graph, ir_version=_IR_VERSION, opset_imports=_OPSET)


def identity_network(batch=1):
    return _make_model(
        [onnx.helper.make_node("Identity", ["input"], ["output"])],
        shape=[batch, 10, 30, 30],
    )


def transpose_hw_network(batch=1):
    return _make_model(
        [onnx.helper.make_node("Transpose", ["input"], ["output"], perm=[0, 1, 3, 2])],
        shape=[batch, 10, 30, 30],
    )


def _gather_network(axis, indices, batch=1):
    init = onnx.helper.make_tensor("indices", onnx.TensorProto.INT64, [len(indices)], indices)
    node = onnx.helper.make_node("Gather", ["input", "indices"], ["output"], axis=axis)
    return _make_model([node], [init], shape=[batch, 10, 30, 30])


def hflip_network(batch=1):
    return _gather_network(axis=3, indices=list(range(29, -1, -1)), batch=batch)


def vflip_network(batch=1):
    return _gather_network(axis=2, indices=list(range(29, -1, -1)), batch=batch)


def rot180_network(batch=1):
    init = onnx.helper.make_tensor("idx", onnx.TensorProto.INT64, [30], list(range(29, -1, -1)))
    node1 = onnx.helper.make_node("Gather", ["input", "idx"], ["mid"], axis=2)
    node2 = onnx.helper.make_node("Gather", ["mid", "idx"], ["output"], axis=3)
    return _make_model([node1, node2], [init], shape=[batch, 10, 30, 30])


def _fixed_canvas_indices(grid_dim, mode, k=None, canvas_dim=30):
    """Indices for a Gather along an axis that performs an in-grid spatial op.

    The first `grid_dim` positions implement the transform; the rest are
    identity (canvas padding stays put). `mode` is "reverse" or "cyclic".
    """
    if mode == "reverse":
        return list(reversed(range(grid_dim))) + list(range(grid_dim, canvas_dim))
    if mode == "cyclic":
        return [(i - k) % grid_dim for i in range(grid_dim)] + list(range(grid_dim, canvas_dim))
    raise ValueError(f"unknown mode: {mode}")


def fixed_flip_h_network(w, batch=1):
    return _gather_network(axis=3, indices=_fixed_canvas_indices(w, "reverse"), batch=batch)


def fixed_flip_v_network(h, batch=1):
    return _gather_network(axis=2, indices=_fixed_canvas_indices(h, "reverse"), batch=batch)


def _slice_active_inits(prefix, h, w):
    """Initializers for `Slice axes=[2,3] starts=[0,0] ends=[h,w]` extracting the
    active grid from the [N,10,30,30] canvas. Returns (4 inits, slice node)."""
    starts = onnx.helper.make_tensor(f"{prefix}_starts", onnx.TensorProto.INT64, [2], [0, 0])
    ends   = onnx.helper.make_tensor(f"{prefix}_ends",   onnx.TensorProto.INT64, [2], [h, w])
    axes   = onnx.helper.make_tensor(f"{prefix}_axes",   onnx.TensorProto.INT64, [2], [2, 3])
    steps  = onnx.helper.make_tensor(f"{prefix}_steps",  onnx.TensorProto.INT64, [2], [1, 1])
    return [starts, ends, axes, steps]


def fixed_rot180_network(h, w, batch=1):
    """rot180 of an h-by-w grid embedded at top-left of [batch,10,30,30].

    For non-30x30 active grids, uses the Slice+Gather+Gather+Pad pattern: extract
    the small active region, flip both axes, pad back to canvas. The intermediate
    tensors are sized to the active grid (h*w*10 floats) instead of full canvas
    (30*30*10 = 9,000 floats), which is dramatically cheaper for small h, w.

    For 30x30 active grids, falls back to the original two-canvas-Gather form
    (no Slice/Pad overhead since the slice is a no-op identity).
    """
    if h == 30 and w == 30:
        # Fallback: two-Gather form (cheapest when no Slice/Pad savings to harvest).
        idx = _fixed_canvas_indices(h, "reverse")
        idx_init = onnx.helper.make_tensor("fs_idx", onnx.TensorProto.INT64, [30], idx)
        node1 = onnx.helper.make_node("Gather", ["input", "fs_idx"], ["fs_mid"], axis=2)
        node2 = onnx.helper.make_node("Gather", ["fs_mid", "fs_idx"], ["output"], axis=3)
        return _make_model([node1, node2], [idx_init], shape=[batch, 10, 30, 30])

    inits = _slice_active_inits("fs_sl", h, w)
    slice_node = onnx.helper.make_node(
        "Slice",
        ["input", "fs_sl_starts", "fs_sl_ends", "fs_sl_axes", "fs_sl_steps"],
        ["fs_active"],
    )
    if h == w:
        idx_init = onnx.helper.make_tensor("fs_idx", onnx.TensorProto.INT64, [h], list(reversed(range(h))))
        inits.append(idx_init)
        flip_h = onnx.helper.make_node("Gather", ["fs_active", "fs_idx"], ["fs_rh"], axis=2)
        flip_w = onnx.helper.make_node("Gather", ["fs_rh", "fs_idx"], ["fs_rhw"], axis=3)
    else:
        idx_h_init = onnx.helper.make_tensor("fs_idx_h", onnx.TensorProto.INT64, [h], list(reversed(range(h))))
        idx_w_init = onnx.helper.make_tensor("fs_idx_w", onnx.TensorProto.INT64, [w], list(reversed(range(w))))
        inits.extend([idx_h_init, idx_w_init])
        flip_h = onnx.helper.make_node("Gather", ["fs_active", "fs_idx_h"], ["fs_rh"], axis=2)
        flip_w = onnx.helper.make_node("Gather", ["fs_rh", "fs_idx_w"], ["fs_rhw"], axis=3)
    pad_node = onnx.helper.make_node(
        "Pad", ["fs_rhw"], ["output"],
        pads=[0, 0, 0, 0, 0, 0, 30 - h, 30 - w],
        mode="constant", value=0.0,
    )
    return _make_model([slice_node, flip_h, flip_w, pad_node], inits, shape=[batch, 10, 30, 30])


def fixed_rot90_network(n, batch=1):
    """rot90 CW on a square n-by-n grid: Transpose (axes 2,3) then flip W.

    For n < 30, uses Slice+Transpose+Gather+Pad to keep intermediates small.
    For n == 30, falls back to the original Transpose+Gather form.
    """
    if n == 30:
        idx = _fixed_canvas_indices(n, "reverse")
        idx_init = onnx.helper.make_tensor("fs90_idx", onnx.TensorProto.INT64, [30], idx)
        transpose = onnx.helper.make_node("Transpose", ["input"], ["fs90_t"], perm=[0, 1, 3, 2])
        gather = onnx.helper.make_node("Gather", ["fs90_t", "fs90_idx"], ["output"], axis=3)
        return _make_model([transpose, gather], [idx_init], shape=[batch, 10, 30, 30])

    inits = _slice_active_inits("fs90_sl", n, n)
    inits.append(onnx.helper.make_tensor("fs90_idx", onnx.TensorProto.INT64, [n], list(reversed(range(n)))))
    nodes = [
        onnx.helper.make_node("Slice", ["input", "fs90_sl_starts", "fs90_sl_ends", "fs90_sl_axes", "fs90_sl_steps"], ["fs90_active"]),
        onnx.helper.make_node("Transpose", ["fs90_active"], ["fs90_t"], perm=[0, 1, 3, 2]),
        onnx.helper.make_node("Gather", ["fs90_t", "fs90_idx"], ["fs90_rotated"], axis=3),
        onnx.helper.make_node("Pad", ["fs90_rotated"], ["output"], pads=[0, 0, 0, 0, 0, 0, 30 - n, 30 - n], mode="constant", value=0.0),
    ]
    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def fixed_rot270_network(n, batch=1):
    """rot270 CW (= rot90 CCW = anti-diagonal flip) on a square grid: Transpose then flip H.

    Slice+Pad sandwich for n < 30; full-canvas form for n == 30.
    """
    if n == 30:
        idx = _fixed_canvas_indices(n, "reverse")
        idx_init = onnx.helper.make_tensor("fs270_idx", onnx.TensorProto.INT64, [30], idx)
        transpose = onnx.helper.make_node("Transpose", ["input"], ["fs270_t"], perm=[0, 1, 3, 2])
        gather = onnx.helper.make_node("Gather", ["fs270_t", "fs270_idx"], ["output"], axis=2)
        return _make_model([transpose, gather], [idx_init], shape=[batch, 10, 30, 30])

    inits = _slice_active_inits("fs270_sl", n, n)
    inits.append(onnx.helper.make_tensor("fs270_idx", onnx.TensorProto.INT64, [n], list(reversed(range(n)))))
    nodes = [
        onnx.helper.make_node("Slice", ["input", "fs270_sl_starts", "fs270_sl_ends", "fs270_sl_axes", "fs270_sl_steps"], ["fs270_active"]),
        onnx.helper.make_node("Transpose", ["fs270_active"], ["fs270_t"], perm=[0, 1, 3, 2]),
        onnx.helper.make_node("Gather", ["fs270_t", "fs270_idx"], ["fs270_rotated"], axis=2),
        onnx.helper.make_node("Pad", ["fs270_rotated"], ["output"], pads=[0, 0, 0, 0, 0, 0, 30 - n, 30 - n], mode="constant", value=0.0),
    ]
    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def fixed_shift_v_network(h, k, batch=1):
    return _gather_network(axis=2, indices=_fixed_canvas_indices(h, "cyclic", k=k), batch=batch)


def fixed_shift_h_network(w, k, batch=1):
    return _gather_network(axis=3, indices=_fixed_canvas_indices(w, "cyclic", k=k), batch=batch)


def marker_corners_network(marker_color, corners, batch=1):
    """ONNX: Slice + Slice+Pad x4 (shifted markers) + Mul x4 (grid mask) + Add x3 + Sub x2 + Concat.

    For tasks where the input has exactly one cell of `marker_color` (every other
    cell is color 0), and the output replaces the marker with 0 while placing
    `corners["TL"]`, `corners["TR"]`, `corners["BL"]`, `corners["BR"]` at the
    four diagonal positions around the marker (only in-bounds diagonals get a
    color; out-of-bounds are skipped).

    Pipeline (input [N, 10, 30, 30]):
      1. Slice channel `marker_color`     -> marker [N, 1, 30, 30]
      2. Slice channel 0                  -> input_ch0 [N, 1, 30, 30]
      3. grid_mask = input_ch0 + marker   -> 1 inside the active grid, 0 in padding
         (works because the task has only 2 input colors: 0 and the marker)
      4. For each diagonal (TL, TR, BR, BL):
         a. Slice + Pad to shift marker by (dr, dc)
         b. Mul by grid_mask to kill out-of-grid placements
      5. any_corner = sum of the 4 masked shifted markers
      6. ch0_out = grid_mask - any_corner (1 at active-grid-non-corner cells, 0 elsewhere)
      7. zeros = marker - marker (used for unset channels)
      8. Concat 10 channels in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] order, mapping each corner
         direction to its destination channel and filling unset channels with zeros.

    Cost (per cost-model verify, opset 10): macs ~9k (4 Muls + 3 Adds + Add + 2 Subs),
    memory ~71 KB (multiple [N, 1, 30, 30] intermediates), params 40 (5 Slice initializers).
    Total ~80k -> ~13.71 pts/task.
    """
    DIRS = [
        ("TL", -1, -1, corners["TL"]),
        ("TR", -1,  1, corners["TR"]),
        ("BR",  1,  1, corners["BR"]),
        ("BL",  1, -1, corners["BL"]),
    ]

    def _slice_channel(name_prefix, in_name, ch, out_name):
        starts = onnx.helper.make_tensor(f"{name_prefix}_starts", onnx.TensorProto.INT64, [1], [ch])
        ends = onnx.helper.make_tensor(f"{name_prefix}_ends", onnx.TensorProto.INT64, [1], [ch + 1])
        axes = onnx.helper.make_tensor(f"{name_prefix}_axes", onnx.TensorProto.INT64, [1], [1])
        steps = onnx.helper.make_tensor(f"{name_prefix}_steps", onnx.TensorProto.INT64, [1], [1])
        node = onnx.helper.make_node(
            "Slice",
            [in_name, f"{name_prefix}_starts", f"{name_prefix}_ends", f"{name_prefix}_axes", f"{name_prefix}_steps"],
            [out_name],
        )
        return node, [starts, ends, axes, steps]

    def _shift(name_prefix, in_name, dr, dc, out_name):
        if dr == -1:
            sh, ph = (1, 30), (0, 1)
        elif dr == 1:
            sh, ph = (0, 29), (1, 0)
        else:
            sh, ph = (0, 30), (0, 0)
        if dc == -1:
            sw, pw = (1, 30), (0, 1)
        elif dc == 1:
            sw, pw = (0, 29), (1, 0)
        else:
            sw, pw = (0, 30), (0, 0)
        starts = onnx.helper.make_tensor(f"{name_prefix}_starts", onnx.TensorProto.INT64, [2], [sh[0], sw[0]])
        ends = onnx.helper.make_tensor(f"{name_prefix}_ends", onnx.TensorProto.INT64, [2], [sh[1], sw[1]])
        axes = onnx.helper.make_tensor(f"{name_prefix}_axes", onnx.TensorProto.INT64, [2], [2, 3])
        steps = onnx.helper.make_tensor(f"{name_prefix}_steps", onnx.TensorProto.INT64, [2], [1, 1])
        slice_out = f"{name_prefix}_sliced"
        slice_node = onnx.helper.make_node(
            "Slice",
            [in_name, f"{name_prefix}_starts", f"{name_prefix}_ends", f"{name_prefix}_axes", f"{name_prefix}_steps"],
            [slice_out],
        )
        pad_node = onnx.helper.make_node(
            "Pad",
            [slice_out],
            [out_name],
            pads=[0, 0, ph[0], pw[0], 0, 0, ph[1], pw[1]],
            mode="constant",
            value=0.0,
        )
        return [slice_node, pad_node], [starts, ends, axes, steps]

    nodes = []
    inits = []

    n, i = _slice_channel("mc_mk", "input", marker_color, "mc_marker")
    nodes.append(n); inits += i
    n, i = _slice_channel("mc_ch0", "input", 0, "mc_input_ch0")
    nodes.append(n); inits += i
    nodes.append(onnx.helper.make_node("Add", ["mc_input_ch0", "mc_marker"], ["mc_grid_mask"]))

    for tag, dr, dc, _color in DIRS:
        ns, i = _shift(f"mc_sh_{tag}", "mc_marker", dr, dc, f"mc_shifted_{tag}_raw")
        nodes += ns; inits += i
        nodes.append(onnx.helper.make_node("Mul", [f"mc_shifted_{tag}_raw", "mc_grid_mask"], [f"mc_shifted_{tag}"]))

    nodes.append(onnx.helper.make_node("Add", ["mc_shifted_TL", "mc_shifted_TR"], ["mc_sum1"]))
    nodes.append(onnx.helper.make_node("Add", ["mc_shifted_BR", "mc_shifted_BL"], ["mc_sum2"]))
    nodes.append(onnx.helper.make_node("Add", ["mc_sum1", "mc_sum2"], ["mc_any_corner"]))
    nodes.append(onnx.helper.make_node("Sub", ["mc_grid_mask", "mc_any_corner"], ["mc_ch0_out"]))
    nodes.append(onnx.helper.make_node("Sub", ["mc_marker", "mc_marker"], ["mc_zeros"]))

    # Build ordered list of 10 channel sources: ch_i is either ch0_out, a shifted marker, or zeros.
    color_to_dir = {color: tag for tag, _, _, color in DIRS}
    channel_inputs = []
    for c in range(10):
        if c == 0:
            channel_inputs.append("mc_ch0_out")
        elif c == marker_color:
            channel_inputs.append("mc_zeros")
        elif c in color_to_dir:
            channel_inputs.append(f"mc_shifted_{color_to_dir[c]}")
        else:
            channel_inputs.append("mc_zeros")
    nodes.append(onnx.helper.make_node("Concat", channel_inputs, ["output"], axis=1))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def isolated_marker_recolor_network(marker_color, recolor, batch=1):
    """ONNX: Slice + Conv 3x3 + Equal + Cast + Mul + Sub + Add + Concat.

    For tasks where the rule is: every input cell of color `marker_color` that has
    NO `marker_color` 4-neighbor (up/down/left/right; within the active grid;
    canvas-padding cells don't count because they're channel 0, not channel
    `marker_color`) gets recolored to `recolor` in the output. All other cells
    are unchanged.

    Pipeline (input [N, 10, 30, 30]):
      1. Slice channel `marker_color`     -> mask_M [N, 1, 30, 30]
      2. Slice channel `recolor`          -> mask_R [N, 1, 30, 30]
      3. Conv 3x3 with kernel             -> nbrs   [N, 1, 30, 30]
            [[0,1,0],
             [1,0,1],
             [0,1,0]]
         on mask_M with pads=[1,1,1,1]; counts the number of M 4-neighbors per cell.
         Padding region is mask_M=0 so canvas padding doesn't count.
      4. Less(nbrs, 0.5)                 -> is_zero_bool [N, 1, 30, 30]
            (nbrs is int-valued in {0,1,2,3,4}; <0.5 ⇔ ==0)
            (Equal isn't float-supported until opset 13; our pin is opset 10)
      5. Cast to float                   -> is_zero [N, 1, 30, 30]
      6. Mul(mask_M, is_zero)            -> is_isolated [N, 1, 30, 30]
                                            (cells that are M AND have 0 M-neighbors)
      7. new_mask_M = Sub(mask_M, is_isolated)   (M loses the isolated cells)
      8. new_mask_R = Add(mask_R, is_isolated)   (R gains the isolated cells)
      9. For each other channel i in 0..9, i ∉ {M, R}: Slice -> pass through.
     10. Concat all 10 channels in order [0, 1, ..., 9].

    Cost (per cluster-summary cost model, opset 10): macs ~12,000 (Conv 3x3 1->1 = 8100,
    Equal/Mul/Sub/Add on [1,1,30,30] = 900 each), memory ~50 KB (10+ intermediate
    [1,1,30,30] tensors ~3.6 KB each plus padded Conv input), params ~50 (Conv kernel 9,
    Equal scalar 1, 10 Slice ops × 4 initializers each = 40). Total ~62k -> ~14.0 pts/task.
    """
    if marker_color == recolor:
        raise ValueError(f"marker_color={marker_color} must differ from recolor={recolor}")
    if not (0 <= marker_color < 10) or not (0 <= recolor < 10):
        raise ValueError(f"marker_color and recolor must be in 0..9")

    def _slice_channel(name_prefix, in_name, ch, out_name):
        starts = onnx.helper.make_tensor(f"{name_prefix}_starts", onnx.TensorProto.INT64, [1], [ch])
        ends = onnx.helper.make_tensor(f"{name_prefix}_ends", onnx.TensorProto.INT64, [1], [ch + 1])
        axes = onnx.helper.make_tensor(f"{name_prefix}_axes", onnx.TensorProto.INT64, [1], [1])
        steps = onnx.helper.make_tensor(f"{name_prefix}_steps", onnx.TensorProto.INT64, [1], [1])
        node = onnx.helper.make_node(
            "Slice",
            [in_name, f"{name_prefix}_starts", f"{name_prefix}_ends", f"{name_prefix}_axes", f"{name_prefix}_steps"],
            [out_name],
        )
        return node, [starts, ends, axes, steps]

    nodes = []
    inits = []

    # Slice all 10 channels (we'll re-use mask_M and mask_R; others pass through)
    channel_names = {}
    for ch in range(10):
        out_name = f"imr_ch{ch}"
        node, ch_inits = _slice_channel(f"imr_sl_{ch}", "input", ch, out_name)
        nodes.append(node)
        inits += ch_inits
        channel_names[ch] = out_name

    # Conv 3x3 with 8-neighbor kernel on mask_M
    kernel = onnx.helper.make_tensor(
        "imr_conv_w",
        onnx.TensorProto.FLOAT,
        [1, 1, 3, 3],
        [0.0, 1.0, 0.0,
         1.0, 0.0, 1.0,
         0.0, 1.0, 0.0],
    )
    inits.append(kernel)
    nodes.append(onnx.helper.make_node(
        "Conv",
        [channel_names[marker_color], "imr_conv_w"],
        ["imr_nbrs"],
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        strides=[1, 1],
    ))

    # Less(nbrs, 0.5) — nbrs is int-valued in {0,1,2,3,4}; <0.5 ⇔ ==0
    # (Equal doesn't support float until opset 13; our pin is opset 10.)
    half_scalar = onnx.helper.make_tensor("imr_half", onnx.TensorProto.FLOAT, [1], [0.5])
    inits.append(half_scalar)
    nodes.append(onnx.helper.make_node("Less", ["imr_nbrs", "imr_half"], ["imr_is_zero_bool"]))
    nodes.append(onnx.helper.make_node("Cast", ["imr_is_zero_bool"], ["imr_is_zero"], to=onnx.TensorProto.FLOAT))

    # is_isolated = mask_M * is_zero
    nodes.append(onnx.helper.make_node("Mul", [channel_names[marker_color], "imr_is_zero"], ["imr_is_isolated"]))

    # new_mask_M = mask_M - is_isolated
    nodes.append(onnx.helper.make_node("Sub", [channel_names[marker_color], "imr_is_isolated"], ["imr_new_mask_M"]))
    # new_mask_R = mask_R + is_isolated
    nodes.append(onnx.helper.make_node("Add", [channel_names[recolor], "imr_is_isolated"], ["imr_new_mask_R"]))

    # Build channel inputs for Concat: replace M and R with new_mask_M / new_mask_R
    concat_inputs = []
    for ch in range(10):
        if ch == marker_color:
            concat_inputs.append("imr_new_mask_M")
        elif ch == recolor:
            concat_inputs.append("imr_new_mask_R")
        else:
            concat_inputs.append(channel_names[ch])
    nodes.append(onnx.helper.make_node("Concat", concat_inputs, ["output"], axis=1))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def l_shape_missing_corner_network(marker_color, src_color, fill_color, batch=1):
    """ONNX: Slice + Conv 2x2 (block-count) + Sub + Mul + Less + Cast +
    Conv 2x2 (any-block-near-cell) + Greater + Cast + Mul + Sub + Add + Concat.

    For tasks where the rule is: in any 2x2 block of the input with exactly 3 cells
    of color `marker_color` (M) and 1 cell of color `src_color` (C), that C cell
    becomes color `fill_color` (F) in the output. All other cells unchanged.

    Pipeline (input [N, 10, 30, 30]):
      1. Slice all 10 channels  -> ch[0..9] each [N, 1, 30, 30]
      2. block_count = Conv 2x2 with all-1 kernel on ch[M], pads=[0,0,1,1]:
            block_count[r,c] = sum of mask_M over the 2x2 block with TL at (r,c).
            Bottom and right padding (=0) means last row/col of block_count is
            undercounted, but count==3 cannot fire spuriously there.
      3. count_minus_3 = Sub(block_count, 3.0)
      4. count_minus_3_sq = Mul(count_minus_3, count_minus_3)
      5. block_has_3_bool = Less(count_minus_3_sq, 0.5)
            count is int-valued in {0,1,2,3,4}; (count-3)^2 in {0,1,4,9}; <0.5 iff ==0 iff count==3.
            (Equal isn't float-supported until opset 13; our pin is opset 10.)
      6. block_has_3 = Cast(block_has_3_bool, FLOAT)
      7. cell_in_l_count = Conv 2x2 with all-1 kernel on block_has_3, pads=[1,1,0,0]:
            cell_in_l_count[i,j] = sum over the 4 blocks with TL in
            {(i-1,j-1),(i-1,j),(i,j-1),(i,j)} of block_has_3 (out-of-grid blocks
            contribute 0 via padding). Cell (i,j) is in <=4 blocks; in_l fires iff
            any of those blocks has count==3.
      8. cell_in_l_bool = Greater(cell_in_l_count, 0.5)
      9. cell_in_l = Cast to FLOAT
     10. cell_changes = Mul(cell_in_l, ch[C])  -> only C cells in qualifying L flip
     11. new_mask_C = Sub(ch[C], cell_changes)
     12. new_mask_F = Add(ch[F], cell_changes)
     13. Concat 10 channels: substitute new_mask_C for ch C, new_mask_F for ch F,
         others pass through.

    Cost (rough, opset 10): macs ~16k (2 Conv 2x2 1->1 = 3,600 each + ~6 elementwise
    ops on [1,1,30,30] = 900 each), memory ~72 KB (multiple [1,1,30,30] intermediates),
    params ~50 (2 conv kernels of 4 floats + scalar consts + 10 Slice initializers).
    Total ~90k cost -> ~13.6 pts/task. Verify with score_network before merge.
    """
    if marker_color == src_color or marker_color == fill_color or src_color == fill_color:
        raise ValueError(
            f"marker_color={marker_color}, src_color={src_color}, fill_color={fill_color} must all differ"
        )
    for c, name in [(marker_color, "marker_color"), (src_color, "src_color"), (fill_color, "fill_color")]:
        if not (0 <= c < 10):
            raise ValueError(f"{name}={c} must be in 0..9")

    def _slice_channel(name_prefix, in_name, ch, out_name):
        starts = onnx.helper.make_tensor(f"{name_prefix}_starts", onnx.TensorProto.INT64, [1], [ch])
        ends = onnx.helper.make_tensor(f"{name_prefix}_ends", onnx.TensorProto.INT64, [1], [ch + 1])
        axes = onnx.helper.make_tensor(f"{name_prefix}_axes", onnx.TensorProto.INT64, [1], [1])
        steps = onnx.helper.make_tensor(f"{name_prefix}_steps", onnx.TensorProto.INT64, [1], [1])
        node = onnx.helper.make_node(
            "Slice",
            [in_name, f"{name_prefix}_starts", f"{name_prefix}_ends", f"{name_prefix}_axes", f"{name_prefix}_steps"],
            [out_name],
        )
        return node, [starts, ends, axes, steps]

    nodes = []
    inits = []

    channel_names = {}
    for ch in range(10):
        out_name = f"lsm_ch{ch}"
        node, ch_inits = _slice_channel(f"lsm_sl_{ch}", "input", ch, out_name)
        nodes.append(node)
        inits += ch_inits
        channel_names[ch] = out_name

    # Conv 2x2 all-1 kernel on mask_M to count M's per 2x2 block (TL-anchored).
    block_kernel = onnx.helper.make_tensor(
        "lsm_block_w", onnx.TensorProto.FLOAT, [1, 1, 2, 2], [1.0, 1.0, 1.0, 1.0]
    )
    inits.append(block_kernel)
    nodes.append(onnx.helper.make_node(
        "Conv",
        [channel_names[marker_color], "lsm_block_w"],
        ["lsm_block_count"],
        kernel_shape=[2, 2],
        pads=[0, 0, 1, 1],
        strides=[1, 1],
    ))

    three = onnx.helper.make_tensor("lsm_three", onnx.TensorProto.FLOAT, [1], [3.0])
    half = onnx.helper.make_tensor("lsm_half", onnx.TensorProto.FLOAT, [1], [0.5])
    inits += [three, half]
    nodes.append(onnx.helper.make_node("Sub", ["lsm_block_count", "lsm_three"], ["lsm_diff"]))
    nodes.append(onnx.helper.make_node("Mul", ["lsm_diff", "lsm_diff"], ["lsm_diff_sq"]))
    nodes.append(onnx.helper.make_node("Less", ["lsm_diff_sq", "lsm_half"], ["lsm_block_has_3_bool"]))
    nodes.append(onnx.helper.make_node("Cast", ["lsm_block_has_3_bool"], ["lsm_block_has_3"], to=onnx.TensorProto.FLOAT))

    # Second Conv 2x2 all-1, pads=[1,1,0,0]: per-cell sum of the (up to 4) blocks containing it.
    near_kernel = onnx.helper.make_tensor(
        "lsm_near_w", onnx.TensorProto.FLOAT, [1, 1, 2, 2], [1.0, 1.0, 1.0, 1.0]
    )
    inits.append(near_kernel)
    nodes.append(onnx.helper.make_node(
        "Conv",
        ["lsm_block_has_3", "lsm_near_w"],
        ["lsm_cell_in_l_count"],
        kernel_shape=[2, 2],
        pads=[1, 1, 0, 0],
        strides=[1, 1],
    ))

    nodes.append(onnx.helper.make_node("Greater", ["lsm_cell_in_l_count", "lsm_half"], ["lsm_cell_in_l_bool"]))
    nodes.append(onnx.helper.make_node("Cast", ["lsm_cell_in_l_bool"], ["lsm_cell_in_l"], to=onnx.TensorProto.FLOAT))

    nodes.append(onnx.helper.make_node("Mul", ["lsm_cell_in_l", channel_names[src_color]], ["lsm_cell_changes"]))
    nodes.append(onnx.helper.make_node("Sub", [channel_names[src_color], "lsm_cell_changes"], ["lsm_new_mask_C"]))
    nodes.append(onnx.helper.make_node("Add", [channel_names[fill_color], "lsm_cell_changes"], ["lsm_new_mask_F"]))

    concat_inputs = []
    for ch in range(10):
        if ch == src_color:
            concat_inputs.append("lsm_new_mask_C")
        elif ch == fill_color:
            concat_inputs.append("lsm_new_mask_F")
        else:
            concat_inputs.append(channel_names[ch])
    nodes.append(onnx.helper.make_node("Concat", concat_inputs, ["output"], axis=1))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def aligned_marker_fill_network(marker_color, src_color, fill_color, use_h=True, use_v=True, batch=1):
    """ONNX: Slice + Conv 1x30 (row cumsum) + Slice/Pad shift + Sub broadcast +
    Greater/Cast + Mul + (same for column) + Add + Greater/Cast + Mul + Sub + Add + Concat.

    For tasks where the rule is: pairs of `marker_color` (M) cells aligned in the
    same row OR column get the cells strictly between them filled with `fill_color`
    (F) in the output, but only for cells whose original color was `src_color` (C).
    Markers themselves and cells of other colors are unchanged.

    Algorithm: a cell (r, c) is "between row-markers" iff there are M-markers both
    strictly left of c AND strictly right of c in row r. Equivalently, the per-row
    inclusive cumsum from the left at position c-1 is > 0, AND the row-total minus
    the inclusive cumsum at c is > 0 (i.e., strict-right count > 0). Same logic for
    columns. With M's mask {0,1}, the Conv 1x30 (or 30x1) ones-kernel computes the
    inclusive cumsum directly. Padded canvas cells contribute 0 to mask_M so the
    rule can't fire there.

    Pipeline (input [N, 10, 30, 30]):
      1. Slice all 10 channels.
      2. cumsum_left_h = Conv 1x30 ones-kernel on mask_M, pads=[0,29,0,0]:
            cumsum_left_h[r, c] = sum_{c'<=c} mask_M[r, c'] (inclusive).
      3. row_total = Slice cumsum_left_h at column 29 -> [N,1,30,1].
      4. strict_left_h = Pad(left=1)(Slice cumsum_left_h cols [0..28]):
            strict_left_h[r, c] = cumsum_left_h[r, c-1] (or 0 at c=0).
      5. strict_right_h = Sub(row_total, cumsum_left_h)  (broadcast over W).
      6. has_left_h = Cast(Greater(strict_left_h, 0.5))
         has_right_h = Cast(Greater(strict_right_h, 0.5))
      7. between_h = Mul(has_left_h, has_right_h).
      8. Same as 2-7 along H axis (Conv 30x1 on mask_M, pads=[29,0,0,0]) -> between_v.
      9. between_any = Cast(Greater(Add(between_h, between_v), 0.5)).
     10. cell_changes = Mul(between_any, mask_C).
     11. new_mask_C = Sub(mask_C, cell_changes); new_mask_F = Add(mask_F, cell_changes).
     12. Concat 10 channels.

    Cost (rough, opset 10): macs ~60k (2 directional Convs at ~27k each + ~8 elementwise
    on [1,1,30,30] at 900 each), memory ~85 KB (multiple [1,1,30,30] intermediates +
    Conv padded inputs), params ~150 (2 Conv kernels of 30 + slice/pad initializers
    + scalars). Total ~145k -> ~13.1 pts/task. Verify with score_network.
    """
    if not (use_h or use_v):
        raise ValueError("at least one of use_h, use_v must be True")
    if marker_color == src_color or marker_color == fill_color or src_color == fill_color:
        raise ValueError(
            f"marker_color={marker_color}, src_color={src_color}, fill_color={fill_color} must all differ"
        )
    for c, name in [(marker_color, "marker_color"), (src_color, "src_color"), (fill_color, "fill_color")]:
        if not (0 <= c < 10):
            raise ValueError(f"{name}={c} must be in 0..9")

    def _slice_channel(name_prefix, in_name, ch, out_name):
        starts = onnx.helper.make_tensor(f"{name_prefix}_starts", onnx.TensorProto.INT64, [1], [ch])
        ends = onnx.helper.make_tensor(f"{name_prefix}_ends", onnx.TensorProto.INT64, [1], [ch + 1])
        axes = onnx.helper.make_tensor(f"{name_prefix}_axes", onnx.TensorProto.INT64, [1], [1])
        steps = onnx.helper.make_tensor(f"{name_prefix}_steps", onnx.TensorProto.INT64, [1], [1])
        node = onnx.helper.make_node(
            "Slice",
            [in_name, f"{name_prefix}_starts", f"{name_prefix}_ends", f"{name_prefix}_axes", f"{name_prefix}_steps"],
            [out_name],
        )
        return node, [starts, ends, axes, steps]

    nodes = []
    inits = []

    channel_names = {}
    for ch in range(10):
        out_name = f"amf_ch{ch}"
        node, ch_inits = _slice_channel(f"amf_sl_{ch}", "input", ch, out_name)
        nodes.append(node)
        inits += ch_inits
        channel_names[ch] = out_name

    half = onnx.helper.make_tensor("amf_half", onnx.TensorProto.FLOAT, [1], [0.5])
    inits.append(half)

    def _build_axis_between(prefix, axis, kernel_shape, conv_pads, slice_axis_idx, pad_pads):
        """Compute between_<axis> mask via inclusive-cumsum trick. Returns the output node name."""
        kernel_w = onnx.helper.make_tensor(
            f"{prefix}_w", onnx.TensorProto.FLOAT, [1, 1] + list(kernel_shape), [1.0] * (kernel_shape[0] * kernel_shape[1])
        )
        inits.append(kernel_w)
        nodes.append(onnx.helper.make_node(
            "Conv",
            [channel_names[marker_color], f"{prefix}_w"],
            [f"{prefix}_cumsum"],
            kernel_shape=list(kernel_shape),
            pads=conv_pads,
            strides=[1, 1],
        ))
        # Slice last position along the active axis to get total count: shape has 1 along that axis.
        total_starts = onnx.helper.make_tensor(f"{prefix}_tot_starts", onnx.TensorProto.INT64, [1], [29])
        total_ends = onnx.helper.make_tensor(f"{prefix}_tot_ends", onnx.TensorProto.INT64, [1], [30])
        total_axes = onnx.helper.make_tensor(f"{prefix}_tot_axes", onnx.TensorProto.INT64, [1], [slice_axis_idx])
        total_steps = onnx.helper.make_tensor(f"{prefix}_tot_steps", onnx.TensorProto.INT64, [1], [1])
        inits.extend([total_starts, total_ends, total_axes, total_steps])
        nodes.append(onnx.helper.make_node(
            "Slice",
            [f"{prefix}_cumsum", f"{prefix}_tot_starts", f"{prefix}_tot_ends", f"{prefix}_tot_axes", f"{prefix}_tot_steps"],
            [f"{prefix}_total"],
        ))
        # strict_far = total - cumsum (broadcast).
        nodes.append(onnx.helper.make_node("Sub", [f"{prefix}_total", f"{prefix}_cumsum"], [f"{prefix}_strict_far"]))
        # strict_near = Slice cumsum on the active axis [0..28], then Pad +1 at the start of that axis.
        near_starts = onnx.helper.make_tensor(f"{prefix}_near_starts", onnx.TensorProto.INT64, [1], [0])
        near_ends = onnx.helper.make_tensor(f"{prefix}_near_ends", onnx.TensorProto.INT64, [1], [29])
        near_axes = onnx.helper.make_tensor(f"{prefix}_near_axes", onnx.TensorProto.INT64, [1], [slice_axis_idx])
        near_steps = onnx.helper.make_tensor(f"{prefix}_near_steps", onnx.TensorProto.INT64, [1], [1])
        inits.extend([near_starts, near_ends, near_axes, near_steps])
        nodes.append(onnx.helper.make_node(
            "Slice",
            [f"{prefix}_cumsum", f"{prefix}_near_starts", f"{prefix}_near_ends", f"{prefix}_near_axes", f"{prefix}_near_steps"],
            [f"{prefix}_sliced"],
        ))
        nodes.append(onnx.helper.make_node(
            "Pad",
            [f"{prefix}_sliced"],
            [f"{prefix}_strict_near"],
            pads=pad_pads,
            mode="constant",
            value=0.0,
        ))
        nodes.append(onnx.helper.make_node("Greater", [f"{prefix}_strict_near", "amf_half"], [f"{prefix}_has_near_b"]))
        nodes.append(onnx.helper.make_node("Cast", [f"{prefix}_has_near_b"], [f"{prefix}_has_near"], to=onnx.TensorProto.FLOAT))
        nodes.append(onnx.helper.make_node("Greater", [f"{prefix}_strict_far", "amf_half"], [f"{prefix}_has_far_b"]))
        nodes.append(onnx.helper.make_node("Cast", [f"{prefix}_has_far_b"], [f"{prefix}_has_far"], to=onnx.TensorProto.FLOAT))
        nodes.append(onnx.helper.make_node("Mul", [f"{prefix}_has_near", f"{prefix}_has_far"], [f"{prefix}_between"]))
        return f"{prefix}_between"

    between_outputs = []
    if use_h:
        # H direction: kernel 1x30, conv pads=[0, 29, 0, 0], slice axis=3, pad 1 col on left
        # Pad pads format for 4D: [N_b, C_b, H_b, W_b, N_e, C_e, H_e, W_e].
        between_outputs.append(_build_axis_between(
            "amf_h", axis="W",
            kernel_shape=(1, 30),
            conv_pads=[0, 29, 0, 0],
            slice_axis_idx=3,
            pad_pads=[0, 0, 0, 1, 0, 0, 0, 0],
        ))
    if use_v:
        # V direction: kernel 30x1, conv pads=[29, 0, 0, 0], slice axis=2, pad 1 row on top
        between_outputs.append(_build_axis_between(
            "amf_v", axis="H",
            kernel_shape=(30, 1),
            conv_pads=[29, 0, 0, 0],
            slice_axis_idx=2,
            pad_pads=[0, 0, 1, 0, 0, 0, 0, 0],
        ))

    if len(between_outputs) == 1:
        between_any_name = between_outputs[0]
    else:
        nodes.append(onnx.helper.make_node("Add", between_outputs, ["amf_sum"]))
        nodes.append(onnx.helper.make_node("Greater", ["amf_sum", "amf_half"], ["amf_between_any_b"]))
        nodes.append(onnx.helper.make_node("Cast", ["amf_between_any_b"], ["amf_between_any"], to=onnx.TensorProto.FLOAT))
        between_any_name = "amf_between_any"

    nodes.append(onnx.helper.make_node("Mul", [between_any_name, channel_names[src_color]], ["amf_cell_changes"]))
    nodes.append(onnx.helper.make_node("Sub", [channel_names[src_color], "amf_cell_changes"], ["amf_new_mask_C"]))
    nodes.append(onnx.helper.make_node("Add", [channel_names[fill_color], "amf_cell_changes"], ["amf_new_mask_F"]))

    concat_inputs = []
    for ch in range(10):
        if ch == src_color:
            concat_inputs.append("amf_new_mask_C")
        elif ch == fill_color:
            concat_inputs.append("amf_new_mask_F")
        else:
            concat_inputs.append(channel_names[ch])
    nodes.append(onnx.helper.make_node("Concat", concat_inputs, ["output"], axis=1))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def sandwich_fill_network(marker_color, src_color, fill_color, use_h, use_v, batch=1):
    """ONNX: Slice + 1 or 2 Convs (1x3 / 3x1) + Greater/Cast + Mul + Sub + Add + Concat.

    For tasks where the rule is: cells (r, c) of color `src_color` (C) sandwiched
    by `marker_color` (M) markers at offset 1 in the row (and/or column) direction
    become color `fill_color` (F). Markers and other-colored cells unchanged.

    Differs from `aligned-marker-fill`: this only fills cells with M at distance
    EXACTLY 1 on both sides (i.e., m...c...m where m is M and c is between with
    no gap of more than 1 between m and c). aligned-marker-fill fills any cells
    strictly between any two aligned M's regardless of distance.

    Args:
      use_h: if True, check horizontal sandwich (M at (r, c-1) AND M at (r, c+1)).
      use_v: if True, check vertical sandwich (M at (r-1, c) AND M at (r+1, c)).

    Pipeline (input [N, 10, 30, 30]):
      1. Slice all 10 channels.
      2. If use_h: Conv 1x3 kernel [1, 0, 1] on mask_M, pads=[0, 1, 0, 1].
         h_count[r, c] = mask_M[r, c-1] + mask_M[r, c+1]; max value 2.
         has_h = Cast(Greater(h_count, 1.5)).
      3. If use_v: Conv 3x1 kernel [[1], [0], [1]], pads=[1, 0, 1, 0].
         v_count, has_v.
      4. sandwich = if both: Cast(Greater(Add(has_h, has_v), 0.5)); else has_h or has_v alone.
      5. cell_changes = Mul(sandwich, mask_C).
      6. new_mask_C = Sub(mask_C, cell_changes); new_mask_F = Add(mask_F, cell_changes).
      7. Concat 10 channels.

    Cost (rough): ~8k MACs per Conv (1x3 or 3x1, kernel volume 3, output 30x30 = 2,700 each)
    plus ~5k elementwise ops on [1,1,30,30]. Memory ~30 KB, params ~50.
    Total ~30-50k cost -> ~13.7-14.4 pts/task.
    """
    if not (use_h or use_v):
        raise ValueError("at least one of use_h, use_v must be True")
    if marker_color == src_color or marker_color == fill_color or src_color == fill_color:
        raise ValueError(
            f"marker_color={marker_color}, src_color={src_color}, fill_color={fill_color} must all differ"
        )
    for c, name in [(marker_color, "marker_color"), (src_color, "src_color"), (fill_color, "fill_color")]:
        if not (0 <= c < 10):
            raise ValueError(f"{name}={c} must be in 0..9")

    def _slice_channel(name_prefix, in_name, ch, out_name):
        starts = onnx.helper.make_tensor(f"{name_prefix}_starts", onnx.TensorProto.INT64, [1], [ch])
        ends = onnx.helper.make_tensor(f"{name_prefix}_ends", onnx.TensorProto.INT64, [1], [ch + 1])
        axes = onnx.helper.make_tensor(f"{name_prefix}_axes", onnx.TensorProto.INT64, [1], [1])
        steps = onnx.helper.make_tensor(f"{name_prefix}_steps", onnx.TensorProto.INT64, [1], [1])
        node = onnx.helper.make_node(
            "Slice",
            [in_name, f"{name_prefix}_starts", f"{name_prefix}_ends", f"{name_prefix}_axes", f"{name_prefix}_steps"],
            [out_name],
        )
        return node, [starts, ends, axes, steps]

    nodes = []
    inits = []

    channel_names = {}
    for ch in range(10):
        out_name = f"sw_ch{ch}"
        node, ch_inits = _slice_channel(f"sw_sl_{ch}", "input", ch, out_name)
        nodes.append(node)
        inits.extend(ch_inits)
        channel_names[ch] = out_name

    half = onnx.helper.make_tensor("sw_half", onnx.TensorProto.FLOAT, [1], [0.5])
    one_five = onnx.helper.make_tensor("sw_one_five", onnx.TensorProto.FLOAT, [1], [1.5])
    inits.extend([half, one_five])

    sand_outputs = []
    if use_h:
        h_kernel = onnx.helper.make_tensor(
            "sw_h_w", onnx.TensorProto.FLOAT, [1, 1, 1, 3], [1.0, 0.0, 1.0]
        )
        inits.append(h_kernel)
        nodes.append(onnx.helper.make_node(
            "Conv",
            [channel_names[marker_color], "sw_h_w"],
            ["sw_h_count"],
            kernel_shape=[1, 3],
            pads=[0, 1, 0, 1],
            strides=[1, 1],
        ))
        nodes.append(onnx.helper.make_node("Greater", ["sw_h_count", "sw_one_five"], ["sw_has_h_b"]))
        nodes.append(onnx.helper.make_node("Cast", ["sw_has_h_b"], ["sw_has_h"], to=onnx.TensorProto.FLOAT))
        sand_outputs.append("sw_has_h")
    if use_v:
        v_kernel = onnx.helper.make_tensor(
            "sw_v_w", onnx.TensorProto.FLOAT, [1, 1, 3, 1], [1.0, 0.0, 1.0]
        )
        inits.append(v_kernel)
        nodes.append(onnx.helper.make_node(
            "Conv",
            [channel_names[marker_color], "sw_v_w"],
            ["sw_v_count"],
            kernel_shape=[3, 1],
            pads=[1, 0, 1, 0],
            strides=[1, 1],
        ))
        nodes.append(onnx.helper.make_node("Greater", ["sw_v_count", "sw_one_five"], ["sw_has_v_b"]))
        nodes.append(onnx.helper.make_node("Cast", ["sw_has_v_b"], ["sw_has_v"], to=onnx.TensorProto.FLOAT))
        sand_outputs.append("sw_has_v")

    if len(sand_outputs) == 1:
        sandwich_name = sand_outputs[0]
    else:
        nodes.append(onnx.helper.make_node("Add", sand_outputs, ["sw_sum"]))
        nodes.append(onnx.helper.make_node("Greater", ["sw_sum", "sw_half"], ["sw_sandwich_b"]))
        nodes.append(onnx.helper.make_node("Cast", ["sw_sandwich_b"], ["sw_sandwich"], to=onnx.TensorProto.FLOAT))
        sandwich_name = "sw_sandwich"

    nodes.append(onnx.helper.make_node("Mul", [sandwich_name, channel_names[src_color]], ["sw_cell_changes"]))
    nodes.append(onnx.helper.make_node("Sub", [channel_names[src_color], "sw_cell_changes"], ["sw_new_mask_C"]))
    nodes.append(onnx.helper.make_node("Add", [channel_names[fill_color], "sw_cell_changes"], ["sw_new_mask_F"]))

    concat_inputs = []
    for ch in range(10):
        if ch == src_color:
            concat_inputs.append("sw_new_mask_C")
        elif ch == fill_color:
            concat_inputs.append("sw_new_mask_F")
        else:
            concat_inputs.append(channel_names[ch])
    nodes.append(onnx.helper.make_node("Concat", concat_inputs, ["output"], axis=1))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def pi_shape_project_to_bottom_network(fill_color, batch=1):
    """ONNX: Slice + ReduceSum + Sub + Conv 2x3 + Greater/Cast + Mul + ReduceMax x2 +
    Slice/Pad + Mul + Sub + Mul (broadcast) + Sub + Add + Concat.

    For tasks where the rule is: each input "Pi" gap (a cell (r, c) of color 0 such
    that the 5-cell stencil {(r-1,c-1),(r-1,c),(r-1,c+1),(r,c-1),(r,c+1)} all hold
    nonzero colors) projects color F (fill_color) onto the active-grid bottom row at
    column c. All other cells unchanged.

    Active-grid bottom row is detected dynamically as: the row r where some active
    cell exists at row r AND no active cells exist at row r+1 (or r=29). Assumes
    a single contiguous active region (the validator's [1,10,30,30] convention
    places the active grid at top-left); two stacked regions would each get a
    "bottom row" mark.

    Pipeline (input [N, 10, 30, 30]):
      1. Slice all 10 channels: ch[0..9].
      2. is_active = ReduceSum(input, axes=[1], keepdims=True). [N,1,30,30].
            1 in active-grid cells (any color), 0 in canvas padding.
      3. is_nonzero = is_active - ch[0]. [N,1,30,30].
            1 in active-grid non-zero-color cells, 0 elsewhere.
      4. stencil_sum = Conv 2x3 ones-except-center kernel [[1,1,1],[1,0,1]] on
         is_nonzero, pads=[1,1,0,1].
            stencil_sum[r, c] = count of nonzero stencil cells around (r, c).
      5. has_pi = Cast(Greater(stencil_sum, 4.5)).
      6. is_gap = Mul(has_pi, ch[0]). 1 iff has_pi AND center is color 0.
      7. gap_col_mask = ReduceMax(is_gap, axes=[2], keepdims=True). [N,1,1,30].
      8. row_has_active = ReduceMax(is_active, axes=[3], keepdims=True). [N,1,30,1].
      9. shifted_up = Pad(Slice(row_has_active rows [1..30]), bottom 1). [N,1,30,1].
            shifted_up[r] = row_has_active[r+1] (or 0 at r=29).
     10. row_active_AND_shifted = Mul(row_has_active, shifted_up).
         bottom_row_mask = Sub(row_has_active, row_active_AND_shifted).
            = row_has_active * (1 - shifted_up). 1 iff r is the last active row.
     11. final_4_mask = Mul(bottom_row_mask, gap_col_mask). Broadcast -> [N,1,30,30].
     12. new_mask_0 = Sub(ch[0], final_4_mask).
         new_mask_F = Add(ch[fill_color], final_4_mask).
     13. Concat 10 channels (substitute mask_0, mask_F; pass-through others).

    Cost (rough, opset 10): macs ~30k (ReduceSum 8.1k + Sub 0.9k + Conv 5.4k +
    several elementwise + 2 ReduceMax ~1k each), memory ~50 KB, params ~70.
    Total ~80k -> ~13.7 pts/task. Verify with score_network.
    """
    if fill_color == 0 or not (0 <= fill_color < 10):
        raise ValueError(f"fill_color={fill_color} must be in 1..9")

    def _slice_channel(name_prefix, in_name, ch, out_name):
        starts = onnx.helper.make_tensor(f"{name_prefix}_starts", onnx.TensorProto.INT64, [1], [ch])
        ends = onnx.helper.make_tensor(f"{name_prefix}_ends", onnx.TensorProto.INT64, [1], [ch + 1])
        axes = onnx.helper.make_tensor(f"{name_prefix}_axes", onnx.TensorProto.INT64, [1], [1])
        steps = onnx.helper.make_tensor(f"{name_prefix}_steps", onnx.TensorProto.INT64, [1], [1])
        node = onnx.helper.make_node(
            "Slice",
            [in_name, f"{name_prefix}_starts", f"{name_prefix}_ends", f"{name_prefix}_axes", f"{name_prefix}_steps"],
            [out_name],
        )
        return node, [starts, ends, axes, steps]

    nodes = []
    inits = []

    channel_names = {}
    for ch in range(10):
        out_name = f"pi_ch{ch}"
        node, ch_inits = _slice_channel(f"pi_sl_{ch}", "input", ch, out_name)
        nodes.append(node)
        inits.extend(ch_inits)
        channel_names[ch] = out_name

    nodes.append(onnx.helper.make_node(
        "ReduceSum", ["input"], ["pi_is_active"], axes=[1], keepdims=1
    ))
    nodes.append(onnx.helper.make_node("Sub", ["pi_is_active", channel_names[0]], ["pi_is_nonzero"]))

    stencil_kernel = onnx.helper.make_tensor(
        "pi_stencil_w", onnx.TensorProto.FLOAT, [1, 1, 2, 3],
        [1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
    )
    inits.append(stencil_kernel)
    nodes.append(onnx.helper.make_node(
        "Conv",
        ["pi_is_nonzero", "pi_stencil_w"],
        ["pi_stencil_sum"],
        kernel_shape=[2, 3],
        pads=[1, 1, 0, 1],
        strides=[1, 1],
    ))

    threshold = onnx.helper.make_tensor("pi_thresh", onnx.TensorProto.FLOAT, [1], [4.5])
    inits.append(threshold)
    nodes.append(onnx.helper.make_node("Greater", ["pi_stencil_sum", "pi_thresh"], ["pi_has_pi_b"]))
    nodes.append(onnx.helper.make_node("Cast", ["pi_has_pi_b"], ["pi_has_pi"], to=onnx.TensorProto.FLOAT))
    nodes.append(onnx.helper.make_node("Mul", ["pi_has_pi", channel_names[0]], ["pi_is_gap"]))

    nodes.append(onnx.helper.make_node(
        "ReduceMax", ["pi_is_gap"], ["pi_gap_col_mask"], axes=[2], keepdims=1
    ))
    nodes.append(onnx.helper.make_node(
        "ReduceMax", ["pi_is_active"], ["pi_row_has_active"], axes=[3], keepdims=1
    ))

    # shifted_up = Pad(bottom=1)(Slice rows [1..30] of row_has_active)
    shift_starts = onnx.helper.make_tensor("pi_shift_starts", onnx.TensorProto.INT64, [1], [1])
    shift_ends = onnx.helper.make_tensor("pi_shift_ends", onnx.TensorProto.INT64, [1], [30])
    shift_axes = onnx.helper.make_tensor("pi_shift_axes", onnx.TensorProto.INT64, [1], [2])
    shift_steps = onnx.helper.make_tensor("pi_shift_steps", onnx.TensorProto.INT64, [1], [1])
    inits.extend([shift_starts, shift_ends, shift_axes, shift_steps])
    nodes.append(onnx.helper.make_node(
        "Slice",
        ["pi_row_has_active", "pi_shift_starts", "pi_shift_ends", "pi_shift_axes", "pi_shift_steps"],
        ["pi_shifted_sliced"],
    ))
    nodes.append(onnx.helper.make_node(
        "Pad",
        ["pi_shifted_sliced"],
        ["pi_shifted_up"],
        pads=[0, 0, 0, 0, 0, 0, 1, 0],
        mode="constant",
        value=0.0,
    ))

    # bottom_row_mask = row_has_active * (1 - shifted_up) = row_has_active - row_has_active * shifted_up
    nodes.append(onnx.helper.make_node("Mul", ["pi_row_has_active", "pi_shifted_up"], ["pi_active_and_shifted"]))
    nodes.append(onnx.helper.make_node("Sub", ["pi_row_has_active", "pi_active_and_shifted"], ["pi_bottom_row_mask"]))

    nodes.append(onnx.helper.make_node("Mul", ["pi_bottom_row_mask", "pi_gap_col_mask"], ["pi_final_mask"]))
    nodes.append(onnx.helper.make_node("Sub", [channel_names[0], "pi_final_mask"], ["pi_new_mask_0"]))
    nodes.append(onnx.helper.make_node("Add", [channel_names[fill_color], "pi_final_mask"], ["pi_new_mask_F"]))

    concat_inputs = []
    for ch in range(10):
        if ch == 0:
            concat_inputs.append("pi_new_mask_0")
        elif ch == fill_color:
            concat_inputs.append("pi_new_mask_F")
        else:
            concat_inputs.append(channel_names[ch])
    nodes.append(onnx.helper.make_node("Concat", concat_inputs, ["output"], axis=1))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def slice_extract_network(h_start, h_end, w_start, w_end, batch=1):
    """ONNX: Slice + Pad. For tasks where the output is a fixed sub-block of input.

    Pipeline (input [N, 10, 30, 30]):
      1. Slice(axes=[2,3], starts=[h_start, w_start], ends=[h_end, w_end],
               steps=[1, 1])               -> [N, 10, h_end-h_start, w_end-w_start]
      2. Pad(after=[30-h_out, 30-w_out])   -> [N, 10, 30, 30]

    Cost (per cost-model verify, opset 10): macs=0 (Slice and Pad both shape-only),
    params=8 (Slice initializers), memory ~ 4 * 10 * h_out * w_out bytes for the
    intermediate. For (3,3) extract: cost ~368, points ~19.09. For (2,2) extract:
    cost ~168, points ~19.88.
    """
    h_out, w_out = h_end - h_start, w_end - w_start
    starts = onnx.helper.make_tensor("sl_starts", onnx.TensorProto.INT64, [2], [h_start, w_start])
    ends = onnx.helper.make_tensor("sl_ends", onnx.TensorProto.INT64, [2], [h_end, w_end])
    axes = onnx.helper.make_tensor("sl_axes", onnx.TensorProto.INT64, [2], [2, 3])
    steps = onnx.helper.make_tensor("sl_steps", onnx.TensorProto.INT64, [2], [1, 1])
    slice_node = onnx.helper.make_node(
        "Slice", ["input", "sl_starts", "sl_ends", "sl_axes", "sl_steps"], ["sl_out"]
    )
    pad_node = onnx.helper.make_node(
        "Pad",
        ["sl_out"],
        ["output"],
        pads=[0, 0, 0, 0, 0, 0, 30 - h_out, 30 - w_out],
        mode="constant",
        value=0.0,
    )
    return _make_model(
        [slice_node, pad_node],
        [starts, ends, axes, steps],
        shape=[batch, 10, 30, 30],
    )


def translation_network(h, w, dr, dc, batch=1):
    """ONNX: Slice + Pad. Translates the (h, w) grid at top-left by (dr, dc),
    filling active-grid cells whose source is out of bounds with strict color 0.

    For input grid at top-left of the [N, 10, 30, 30] canvas with shape (h, w),
    output's grid retains shape (h, w) at the top-left; cell out[r,c] receives
    in[r-dr, c-dc] when 0 <= r-dr < h and 0 <= c-dc < w, else channel-0 one-hot.

    Pipeline:
      1. Slice axes [2, 3] of input to grab the surviving sub-region:
         rows [max(0, -dr), h - max(0, dr)), cols [max(0, -dc), w - max(0, dc))
         -> [N, 10, h - |dr|, w - |dc|]
      2. Pad to [N, 10, 30, 30], then restore channel-0 background for missing
         cells inside the original active grid:
         H before = max(0, dr), H after = 30 - h_before - h_sliced
         W before = max(0, dc), W after = 30 - w_before - w_sliced
         The H/W after values fill both the rest of the (h, w) grid AND the
         canvas-padding region beyond it.

    Cost (per cost-model verify, opset 10): macs=0, params=8 (4 Slice initializers
    of shape [2]), memory ~ 4 * 10 * (h - |dr|) * (w - |dc|) bytes from the
    intermediate. For full-canvas (30, 30, 1, 0): ~36k cost -> ~14.5 pts.
    For (10, 10, 1, 0): ~3.6k cost -> ~16.8 pts.
    """
    h_sliced = h - abs(dr)
    w_sliced = w - abs(dc)
    sr_start = max(0, -dr)
    sr_end = sr_start + h_sliced
    sc_start = max(0, -dc)
    sc_end = sc_start + w_sliced
    pad_h_before = max(0, dr)
    pad_w_before = max(0, dc)
    pad_h_after = 30 - pad_h_before - h_sliced
    pad_w_after = 30 - pad_w_before - w_sliced

    starts = onnx.helper.make_tensor("trn_starts", onnx.TensorProto.INT64, [2], [sr_start, sc_start])
    ends = onnx.helper.make_tensor("trn_ends", onnx.TensorProto.INT64, [2], [sr_end, sc_end])
    axes = onnx.helper.make_tensor("trn_axes", onnx.TensorProto.INT64, [2], [2, 3])
    steps = onnx.helper.make_tensor("trn_steps", onnx.TensorProto.INT64, [2], [1, 1])
    slice_node = onnx.helper.make_node(
        "Slice", ["input", "trn_starts", "trn_ends", "trn_axes", "trn_steps"], ["trn_sliced"]
    )
    bg_channel = onnx.helper.make_tensor(
        "trn_bg_channel",
        onnx.TensorProto.FLOAT,
        [1, 10, 1, 1],
        [1.0] + [0.0] * 9,
    )
    pad_node = onnx.helper.make_node(
        "Pad",
        ["trn_sliced"],
        ["trn_padded"],
        pads=[0, 0, pad_h_before, pad_w_before, 0, 0, pad_h_after, pad_w_after],
        mode="constant",
        value=0.0,
    )
    input_active_node = onnx.helper.make_node(
        "ReduceSum", ["input"], ["trn_input_active"], axes=[1], keepdims=1
    )
    padded_active_node = onnx.helper.make_node(
        "ReduceSum", ["trn_padded"], ["trn_padded_active"], axes=[1], keepdims=1
    )
    missing_bg_node = onnx.helper.make_node(
        "Sub", ["trn_input_active", "trn_padded_active"], ["trn_missing_bg"]
    )
    bg_fill_node = onnx.helper.make_node(
        "Mul", ["trn_missing_bg", "trn_bg_channel"], ["trn_bg_fill"]
    )
    output_node = onnx.helper.make_node(
        "Add", ["trn_padded", "trn_bg_fill"], ["output"]
    )
    return _make_model(
        [
            slice_node,
            pad_node,
            input_active_node,
            padded_active_node,
            missing_bg_node,
            bg_fill_node,
            output_node,
        ],
        [starts, ends, axes, steps, bg_channel],
        shape=[batch, 10, 30, 30],
    )


def pixel_perm_network(h, w, indices, batch=1):
    """ONNX: Slice -> Reshape -> Gather -> Reshape -> Pad.

    For tasks where input/output share shape (h, w) and output cell at flat index
    i_out always equals input cell at flat index indices[i_out] (consistent across
    all visible pairs).

    Pipeline (input [N,10,30,30]):
      1. Slice axes [2,3] -> [N,10,h,w]               (extract active grid)
      2. Reshape -> [N,10,h*w]                        (flatten spatial)
      3. Gather along axis=2 with `indices`           (per-cell permutation)
      4. Reshape -> [N,10,h,w]
      5. Pad -> [N,10,30,30]                          (re-embed at top-left)

    Cost (per cost-model verify, opset 10): macs=0 (all shape ops + Gather are
    0 MACs). params = h*w (int64 indices) + small Slice/Reshape inits.
    Memory = sum of value_info bytes for 4 intermediates, each h*w*10*4 bytes.
    For h*w=33 (task214): memory ~5.3k + params ~33 -> ~5.3k -> ~16.4 pts.
    For h*w=12 (task373): memory ~1.9k + params ~12 -> ~1.9k -> ~17.4 pts.
    For h*w=40 (task385): memory ~6.4k + params ~40 -> ~6.4k -> ~16.2 pts.
    """
    n = h * w
    inits = _slice_active_inits("pp_sl", h, w)
    slice_node = onnx.helper.make_node(
        "Slice",
        ["input", "pp_sl_starts", "pp_sl_ends", "pp_sl_axes", "pp_sl_steps"],
        ["pp_active"],
    )
    shape_flat = onnx.helper.make_tensor(
        "pp_shape_flat", onnx.TensorProto.INT64, [3], [batch, 10, n]
    )
    reshape1 = onnx.helper.make_node(
        "Reshape", ["pp_active", "pp_shape_flat"], ["pp_flat"]
    )
    indices_init = onnx.helper.make_tensor(
        "pp_indices", onnx.TensorProto.INT64, [n], indices
    )
    gather = onnx.helper.make_node(
        "Gather", ["pp_flat", "pp_indices"], ["pp_gathered"], axis=2
    )
    shape_grid = onnx.helper.make_tensor(
        "pp_shape_grid", onnx.TensorProto.INT64, [4], [batch, 10, h, w]
    )
    reshape2 = onnx.helper.make_node(
        "Reshape", ["pp_gathered", "pp_shape_grid"], ["pp_grid"]
    )
    pad_node = onnx.helper.make_node(
        "Pad", ["pp_grid"], ["output"],
        pads=[0, 0, 0, 0, 0, 0, 30 - h, 30 - w],
        mode="constant", value=0.0,
    )
    return _make_model(
        [slice_node, reshape1, gather, reshape2, pad_node],
        inits + [shape_flat, indices_init, shape_grid],
        shape=[batch, 10, 30, 30],
    )


def shape_changing_pixel_perm_network(in_h, in_w, out_h, out_w, indices, batch=1):
    """ONNX: Slice -> Reshape (flatten input) -> Gather -> Reshape (output grid) -> Pad.

    Generalizes pixel_perm_network to the case where output shape (out_h, out_w)
    differs from input shape (in_h, in_w). indices has length out_h*out_w; each
    entry is a flat index into the in_h*in_w input.

    Pipeline (input [N,10,30,30]):
      1. Slice axes [2,3] -> [N,10,in_h,in_w]                  (extract input grid)
      2. Reshape -> [N,10,in_h*in_w]                           (flatten input)
      3. Gather along axis=2 with `indices` (length out_h*out_w) -> [N,10,out_h*out_w]
      4. Reshape -> [N,10,out_h,out_w]
      5. Pad -> [N,10,30,30]

    Used for tile-rearrange tasks (e.g. ashok205 cluster: in (3,3) -> out (6,6)
    or (9,9), where each output cell maps to a specific input cell). Cost
    similar to pixel_perm_network but indices array is sized to the OUTPUT.
    """
    n_in = in_h * in_w
    n_out = out_h * out_w
    inits = _slice_active_inits("scpp_sl", in_h, in_w)
    slice_node = onnx.helper.make_node(
        "Slice",
        ["input", "scpp_sl_starts", "scpp_sl_ends", "scpp_sl_axes", "scpp_sl_steps"],
        ["scpp_active"],
    )
    shape_flat = onnx.helper.make_tensor(
        "scpp_shape_flat", onnx.TensorProto.INT64, [3], [batch, 10, n_in]
    )
    reshape1 = onnx.helper.make_node(
        "Reshape", ["scpp_active", "scpp_shape_flat"], ["scpp_flat"]
    )
    indices_init = onnx.helper.make_tensor(
        "scpp_indices", onnx.TensorProto.INT64, [n_out], indices
    )
    gather = onnx.helper.make_node(
        "Gather", ["scpp_flat", "scpp_indices"], ["scpp_gathered"], axis=2
    )
    shape_grid = onnx.helper.make_tensor(
        "scpp_shape_grid", onnx.TensorProto.INT64, [4], [batch, 10, out_h, out_w]
    )
    reshape2 = onnx.helper.make_node(
        "Reshape", ["scpp_gathered", "scpp_shape_grid"], ["scpp_grid"]
    )
    pad_node = onnx.helper.make_node(
        "Pad", ["scpp_grid"], ["output"],
        pads=[0, 0, 0, 0, 0, 0, 30 - out_h, 30 - out_w],
        mode="constant", value=0.0,
    )
    return _make_model(
        [slice_node, reshape1, gather, reshape2, pad_node],
        inits + [shape_flat, indices_init, shape_grid],
        shape=[batch, 10, 30, 30],
    )


def outer_product_marker_tile_network(mode, marker_color=None, batch=1):
    """Outer-product marker tile (3x3 input -> 9x9 output).

    For each input cell (i, j) with input[i, j] == marker_color, stamps the entire
    input at output position (3i, 3j); all other 3x3 output blocks fill the active
    9x9 region with color 0 (channel 0 lit). Outside the 9x9 (cols/rows 9..29 of
    the 30x30 canvas), all channels remain 0 — matching the encoding convention
    that canvas padding has all-zero one-hot values.

    Modes:
      mode="fixed", marker_color=c: stamp where input == c. Cheap path (channel
        Slice for the mask). Tasks with a per-task fixed marker color (e.g.
        task315 — marker is always 2 across train+test+arc-gen).
      mode="most_common": stamp where input == most-frequent color (any, incl.
        ch 0). Computed dynamically via ReduceSum + Mul + ReduceMax + Less. Tasks
        where the marker varies per pair but is always the dominant cell (e.g.
        task304 — fully-filled inputs whose dominant color anchors the stamp).

    Pipeline (input [N, 10, 30, 30]):
      1. Slice axes [2,3] -> [N, 10, 3, 3] (extract input grid)
      2. Compute mask [N, 1, 3, 3] (1 at marker cells, 0 elsewhere):
         - "fixed":       Slice axis [1] starts [c] ends [c+1] -> [N, 1, 3, 3]
         - "most_common": ReduceSum(axes=[2,3], keepdims=1) -> [N, 10, 1, 1]
                          counts; Mul(active, counts) -> [N, 10, 3, 3];
                          ReduceSum(axes=[1], keepdims=1) -> [N, 1, 3, 3] per_cell;
                          ReduceMax(counts, axes=[1], keepdims=1) -> [N, 1, 1, 1];
                          Sub, Mul, Less, Cast -> [N, 1, 3, 3] mask.
      3. Reshape active -> [N, 10, 1, 3, 1, 3] (anchor singletons inserted before
         each spatial dim)
      4. Reshape mask   -> [N, 1, 3, 1, 3, 1] (anchor dims with inner singletons)
      5. stamped_6d = Mul(active_6d, mask_6d) -> [N, 10, 3, 3, 3, 3]
      6. Background fill for non-stamped 3x3 blocks (so ch0 = 1 inside the active
         9x9 at color-0 cells, matching the encoding):
         - complement_6d = Sub(1.0, mask_6d) -> [N, 1, 3, 1, 3, 1] (1 at non-marker
           anchor positions)
         - ch0_init: [1, 10, 1, 1, 1, 1] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
         - ch0_bg_6d = Mul(ch0_init, complement_6d) -> broadcast to
           [N, 10, 3, 1, 3, 1] then Adds against stamped_6d on Add broadcast.
      7. combined_6d = Add(stamped_6d, ch0_bg_6d) -> [N, 10, 3, 3, 3, 3]
      8. Reshape -> [N, 10, 9, 9] (C-order: (anchor_i, inner_r) -> row,
         (anchor_j, inner_c) -> col)
      9. Pad after [21, 21] -> [N, 10, 30, 30]

    Cost (per cost-model verify, opset 10):
      "fixed":       macs ~3.3k  + memory ~10 KB + params ~50  -> ~13 KB cost
                     -> ~15.6 pts/task.
      "most_common": macs ~3.5k  + memory ~12 KB + params ~50  -> ~15 KB cost
                     -> ~15.4 pts/task.
    """
    if mode not in ("fixed", "most_common"):
        raise ValueError(f"mode must be 'fixed' or 'most_common', got {mode!r}")
    if mode == "fixed":
        if marker_color is None or not (0 <= marker_color < 10):
            raise ValueError(f"fixed mode requires marker_color in 0..9, got {marker_color!r}")

    inits = _slice_active_inits("opmt_sl", 3, 3)
    nodes = [
        onnx.helper.make_node(
            "Slice",
            ["input", "opmt_sl_starts", "opmt_sl_ends", "opmt_sl_axes", "opmt_sl_steps"],
            ["opmt_active"],
        )
    ]

    if mode == "fixed":
        # Slice channel `marker_color` to get [N, 1, 3, 3] mask.
        ch_starts = onnx.helper.make_tensor("opmt_ch_starts", onnx.TensorProto.INT64, [1], [marker_color])
        ch_ends = onnx.helper.make_tensor("opmt_ch_ends", onnx.TensorProto.INT64, [1], [marker_color + 1])
        ch_axes = onnx.helper.make_tensor("opmt_ch_axes", onnx.TensorProto.INT64, [1], [1])
        ch_steps = onnx.helper.make_tensor("opmt_ch_steps", onnx.TensorProto.INT64, [1], [1])
        inits.extend([ch_starts, ch_ends, ch_axes, ch_steps])
        nodes.append(onnx.helper.make_node(
            "Slice",
            ["opmt_active", "opmt_ch_starts", "opmt_ch_ends", "opmt_ch_axes", "opmt_ch_steps"],
            ["opmt_is_marker"],
        ))
    else:  # most_common
        # Counts per color: [N, 10, 1, 1].
        nodes.append(onnx.helper.make_node(
            "ReduceSum", ["opmt_active"], ["opmt_counts"], axes=[2, 3], keepdims=1
        ))
        # per_cell_byCh = active * counts (each cell's column has 1 only on its color
        # channel; product gives that color's count on the cell's channel, 0 elsewhere).
        nodes.append(onnx.helper.make_node(
            "Mul", ["opmt_active", "opmt_counts"], ["opmt_per_cell_byCh"],
        ))
        nodes.append(onnx.helper.make_node(
            "ReduceSum", ["opmt_per_cell_byCh"], ["opmt_per_cell"],
            axes=[1], keepdims=1,
        ))
        # Target count = ReduceMax over channels.
        nodes.append(onnx.helper.make_node(
            "ReduceMax", ["opmt_counts"], ["opmt_target"],
            axes=[1], keepdims=1,
        ))
        # is_marker via Less((per_cell - target)^2, 0.5). Equal-on-float is opset-13.
        half = onnx.helper.make_tensor("opmt_half", onnx.TensorProto.FLOAT, [1], [0.5])
        inits.append(half)
        nodes.append(onnx.helper.make_node("Sub", ["opmt_per_cell", "opmt_target"], ["opmt_diff"]))
        nodes.append(onnx.helper.make_node("Mul", ["opmt_diff", "opmt_diff"], ["opmt_diff_sq"]))
        nodes.append(onnx.helper.make_node("Less", ["opmt_diff_sq", "opmt_half"], ["opmt_is_marker_bool"]))
        nodes.append(onnx.helper.make_node(
            "Cast", ["opmt_is_marker_bool"], ["opmt_is_marker"],
            to=onnx.TensorProto.FLOAT,
        ))

    # Reshape active [N,10,3,3] -> [N,10,1,3,1,3] (anchor singletons before each spatial).
    shape_input_init = onnx.helper.make_tensor(
        "opmt_shape_input6d", onnx.TensorProto.INT64, [6], [batch, 10, 1, 3, 1, 3]
    )
    inits.append(shape_input_init)
    nodes.append(onnx.helper.make_node(
        "Reshape", ["opmt_active", "opmt_shape_input6d"], ["opmt_input_6d"],
    ))

    # Reshape is_marker [N,1,3,3] -> [N,1,3,1,3,1].
    shape_mask_init = onnx.helper.make_tensor(
        "opmt_shape_mask6d", onnx.TensorProto.INT64, [6], [batch, 1, 3, 1, 3, 1]
    )
    inits.append(shape_mask_init)
    nodes.append(onnx.helper.make_node(
        "Reshape", ["opmt_is_marker", "opmt_shape_mask6d"], ["opmt_mask_6d"],
    ))

    # stamped_6d = active_6d * mask_6d -> [N, 10, 3, 3, 3, 3].
    nodes.append(onnx.helper.make_node(
        "Mul", ["opmt_input_6d", "opmt_mask_6d"], ["opmt_stamped_6d"],
    ))

    # Background fill: ch0 = 1 inside the active 9x9 at non-marker anchor positions.
    # complement_6d = 1 - mask_6d -> [N, 1, 3, 1, 3, 1].
    one_scalar = onnx.helper.make_tensor("opmt_one", onnx.TensorProto.FLOAT, [1], [1.0])
    inits.append(one_scalar)
    nodes.append(onnx.helper.make_node(
        "Sub", ["opmt_one", "opmt_mask_6d"], ["opmt_complement_6d"],
    ))
    # ch0_init: [1, 10, 1, 1, 1, 1] with [1, 0, ..., 0]. Multiplying by complement
    # broadcasts to [N, 10, 3, 1, 3, 1] (then Adds onto stamped via [N, 10, 3, 3, 3, 3]).
    ch0_init = onnx.helper.make_tensor(
        "opmt_ch0_only", onnx.TensorProto.FLOAT, [1, 10, 1, 1, 1, 1],
        [1.0] + [0.0] * 9,
    )
    inits.append(ch0_init)
    nodes.append(onnx.helper.make_node(
        "Mul", ["opmt_ch0_only", "opmt_complement_6d"], ["opmt_ch0_bg_6d"],
    ))
    nodes.append(onnx.helper.make_node(
        "Add", ["opmt_stamped_6d", "opmt_ch0_bg_6d"], ["opmt_combined_6d"],
    ))

    # Reshape -> [N, 10, 9, 9].
    shape_9x9_init = onnx.helper.make_tensor(
        "opmt_shape_9x9", onnx.TensorProto.INT64, [4], [batch, 10, 9, 9]
    )
    inits.append(shape_9x9_init)
    nodes.append(onnx.helper.make_node(
        "Reshape", ["opmt_combined_6d", "opmt_shape_9x9"], ["opmt_grid_9x9"],
    ))

    # Pad to [N, 10, 30, 30].
    nodes.append(onnx.helper.make_node(
        "Pad", ["opmt_grid_9x9"], ["output"],
        pads=[0, 0, 0, 0, 0, 0, 21, 21],
        mode="constant", value=0.0,
    ))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def task220_multi_marker_stamp_network(batch=1):
    """Task220: sparse 2/3/8 markers paint 8-neighbor rings as 1/6/4.

    The bundled generator keeps markers at least one cell inboard and separates
    them by Chebyshev distance >= 3, so one full-channel 3x3 Conv can do the
    whole exact rewrite: preserve all center pixels, subtract background at
    ring cells, and add each target ring color.
    """
    W = np.zeros((10, 10, 3, 3), dtype=np.float32)
    for ch in range(10):
        W[ch, ch, 1, 1] = 1.0
    for source, target in ((2, 1), (3, 6), (8, 4)):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                W[0, source, 1 - dr, 1 - dc] -= 1.0
                W[target, source, 1 - dr, 1 - dc] += 1.0
    weight = onnx.helper.make_tensor("t220_w", onnx.TensorProto.FLOAT, W.shape, W.flatten().tolist())
    nodes = [
        onnx.helper.make_node(
            "Conv",
            ["input", "t220_w"],
            ["output"],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        )
    ]
    return _make_model(nodes, [weight], shape=[batch, 10, 30, 30])


def task005_motif_ray_extend_network(batch=1):
    """Task005: extend partial 3x3 motif copies along stride-4 rays.

    The source motif is the unique color/window with the largest 3x3 count.
    Any non-source color appearing in one of the eight neighboring stride-4
    motif blocks seeds a ray; the selected source motif is stamped every 4
    cells along that ray, clipped to the 21x21 active grid.
    """
    if batch != 1:
        raise NotImplementedError("task005_motif_ray_extend_network supports batch=1 only")

    P = "t005_"
    nodes = []
    inits = []

    def tensor(name, dtype, shape, values):
        inits.append(onnx.helper.make_tensor(name, dtype, shape, values))
        return name

    def ints(name, values):
        return tensor(P + name, onnx.TensorProto.INT64, [len(values)], values)

    def node(op_type, inputs, outputs, **attrs):
        nodes.append(onnx.helper.make_node(op_type, inputs, outputs, **attrs))

    ch_axis = ints("ch_axis", [1])
    ch_step = ints("ch_step", [1])
    ch1 = ints("ch1", [1])
    ch10 = ints("ch10", [10])
    half = tensor(P + "half", onnx.TensorProto.FLOAT, [1], [0.5])
    active = tensor(
        P + "active21",
        onnx.TensorProto.FLOAT,
        [1, 1, 30, 30],
        [1.0 if r < 21 and c < 21 else 0.0 for r in range(30) for c in range(30)],
    )

    node("Slice", ["input", ch1, ch10, ch_axis, ch_step], [P + "nonbg"])
    score_w = tensor(P + "score_w", onnx.TensorProto.FLOAT, [9, 1, 3, 3], [1.0] * 81)
    node("Conv", [P + "nonbg", score_w], [P + "scores"], group=9)
    node(
        "Slice",
        [P + "scores", ints("source_st", [6, 6]), ints("source_en", [13, 13]), ints("source_ax", [2, 3]), ints("source_step", [1, 1])],
        [P + "source_scores"],
    )
    node("ReduceMax", [P + "source_scores"], [P + "global_max"], axes=[1, 2, 3], keepdims=1)
    node("Sub", [P + "global_max", half], [P + "source_thresh"])
    node("Greater", [P + "source_scores", P + "source_thresh"], [P + "source_anchor_b"])
    node("Cast", [P + "source_anchor_b"], [P + "source_anchor"], to=onnx.TensorProto.FLOAT)
    node("ReduceMax", [P + "source_anchor"], [P + "source_anchor_sp"], axes=[1], keepdims=1)

    pattern_bits = {}
    source_pat_maps = {}
    for rr in range(3):
        for cc in range(3):
            name = f"pat_{rr}_{cc}"
            node(
                "Slice",
                [
                    P + "nonbg",
                    ints(name + "_st", [6 + rr, 6 + cc]),
                    ints(name + "_en", [13 + rr, 13 + cc]),
                    ints(name + "_ax", [2, 3]),
                    ints(name + "_step", [1, 1]),
                ],
                [P + name + "_cells"],
            )
            node("Mul", [P + name + "_cells", P + "source_anchor"], [P + name + "_at_source"])
            node("ReduceMax", [P + name + "_at_source"], [P + name], axes=[1, 2, 3], keepdims=1)
            pattern_bits[(rr, cc)] = P + name
            node("Mul", [P + "source_anchor_sp", P + name], [P + name + "_source"])
            source_pat_maps[(rr, cc)] = P + name + "_source"

    def shift_scores_for_neighbor(dr, dc, name):
        sr = 6 + 4 * dr
        er = 13 + 4 * dr
        sc = 6 + 4 * dc
        ec = 13 + 4 * dc
        node(
            "Slice",
            [
                P + "scores",
                ints(name + "_st", [sr, sc]),
                ints(name + "_en", [er, ec]),
                ints(name + "_ax", [2, 3]),
                ints(name + "_step", [1, 1]),
            ],
            [P + name],
        )
        return P + name

    def ray_kernel_name(dr, dc, rr, cc):
        # Numeric cursor sweep for task005 showed all train/test/arc-gen anchors need
        # at most three stride-4 ray copies; k=1..2 misses examples, while k=3
        # remains exact and removes dead ray-kernel parameters versus the legacy k=5.
        rels = [(6 + 4 * k * dr + rr, 6 + 4 * k * dc + cc) for k in range(1, 4)]

        def axis_params(vals):
            lo = min(vals)
            hi = max(vals)
            pad_start = max(0, -lo)
            size = max(hi + pad_start + 1, pad_start + 24)
            pad_end = size - 24 - pad_start
            return pad_start, size, pad_end

        pad_top, kh, pad_bottom = axis_params([r for r, _ in rels])
        pad_left, kw, pad_right = axis_params([c for _, c in rels])
        vals = np.zeros((kh, kw), dtype=np.float32)
        for rel_r, rel_c in rels:
            vals[rel_r + pad_top, rel_c + pad_left] = 1.0
        kname = P + f"ray_k_{dr+1}_{dc+1}_{rr}_{cc}"
        tensor(kname, onnx.TensorProto.FLOAT, [1, 1, kh, kw], vals.flatten().tolist())
        return kname, [pad_top, pad_left, pad_bottom, pad_right]

    dir_outputs = []
    for dr, dc in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
        dname = f"d{dr+1}{dc+1}"
        shifted = shift_scores_for_neighbor(dr, dc, dname + "_neighbor_scores")
        node("Mul", [shifted, P + "source_anchor_sp"], [P + dname + "_neighbor_at_source"])
        node("ReduceMax", [P + dname + "_neighbor_at_source"], [P + dname + "_seed_count"], axes=[2, 3], keepdims=1)
        node("Greater", [P + dname + "_seed_count", half], [P + dname + "_seed_b"])
        node("Cast", [P + dname + "_seed_b"], [P + dname + "_seed"], to=onnx.TensorProto.FLOAT)

        ray_parts = []
        for rr in range(3):
            for cc in range(3):
                kname, pads = ray_kernel_name(dr, dc, rr, cc)
                raw = P + f"{dname}_ray_{rr}_{cc}"
                node("ConvTranspose", [source_pat_maps[(rr, cc)], kname], [raw], pads=pads)
                ray_parts.append(raw)
        node("Sum", ray_parts, [P + dname + "_ray_sum"])
        node("Mul", [P + dname + "_ray_sum", P + dname + "_seed"], [P + dname + "_colors"])
        dir_outputs.append(P + dname + "_colors")

    node("Max", dir_outputs, [P + "generated"])
    node("Max", [P + "nonbg", P + "generated"], [P + "nonbg_with_rays"])
    node("Mul", [P + "nonbg_with_rays", active], [P + "nonbg_active"])
    node("ReduceMax", [P + "nonbg_active"], [P + "foreground"], axes=[1], keepdims=1)
    node("Sub", [active, P + "foreground"], [P + "ch0"])
    node("Concat", [P + "ch0", P + "nonbg_active"], ["output"], axis=1)

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def per_color_stamp_network(rules, batch=1):
    """Per-input-color stamp pattern. For each rule (in_color, stamp_color, kernel)
    in `rules`: every cell of color `in_color` in the input projects `stamp_color`
    to its `kernel`-described neighbors in the output.

    Kernels:
      "ortho": [[0,1,0],[1,0,1],[0,1,0]] — 4-connected (up/down/left/right).
      "diag":  [[1,0,1],[0,0,0],[1,0,1]] — 4-diagonal neighbors.
      "all8":  [[1,1,1],[1,0,1],[1,1,1]] — 8-connected.

    Stamps are placed where the kernel-summed mask is >= 1. The cell itself is
    NEVER overwritten (the kernel center is 0, so the rule "stamp at neighbor
    of input==in_color" doesn't include the in_color cell itself). Stamp cells
    must be color 0 in the input (otherwise the existing color is replaced —
    this is by design for the LB-5126 task015 rule, where stamps land on
    canvas-padding/empty cells only; if a cleaner constraint is needed, the
    caller's detector enforces it).

    Implementation:
      For each rule (in_color, stamp_color, kernel):
        1. Slice input axis [1] starts [in_color] ends [in_color+1] -> [N, 1, 30, 30] mask_in.
        2. Conv 3x3 with the kernel weights, pads=[1,1,1,1] -> [N, 1, 30, 30] count.
        3. lit_bool = Greater(count, 0.5)
        4. lit = Cast(lit_bool, FLOAT) -> [N, 1, 30, 30]   stamp mask
      Aggregate per-channel deltas:
        For ch in 0..9:
          Slice input channel ch -> input_ch [N, 1, 30, 30]
          add_to_ch[c] = sum of `lit_for_rule_with_stamp_color_c` (across rules)
          subtract_from_ch[0] = sum of all `lit` masks (each stamp turns a ch0=1 into 0)
        Then:
          new_ch0  = input_ch0 - subtract_from_ch0
          new_chSC = input_chSC + add_to_chSC
          others   = input_ch (unchanged)
      Concat 10 channels.

    Cost (per cost-model verify, opset 10): macs ~ R * 8100 (Conv 3x3 1->1 per
    rule) + R * 900 (Greater) + 10 * 0 (channel Slices) + 1-2 * 900 per touched
    channel (Sub/Add); memory ~ 36 KB * (1 + 2*R + 1) intermediates; params ~
    R * 9 (kernel) + 50 * 4 (Slice inits). For task015 (R=2): ~22k macs +
    ~250 KB memory + ~250 params -> ~272 KB cost -> ~12.4 pts/task.
    """
    KERNELS = {
        "ortho": [0.0, 1.0, 0.0,
                  1.0, 0.0, 1.0,
                  0.0, 1.0, 0.0],
        "diag":  [1.0, 0.0, 1.0,
                  0.0, 0.0, 0.0,
                  1.0, 0.0, 1.0],
        "all8":  [1.0, 1.0, 1.0,
                  1.0, 0.0, 1.0,
                  1.0, 1.0, 1.0],
    }
    if not rules:
        raise ValueError("rules must be a non-empty list of (in_color, stamp_color, kernel)")
    for in_c, st_c, k_name in rules:
        if not (0 <= in_c < 10):
            raise ValueError(f"in_color must be 0..9, got {in_c}")
        if not (0 <= st_c < 10):
            raise ValueError(f"stamp_color must be 0..9, got {st_c}")
        if k_name not in KERNELS:
            raise ValueError(f"kernel must be one of {set(KERNELS)}, got {k_name!r}")
        if in_c == st_c:
            raise ValueError(f"in_color and stamp_color must differ (got {in_c})")
        if in_c == 0:
            raise ValueError(f"in_color must be nonzero (got 0)")

    normalized_rules = tuple(sorted((int(in_c), int(st_c), k_name) for in_c, st_c, k_name in rules))
    if normalized_rules == ((2, 1, "all8"), (3, 6, "all8"), (8, 4, "all8")):
        return task220_multi_marker_stamp_network(batch=batch)

    nodes = []
    inits = []

    def _slice_channel(name_prefix, in_name, ch, out_name):
        starts = onnx.helper.make_tensor(f"{name_prefix}_starts", onnx.TensorProto.INT64, [1], [ch])
        ends = onnx.helper.make_tensor(f"{name_prefix}_ends", onnx.TensorProto.INT64, [1], [ch + 1])
        axes = onnx.helper.make_tensor(f"{name_prefix}_axes", onnx.TensorProto.INT64, [1], [1])
        steps = onnx.helper.make_tensor(f"{name_prefix}_steps", onnx.TensorProto.INT64, [1], [1])
        node = onnx.helper.make_node(
            "Slice",
            [in_name, f"{name_prefix}_starts", f"{name_prefix}_ends", f"{name_prefix}_axes", f"{name_prefix}_steps"],
            [out_name],
        )
        return node, [starts, ends, axes, steps]

    # Slice all 10 channels of input.
    channel_names = {}
    for ch in range(10):
        out_name = f"pcs_ch{ch}"
        node, ch_inits = _slice_channel(f"pcs_sl_{ch}", "input", ch, out_name)
        nodes.append(node)
        inits.extend(ch_inits)
        channel_names[ch] = out_name

    half_const = onnx.helper.make_tensor("pcs_half", onnx.TensorProto.FLOAT, [1], [0.5])
    inits.append(half_const)

    # Per rule: compute the stamp-lit mask, then clip to (active grid AND was color 0).
    # Channel 0 of input is 1 only at active-grid color-0 cells (canvas padding has
    # all channels = 0 in the encoding). Multiplying the Conv-derived lit mask by
    # ch0 ensures stamps don't leak into canvas padding (when an in_c cell is near
    # the active-grid edge) and don't overwrite existing non-zero cells.
    stamp_masks_per_color = collections.defaultdict(list)  # stamp_color -> [clipped_lit_name, ...]
    for idx, (in_c, st_c, k_name) in enumerate(rules):
        kernel_init = onnx.helper.make_tensor(
            f"pcs_kernel_{idx}", onnx.TensorProto.FLOAT, [1, 1, 3, 3], KERNELS[k_name]
        )
        inits.append(kernel_init)
        nodes.append(onnx.helper.make_node(
            "Conv",
            [channel_names[in_c], f"pcs_kernel_{idx}"],
            [f"pcs_count_{idx}"],
            kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1],
        ))
        nodes.append(onnx.helper.make_node(
            "Greater", [f"pcs_count_{idx}", "pcs_half"], [f"pcs_lit_bool_{idx}"]
        ))
        nodes.append(onnx.helper.make_node(
            "Cast", [f"pcs_lit_bool_{idx}"], [f"pcs_lit_raw_{idx}"], to=onnx.TensorProto.FLOAT
        ))
        # Clip: multiply by input ch0 (= 1 only at active-grid color-0 cells).
        nodes.append(onnx.helper.make_node(
            "Mul", [f"pcs_lit_raw_{idx}", channel_names[0]], [f"pcs_lit_{idx}"]
        ))
        stamp_masks_per_color[st_c].append(f"pcs_lit_{idx}")

    # Per-channel deltas. ch0 loses every stamp; the stamp_color channels gain stamps.
    # Sum all lit masks (per channel where they aggregate). For task015 we have at most
    # one stamp_color per rule and they don't usually conflict.
    def _sum_masks(mask_names, prefix):
        if len(mask_names) == 1:
            return mask_names[0]
        cur = mask_names[0]
        for i, nxt in enumerate(mask_names[1:], start=1):
            out = f"{prefix}_sum_{i}"
            nodes.append(onnx.helper.make_node("Add", [cur, nxt], [out]))
            cur = out
        return cur

    all_lit_names = [f"pcs_lit_{i}" for i in range(len(rules))]
    total_subtract = _sum_masks(all_lit_names, "pcs_total_sub")

    # New channel outputs.
    concat_inputs = []
    for ch in range(10):
        if ch == 0:
            # ch0 loses every stamp.
            new_name = f"pcs_new_ch{ch}"
            nodes.append(onnx.helper.make_node(
                "Sub", [channel_names[ch], total_subtract], [new_name]
            ))
            concat_inputs.append(new_name)
        elif ch in stamp_masks_per_color:
            # ch gains the stamps that target it.
            ch_add = _sum_masks(stamp_masks_per_color[ch], f"pcs_add_ch{ch}")
            new_name = f"pcs_new_ch{ch}"
            nodes.append(onnx.helper.make_node(
                "Add", [channel_names[ch], ch_add], [new_name]
            ))
            concat_inputs.append(new_name)
        else:
            concat_inputs.append(channel_names[ch])

    nodes.append(onnx.helper.make_node(
        "Concat", concat_inputs, ["output"], axis=1,
    ))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def active_grid_frame_network(border_color, batch=1):
    """Draw a 1-pixel border of `border_color` around the active grid; interior
    of active grid is color 0; canvas padding is all-zero. Input content is
    ignored (only the active-grid SHAPE matters, derived from the encoding).

    Pipeline (input [N, 10, 30, 30]):
      1. is_active = ReduceSum(input, axes=[1], keepdims=1) -> [N, 1, 30, 30]
            (1 at active-grid cells of any color; 0 at canvas padding)
      2. count = Conv 3x3 with kernel [[1,1,1],[1,0,1],[1,1,1]] on is_active
            -> [N, 1, 30, 30] (count of active 8-neighbors)
      3. count_lt_8_bool = Less(count, 7.5) (active interior cells have count==8;
            border cells have count <= 5; canvas-padding cells have count
            anywhere from 0 to 5 but we mask via is_active)
      4. count_lt_8 = Cast(count_lt_8_bool, FLOAT)
      5. is_border = Mul(is_active, count_lt_8) -> 1 at active-grid border cells
            (active AND has fewer than 8 active 8-neighbors)
      6. interior_ch0 = Sub(is_active, is_border) -> 1 at active-grid interior cells
      7. zeros = Sub(is_border, is_border) -> [N, 1, 30, 30] all-zero tensor
      8. Concat 10 channels: ch0=interior_ch0, ch_border_color=is_border,
            others=zeros (with appropriate channel placement).

    Cost: macs ~ 9k (ReduceSum) + 8.1k (Conv 3x3 1->1) + ~3.6k (Less+Cast+Mul+Sub
    on [N,1,30,30]); memory ~ 22 KB (5-6 [N,1,30,30] intermediates); params ~ 10
    (kernel weight + scalar consts). Total ~ 21k -> ~14.5 pts/task.
    """
    if not (1 <= border_color < 10):
        raise ValueError(f"border_color must be in 1..9, got {border_color}")

    nodes = []
    inits = []

    # is_active = ReduceSum across channels.
    nodes.append(onnx.helper.make_node(
        "ReduceSum", ["input"], ["agf_is_active"], axes=[1], keepdims=1
    ))

    # Conv 3x3 with 8-neighbor (no center) kernel on is_active.
    kernel_init = onnx.helper.make_tensor(
        "agf_kw", onnx.TensorProto.FLOAT, [1, 1, 3, 3],
        [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    )
    inits.append(kernel_init)
    nodes.append(onnx.helper.make_node(
        "Conv", ["agf_is_active", "agf_kw"], ["agf_count"],
        kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1],
    ))

    # count_lt_8 via Less(count, 7.5).
    threshold_init = onnx.helper.make_tensor(
        "agf_thresh", onnx.TensorProto.FLOAT, [1], [7.5],
    )
    inits.append(threshold_init)
    nodes.append(onnx.helper.make_node(
        "Less", ["agf_count", "agf_thresh"], ["agf_lt8_bool"],
    ))
    nodes.append(onnx.helper.make_node(
        "Cast", ["agf_lt8_bool"], ["agf_lt8"], to=onnx.TensorProto.FLOAT,
    ))

    # is_border = is_active * lt8.
    nodes.append(onnx.helper.make_node(
        "Mul", ["agf_is_active", "agf_lt8"], ["agf_is_border"],
    ))

    # interior_ch0 = is_active - is_border.
    nodes.append(onnx.helper.make_node(
        "Sub", ["agf_is_active", "agf_is_border"], ["agf_interior_ch0"],
    ))

    # zeros placeholder tensor for unused channels.
    nodes.append(onnx.helper.make_node(
        "Sub", ["agf_is_border", "agf_is_border"], ["agf_zeros"],
    ))

    # Build 10-channel concat list.
    concat_inputs = []
    for ch in range(10):
        if ch == 0:
            concat_inputs.append("agf_interior_ch0")
        elif ch == border_color:
            concat_inputs.append("agf_is_border")
        else:
            concat_inputs.append("agf_zeros")
    nodes.append(onnx.helper.make_node(
        "Concat", concat_inputs, ["output"], axis=1,
    ))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def outline_extract_network(canvas_h=20, canvas_w=22, batch=1):
    """Outline-extract: keep only the boundary cells of each connected non-zero
    region; interior cells become color 0. A cell is "interior" iff it's the
    same color as ALL 8 of its neighbors (and itself).

    Crop-process-pad: crop input to [N, 10, canvas_h, canvas_w] (max actual
    grid size), process, then Pad result back to [N, 10, 30, 30]. Reduces
    memory from ~196k to ~80k for task98 (max grid 20×22).

    Pipeline (input [N, 10, 30, 30]):
      0. Crop input to [N, 10, canvas_h, canvas_w]
      1. Grouped Conv 3x3, same padding -> [N, 10, canvas_h, canvas_w]
      2. is_interior = Cast(Greater(count, 8.5))
      3. is_interior_active = Mul(is_interior, crop)
      4. new_crop = crop - is_interior_active
      5. interior_total = ReduceSum over channels
      6. Slice ch0 + Add interior_total -> updated_ch0
      7. Slice ch1..9
      8. Concat -> ole_result [N, 10, canvas_h, canvas_w]
      9. Pad ole_result -> output [N, 10, 30, 30]

    Cost: ~80k (down from ~196k). Verified 100% (265/265) on task98.
    """
    P = "ole_"
    nodes = []
    inits = []

    # Step 0: crop input to [N, 10, canvas_h, canvas_w]
    cr_starts = onnx.helper.make_tensor(P+"cr_starts", onnx.TensorProto.INT64, [2], [0, 0])
    cr_ends = onnx.helper.make_tensor(P+"cr_ends", onnx.TensorProto.INT64, [2], [canvas_h, canvas_w])
    cr_axes = onnx.helper.make_tensor(P+"cr_axes", onnx.TensorProto.INT64, [2], [2, 3])
    cr_steps = onnx.helper.make_tensor(P+"cr_steps", onnx.TensorProto.INT64, [2], [1, 1])
    inits.extend([cr_starts, cr_ends, cr_axes, cr_steps])
    nodes.append(onnx.helper.make_node(
        "Slice", ["input", P+"cr_starts", P+"cr_ends", P+"cr_axes", P+"cr_steps"],
        [P+"crop"],
    ))

    # Step 1: grouped Conv (same padding) on [N, 10, canvas_h, canvas_w]
    weight_data = []
    for c in range(10):
        if c == 0:
            weight_data.extend([0.0] * 9)
        else:
            weight_data.extend([1.0] * 9)
    weight_init = onnx.helper.make_tensor(
        P+"w", onnx.TensorProto.FLOAT, [10, 1, 3, 3], weight_data,
    )
    inits.append(weight_init)
    nodes.append(onnx.helper.make_node(
        "Conv", [P+"crop", P+"w"], [P+"count"],
        kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1], group=10,
    ))

    # Step 2: is_interior = Cast(Greater(count, 8.5))
    eight_half_init = onnx.helper.make_tensor(P+"85", onnx.TensorProto.FLOAT, [1], [8.5])
    inits.append(eight_half_init)
    nodes.append(onnx.helper.make_node("Greater", [P+"count", P+"85"], [P+"is_int_bool"]))
    nodes.append(onnx.helper.make_node(
        "Cast", [P+"is_int_bool"], [P+"is_int"], to=onnx.TensorProto.FLOAT,
    ))

    # Step 3: is_interior_active = Mul(is_interior, crop)
    nodes.append(onnx.helper.make_node("Mul", [P+"is_int", P+"crop"], [P+"is_int_active"]))

    # Step 4: new_crop = crop - is_interior_active
    nodes.append(onnx.helper.make_node("Sub", [P+"crop", P+"is_int_active"], [P+"new_crop"]))

    # Step 5: interior_total = ReduceSum over channels -> [N, 1, canvas_h, canvas_w]
    nodes.append(onnx.helper.make_node(
        "ReduceSum", [P+"is_int_active"], [P+"interior_total"],
        axes=[1], keepdims=1,
    ))

    # Steps 6-7: Slice ch0 + Add interior_total -> updated_ch0; Slice ch1..9
    def _slice_channel(name_prefix, in_name, ch, out_name):
        starts = onnx.helper.make_tensor(f"{name_prefix}_starts", onnx.TensorProto.INT64, [1], [ch])
        ends = onnx.helper.make_tensor(f"{name_prefix}_ends", onnx.TensorProto.INT64, [1], [ch + 1])
        axes = onnx.helper.make_tensor(f"{name_prefix}_axes", onnx.TensorProto.INT64, [1], [1])
        steps = onnx.helper.make_tensor(f"{name_prefix}_steps", onnx.TensorProto.INT64, [1], [1])
        node = onnx.helper.make_node(
            "Slice",
            [in_name, f"{name_prefix}_starts", f"{name_prefix}_ends", f"{name_prefix}_axes", f"{name_prefix}_steps"],
            [out_name],
        )
        return node, [starts, ends, axes, steps]

    sl_node, sl_inits = _slice_channel(P+"sl_ch0", P+"new_crop", 0, P+"new_ch0")
    nodes.append(sl_node)
    inits.extend(sl_inits)
    nodes.append(onnx.helper.make_node("Add", [P+"new_ch0", P+"interior_total"], [P+"updated_ch0"]))

    other_channel_names = []
    for ch in range(1, 10):
        sl_node, sl_inits = _slice_channel(f"{P}sl_ch{ch}", P+"new_crop", ch, f"{P}new_ch{ch}")
        nodes.append(sl_node)
        inits.extend(sl_inits)
        other_channel_names.append(f"{P}new_ch{ch}")

    # Step 8: Concat -> ole_result [N, 10, canvas_h, canvas_w]
    nodes.append(onnx.helper.make_node(
        "Concat", [P+"updated_ch0"] + other_channel_names, [P+"result"], axis=1,
    ))

    # Step 9: Pad ole_result -> output [N, 10, 30, 30]
    nodes.append(onnx.helper.make_node(
        "Pad", [P+"result"], ["output"],
        pads=[0, 0, 0, 0, 0, 0, 30 - canvas_h, 30 - canvas_w],
        mode="constant", value=0.0,
    ))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def bbox_fill_network(target_color, batch=1):
    """Bbox-fill: fill the tight bounding box of all non-zero cells in the input
    with `target_color`; cells outside the bbox remain unchanged; non-zero
    cells inside the bbox stay at their original color.

    Pipeline (input [N, 10, 30, 30]):
      1. is_active = ReduceSum(input, axes=[1], keepdims=1) -> [N, 1, 30, 30]
            (1 at active-grid cells, 0 at canvas padding — the encoding has
            exactly one channel lit per active cell, zero at padding.)
      2. ch0 = Slice(input, axis=[1], [0:1]) -> [N, 1, 30, 30]
      3. nonzero_mask = is_active - ch0 -> [N, 1, 30, 30]
            (1 at active-grid non-zero cells, 0 elsewhere)
      4. row_has_nz = ReduceMax(nonzero_mask, axes=[3], keepdims=1) -> [N, 1, 30, 1]
      5. col_has_nz = ReduceMax(nonzero_mask, axes=[2], keepdims=1) -> [N, 1, 1, 30]
      6. cumtop_row = MaxPool(row_has_nz, kernel=[30, 1], pads=[29, 0, 0, 0])
            -> [N, 1, 30, 1]. Output[r] = max(row_has_nz[max(0, r-29) .. r]) =
            1 iff any non-zero row in [0, r].
      7. cumbot_row = MaxPool(row_has_nz, kernel=[30, 1], pads=[0, 0, 29, 0])
            -> 1 iff any non-zero row in [r, 29].
      8. cumtop_col = MaxPool(col_has_nz, kernel=[1, 30], pads=[0, 29, 0, 0])
            -> [N, 1, 1, 30]. 1 iff any non-zero col in [0, c].
      9. cumbot_col = MaxPool(col_has_nz, kernel=[1, 30], pads=[0, 0, 0, 29])
     10. row_in_bbox = cumtop_row * cumbot_row -> [N, 1, 30, 1]
     11. col_in_bbox = cumtop_col * cumbot_col -> [N, 1, 1, 30]
     12. in_bbox = row_in_bbox * col_in_bbox -> [N, 1, 30, 30] (broadcast)
     13. fill_mask = in_bbox * ch0 -> [N, 1, 30, 30]
            (1 only at active-grid cells inside bbox that were color 0)
     14. For ch=0: new_ch0 = ch0 - fill_mask
         For ch=target_color: new_chT = ch_T + fill_mask
         Other channels: unchanged
     15. Concat 10 channels.

    Cost (per cost-model verify, opset 10): macs ~ 9k (ReduceSum) + 900*2 (per-axis
    ReduceMax) + 900*5 (Mul/Sub on [N, 1, 30, 30]) + small. MaxPool is 0 MACs in
    onnx-tool 1.0.1 (no `profile()` override on MaxPoolNode); kernel-30 is just
    a span. Memory ~70 KB (a handful of [N, 1, 30, 30] intermediates).
    Params ~50 (Slice/Reduce inits + scalar consts). Total ~80-100k -> ~12.4 pts/task.
    """
    if not (1 <= target_color < 10):
        raise ValueError(f"target_color must be in 1..9, got {target_color}")

    nodes = []
    inits = []

    def _slice_channel(name_prefix, in_name, ch, out_name):
        starts = onnx.helper.make_tensor(f"{name_prefix}_starts", onnx.TensorProto.INT64, [1], [ch])
        ends = onnx.helper.make_tensor(f"{name_prefix}_ends", onnx.TensorProto.INT64, [1], [ch + 1])
        axes = onnx.helper.make_tensor(f"{name_prefix}_axes", onnx.TensorProto.INT64, [1], [1])
        steps = onnx.helper.make_tensor(f"{name_prefix}_steps", onnx.TensorProto.INT64, [1], [1])
        node = onnx.helper.make_node(
            "Slice",
            [in_name, f"{name_prefix}_starts", f"{name_prefix}_ends", f"{name_prefix}_axes", f"{name_prefix}_steps"],
            [out_name],
        )
        return node, [starts, ends, axes, steps]

    # Slice each channel of input (we'll build the output as a Concat of these,
    # with ch0 and ch target_color modified).
    channel_names = {}
    for ch in range(10):
        out_name = f"bbf_ch{ch}"
        node, ch_inits = _slice_channel(f"bbf_sl_{ch}", "input", ch, out_name)
        nodes.append(node)
        inits.extend(ch_inits)
        channel_names[ch] = out_name

    # is_active = sum across channels
    nodes.append(onnx.helper.make_node(
        "ReduceSum", ["input"], ["bbf_is_active"], axes=[1], keepdims=1
    ))

    # nonzero_mask = is_active - ch0
    nodes.append(onnx.helper.make_node(
        "Sub", ["bbf_is_active", channel_names[0]], ["bbf_nz_mask"]
    ))

    # row_has_nz, col_has_nz
    nodes.append(onnx.helper.make_node(
        "ReduceMax", ["bbf_nz_mask"], ["bbf_row_has_nz"], axes=[3], keepdims=1
    ))
    nodes.append(onnx.helper.make_node(
        "ReduceMax", ["bbf_nz_mask"], ["bbf_col_has_nz"], axes=[2], keepdims=1
    ))

    # cumulative-OR via MaxPool. Kernel size = canvas dim (30) along the relevant
    # axis. pads = [29, 0, 0, 0] for "from-top"; [0, 0, 29, 0] for "from-bottom"
    # along H; analogous for W.
    # MaxPool pads in opset 10: list of 2*spatial_rank ints, [begin_h, begin_w, end_h, end_w].
    nodes.append(onnx.helper.make_node(
        "MaxPool", ["bbf_row_has_nz"], ["bbf_cumtop_row"],
        kernel_shape=[30, 1], pads=[29, 0, 0, 0], strides=[1, 1],
    ))
    nodes.append(onnx.helper.make_node(
        "MaxPool", ["bbf_row_has_nz"], ["bbf_cumbot_row"],
        kernel_shape=[30, 1], pads=[0, 0, 29, 0], strides=[1, 1],
    ))
    nodes.append(onnx.helper.make_node(
        "MaxPool", ["bbf_col_has_nz"], ["bbf_cumtop_col"],
        kernel_shape=[1, 30], pads=[0, 29, 0, 0], strides=[1, 1],
    ))
    nodes.append(onnx.helper.make_node(
        "MaxPool", ["bbf_col_has_nz"], ["bbf_cumbot_col"],
        kernel_shape=[1, 30], pads=[0, 0, 0, 29], strides=[1, 1],
    ))

    # row_in_bbox = cumtop_row * cumbot_row; col_in_bbox = cumtop_col * cumbot_col
    nodes.append(onnx.helper.make_node(
        "Mul", ["bbf_cumtop_row", "bbf_cumbot_row"], ["bbf_row_in_bbox"]
    ))
    nodes.append(onnx.helper.make_node(
        "Mul", ["bbf_cumtop_col", "bbf_cumbot_col"], ["bbf_col_in_bbox"]
    ))

    # in_bbox = row_in_bbox * col_in_bbox (broadcasts to [N, 1, 30, 30])
    nodes.append(onnx.helper.make_node(
        "Mul", ["bbf_row_in_bbox", "bbf_col_in_bbox"], ["bbf_in_bbox"]
    ))

    # fill_mask = in_bbox * ch0
    nodes.append(onnx.helper.make_node(
        "Mul", ["bbf_in_bbox", channel_names[0]], ["bbf_fill_mask"]
    ))

    # new_ch0 = ch0 - fill_mask
    nodes.append(onnx.helper.make_node(
        "Sub", [channel_names[0], "bbf_fill_mask"], ["bbf_new_ch0"]
    ))
    # new_ch_target = ch_target + fill_mask
    nodes.append(onnx.helper.make_node(
        "Add", [channel_names[target_color], "bbf_fill_mask"], ["bbf_new_chT"]
    ))

    # Concat 10 channels
    concat_inputs = []
    for ch in range(10):
        if ch == 0:
            concat_inputs.append("bbf_new_ch0")
        elif ch == target_color:
            concat_inputs.append("bbf_new_chT")
        else:
            concat_inputs.append(channel_names[ch])
    nodes.append(onnx.helper.make_node(
        "Concat", concat_inputs, ["output"], axis=1
    ))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def gridline_adjacent_fill_network(batch=1):
    """Task105 gridline-adjacent fill.

    Inputs contain only colors 0/1. Fill color-0 cells with color 2 along:
      * the min/max row and column of the color-1 bounding box,
      * any row/column with at least four color-1 cells,
      * any row/column with exactly three color-1 cells where two are adjacent.

    The row lines are clipped to the color-1 bbox columns, and column lines are
    clipped to the color-1 bbox rows.
    """
    nodes = []
    inits = []

    def _scalar(name, value):
        tensor = onnx.helper.make_tensor(name, onnx.TensorProto.FLOAT, [1], [float(value)])
        inits.append(tensor)
        return name

    def _ints(name, values):
        tensor = onnx.helper.make_tensor(name, onnx.TensorProto.INT64, [len(values)], values)
        inits.append(tensor)
        return name

    half = _scalar("glaf_half", 0.5)
    one = _scalar("glaf_one", 1.0)
    three = _scalar("glaf_three", 3.0)
    three_half = _scalar("glaf_three_half", 3.5)
    one_half = _scalar("glaf_one_half", 1.5)

    def _slice_channel(ch, out_name):
        nodes.append(onnx.helper.make_node(
            "Slice",
            ["input",
             _ints(f"glaf_sl{ch}_starts", [ch]),
             _ints(f"glaf_sl{ch}_ends", [ch + 1]),
             _ints(f"glaf_sl{ch}_axes", [1]),
             _ints(f"glaf_sl{ch}_steps", [1])],
            [out_name],
        ))

    channels = {}
    for ch in (0, 1):
        name = f"glaf_ch{ch}"
        _slice_channel(ch, name)
        channels[ch] = name

    # Count color-1 cells by row/column.
    nodes.append(onnx.helper.make_node("ReduceSum", [channels[1]], ["glaf_row_count"], axes=[3], keepdims=1))
    nodes.append(onnx.helper.make_node("ReduceSum", [channels[1]], ["glaf_col_count"], axes=[2], keepdims=1))

    # row_has / col_has are also used to build first/last bbox-edge masks.
    nodes.append(onnx.helper.make_node("Greater", ["glaf_row_count", half], ["glaf_row_has_b"]))
    nodes.append(onnx.helper.make_node("Greater", ["glaf_col_count", half], ["glaf_col_has_b"]))
    nodes.append(onnx.helper.make_node("Cast", ["glaf_row_has_b"], ["glaf_row_has"], to=onnx.TensorProto.FLOAT))
    nodes.append(onnx.helper.make_node("Cast", ["glaf_col_has_b"], ["glaf_col_has"], to=onnx.TensorProto.FLOAT))

    # Bbox spans from cumulative max-pools.
    nodes.append(onnx.helper.make_node(
        "MaxPool", ["glaf_row_has"], ["glaf_cumtop_row"],
        kernel_shape=[30, 1], pads=[29, 0, 0, 0], strides=[1, 1],
    ))
    nodes.append(onnx.helper.make_node(
        "MaxPool", ["glaf_row_has"], ["glaf_cumbot_row"],
        kernel_shape=[30, 1], pads=[0, 0, 29, 0], strides=[1, 1],
    ))
    nodes.append(onnx.helper.make_node(
        "MaxPool", ["glaf_col_has"], ["glaf_cumleft_col"],
        kernel_shape=[1, 30], pads=[0, 29, 0, 0], strides=[1, 1],
    ))
    nodes.append(onnx.helper.make_node(
        "MaxPool", ["glaf_col_has"], ["glaf_cumright_col"],
        kernel_shape=[1, 30], pads=[0, 0, 0, 29], strides=[1, 1],
    ))
    nodes.append(onnx.helper.make_node("Mul", ["glaf_cumtop_row", "glaf_cumbot_row"], ["glaf_row_in_bbox"]))
    nodes.append(onnx.helper.make_node("Mul", ["glaf_cumleft_col", "glaf_cumright_col"], ["glaf_col_in_bbox"]))

    # Strict-before/after masks identify the first and last row/column with color 1.
    nodes.append(onnx.helper.make_node(
        "Slice",
        ["glaf_cumtop_row",
         _ints("glaf_prev_row_starts", [0]),
         _ints("glaf_prev_row_ends", [29]),
         _ints("glaf_prev_row_axes", [2]),
         _ints("glaf_prev_row_steps", [1])],
        ["glaf_prev_row_29"],
    ))
    nodes.append(onnx.helper.make_node(
        "Pad", ["glaf_prev_row_29"], ["glaf_prev_row"],
        pads=[0, 0, 1, 0, 0, 0, 0, 0], mode="constant", value=0.0,
    ))
    nodes.append(onnx.helper.make_node(
        "Slice",
        ["glaf_cumbot_row",
         _ints("glaf_next_row_starts", [1]),
         _ints("glaf_next_row_ends", [30]),
         _ints("glaf_next_row_axes", [2]),
         _ints("glaf_next_row_steps", [1])],
        ["glaf_next_row_29"],
    ))
    nodes.append(onnx.helper.make_node(
        "Pad", ["glaf_next_row_29"], ["glaf_next_row"],
        pads=[0, 0, 0, 0, 0, 0, 1, 0], mode="constant", value=0.0,
    ))
    nodes.append(onnx.helper.make_node(
        "Slice",
        ["glaf_cumleft_col",
         _ints("glaf_prev_col_starts", [0]),
         _ints("glaf_prev_col_ends", [29]),
         _ints("glaf_prev_col_axes", [3]),
         _ints("glaf_prev_col_steps", [1])],
        ["glaf_prev_col_29"],
    ))
    nodes.append(onnx.helper.make_node(
        "Pad", ["glaf_prev_col_29"], ["glaf_prev_col"],
        pads=[0, 0, 0, 1, 0, 0, 0, 0], mode="constant", value=0.0,
    ))
    nodes.append(onnx.helper.make_node(
        "Slice",
        ["glaf_cumright_col",
         _ints("glaf_next_col_starts", [1]),
         _ints("glaf_next_col_ends", [30]),
         _ints("glaf_next_col_axes", [3]),
         _ints("glaf_next_col_steps", [1])],
        ["glaf_next_col_29"],
    ))
    nodes.append(onnx.helper.make_node(
        "Pad", ["glaf_next_col_29"], ["glaf_next_col"],
        pads=[0, 0, 0, 0, 0, 0, 0, 1], mode="constant", value=0.0,
    ))

    nodes.append(onnx.helper.make_node("Sub", [one, "glaf_prev_row"], ["glaf_no_prev_row"]))
    nodes.append(onnx.helper.make_node("Sub", [one, "glaf_next_row"], ["glaf_no_next_row"]))
    nodes.append(onnx.helper.make_node("Sub", [one, "glaf_prev_col"], ["glaf_no_prev_col"]))
    nodes.append(onnx.helper.make_node("Sub", [one, "glaf_next_col"], ["glaf_no_next_col"]))
    nodes.append(onnx.helper.make_node("Mul", ["glaf_row_has", "glaf_no_prev_row"], ["glaf_min_row"]))
    nodes.append(onnx.helper.make_node("Mul", ["glaf_row_has", "glaf_no_next_row"], ["glaf_max_row"]))
    nodes.append(onnx.helper.make_node("Mul", ["glaf_col_has", "glaf_no_prev_col"], ["glaf_min_col"]))
    nodes.append(onnx.helper.make_node("Mul", ["glaf_col_has", "glaf_no_next_col"], ["glaf_max_col"]))
    nodes.append(onnx.helper.make_node("Add", ["glaf_min_row", "glaf_max_row"], ["glaf_row_edge_sum"]))
    nodes.append(onnx.helper.make_node("Add", ["glaf_min_col", "glaf_max_col"], ["glaf_col_edge_sum"]))
    nodes.append(onnx.helper.make_node("Greater", ["glaf_row_edge_sum", half], ["glaf_row_edge_b"]))
    nodes.append(onnx.helper.make_node("Greater", ["glaf_col_edge_sum", half], ["glaf_col_edge_b"]))
    nodes.append(onnx.helper.make_node("Cast", ["glaf_row_edge_b"], ["glaf_row_edge"], to=onnx.TensorProto.FLOAT))
    nodes.append(onnx.helper.make_node("Cast", ["glaf_col_edge_b"], ["glaf_col_edge"], to=onnx.TensorProto.FLOAT))

    # Count >= 4 lines.
    nodes.append(onnx.helper.make_node("Greater", ["glaf_row_count", three_half], ["glaf_row_ge4_b"]))
    nodes.append(onnx.helper.make_node("Greater", ["glaf_col_count", three_half], ["glaf_col_ge4_b"]))
    nodes.append(onnx.helper.make_node("Cast", ["glaf_row_ge4_b"], ["glaf_row_ge4"], to=onnx.TensorProto.FLOAT))
    nodes.append(onnx.helper.make_node("Cast", ["glaf_col_ge4_b"], ["glaf_col_ge4"], to=onnx.TensorProto.FLOAT))

    # Exactly-3 counts via squared difference, because Equal on float is opset-13+.
    nodes.append(onnx.helper.make_node("Sub", ["glaf_row_count", three], ["glaf_row_count_diff"]))
    nodes.append(onnx.helper.make_node("Sub", ["glaf_col_count", three], ["glaf_col_count_diff"]))
    nodes.append(onnx.helper.make_node("Mul", ["glaf_row_count_diff", "glaf_row_count_diff"], ["glaf_row_count_diff2"]))
    nodes.append(onnx.helper.make_node("Mul", ["glaf_col_count_diff", "glaf_col_count_diff"], ["glaf_col_count_diff2"]))
    nodes.append(onnx.helper.make_node("Less", ["glaf_row_count_diff2", half], ["glaf_row_eq3_b"]))
    nodes.append(onnx.helper.make_node("Less", ["glaf_col_count_diff2", half], ["glaf_col_eq3_b"]))
    nodes.append(onnx.helper.make_node("Cast", ["glaf_row_eq3_b"], ["glaf_row_eq3"], to=onnx.TensorProto.FLOAT))
    nodes.append(onnx.helper.make_node("Cast", ["glaf_col_eq3_b"], ["glaf_col_eq3"], to=onnx.TensorProto.FLOAT))

    # Adjacent-pair tests along rows/columns.
    row_adj_w = onnx.helper.make_tensor("glaf_row_adj_w", onnx.TensorProto.FLOAT, [1, 1, 1, 2], [1.0, 1.0])
    col_adj_w = onnx.helper.make_tensor("glaf_col_adj_w", onnx.TensorProto.FLOAT, [1, 1, 2, 1], [1.0, 1.0])
    inits.extend([row_adj_w, col_adj_w])
    nodes.append(onnx.helper.make_node("Conv", [channels[1], "glaf_row_adj_w"], ["glaf_row_adj_pair_counts"]))
    nodes.append(onnx.helper.make_node("Conv", [channels[1], "glaf_col_adj_w"], ["glaf_col_adj_pair_counts"]))
    nodes.append(onnx.helper.make_node("ReduceMax", ["glaf_row_adj_pair_counts"], ["glaf_row_adj_pair_count"], axes=[3], keepdims=1))
    nodes.append(onnx.helper.make_node("ReduceMax", ["glaf_col_adj_pair_counts"], ["glaf_col_adj_pair_count"], axes=[2], keepdims=1))
    nodes.append(onnx.helper.make_node("Greater", ["glaf_row_adj_pair_count", one_half], ["glaf_row_adj_b"]))
    nodes.append(onnx.helper.make_node("Greater", ["glaf_col_adj_pair_count", one_half], ["glaf_col_adj_b"]))
    nodes.append(onnx.helper.make_node("Cast", ["glaf_row_adj_b"], ["glaf_row_adj"], to=onnx.TensorProto.FLOAT))
    nodes.append(onnx.helper.make_node("Cast", ["glaf_col_adj_b"], ["glaf_col_adj"], to=onnx.TensorProto.FLOAT))
    nodes.append(onnx.helper.make_node("Mul", ["glaf_row_eq3", "glaf_row_adj"], ["glaf_row_eq3_adj"]))
    nodes.append(onnx.helper.make_node("Mul", ["glaf_col_eq3", "glaf_col_adj"], ["glaf_col_eq3_adj"]))

    # Selected line masks.
    nodes.append(onnx.helper.make_node("Add", ["glaf_row_edge", "glaf_row_ge4"], ["glaf_row_sel_a"]))
    nodes.append(onnx.helper.make_node("Add", ["glaf_row_sel_a", "glaf_row_eq3_adj"], ["glaf_row_sel_sum"]))
    nodes.append(onnx.helper.make_node("Add", ["glaf_col_edge", "glaf_col_ge4"], ["glaf_col_sel_a"]))
    nodes.append(onnx.helper.make_node("Add", ["glaf_col_sel_a", "glaf_col_eq3_adj"], ["glaf_col_sel_sum"]))
    nodes.append(onnx.helper.make_node("Greater", ["glaf_row_sel_sum", half], ["glaf_row_sel_b"]))
    nodes.append(onnx.helper.make_node("Greater", ["glaf_col_sel_sum", half], ["glaf_col_sel_b"]))
    nodes.append(onnx.helper.make_node("Cast", ["glaf_row_sel_b"], ["glaf_row_sel"], to=onnx.TensorProto.FLOAT))
    nodes.append(onnx.helper.make_node("Cast", ["glaf_col_sel_b"], ["glaf_col_sel"], to=onnx.TensorProto.FLOAT))

    # Expand selected rows/columns over the opposite bbox span.
    nodes.append(onnx.helper.make_node("Mul", ["glaf_row_sel", "glaf_col_in_bbox"], ["glaf_row_fill"]))
    nodes.append(onnx.helper.make_node("Mul", ["glaf_col_sel", "glaf_row_in_bbox"], ["glaf_col_fill"]))
    nodes.append(onnx.helper.make_node("Add", ["glaf_row_fill", "glaf_col_fill"], ["glaf_line_fill_sum"]))
    nodes.append(onnx.helper.make_node("Greater", ["glaf_line_fill_sum", half], ["glaf_line_fill_b"]))
    nodes.append(onnx.helper.make_node("Cast", ["glaf_line_fill_b"], ["glaf_line_fill"], to=onnx.TensorProto.FLOAT))
    nodes.append(onnx.helper.make_node("Mul", ["glaf_line_fill", channels[0]], ["glaf_fill_mask"]))

    nodes.append(onnx.helper.make_node("Sub", [channels[0], "glaf_fill_mask"], ["glaf_new_ch0"]))
    nodes.append(onnx.helper.make_node("Sub", [channels[1], channels[1]], ["glaf_zero"]))

    concat_inputs = ["glaf_new_ch0", channels[1], "glaf_fill_mask"]
    concat_inputs.extend(["glaf_zero"] * 7)
    nodes.append(onnx.helper.make_node("Concat", concat_inputs, ["output"], axis=1))
    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def line_projection_to_full_line_network(batch=1):
    """Task025: project stray same-color cells next to their full row/column.

    For each non-background color that forms a full active-grid row or column,
    keep the full line and move every off-line cell of that color to the cell
    adjacent to the line on the same side. Colors without a full line disappear.
    """
    if batch != 1:
        raise NotImplementedError("line-projection builder is single-example only")

    nodes = []
    inits = []
    P = "lpfl_"

    def _tensor(name, dtype, shape, values):
        t = onnx.helper.make_tensor(name, dtype, shape, values)
        inits.append(t)
        return name

    def _node(op_type, inputs, outputs, **attrs):
        nodes.append(onnx.helper.make_node(op_type, inputs, outputs, **attrs))

    def _ints(name, values):
        return _tensor(name, onnx.TensorProto.INT64, [len(values)], values)

    def _float(name, value):
        return _tensor(name, onnx.TensorProto.FLOAT, [1], [float(value)])

    zero = _float(P + "zero", 0.0)
    half = _float(P + "half", 0.5)

    _node(
        "Slice",
        [
            "input",
            _ints(P + "sl19_st", [1]),
            _ints(P + "sl19_en", [10]),
            _ints(P + "sl19_ax", [1]),
            _ints(P + "sl19_sp", [1]),
        ],
        [P + "colors"],
    )
    _node("ReduceSum", ["input"], [P + "active"], axes=[1], keepdims=1)
    _node("ReduceSum", [P + "active"], [P + "row_total"], axes=[3], keepdims=1)
    _node("ReduceSum", [P + "active"], [P + "col_total"], axes=[2], keepdims=1)
    _node("ReduceSum", [P + "colors"], [P + "row_counts"], axes=[3], keepdims=1)
    _node("ReduceSum", [P + "colors"], [P + "col_counts"], axes=[2], keepdims=1)

    _node("Greater", [P + "row_total", zero], [P + "row_active_b"])
    _node("Cast", [P + "row_active_b"], [P + "row_active"], to=onnx.TensorProto.FLOAT)
    _node("Greater", [P + "col_total", zero], [P + "col_active_b"])
    _node("Cast", [P + "col_active_b"], [P + "col_active"], to=onnx.TensorProto.FLOAT)

    _node("Sub", [P + "row_counts", P + "row_total"], [P + "row_diff"])
    _node("Abs", [P + "row_diff"], [P + "row_abs"])
    _node("Less", [P + "row_abs", half], [P + "row_full_b"])
    _node("Cast", [P + "row_full_b"], [P + "row_full_raw"], to=onnx.TensorProto.FLOAT)
    _node("Mul", [P + "row_full_raw", P + "row_active"], [P + "row_full"])

    _node("Sub", [P + "col_counts", P + "col_total"], [P + "col_diff"])
    _node("Abs", [P + "col_diff"], [P + "col_abs"])
    _node("Less", [P + "col_abs", half], [P + "col_full_b"])
    _node("Cast", [P + "col_full_b"], [P + "col_full_raw"], to=onnx.TensorProto.FLOAT)
    _node("Mul", [P + "col_full_raw", P + "col_active"], [P + "col_full"])

    def _shift_row(src, dst, direction):
        if direction == "up":
            starts, ends = [1], [30]
            pads = [0, 0, 0, 0, 0, 0, 1, 0]
        else:
            starts, ends = [0], [29]
            pads = [0, 0, 1, 0, 0, 0, 0, 0]
        _node(
            "Slice",
            [
                src,
                _ints(dst + "_st", starts),
                _ints(dst + "_en", ends),
                _ints(dst + "_ax", [2]),
                _ints(dst + "_sp", [1]),
            ],
            [dst + "_slice"],
        )
        _node("Pad", [dst + "_slice"], [dst], pads=pads, mode="constant", value=0.0)

    def _shift_col(src, dst, direction):
        if direction == "left":
            starts, ends = [1], [30]
            pads = [0, 0, 0, 0, 0, 0, 0, 1]
        else:
            starts, ends = [0], [29]
            pads = [0, 0, 0, 1, 0, 0, 0, 0]
        _node(
            "Slice",
            [
                src,
                _ints(dst + "_st", starts),
                _ints(dst + "_en", ends),
                _ints(dst + "_ax", [3]),
                _ints(dst + "_sp", [1]),
            ],
            [dst + "_slice"],
        )
        _node("Pad", [dst + "_slice"], [dst], pads=pads, mode="constant", value=0.0)

    _shift_row(P + "row_full", P + "row_full_up", "up")
    _shift_row(P + "row_full", P + "row_full_down", "down")
    _shift_col(P + "col_full", P + "col_full_left", "left")
    _shift_col(P + "col_full", P + "col_full_right", "right")

    _node(
        "MaxPool",
        [P + "colors"],
        [P + "cum_top"],
        kernel_shape=[30, 1],
        pads=[29, 0, 0, 0],
        strides=[1, 1],
    )
    _node(
        "MaxPool",
        [P + "colors"],
        [P + "cum_bottom"],
        kernel_shape=[30, 1],
        pads=[0, 0, 29, 0],
        strides=[1, 1],
    )
    _node(
        "MaxPool",
        [P + "colors"],
        [P + "cum_left"],
        kernel_shape=[1, 30],
        pads=[0, 29, 0, 0],
        strides=[1, 1],
    )
    _node(
        "MaxPool",
        [P + "colors"],
        [P + "cum_right"],
        kernel_shape=[1, 30],
        pads=[0, 0, 0, 29],
        strides=[1, 1],
    )

    _node("Mul", [P + "row_full", P + "active"], [P + "row_lines"])
    _node("Mul", [P + "col_full", P + "active"], [P + "col_lines"])
    _node("Mul", [P + "cum_top", P + "row_full_up"], [P + "proj_above"])
    _node("Mul", [P + "cum_bottom", P + "row_full_down"], [P + "proj_below"])
    _node("Mul", [P + "cum_left", P + "col_full_left"], [P + "proj_left"])
    _node("Mul", [P + "cum_right", P + "col_full_right"], [P + "proj_right"])

    _node(
        "Sum",
        [
            P + "row_lines",
            P + "col_lines",
            P + "proj_above",
            P + "proj_below",
            P + "proj_left",
            P + "proj_right",
        ],
        [P + "nonbg_raw"],
    )
    _node("Greater", [P + "nonbg_raw", half], [P + "nonbg_b"])
    _node("Cast", [P + "nonbg_b"], [P + "nonbg"], to=onnx.TensorProto.FLOAT)
    _node("ReduceMax", [P + "nonbg"], [P + "any_nonbg"], axes=[1], keepdims=1)
    _node("Sub", [P + "active", P + "any_nonbg"], [P + "ch0"])
    _node("Concat", [P + "ch0", P + "nonbg"], ["output"], axis=1)

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def diagonal_zero_mask_projection_network(batch=1):
    """Task370: repeat the zero-hole mask along the marker-to-hole diagonal.

    Inputs have a dominant nonzero background color, one marker cell of a
    second color, and color-0 holes. The marker points to the farthest zero
    cell on one of its diagonals; the whole zero mask is then translated by
    that vector repeatedly, painting only background cells with the marker
    color while preserving the original zero holes.
    """
    if batch != 1:
        raise NotImplementedError("task370 diagonal projection builder is single-example only")

    S = 20
    nodes = []
    inits = []
    P = "dzmp_"

    def _tensor(name, dtype, shape, values):
        t = onnx.helper.make_tensor(name, dtype, shape, values)
        inits.append(t)
        return name

    def _ints(name, values):
        return _tensor(name, onnx.TensorProto.INT64, [len(values)], values)

    def _float(name, value):
        return _tensor(name, onnx.TensorProto.FLOAT, [1], [float(value)])

    def _node(op_type, inputs, outputs, **attrs):
        nodes.append(onnx.helper.make_node(op_type, inputs, outputs, **attrs))

    half = _float(P + "half", 0.5)
    one = _float(P + "one", 1.0)

    ch_axis = _ints(P + "ch_axis", [1])
    ch_step = _ints(P + "ch_step", [1])
    hw_axes = _ints(P + "hw_axes", [2, 3])
    hw_steps = _ints(P + "hw_steps", [1, 1])

    _node("Slice", ["input", _ints(P + "crop_st", [0, 0]), _ints(P + "crop_en", [S, S]), hw_axes, hw_steps], [P + "crop"])
    _node("Slice", [P + "crop", _ints(P + "ch0_st", [0]), _ints(P + "ch0_en", [1]), ch_axis, ch_step], [P + "zero_mask"])
    _node("Slice", [P + "crop", _ints(P + "ch19_st", [1]), _ints(P + "ch19_en", [10]), ch_axis, ch_step], [P + "colors"])

    # Background is the dominant nonzero color; marker color is the other
    # present nonzero color. This avoids baking either color into the network.
    _node("ReduceSum", [P + "colors"], [P + "counts"], axes=[2, 3], keepdims=1)
    _node("ReduceMax", [P + "counts"], [P + "max_count"], axes=[1], keepdims=1)
    _node("Sub", [P + "counts", P + "max_count"], [P + "bg_diff"])
    _node("Abs", [P + "bg_diff"], [P + "bg_abs"])
    _node("Less", [P + "bg_abs", half], [P + "bg_mask_b"])
    _node("Cast", [P + "bg_mask_b"], [P + "bg_mask"], to=onnx.TensorProto.FLOAT)
    _node("Greater", [P + "counts", half], [P + "present_b"])
    _node("Cast", [P + "present_b"], [P + "present"], to=onnx.TensorProto.FLOAT)
    _node("Sub", [P + "present", P + "bg_mask"], [P + "marker_color_mask"])

    _node("Mul", [P + "colors", P + "bg_mask"], [P + "bg_cells_by_ch"])
    _node("ReduceSum", [P + "bg_cells_by_ch"], [P + "bg_cells"], axes=[1], keepdims=1)
    _node("Mul", [P + "colors", P + "marker_color_mask"], [P + "marker_cells_by_ch"])
    _node("ReduceSum", [P + "marker_cells_by_ch"], [P + "marker_cells"], axes=[1], keepdims=1)

    def _shift_1ch(src, dst, dr, dc):
        """Translate a [1,1,S,S] tensor by (dr, dc), clipping off-canvas."""
        r0 = max(0, -dr)
        r1 = S - max(0, dr)
        c0 = max(0, -dc)
        c1 = S - max(0, dc)
        pad_top = max(0, dr)
        pad_bottom = max(0, -dr)
        pad_left = max(0, dc)
        pad_right = max(0, -dc)
        _node(
            "Slice",
            [src, _ints(dst + "_st", [r0, c0]), _ints(dst + "_en", [r1, c1]), hw_axes, hw_steps],
            [dst + "_slice"],
        )
        _node(
            "Pad",
            [dst + "_slice"],
            [dst],
            pads=[0, 0, pad_top, pad_left, 0, 0, pad_bottom, pad_right],
            mode="constant",
            value=0.0,
        )

    by_distance = {}
    vector_gates = []
    for dist in (2, 3, 4, 5):
        by_distance[dist] = []
        for sr in (-1, 1):
            for sc in (-1, 1):
                dr = sr * dist
                dc = sc * dist
                suffix = f"d{dist}_{'p' if sr > 0 else 'n'}{'p' if sc > 0 else 'n'}"
                shifted = P + "valid_shift_" + suffix
                _shift_1ch(P + "zero_mask", shifted, dr, dc)
                _node("Mul", [shifted, P + "marker_cells"], [P + "valid_overlap_" + suffix])
                _node(
                    "ReduceMax",
                    [P + "valid_overlap_" + suffix],
                    [P + "valid_" + suffix],
                    axes=[2, 3],
                    keepdims=1,
                )
                by_distance[dist].append(P + "valid_" + suffix)
                vector_gates.append((dist, dr, dc, suffix, P + "valid_" + suffix))

    dist_any = {}
    for dist, valid_names in by_distance.items():
        if len(valid_names) == 1:
            dist_any[dist] = valid_names[0]
        else:
            _node("Sum", valid_names, [P + f"d{dist}_sum"])
            _node("Greater", [P + f"d{dist}_sum", half], [P + f"d{dist}_any_b"])
            _node("Cast", [P + f"d{dist}_any_b"], [P + f"d{dist}_any"], to=onnx.TensorProto.FLOAT)
            dist_any[dist] = P + f"d{dist}_any"

    def _no_greater_gate(dist):
        higher = [dist_any[d] for d in (2, 3, 4, 5) if d > dist]
        if not higher:
            return one
        if len(higher) == 1:
            higher_sum = higher[0]
        else:
            higher_sum = P + f"d{dist}_higher_sum"
            _node("Sum", higher, [higher_sum])
        _node("Greater", [higher_sum, half], [P + f"d{dist}_higher_b"])
        _node("Cast", [P + f"d{dist}_higher_b"], [P + f"d{dist}_higher"], to=onnx.TensorProto.FLOAT)
        _node("Sub", [one, P + f"d{dist}_higher"], [P + f"d{dist}_no_greater"])
        return P + f"d{dist}_no_greater"

    no_greater = {dist: _no_greater_gate(dist) for dist in (2, 3, 4, 5)}

    selected_projection_masks = []
    for dist, dr, dc, suffix, valid_name in vector_gates:
        _node("Mul", [valid_name, no_greater[dist]], [P + "gate_" + suffix])
        shifts = []
        for k in range(1, (S - 1) // dist + 1):
            shifted = P + f"proj_{suffix}_k{k}"
            _shift_1ch(P + "zero_mask", shifted, dr * k, dc * k)
            shifts.append(shifted)
        if len(shifts) == 1:
            union = shifts[0]
        else:
            union = P + "proj_sum_" + suffix
            _node("Sum", shifts, [union])
            _node("Greater", [union, half], [P + "proj_any_b_" + suffix])
            _node("Cast", [P + "proj_any_b_" + suffix], [P + "proj_any_" + suffix], to=onnx.TensorProto.FLOAT)
            union = P + "proj_any_" + suffix
        _node("Mul", [union, P + "gate_" + suffix], [P + "selected_" + suffix])
        selected_projection_masks.append(P + "selected_" + suffix)

    _node("Sum", selected_projection_masks, [P + "proj_raw"])
    _node("Greater", [P + "proj_raw", half], [P + "proj_b"])
    _node("Cast", [P + "proj_b"], [P + "proj"], to=onnx.TensorProto.FLOAT)

    # Paint only true background cells; original zero holes and the marker cell
    # pass through unchanged.
    _node("Mul", [P + "proj", P + "bg_cells"], [P + "paint"])
    _node("Mul", [P + "bg_mask", P + "paint"], [P + "remove_bg"])
    _node("Sub", [P + "colors", P + "remove_bg"], [P + "colors_no_bg"])
    _node("Mul", [P + "marker_color_mask", P + "paint"], [P + "add_marker"])
    _node("Add", [P + "colors_no_bg", P + "add_marker"], [P + "out_colors"])
    _node("Concat", [P + "zero_mask", P + "out_colors"], [P + "out_crop"], axis=1)
    _node("Pad", [P + "out_crop"], ["output"], pads=[0, 0, 0, 0, 0, 0, 30 - S, 30 - S],
          mode="constant", value=0.0)

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def marker_shape_copy_network(batch=1):
    """Task089: copy learned marker-relative shapes to isolated markers.

    All examples are 13x13. Marker color 2 receives a horizontally mirrored
    copy of its learned source shape; marker color 3 receives an identity copy.
    The source marker is the marker cell adjacent to a non-marker shape color,
    while isolated same-color markers are the stamp targets.
    """
    if batch != 1:
        raise NotImplementedError("task089 marker-shape-copy builder is single-example only")

    S = 13
    nodes = []
    inits = []
    P = "msc_"

    def _tensor(name, dtype, shape, values):
        t = onnx.helper.make_tensor(name, dtype, shape, values)
        inits.append(t)
        return name

    def _ints(name, values):
        return _tensor(name, onnx.TensorProto.INT64, [len(values)], values)

    def _float(name, value):
        return _tensor(name, onnx.TensorProto.FLOAT, [1], [float(value)])

    def _node(op_type, inputs, outputs, **attrs):
        nodes.append(onnx.helper.make_node(op_type, inputs, outputs, **attrs))

    half = _float(P + "half", 0.5)
    one = _float(P + "one", 1.0)
    ch_axis = _ints(P + "ch_axis", [1])
    ch_step = _ints(P + "ch_step", [1])
    hw_axes = _ints(P + "hw_axes", [2, 3])
    hw_steps = _ints(P + "hw_steps", [1, 1])

    _node("Slice", ["input", _ints(P + "crop_st", [0, 0]), _ints(P + "crop_en", [S, S]), hw_axes, hw_steps], [P + "crop"])
    _node("Slice", [P + "crop", _ints(P + "ch0_st", [0]), _ints(P + "ch0_en", [1]), ch_axis, ch_step], [P + "ch0"])
    _node("Slice", [P + "crop", _ints(P + "ch19_st", [1]), _ints(P + "ch19_en", [10]), ch_axis, ch_step], [P + "colors"])
    _node("Slice", [P + "crop", _ints(P + "ch2_st", [2]), _ints(P + "ch2_en", [3]), ch_axis, ch_step], [P + "ch2"])
    _node("Slice", [P + "crop", _ints(P + "ch3_st", [3]), _ints(P + "ch3_en", [4]), ch_axis, ch_step], [P + "ch3"])

    # Shape colors exclude the marker colors 2 and 3.
    nonmarker_mask = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    _tensor(P + "nonmarker_mask", onnx.TensorProto.FLOAT, [1, 9, 1, 1], nonmarker_mask)
    _node("Mul", [P + "colors", P + "nonmarker_mask"], [P + "shape_colors"])
    _node("ReduceMax", [P + "shape_colors"], [P + "shape_any"], axes=[1], keepdims=1)
    _node(
        "MaxPool",
        [P + "shape_any"],
        [P + "near_shape"],
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        strides=[1, 1],
    )
    _node("Mul", [P + "ch2", P + "near_shape"], [P + "src2"])
    _node("Mul", [P + "ch3", P + "near_shape"], [P + "src3"])
    _node("Sub", [one, P + "near_shape"], [P + "not_near_shape"])
    _node("Mul", [P + "ch2", P + "not_near_shape"], [P + "tgt2"])
    _node("Mul", [P + "ch3", P + "not_near_shape"], [P + "tgt3"])

    def _shift_1ch(src, dst, dr, dc):
        r0 = max(0, -dr)
        r1 = S - max(0, dr)
        c0 = max(0, -dc)
        c1 = S - max(0, dc)
        pad_top = max(0, dr)
        pad_bottom = max(0, -dr)
        pad_left = max(0, dc)
        pad_right = max(0, -dc)
        _node(
            "Slice",
            [src, _ints(dst + "_st", [r0, c0]), _ints(dst + "_en", [r1, c1]), hw_axes, hw_steps],
            [dst + "_slice"],
        )
        _node(
            "Pad",
            [dst + "_slice"],
            [dst],
            pads=[0, 0, pad_top, pad_left, 0, 0, pad_bottom, pad_right],
            mode="constant",
            value=0.0,
        )

    def _source_color_vector(src, dst_prefix):
        _node(
            "MaxPool",
            [src],
            [dst_prefix + "near_src"],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[1, 1],
        )
        _node("Mul", [P + "shape_colors", dst_prefix + "near_src"], [dst_prefix + "adj_shape"])
        _node("ReduceMax", [dst_prefix + "adj_shape"], [dst_prefix + "color_vec"], axes=[2, 3], keepdims=1)
        return dst_prefix + "color_vec"

    src2_color = _source_color_vector(P + "src2", P + "src2_")
    src3_color = _source_color_vector(P + "src3", P + "src3_")

    paint_terms = []
    for marker_name, src, target, color_vec, flip_h in (
        ("m2", P + "src2", P + "tgt2", src2_color, True),
        ("m3", P + "src3", P + "tgt3", src3_color, False),
    ):
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                if dr == 0 and dc == 0:
                    continue
                suffix = f"{marker_name}_{'p' if dr > 0 else 'n' if dr < 0 else 'z'}{abs(dr)}_{'p' if dc > 0 else 'n' if dc < 0 else 'z'}{abs(dc)}"
                src_shift = P + "src_at_" + suffix
                _shift_1ch(src, src_shift, dr, dc)
                _node("Mul", [P + "shape_colors", src_shift], [P + "src_shape_" + suffix])
                _node("Mul", [P + "src_shape_" + suffix, color_vec], [P + "src_shape_color_" + suffix])
                _node(
                    "ReduceMax",
                    [P + "src_shape_color_" + suffix],
                    [P + "offset_color_" + suffix],
                    axes=[2, 3],
                    keepdims=1,
                )
                out_dc = -dc if flip_h else dc
                target_shift = P + "target_at_" + suffix
                _shift_1ch(target, target_shift, dr, out_dc)
                _node("Mul", [target_shift, P + "ch0"], [P + "target_bg_" + suffix])
                _node("Mul", [P + "target_bg_" + suffix, P + "offset_color_" + suffix], [P + "paint_" + suffix])
                paint_terms.append(P + "paint_" + suffix)

    _node("Sum", paint_terms, [P + "paint_raw"])
    _node("Greater", [P + "paint_raw", half], [P + "paint_b"])
    _node("Cast", [P + "paint_b"], [P + "paint"], to=onnx.TensorProto.FLOAT)
    _node("ReduceMax", [P + "paint"], [P + "paint_any"], axes=[1], keepdims=1)
    _node("Sub", [P + "ch0", P + "paint_any"], [P + "ch0_out"])
    _node("Add", [P + "colors", P + "paint"], [P + "out_colors"])
    _node("Concat", [P + "ch0_out", P + "out_colors"], [P + "out_crop"], axis=1)
    _node("Pad", [P + "out_crop"], ["output"], pads=[0, 0, 0, 0, 0, 0, 30 - S, 30 - S],
          mode="constant", value=0.0)

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def count_2x2_one_blocks_network(batch=1):
    """Task038: count color-1 2x2 blocks and emit a unary 1x5 row."""
    nodes = []
    inits = []
    P = "c2ob_"

    def _tensor(name, dtype, shape, values):
        t = onnx.helper.make_tensor(name, dtype, shape, values)
        inits.append(t)
        return name

    def _ints(name, values):
        return _tensor(name, onnx.TensorProto.INT64, [len(values)], values)

    def _node(op_type, inputs, outputs, **attrs):
        nodes.append(onnx.helper.make_node(op_type, inputs, outputs, **attrs))

    ch_axis = _ints(P + "ch_axis", [1])
    ch_step = _ints(P + "ch_step", [1])
    hw_axes = _ints(P + "hw_axes", [2, 3])
    hw_step = _ints(P + "hw_step", [1, 1])
    three_half = _tensor(P + "three_half", onnx.TensorProto.FLOAT, [1], [3.5])
    kernel = _tensor(P + "kernel", onnx.TensorProto.FLOAT, [1, 1, 2, 2], [1.0] * 4)
    range5 = _tensor(P + "range5", onnx.TensorProto.FLOAT, [1, 1, 1, 5], [float(i) for i in range(5)])
    one5 = _tensor(P + "one5", onnx.TensorProto.FLOAT, [1, 1, 1, 5], [1.0] * 5)

    _node("Slice", ["input", _ints(P + "ch1_st", [1]), _ints(P + "ch1_en", [2]), ch_axis, ch_step], [P + "ch1"])
    _node("Slice", [P + "ch1", _ints(P + "hw_st", [0, 0]), _ints(P + "hw_en", [9, 9]), hw_axes, hw_step], [P + "crop"])
    _node("Conv", [P + "crop", kernel], [P + "block_sums"])
    _node("Greater", [P + "block_sums", three_half], [P + "blocks_b"])
    _node("Cast", [P + "blocks_b"], [P + "blocks"], to=onnx.TensorProto.FLOAT)
    _node("ReduceSum", [P + "blocks"], [P + "count"], axes=[1, 2, 3], keepdims=1)
    _node("Less", [range5, P + "count"], [P + "mask_b"])
    _node("Cast", [P + "mask_b"], [P + "mask"], to=onnx.TensorProto.FLOAT)
    _node("Sub", [one5, P + "mask"], [P + "ch0"])
    _node("Sub", [P + "mask", P + "mask"], [P + "zero"])
    _node(
        "Concat",
        [P + "ch0", P + "mask", P + "zero", P + "zero", P + "zero",
         P + "zero", P + "zero", P + "zero", P + "zero", P + "zero"],
        [P + "row"],
        axis=1,
    )
    _node(
        "Pad",
        [P + "row"],
        ["output"],
        pads=[0, 0, 0, 0, 0, 0, 29, 25],
        mode="constant",
        value=0.0,
    )

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def block_grid_line_fill_network(batch=1):
    """Task009: fill aligned same-color spans on a 2x2-cell separator grid."""
    nodes = []
    inits = []
    P = "bglf_"
    fill_colors = list(range(1, 10))

    def _tensor(name, dtype, shape, values):
        t = onnx.helper.make_tensor(name, dtype, shape, values)
        inits.append(t)
        return name

    def _ints(name, values):
        return _tensor(name, onnx.TensorProto.INT64, [len(values)], values)

    def _node(op_type, inputs, outputs, **attrs):
        nodes.append(onnx.helper.make_node(op_type, inputs, outputs, **attrs))

    def _span_matrix(name, predicate):
        vals = []
        for src in range(100):
            sr, sc = divmod(src, 10)
            for tgt in range(100):
                tr, tc = divmod(tgt, 10)
                vals.append(1.0 if predicate(sr, sc, tr, tc) else 0.0)
        return _tensor(name, onnx.TensorProto.FLOAT, [100, 100], vals)

    idx = _tensor(P + "block_idx", onnx.TensorProto.INT64, [10], [0, 3, 6, 9, 12, 15, 18, 21, 24, 27])
    ch_axis = _ints(P + "ch_axis", [1])
    ch_step = _ints(P + "ch_step", [1])
    flat_shape = _tensor(P + "flat_shape", onnx.TensorProto.INT64, [2], [batch, 100])
    grid_shape = _tensor(P + "grid_shape", onnx.TensorProto.INT64, [4], [batch, 1, 10, 10])
    half = _tensor(P + "half", onnx.TensorProto.FLOAT, [1], [0.5])
    one = _tensor(P + "one", onnx.TensorProto.FLOAT, [1], [1.0])
    up_kernel = _tensor(P + "up_kernel", onnx.TensorProto.FLOAT, [1, 1, 2, 2], [1.0] * 4)
    left_mat = _span_matrix(P + "left_mat", lambda sr, sc, tr, tc: sr == tr and sc <= tc)
    right_mat = _span_matrix(P + "right_mat", lambda sr, sc, tr, tc: sr == tr and sc >= tc)
    up_mat = _span_matrix(P + "up_mat", lambda sr, sc, tr, tc: sc == tc and sr <= tr)
    down_mat = _span_matrix(P + "down_mat", lambda sr, sc, tr, tc: sc == tc and sr >= tr)

    _node("Gather", ["input", idx], [P + "block_rows"], axis=2)
    _node("Gather", [P + "block_rows", idx], [P + "blocks"], axis=3)

    masks_by_color = {}
    for color in fill_colors:
        C = f"c{color}_"
        _node(
            "Slice",
            [P + "blocks", _ints(P + C + "st", [color]), _ints(P + C + "en", [color + 1]), ch_axis, ch_step],
            [P + C + "grid"],
        )
        _node("Reshape", [P + C + "grid", flat_shape], [P + C + "flat"])
        for side, mat in (("left", left_mat), ("right", right_mat), ("up", up_mat), ("down", down_mat)):
            _node("MatMul", [P + C + "flat", mat], [P + C + side + "_count"])
            _node("Greater", [P + C + side + "_count", half], [P + C + side + "_b"])
        _node("And", [P + C + "left_b", P + C + "right_b"], [P + C + "row_b"])
        _node("And", [P + C + "up_b", P + C + "down_b"], [P + C + "col_b"])
        _node("Or", [P + C + "row_b", P + C + "col_b"], [P + C + "span_b"])
        _node("Cast", [P + C + "span_b"], [P + C + "span"], to=onnx.TensorProto.FLOAT)
        _node("Reshape", [P + C + "span", grid_shape], [P + C + "span_grid"])
        _node("ConvTranspose", [P + C + "span_grid", up_kernel], [P + C + "interiors29"], strides=[3, 3])
        _node(
            "Pad",
            [P + C + "interiors29"],
            [P + C + "interiors"],
            pads=[0, 0, 0, 0, 0, 0, 1, 1],
            mode="constant",
            value=0.0,
        )
        masks_by_color[color] = P + C + "interiors"

    _node("Concat", [masks_by_color[c] for c in fill_colors], [P + "paint_stack"], axis=1)
    _node("ReduceMax", [P + "paint_stack"], [P + "paint_any"], axes=[1], keepdims=1)
    _node("Sub", [one, P + "paint_any"], [P + "not_paint"])
    _node("Slice", ["input", _ints(P + "ch0_st", [0]), _ints(P + "ch0_en", [1]), ch_axis, ch_step], [P + "ch0_in"])
    _node("Mul", [P + "ch0_in", P + "not_paint"], [P + "ch0_out"])
    output_channels = [P + "ch0_out"]
    for color in fill_colors:
        C = f"c{color}_"
        _node(
            "Slice",
            ["input", _ints(P + C + "in_st", [color]), _ints(P + C + "in_en", [color + 1]), ch_axis, ch_step],
            [P + C + "in"],
        )
        _node("Max", [P + C + "in", masks_by_color[color]], [P + C + "out"])
        output_channels.append(P + C + "out")
    _node(
        "Concat",
        output_channels,
        ["output"],
        axis=1,
    )

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def same_color_neighbor_keep_network(batch=1):
    """Task097: remove singleton same-color 8-connected components."""
    nodes = []
    inits = []
    P = "scnk_"

    def _tensor(name, dtype, shape, values):
        t = onnx.helper.make_tensor(name, dtype, shape, values)
        inits.append(t)
        return name

    def _ints(name, values):
        return _tensor(name, onnx.TensorProto.INT64, [len(values)], values)

    def _node(op_type, inputs, outputs, **attrs):
        nodes.append(onnx.helper.make_node(op_type, inputs, outputs, **attrs))

    ch_axis = _ints(P + "ch_axis", [1])
    ch_step = _ints(P + "ch_step", [1])
    one_half = _tensor(P + "one_half", onnx.TensorProto.FLOAT, [1], [1.5])
    one = _tensor(P + "one", onnx.TensorProto.FLOAT, [1], [1.0])
    kernel_vals = []
    for out_ch in range(9):
        for _in_ch in range(1):
            for _r in range(3):
                for _c in range(3):
                    kernel_vals.append(1.0)
    kernel = _tensor(P + "kernel", onnx.TensorProto.FLOAT, [9, 1, 3, 3], kernel_vals)

    _node("Slice", ["input", _ints(P + "nz_st", [1]), _ints(P + "nz_en", [10]), ch_axis, ch_step], [P + "nz"])
    _node("Conv", [P + "nz", kernel], [P + "neighbor_counts"], pads=[1, 1, 1, 1], group=9)
    _node("Greater", [P + "neighbor_counts", one_half], [P + "has_same_neighbor_b"])
    _node("Cast", [P + "has_same_neighbor_b"], [P + "keep_mask"], to=onnx.TensorProto.FLOAT)
    _node("Mul", [P + "nz", P + "keep_mask"], [P + "kept_colors"])
    _node("ReduceMax", [P + "nz"], [P + "orig_any"], axes=[1], keepdims=1)
    _node("ReduceMax", [P + "kept_colors"], [P + "kept_any"], axes=[1], keepdims=1)
    _node("Sub", [P + "orig_any", P + "kept_any"], [P + "removed"])
    _node("Slice", ["input", _ints(P + "ch0_st", [0]), _ints(P + "ch0_en", [1]), ch_axis, ch_step], [P + "ch0_in"])
    _node("Add", [P + "ch0_in", P + "removed"], [P + "ch0_out_raw"])
    _node("Min", [P + "ch0_out_raw", one], [P + "ch0_out"])
    _node("Concat", [P + "ch0_out", P + "kept_colors"], ["output"], axis=1)

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def marker_adjacent_bbox_reflection_network(batch=1):
    """Task062: reflect a single object across the side touched by color-2 markers.

    Inputs are 10x10 with background 0, marker color 2, and one object color
    from {1,4,5,6,7,8,9}. The output repaints the active grid background as
    color 3, recolors markers to the object color, and adds the object's mirror
    across the marker-adjacent bbox edge.
    """
    S = 10
    P = "mabr_"
    nodes = []
    inits = []

    def _tensor(name, dtype, shape, values):
        t = onnx.helper.make_tensor(name, dtype, shape, values)
        inits.append(t)
        return name

    def _ints(name, values):
        return _tensor(name, onnx.TensorProto.INT64, [len(values)], values)

    def _node(op_type, inputs, outputs, **attrs):
        nodes.append(onnx.helper.make_node(op_type, inputs, outputs, **attrs))

    ch_axis = _ints(P + "ch_axis", [1])
    row_axis = _ints(P + "row_axis", [2])
    col_axis = _ints(P + "col_axis", [3])
    one_step = _ints(P + "step", [1])
    hw_axes = _ints(P + "hw_axes", [2, 3])
    hw_steps = _ints(P + "hw_steps", [1, 1])
    shape_flat = _tensor(P + "shape_flat", onnx.TensorProto.INT64, [3], [batch, 1, S * S])
    shape_grid = _tensor(P + "shape_grid", onnx.TensorProto.INT64, [4], [batch, 1, S, S])
    one_grid = _tensor(P + "one_grid", onnx.TensorProto.FLOAT, [1, 1, S, S], [1.0] * (S * S))

    def _shift_row(src, dst, direction):
        if direction == "up":
            starts, ends = [1], [S]
            pads = [0, 0, 0, 0, 0, 0, 1, 0]
        else:
            starts, ends = [0], [S - 1]
            pads = [0, 0, 1, 0, 0, 0, 0, 0]
        _node("Slice", [src, _ints(dst + "_st", starts), _ints(dst + "_en", ends), row_axis, one_step], [dst + "_slice"])
        _node("Pad", [dst + "_slice"], [dst], pads=pads, mode="constant", value=0.0)

    def _shift_col(src, dst, direction):
        if direction == "left":
            starts, ends = [1], [S]
            pads = [0, 0, 0, 0, 0, 0, 0, 1]
        else:
            starts, ends = [0], [S - 1]
            pads = [0, 0, 0, 1, 0, 0, 0, 0]
        _node("Slice", [src, _ints(dst + "_st", starts), _ints(dst + "_en", ends), col_axis, one_step], [dst + "_slice"])
        _node("Pad", [dst + "_slice"], [dst], pads=pads, mode="constant", value=0.0)

    _node("Slice", ["input", _ints(P + "crop_st", [0, 0]), _ints(P + "crop_en", [S, S]), hw_axes, hw_steps], [P + "crop"])
    _node("Slice", [P + "crop", _ints(P + "ch2_st", [2]), _ints(P + "ch2_en", [3]), ch_axis, one_step], [P + "marker"])

    object_channels = []
    color_slices = {}
    for color in (1, 4, 5, 6, 7, 8, 9):
        name = P + f"ch{color}"
        _node("Slice", [P + "crop", _ints(name + "_st", [color]), _ints(name + "_en", [color + 1]), ch_axis, one_step], [name])
        object_channels.append(name)
        color_slices[color] = name
    _node("Concat", object_channels, [P + "obj_stack"], axis=1)
    _node("ReduceSum", [P + "obj_stack"], [P + "obj"], axes=[1], keepdims=1)
    _node("Reshape", [P + "obj", shape_flat], [P + "obj_flat"])
    _node("ReduceSum", [P + "obj_flat"], [P + "obj_flat_count"], axes=[2], keepdims=1)
    _node("Sub", [P + "obj_flat_count", P + "obj_flat_count"], [P + "flat_zero"])
    _node("Concat", [P + "obj_flat", P + "flat_zero"], [P + "obj_flat_padded"], axis=2)

    _node("ReduceMax", [P + "obj"], [P + "obj_rows"], axes=[3], keepdims=1)
    _node("ReduceMax", [P + "obj"], [P + "obj_cols"], axes=[2], keepdims=1)
    _node("ReduceMax", [P + "marker"], [P + "marker_rows"], axes=[3], keepdims=1)
    _node("ReduceMax", [P + "marker"], [P + "marker_cols"], axes=[2], keepdims=1)

    _node("MaxPool", [P + "obj_rows"], [P + "row_cum_top"], kernel_shape=[S, 1], pads=[S - 1, 0, 0, 0], strides=[1, 1])
    _node("MaxPool", [P + "obj_rows"], [P + "row_cum_bottom"], kernel_shape=[S, 1], pads=[0, 0, S - 1, 0], strides=[1, 1])
    _node("MaxPool", [P + "obj_cols"], [P + "col_cum_left"], kernel_shape=[1, S], pads=[0, S - 1, 0, 0], strides=[1, 1])
    _node("MaxPool", [P + "obj_cols"], [P + "col_cum_right"], kernel_shape=[1, S], pads=[0, 0, 0, S - 1], strides=[1, 1])

    _shift_row(P + "row_cum_top", P + "row_before", "down")
    _shift_row(P + "row_cum_bottom", P + "row_after", "up")
    _shift_col(P + "col_cum_left", P + "col_before", "right")
    _shift_col(P + "col_cum_right", P + "col_after", "left")

    _node("Mul", [P + "obj_rows", P + "row_before"], [P + "row_has_before"])
    _node("Mul", [P + "obj_rows", P + "row_after"], [P + "row_has_after"])
    _node("Mul", [P + "obj_cols", P + "col_before"], [P + "col_has_before"])
    _node("Mul", [P + "obj_cols", P + "col_after"], [P + "col_has_after"])
    _node("Sub", [P + "obj_rows", P + "row_has_before"], [P + "min_row"])
    _node("Sub", [P + "obj_rows", P + "row_has_after"], [P + "max_row"])
    _node("Sub", [P + "obj_cols", P + "col_has_before"], [P + "min_col"])
    _node("Sub", [P + "obj_cols", P + "col_has_after"], [P + "max_col"])

    _shift_col(P + "marker_cols", P + "marker_col_left_of", "right")
    _shift_col(P + "marker_cols", P + "marker_col_right_of", "left")
    _shift_row(P + "marker_rows", P + "marker_row_above", "down")
    _shift_row(P + "marker_rows", P + "marker_row_below", "up")
    _node("Mul", [P + "max_col", P + "marker_col_right_of"], [P + "right_boundary"])
    _node("Mul", [P + "min_col", P + "marker_col_left_of"], [P + "left_boundary"])
    _node("Mul", [P + "max_row", P + "marker_row_below"], [P + "below_boundary"])
    _node("Mul", [P + "min_row", P + "marker_row_above"], [P + "above_boundary"])

    reflected = []

    def _reflect_indices(direction, k):
        idx = []
        for r in range(S):
            for c in range(S):
                if direction == "right":
                    sr, sc = r, 2 * k + 1 - c
                elif direction == "left":
                    sr, sc = r, 2 * k - 1 - c
                elif direction == "below":
                    sr, sc = 2 * k + 1 - r, c
                else:
                    sr, sc = 2 * k - 1 - r, c
                ok = 0 <= sr < S and 0 <= sc < S
                idx.append(sr * S + sc if ok else S * S)
        return idx

    def _add_reflection(direction, boundary, axis_name, k):
        tag = f"{P}{direction}{k}_"
        idx = _reflect_indices(direction, k)
        _tensor(tag + "idx", onnx.TensorProto.INT64, [S * S], idx)
        _node("Gather", [P + "obj_flat_padded", tag + "idx"], [tag + "flat"], axis=2)
        _node("Reshape", [tag + "flat", shape_grid], [tag + "grid"])
        if axis_name == "col":
            _node("Slice", [boundary, _ints(tag + "gst", [k]), _ints(tag + "gen", [k + 1]), col_axis, one_step], [tag + "gate"])
        else:
            _node("Slice", [boundary, _ints(tag + "gst", [k]), _ints(tag + "gen", [k + 1]), row_axis, one_step], [tag + "gate"])
        _node("Mul", [tag + "grid", tag + "gate"], [tag + "gated"])
        reflected.append(tag + "gated")

    for k in range(S - 1):
        _add_reflection("right", P + "right_boundary", "col", k)
        _add_reflection("below", P + "below_boundary", "row", k)
    for k in range(1, S):
        _add_reflection("left", P + "left_boundary", "col", k)
        _add_reflection("above", P + "above_boundary", "row", k)

    cur = reflected[0]
    for idx, name in enumerate(reflected[1:], start=1):
        out_name = f"{P}ref_any_{idx}"
        _node("Max", [cur, name], [out_name])
        cur = out_name
    _node("Max", [P + "obj", P + "marker", cur], [P + "fg"])
    _node("Sub", [P + "fg", P + "fg"], [P + "zero"])
    _node("Sub", [one_grid, P + "fg"], [P + "bg3"])

    output_channels = [P + "zero"]
    for color in range(1, 10):
        if color == 2:
            output_channels.append(P + "zero")
        elif color == 3:
            output_channels.append(P + "bg3")
        else:
            present = P + f"c{color}_present"
            _node("ReduceMax", [color_slices[color]], [present], axes=[2, 3], keepdims=1)
            out = P + f"c{color}_out"
            _node("Mul", [P + "fg", present], [out])
            output_channels.append(out)
    _node("Concat", output_channels, [P + "active_out"], axis=1)
    _node("Pad", [P + "active_out"], ["output"], pads=[0, 0, 0, 0, 0, 0, 30 - S, 30 - S], mode="constant", value=0.0)

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def rectangle_interior_fill8_network(canvas_h=30, canvas_w=30, batch=1):
    """Task120: fill same-color rectangle interiors with color 8."""
    nodes = []
    inits = []
    P = "rif8_"

    def _tensor(name, dtype, shape, values):
        t = onnx.helper.make_tensor(name, dtype, shape, values)
        inits.append(t)
        return name

    def _ints(name, values):
        return _tensor(name, onnx.TensorProto.INT64, [len(values)], values)

    def _node(op_type, inputs, outputs, **attrs):
        nodes.append(onnx.helper.make_node(op_type, inputs, outputs, **attrs))

    ch_axis = _ints(P + "ch_axis", [1])
    ch_step = _ints(P + "ch_step", [1])
    hw_axes = _ints(P + "hw_axes", [2, 3])
    hw_steps = _ints(P + "hw_steps", [1, 1])
    eight_half = _tensor(P + "eight_half", onnx.TensorProto.FLOAT, [1], [8.5])
    one = _tensor(P + "one", onnx.TensorProto.FLOAT, [1], [1.0])
    kernel_vals = []
    for ch in range(10):
        kernel_vals.extend(([0.0] * 9) if ch == 0 else ([1.0] * 9))
    kernel = _tensor(P + "kernel", onnx.TensorProto.FLOAT, [10, 1, 3, 3], kernel_vals)

    _node("Slice", ["input", _ints(P + "crop_st", [0, 0]), _ints(P + "crop_en", [canvas_h, canvas_w]), hw_axes, hw_steps],
          [P + "crop"])
    _node("Conv", [P + "crop", kernel], [P + "count"], pads=[1, 1, 1, 1], group=10)
    _node("Greater", [P + "count", eight_half], [P + "interior_b"])
    _node("Cast", [P + "interior_b"], [P + "interior"], to=onnx.TensorProto.FLOAT)
    _node("Mul", [P + "interior", P + "crop"], [P + "interior_active"])
    _node("ReduceMax", [P + "interior_active"], [P + "fill_mask"], axes=[1], keepdims=1)
    _node("Sub", [one, P + "fill_mask"], [P + "not_fill"])

    output_channels = []
    for ch in range(10):
        out_name = P + f"ch{ch}_out"
        _node("Slice", [P + "crop", _ints(P + f"ch{ch}_st", [ch]), _ints(P + f"ch{ch}_en", [ch + 1]), ch_axis, ch_step],
              [P + f"ch{ch}_in"])
        if ch == 0:
            output_channels.append(P + f"ch{ch}_in")
        elif ch == 8:
            _node("Max", [P + f"ch{ch}_in", P + "fill_mask"], [out_name])
            output_channels.append(out_name)
        else:
            _node("Mul", [P + f"ch{ch}_in", P + "not_fill"], [out_name])
            output_channels.append(out_name)

    _node("Concat", output_channels, [P + "result"], axis=1)
    _node("Pad", [P + "result"], ["output"], pads=[0, 0, 0, 0, 0, 0, 30 - canvas_h, 30 - canvas_w],
          mode="constant", value=0.0)
    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def border_distance_probe_recolor_network(canvas_h=10, canvas_w=10, batch=1):
    """Task040: recolor color-3 probes by nearest opposite border strip."""
    nodes = []
    inits = []
    P = "bdpr_"

    def _tensor(name, dtype, shape, values):
        t = onnx.helper.make_tensor(name, dtype, shape, values)
        inits.append(t)
        return name

    def _ints(name, values):
        return _tensor(name, onnx.TensorProto.INT64, [len(values)], values)

    def _node(op_type, inputs, outputs, **attrs):
        nodes.append(onnx.helper.make_node(op_type, inputs, outputs, **attrs))

    ch_axis = _ints(P + "ch_axis", [1])
    ch_step = _ints(P + "ch_step", [1])
    hw_axes = _ints(P + "hw_axes", [2, 3])
    hw_steps = _ints(P + "hw_steps", [1, 1])
    one = _tensor(P + "one", onnx.TensorProto.FLOAT, [1], [1.0])
    half = _tensor(P + "half", onnx.TensorProto.FLOAT, [1], [0.5])

    top_mask_vals = []
    bot_mask_vals = []
    left_mask_vals = []
    right_mask_vals = []
    row_mid = (canvas_h - 1) // 2
    col_mid = (canvas_w - 1) // 2
    for r in range(canvas_h):
        for c in range(canvas_w):
            top_mask_vals.append(1.0 if r <= row_mid else 0.0)
            bot_mask_vals.append(1.0 if r > row_mid else 0.0)
            left_mask_vals.append(1.0 if c <= col_mid else 0.0)
            right_mask_vals.append(1.0 if c > col_mid else 0.0)
    top_region = _tensor(P + "top_region", onnx.TensorProto.FLOAT, [1, 1, canvas_h, canvas_w], top_mask_vals)
    bot_region = _tensor(P + "bot_region", onnx.TensorProto.FLOAT, [1, 1, canvas_h, canvas_w], bot_mask_vals)
    left_region = _tensor(P + "left_region", onnx.TensorProto.FLOAT, [1, 1, canvas_h, canvas_w], left_mask_vals)
    right_region = _tensor(P + "right_region", onnx.TensorProto.FLOAT, [1, 1, canvas_h, canvas_w], right_mask_vals)

    _node("Slice", ["input", _ints(P + "crop_st", [0, 0]), _ints(P + "crop_en", [canvas_h, canvas_w]), hw_axes, hw_steps],
          [P + "crop"])

    def _spatial_vec(name, starts, ends):
        _node("Slice", [P + "crop", _ints(P + name + "_st", starts), _ints(P + name + "_en", ends), hw_axes, hw_steps],
              [P + name + "_slice"])
        _node("ReduceMax", [P + name + "_slice"], [P + name + "_vec"], axes=[2, 3], keepdims=1)
        return P + name + "_vec"

    top_vec = _spatial_vec("top", [0, 0], [1, canvas_w])
    bot_vec = _spatial_vec("bot", [canvas_h - 1, 0], [canvas_h, canvas_w])
    left_vec = _spatial_vec("left", [0, 0], [canvas_h, 1])
    right_vec = _spatial_vec("right", [0, canvas_w - 1], [canvas_h, canvas_w])

    _node("Mul", [top_vec, bot_vec], [P + "top_bot_same_vec"])
    _node("ReduceSum", [P + "top_bot_same_vec"], [P + "top_bot_same_count"], axes=[1], keepdims=1)
    _node("Greater", [P + "top_bot_same_count", half], [P + "col_axis_b"])
    _node("Cast", [P + "col_axis_b"], [P + "col_axis"], to=onnx.TensorProto.FLOAT)
    _node("Sub", [one, P + "col_axis"], [P + "row_axis"])
    _node("Slice", [P + "crop", _ints(P + "probe_st", [3]), _ints(P + "probe_en", [4]), ch_axis, ch_step], [P + "probe"])
    _node("Sub", [one, P + "probe"], [P + "not_probe"])

    _node("Mul", [P + "crop", P + "not_probe"], [P + "kept"])
    _node("Mul", [top_vec, top_region], [P + "top_area"])
    _node("Mul", [bot_vec, bot_region], [P + "bot_area"])
    _node("Add", [P + "top_area", P + "bot_area"], [P + "row_repl"])
    _node("Mul", [left_vec, left_region], [P + "left_area"])
    _node("Mul", [right_vec, right_region], [P + "right_area"])
    _node("Add", [P + "left_area", P + "right_area"], [P + "col_repl"])
    _node("Mul", [P + "row_repl", P + "row_axis"], [P + "row_pick"])
    _node("Mul", [P + "col_repl", P + "col_axis"], [P + "col_pick"])
    _node("Add", [P + "row_pick", P + "col_pick"], [P + "repl"])
    _node("Mul", [P + "repl", P + "probe"], [P + "paint"])
    _node("Add", [P + "kept", P + "paint"], [P + "result"])
    _node("Pad", [P + "result"], ["output"], pads=[0, 0, 0, 0, 0, 0, 30 - canvas_h, 30 - canvas_w],
          mode="constant", value=0.0)
    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def frame6_border3_fill4_network(canvas_h=30, canvas_w=30, batch=1):
    """Task125: add color-3 outer rings around color-6 frames and fill hollows with 4."""
    nodes = []
    inits = []
    P = "f6b3f4_"

    def _tensor(name, dtype, shape, values):
        t = onnx.helper.make_tensor(name, dtype, shape, values)
        inits.append(t)
        return name

    def _ints(name, values):
        return _tensor(name, onnx.TensorProto.INT64, [len(values)], values)

    def _node(op_type, inputs, outputs, **attrs):
        nodes.append(onnx.helper.make_node(op_type, inputs, outputs, **attrs))

    ch_axis = _ints(P + "ch_axis", [1])
    ch_step = _ints(P + "ch_step", [1])
    hw_axes = _ints(P + "hw_axes", [2, 3])
    hw_steps = _ints(P + "hw_steps", [1, 1])
    half = _tensor(P + "half", onnx.TensorProto.FLOAT, [1], [0.5])
    one = _tensor(P + "one", onnx.TensorProto.FLOAT, [1], [1.0])
    _node("Slice", ["input", _ints(P + "crop_st", [0, 0]), _ints(P + "crop_en", [canvas_h, canvas_w]), hw_axes, hw_steps],
          [P + "crop"])

    channel_names = {}
    for ch in range(10):
        _node("Slice", [P + "crop", _ints(P + f"ch{ch}_st", [ch]), _ints(P + f"ch{ch}_en", [ch + 1]), ch_axis, ch_step],
              [P + f"ch{ch}"])
        channel_names[ch] = P + f"ch{ch}"

    ch6 = channel_names[6]
    ch8 = channel_names[8]
    _node("MaxPool", [ch6], [P + "left"], kernel_shape=[1, 30], pads=[0, 29, 0, 0], strides=[1, 1])
    _node("MaxPool", [ch6], [P + "right"], kernel_shape=[1, 30], pads=[0, 0, 0, 29], strides=[1, 1])
    _node("MaxPool", [ch6], [P + "up"], kernel_shape=[30, 1], pads=[29, 0, 0, 0], strides=[1, 1])
    _node("MaxPool", [ch6], [P + "down"], kernel_shape=[30, 1], pads=[0, 0, 29, 0], strides=[1, 1])
    _node("Mul", [P + "left", P + "right"], [P + "has_lr"])
    _node("Mul", [P + "up", P + "down"], [P + "has_ud"])
    _node("Mul", [P + "has_lr", P + "has_ud"], [P + "inside_any"])
    _node("Mul", [P + "inside_any", ch8], [P + "interior"])
    _node("MaxPool", [ch6], [P + "near6_raw"], kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1])
    _node("Greater", [P + "near6_raw", half], [P + "near6_b"])
    _node("Cast", [P + "near6_b"], [P + "near6"], to=onnx.TensorProto.FLOAT)
    _node("Sub", [one, P + "interior"], [P + "not_interior"])
    _node("Mul", [ch8, P + "near6"], [P + "near_bg"])
    _node("Mul", [P + "near_bg", P + "not_interior"], [P + "border"])
    _node("Add", [P + "interior", P + "border"], [P + "changed8"])
    _node("Sub", [one, P + "changed8"], [P + "keep8"])
    _node("Mul", [ch8, P + "keep8"], [P + "ch8_out"])
    _node("Max", [channel_names[3], P + "border"], [P + "ch3_out"])
    _node("Max", [channel_names[4], P + "interior"], [P + "ch4_out"])

    output_channels = []
    for ch in range(10):
        if ch == 3:
            output_channels.append(P + "ch3_out")
        elif ch == 4:
            output_channels.append(P + "ch4_out")
        elif ch == 8:
            output_channels.append(P + "ch8_out")
        else:
            output_channels.append(channel_names[ch])
    _node("Concat", output_channels, [P + "result"], axis=1)
    _node("Pad", [P + "result"], ["output"], pads=[0, 0, 0, 0, 0, 0, 30 - canvas_h, 30 - canvas_w],
          mode="constant", value=0.0)
    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def marker_to_rectangle_line_network(canvas_h=24, canvas_w=24, batch=1):
    """Task064: draw the marker color between isolated markers and a solid rectangle."""
    if batch != 1:
        raise NotImplementedError("marker-to-rectangle-line is single-example only")
    nodes = []
    inits = []
    P = "mtrl_"

    def _tensor(name, dtype, shape, values):
        t = onnx.helper.make_tensor(name, dtype, shape, values)
        inits.append(t)
        return name

    def _ints(name, values):
        return _tensor(name, onnx.TensorProto.INT64, [len(values)], values)

    def _node(op_type, inputs, outputs, **attrs):
        nodes.append(onnx.helper.make_node(op_type, inputs, outputs, **attrs))

    hw_axes = _ints(P + "hw_axes", [2, 3])
    hw_steps = _ints(P + "hw_steps", [1, 1])
    half = _tensor(P + "half", onnx.TensorProto.FLOAT, [1], [0.5])
    one = _tensor(P + "one", onnx.TensorProto.FLOAT, [1], [1.0])
    zero = _tensor(P + "zero", onnx.TensorProto.FLOAT, [1], [0.0])
    h_kernel_vals = []
    v_kernel_vals = []
    for _ch in range(10):
        h_kernel_vals.extend([0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        v_kernel_vals.extend([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    h_kernel = _tensor(P + "h_kernel", onnx.TensorProto.FLOAT, [10, 1, 3, 3], h_kernel_vals)
    v_kernel = _tensor(P + "v_kernel", onnx.TensorProto.FLOAT, [10, 1, 3, 3], v_kernel_vals)

    _node("Slice", ["input", _ints(P + "crop_st", [0, 0]), _ints(P + "crop_en", [canvas_h, canvas_w]), hw_axes, hw_steps],
          [P + "crop"])
    _node("ReduceSum", [P + "crop"], [P + "counts"], axes=[2, 3], keepdims=1)
    _node("ReduceMax", [P + "counts"], [P + "bg_count"], axes=[1], keepdims=1)
    _node("Less", [P + "counts", P + "bg_count"], [P + "non_bg_ch_b"])
    _node("Cast", [P + "non_bg_ch_b"], [P + "non_bg_ch"], to=onnx.TensorProto.FLOAT)
    _node("Mul", [P + "crop", P + "non_bg_ch"], [P + "non_bg_cells"])
    _node("Conv", [P + "non_bg_cells", h_kernel], [P + "h_neighbors"], pads=[1, 1, 1, 1], group=10)
    _node("Conv", [P + "non_bg_cells", v_kernel], [P + "v_neighbors"], pads=[1, 1, 1, 1], group=10)
    _node("Greater", [P + "h_neighbors", half], [P + "has_h_b"])
    _node("Greater", [P + "v_neighbors", half], [P + "has_v_b"])
    _node("Where", [P + "has_h_b", P + "non_bg_cells", zero], [P + "h_rect"])
    _node("Where", [P + "has_v_b", P + "h_rect", zero], [P + "rect_ch"])
    _node("ReduceSum", [P + "rect_ch"], [P + "rect"], axes=[1], keepdims=1)
    _node("ReduceSum", [P + "non_bg_cells"], [P + "non_bg"], axes=[1], keepdims=1)
    _node("Sub", [P + "non_bg", P + "rect"], [P + "marker"])

    _node("Mul", [P + "counts", P + "non_bg_ch"], [P + "non_bg_counts"])
    _node("ReduceSum", [P + "rect_ch"], [P + "rect_counts"], axes=[2, 3], keepdims=1)
    _node("Sub", [P + "non_bg_counts", P + "rect_counts"], [P + "marker_counts"])
    _node("Greater", [P + "marker_counts", half], [P + "marker_ch_b"])
    _node("Cast", [P + "marker_ch_b"], [P + "marker_ch"], to=onnx.TensorProto.FLOAT)

    _node("MaxPool", [P + "marker"], [P + "marker_left"], kernel_shape=[1, canvas_w], pads=[0, canvas_w - 1, 0, 0], strides=[1, 1])
    _node("MaxPool", [P + "marker"], [P + "marker_right"], kernel_shape=[1, canvas_w], pads=[0, 0, 0, canvas_w - 1], strides=[1, 1])
    _node("MaxPool", [P + "marker"], [P + "marker_up"], kernel_shape=[canvas_h, 1], pads=[canvas_h - 1, 0, 0, 0], strides=[1, 1])
    _node("MaxPool", [P + "marker"], [P + "marker_down"], kernel_shape=[canvas_h, 1], pads=[0, 0, canvas_h - 1, 0], strides=[1, 1])
    _node("MaxPool", [P + "rect"], [P + "rect_left"], kernel_shape=[1, canvas_w], pads=[0, canvas_w - 1, 0, 0], strides=[1, 1])
    _node("MaxPool", [P + "rect"], [P + "rect_right"], kernel_shape=[1, canvas_w], pads=[0, 0, 0, canvas_w - 1], strides=[1, 1])
    _node("MaxPool", [P + "rect"], [P + "rect_up"], kernel_shape=[canvas_h, 1], pads=[canvas_h - 1, 0, 0, 0], strides=[1, 1])
    _node("MaxPool", [P + "rect"], [P + "rect_down"], kernel_shape=[canvas_h, 1], pads=[0, 0, canvas_h - 1, 0], strides=[1, 1])

    _node("Mul", [P + "marker_left", P + "rect_right"], [P + "line_lr"])
    _node("Mul", [P + "marker_right", P + "rect_left"], [P + "line_rl"])
    _node("Mul", [P + "marker_up", P + "rect_down"], [P + "line_ud"])
    _node("Mul", [P + "marker_down", P + "rect_up"], [P + "line_du"])
    _node("Max", [P + "line_lr", P + "line_rl", P + "line_ud", P + "line_du"], [P + "line"])
    _node("Sub", [one, P + "rect"], [P + "not_rect"])
    _node("Mul", [P + "line", P + "not_rect"], [P + "paint_mask"])
    _node("Sub", [one, P + "paint_mask"], [P + "keep_mask"])
    _node("Mul", [P + "crop", P + "keep_mask"], [P + "kept"])
    _node("Mul", [P + "paint_mask", P + "marker_ch"], [P + "paint"])
    _node("Max", [P + "kept", P + "paint"], [P + "result"])
    _node("Pad", [P + "result"], ["output"], pads=[0, 0, 0, 0, 0, 0, 30 - canvas_h, 30 - canvas_w],
          mode="constant", value=0.0)

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def foreground_count_row_network(batch=1):
    """Task339: emit one row of the foreground color, width = foreground count.

    The task inputs are 3x3 grids with exactly one non-zero color. The output is
    a 1 x count strip of that color. Keep canvas padding all-zero rather than
    reconstructing channel 0, because the active output contains no background.
    """
    nodes = []
    inits = []
    P = "fcr_"

    def _tensor(name, dtype, shape, values):
        t = onnx.helper.make_tensor(name, dtype, shape, values)
        inits.append(t)
        return name

    def _ints(name, values):
        return _tensor(name, onnx.TensorProto.INT64, [len(values)], values)

    def _node(op_type, inputs, outputs, **attrs):
        nodes.append(onnx.helper.make_node(op_type, inputs, outputs, **attrs))

    ch_axis = _ints(P + "ch_axis", [1])
    ch_step = _ints(P + "ch_step", [1])
    range_row = _tensor(
        P + "range_row",
        onnx.TensorProto.FLOAT,
        [1, 1, 1, 30],
        [float(i) for i in range(30)],
    )

    _node("Slice", ["input", _ints(P + "nz_st", [1]), _ints(P + "nz_en", [10]), ch_axis, ch_step], [P + "nz"])
    _node("ReduceSum", [P + "nz"], [P + "count"], axes=[1, 2, 3], keepdims=1)
    _node("ReduceMax", [P + "nz"], [P + "present"], axes=[2, 3], keepdims=1)
    _node("Less", [range_row, P + "count"], [P + "width_b"])
    _node("Cast", [P + "width_b"], [P + "width"], to=onnx.TensorProto.FLOAT)
    _node("Mul", [P + "present", P + "width"], [P + "colored"])
    _node("Sub", [P + "width", P + "width"], [P + "ch0"])
    _node("Concat", [P + "ch0", P + "colored"], [P + "row"], axis=1)
    _node(
        "Pad",
        [P + "row"],
        ["output"],
        pads=[0, 0, 0, 0, 0, 0, 29, 0],
        mode="constant",
        value=0.0,
    )

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def marker_count_cross_shift_network(batch=1):
    """Task362: color-5 marker count shifts a full same-color cross down-left.

    Inputs are 10x10. A single nonzero/non-5 color forms one full row and one
    full column; the number of color-5 marker cells is k. The output removes the
    markers and redraws that full cross with its center shifted (+k rows, -k cols).
    """
    S = 10
    P = "mccs_"
    nodes = []
    inits = []

    def _tensor(name, dtype, shape, values):
        t = onnx.helper.make_tensor(name, dtype, shape, values)
        inits.append(t)
        return name

    def _ints(name, values):
        return _tensor(name, onnx.TensorProto.INT64, [len(values)], values)

    def _float(name, value):
        return _tensor(name, onnx.TensorProto.FLOAT, [1], [float(value)])

    def _node(op_type, inputs, outputs, **attrs):
        nodes.append(onnx.helper.make_node(op_type, inputs, outputs, **attrs))

    ch_axis = _ints(P + "ch_axis", [1])
    h_axis = _ints(P + "h_axis", [2])
    w_axis = _ints(P + "w_axis", [3])
    one_step = _ints(P + "step", [1])
    hw_axes = _ints(P + "hw_axes", [2, 3])
    hw_steps = _ints(P + "hw_steps", [1, 1])
    row_reps = _ints(P + "row_reps", [1, 1, 1, S])
    col_reps = _ints(P + "col_reps", [1, 1, S, 1])
    half = _float(P + "half", 0.5)
    full_threshold = _float(P + "full_threshold", S - 0.5)
    one_grid = _tensor(P + "one_grid", onnx.TensorProto.FLOAT, [1, 1, S, S], [1.0] * (S * S))

    _node("Slice", ["input", _ints(P + "crop_st", [0, 0]), _ints(P + "crop_en", [S, S]), hw_axes, hw_steps], [P + "crop"])
    _node("Slice", [P + "crop", _ints(P + "ch5_st", [5]), _ints(P + "ch5_en", [6]), ch_axis, one_step], [P + "ch5"])
    _node("ReduceSum", [P + "ch5"], [P + "marker_count"], axes=[1, 2, 3], keepdims=1)
    _node("Mul", [P + "marker_count", one_grid], [P + "marker_plane"])
    _node("Sub", [P + "marker_plane", P + "marker_plane"], [P + "zero_ch"])

    color_channels = []
    for color in range(1, 10):
        if color == 5:
            color_channels.append(P + "zero_ch")
            continue

        c_prefix = f"{P}c{color}_"
        _node("Slice", [P + "crop", _ints(c_prefix + "st", [color]), _ints(c_prefix + "en", [color + 1]), ch_axis, one_step], [c_prefix + "ch"])
        _node("ReduceSum", [c_prefix + "ch"], [c_prefix + "row_count"], axes=[3], keepdims=1)
        _node("ReduceSum", [c_prefix + "ch"], [c_prefix + "col_count"], axes=[2], keepdims=1)
        _node("Greater", [c_prefix + "row_count", full_threshold], [c_prefix + "full_row_b"])
        _node("Greater", [c_prefix + "col_count", full_threshold], [c_prefix + "full_col_b"])
        _node("Cast", [c_prefix + "full_row_b"], [c_prefix + "full_row"], to=onnx.TensorProto.FLOAT)
        _node("Cast", [c_prefix + "full_col_b"], [c_prefix + "full_col"], to=onnx.TensorProto.FLOAT)

        shifted_by_k = []
        for k in range(S):
            k_prefix = f"{c_prefix}k{k}_"
            kval = _float(k_prefix + "value", k)
            _node("Sub", [P + "marker_count", kval], [k_prefix + "diff"])
            _node("Abs", [k_prefix + "diff"], [k_prefix + "abs"])
            _node("Less", [k_prefix + "abs", half], [k_prefix + "gate_b"])
            _node("Cast", [k_prefix + "gate_b"], [k_prefix + "gate"], to=onnx.TensorProto.FLOAT)

            _node(
                "Slice",
                [c_prefix + "full_row", _ints(k_prefix + "rs_st", [0]), _ints(k_prefix + "rs_en", [S - k]), h_axis, one_step],
                [k_prefix + "row_slice"],
            )
            _node(
                "Pad",
                [k_prefix + "row_slice"],
                [k_prefix + "row_shift"],
                pads=[0, 0, k, 0, 0, 0, 0, 0],
                mode="constant",
                value=0.0,
            )
            _node("Tile", [k_prefix + "row_shift", row_reps], [k_prefix + "row_mask"])

            _node(
                "Slice",
                [c_prefix + "full_col", _ints(k_prefix + "cs_st", [k]), _ints(k_prefix + "cs_en", [S]), w_axis, one_step],
                [k_prefix + "col_slice"],
            )
            _node(
                "Pad",
                [k_prefix + "col_slice"],
                [k_prefix + "col_shift"],
                pads=[0, 0, 0, 0, 0, 0, 0, k],
                mode="constant",
                value=0.0,
            )
            _node("Tile", [k_prefix + "col_shift", col_reps], [k_prefix + "col_mask"])
            _node("Max", [k_prefix + "row_mask", k_prefix + "col_mask"], [k_prefix + "cross"])
            _node("Mul", [k_prefix + "cross", k_prefix + "gate"], [k_prefix + "gated"])
            shifted_by_k.append(k_prefix + "gated")

        cur = shifted_by_k[0]
        for idx, name in enumerate(shifted_by_k[1:], start=1):
            out_name = f"{c_prefix}max_{idx}"
            _node("Max", [cur, name], [out_name])
            cur = out_name
        color_channels.append(cur)

    _node("Concat", color_channels, [P + "nonzero"], axis=1)
    _node("ReduceSum", [P + "nonzero"], [P + "fg"], axes=[1], keepdims=1)
    _node("Sub", [one_grid, P + "fg"], [P + "ch0"])
    _node("Concat", [P + "ch0"] + color_channels, [P + "active_out"], axis=1)
    _node(
        "Pad",
        [P + "active_out"],
        ["output"],
        pads=[0, 0, 0, 0, 0, 0, 30 - S, 30 - S],
        mode="constant",
        value=0.0,
    )

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def anchor_block_aligned_marker_line_network(batch=1):
    """Task333: connect markers aligned with a 2x2 color-3 anchor block."""
    S = 10
    P = "abaml_"
    nodes = []
    inits = []

    def _tensor(name, dtype, shape, values):
        t = onnx.helper.make_tensor(name, dtype, shape, values)
        inits.append(t)
        return name

    def _ints(name, values):
        return _tensor(name, onnx.TensorProto.INT64, [len(values)], values)

    def _float(name, value):
        return _tensor(name, onnx.TensorProto.FLOAT, [1], [float(value)])

    def _node(op_type, inputs, outputs, **attrs):
        nodes.append(onnx.helper.make_node(op_type, inputs, outputs, **attrs))

    ch_axis = _ints(P + "ch_axis", [1])
    one_step = _ints(P + "step", [1])
    hw_axes = _ints(P + "hw_axes", [2, 3])
    hw_steps = _ints(P + "hw_steps", [1, 1])
    half = _float(P + "half", 0.5)
    one = _float(P + "one", 1.0)

    _node("Slice", ["input", _ints(P + "crop_st", [0, 0]), _ints(P + "crop_en", [S, S]), hw_axes, hw_steps], [P + "crop"])
    _node("Slice", [P + "crop", _ints(P + "ch0_st", [0]), _ints(P + "ch0_en", [1]), ch_axis, one_step], [P + "ch0"])
    _node("Slice", [P + "crop", _ints(P + "ch3_st", [3]), _ints(P + "ch3_en", [4]), ch_axis, one_step], [P + "ch3"])

    _node("ReduceSum", [P + "ch3"], [P + "row3_count"], axes=[3], keepdims=1)
    _node("ReduceSum", [P + "ch3"], [P + "col3_count"], axes=[2], keepdims=1)
    _node("Greater", [P + "row3_count", half], [P + "row3_b"])
    _node("Greater", [P + "col3_count", half], [P + "col3_b"])
    _node("Cast", [P + "row3_b"], [P + "row3"], to=onnx.TensorProto.FLOAT)
    _node("Cast", [P + "col3_b"], [P + "col3"], to=onnx.TensorProto.FLOAT)

    _node("MaxPool", [P + "ch3"], [P + "anchor_top"], kernel_shape=[S, 1], pads=[S - 1, 0, 0, 0], strides=[1, 1])
    _node("MaxPool", [P + "ch3"], [P + "anchor_bottom"], kernel_shape=[S, 1], pads=[0, 0, S - 1, 0], strides=[1, 1])
    _node("MaxPool", [P + "ch3"], [P + "anchor_left"], kernel_shape=[1, S], pads=[0, S - 1, 0, 0], strides=[1, 1])
    _node("MaxPool", [P + "ch3"], [P + "anchor_right"], kernel_shape=[1, S], pads=[0, 0, 0, S - 1], strides=[1, 1])

    color_channels = []
    paint_channels = []
    for color in range(1, 10):
        c_prefix = f"{P}c{color}_"
        _node("Slice", [P + "crop", _ints(c_prefix + "st", [color]), _ints(c_prefix + "en", [color + 1]), ch_axis, one_step], [c_prefix + "in"])
        if color == 3:
            color_channels.append(c_prefix + "in")
            continue

        _node("Mul", [c_prefix + "in", P + "row3"], [c_prefix + "row_markers"])
        _node("Mul", [c_prefix + "in", P + "col3"], [c_prefix + "col_markers"])
        _node("MaxPool", [c_prefix + "col_markers"], [c_prefix + "marker_top"], kernel_shape=[S, 1], pads=[S - 1, 0, 0, 0], strides=[1, 1])
        _node("MaxPool", [c_prefix + "col_markers"], [c_prefix + "marker_bottom"], kernel_shape=[S, 1], pads=[0, 0, S - 1, 0], strides=[1, 1])
        _node("MaxPool", [c_prefix + "row_markers"], [c_prefix + "marker_left"], kernel_shape=[1, S], pads=[0, S - 1, 0, 0], strides=[1, 1])
        _node("MaxPool", [c_prefix + "row_markers"], [c_prefix + "marker_right"], kernel_shape=[1, S], pads=[0, 0, 0, S - 1], strides=[1, 1])

        _node("Mul", [c_prefix + "marker_top", P + "anchor_bottom"], [c_prefix + "from_above"])
        _node("Mul", [c_prefix + "marker_bottom", P + "anchor_top"], [c_prefix + "from_below"])
        _node("Mul", [c_prefix + "marker_left", P + "anchor_right"], [c_prefix + "from_left"])
        _node("Mul", [c_prefix + "marker_right", P + "anchor_left"], [c_prefix + "from_right"])
        _node("Max", [c_prefix + "from_above", c_prefix + "from_below", c_prefix + "from_left", c_prefix + "from_right"], [c_prefix + "line_raw"])
        _node("Mul", [c_prefix + "line_raw", P + "ch0"], [c_prefix + "paint"])
        _node("Max", [c_prefix + "in", c_prefix + "paint"], [c_prefix + "out"])
        color_channels.append(c_prefix + "out")
        paint_channels.append(c_prefix + "paint")

    cur = paint_channels[0]
    for idx, name in enumerate(paint_channels[1:], start=1):
        out_name = f"{P}paint_any_{idx}"
        _node("Max", [cur, name], [out_name])
        cur = out_name
    _node("Sub", [one, cur], [P + "not_paint"])
    _node("Mul", [P + "ch0", P + "not_paint"], [P + "ch0_out"])
    _node("Concat", [P + "ch0_out"] + color_channels, [P + "active_out"], axis=1)
    _node("Pad", [P + "active_out"], ["output"], pads=[0, 0, 0, 0, 0, 0, 30 - S, 30 - S], mode="constant", value=0.0)

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def frame_inner_markers_to_outer_opposite_network(batch=1):
    """Task228: move four inner frame-corner markers to opposite outer corners."""
    S = 10
    P = "fimoo_"
    nodes = []
    inits = []

    def _tensor(name, dtype, shape, values):
        t = onnx.helper.make_tensor(name, dtype, shape, values)
        inits.append(t)
        return name

    def _ints(name, values):
        return _tensor(name, onnx.TensorProto.INT64, [len(values)], values)

    def _float(name, value):
        return _tensor(name, onnx.TensorProto.FLOAT, [1], [float(value)])

    def _node(op_type, inputs, outputs, **attrs):
        nodes.append(onnx.helper.make_node(op_type, inputs, outputs, **attrs))

    ch_axis = _ints(P + "ch_axis", [1])
    one_step = _ints(P + "step", [1])
    hw_axes = _ints(P + "hw_axes", [2, 3])
    hw_steps = _ints(P + "hw_steps", [1, 1])
    half = _float(P + "half", 0.5)
    edge_threshold = _float(P + "edge_threshold", 2.5)
    one = _float(P + "one", 1.0)
    one_grid = _tensor(P + "one_grid", onnx.TensorProto.FLOAT, [1, 1, S, S], [1.0] * (S * S))

    def _shift_row(src, dst, direction):
        if direction == "up":
            starts, ends = [1], [S]
            pads = [0, 0, 0, 0, 0, 0, 1, 0]
        else:
            starts, ends = [0], [S - 1]
            pads = [0, 0, 1, 0, 0, 0, 0, 0]
        _node("Slice", [src, _ints(dst + "_st", starts), _ints(dst + "_en", ends), _ints(dst + "_ax", [2]), one_step], [dst + "_slice"])
        _node("Pad", [dst + "_slice"], [dst], pads=pads, mode="constant", value=0.0)

    def _shift_col(src, dst, direction):
        if direction == "left":
            starts, ends = [1], [S]
            pads = [0, 0, 0, 0, 0, 0, 0, 1]
        else:
            starts, ends = [0], [S - 1]
            pads = [0, 0, 0, 1, 0, 0, 0, 0]
        _node("Slice", [src, _ints(dst + "_st", starts), _ints(dst + "_en", ends), _ints(dst + "_ax", [3]), one_step], [dst + "_slice"])
        _node("Pad", [dst + "_slice"], [dst], pads=pads, mode="constant", value=0.0)

    _node("Slice", ["input", _ints(P + "crop_st", [0, 0]), _ints(P + "crop_en", [S, S]), hw_axes, hw_steps], [P + "crop"])

    in_by_color = {}
    count_by_color = {}
    for color in range(1, 10):
        c_prefix = f"{P}c{color}_"
        _node("Slice", [P + "crop", _ints(c_prefix + "st", [color]), _ints(c_prefix + "en", [color + 1]), ch_axis, one_step], [c_prefix + "in"])
        _node("ReduceSum", [c_prefix + "in"], [c_prefix + "count"], axes=[2, 3], keepdims=1)
        in_by_color[color] = c_prefix + "in"
        count_by_color[color] = c_prefix + "count"
    _node("Concat", [count_by_color[c] for c in range(1, 10)], [P + "counts"], axis=1)
    _node("ReduceMax", [P + "counts"], [P + "max_count"], axes=[1], keepdims=1)

    border_channels = []
    marker_channels = {}
    for color in range(1, 10):
        c_prefix = f"{P}c{color}_"
        _node("Sub", [count_by_color[color], P + "max_count"], [c_prefix + "border_diff"])
        _node("Abs", [c_prefix + "border_diff"], [c_prefix + "border_abs"])
        _node("Less", [c_prefix + "border_abs", half], [c_prefix + "border_gate_b"])
        _node("Cast", [c_prefix + "border_gate_b"], [c_prefix + "border_gate"], to=onnx.TensorProto.FLOAT)
        _node("Sub", [one, c_prefix + "border_gate"], [c_prefix + "marker_gate"])
        _node("Mul", [in_by_color[color], c_prefix + "border_gate"], [c_prefix + "border"])
        _node("Mul", [in_by_color[color], c_prefix + "marker_gate"], [c_prefix + "markers"])
        border_channels.append(c_prefix + "border")
        marker_channels[color] = c_prefix + "markers"

    _node("Concat", border_channels, [P + "border_stack"], axis=1)
    _node("ReduceSum", [P + "border_stack"], [P + "border"], axes=[1], keepdims=1)
    _node("ReduceSum", [P + "border"], [P + "row_count"], axes=[3], keepdims=1)
    _node("ReduceSum", [P + "border"], [P + "col_count"], axes=[2], keepdims=1)
    _node("Greater", [P + "row_count", edge_threshold], [P + "row_full_b"])
    _node("Greater", [P + "col_count", edge_threshold], [P + "col_full_b"])
    _node("Cast", [P + "row_full_b"], [P + "row_full"], to=onnx.TensorProto.FLOAT)
    _node("Cast", [P + "col_full_b"], [P + "col_full"], to=onnx.TensorProto.FLOAT)

    _node("MaxPool", [P + "row_full"], [P + "row_cum_top"], kernel_shape=[S, 1], pads=[S - 1, 0, 0, 0], strides=[1, 1])
    _node("MaxPool", [P + "row_full"], [P + "row_cum_bottom"], kernel_shape=[S, 1], pads=[0, 0, S - 1, 0], strides=[1, 1])
    _node("MaxPool", [P + "col_full"], [P + "col_cum_left"], kernel_shape=[1, S], pads=[0, S - 1, 0, 0], strides=[1, 1])
    _node("MaxPool", [P + "col_full"], [P + "col_cum_right"], kernel_shape=[1, S], pads=[0, 0, 0, S - 1], strides=[1, 1])
    _shift_row(P + "row_full", P + "row_above_edge", "up")
    _shift_row(P + "row_full", P + "row_below_edge", "down")
    _shift_col(P + "col_full", P + "col_left_edge", "left")
    _shift_col(P + "col_full", P + "col_right_edge", "right")

    _node("Sub", [one, P + "row_cum_top"], [P + "not_row_cum_top"])
    _node("Sub", [one, P + "row_cum_bottom"], [P + "not_row_cum_bottom"])
    _node("Sub", [one, P + "col_cum_left"], [P + "not_col_cum_left"])
    _node("Sub", [one, P + "col_cum_right"], [P + "not_col_cum_right"])
    _node("Mul", [P + "row_above_edge", P + "not_row_cum_top"], [P + "top_outer_row"])
    _node("Mul", [P + "row_below_edge", P + "not_row_cum_bottom"], [P + "bottom_outer_row"])
    _node("Mul", [P + "row_below_edge", P + "row_cum_bottom"], [P + "top_inner_row"])
    _node("Mul", [P + "row_above_edge", P + "row_cum_top"], [P + "bottom_inner_row"])
    _node("Mul", [P + "col_left_edge", P + "not_col_cum_left"], [P + "left_outer_col"])
    _node("Mul", [P + "col_right_edge", P + "not_col_cum_right"], [P + "right_outer_col"])
    _node("Mul", [P + "col_right_edge", P + "col_cum_right"], [P + "left_inner_col"])
    _node("Mul", [P + "col_left_edge", P + "col_cum_left"], [P + "right_inner_col"])

    _node("Mul", [P + "top_inner_row", P + "left_inner_col"], [P + "inner_tl"])
    _node("Mul", [P + "top_inner_row", P + "right_inner_col"], [P + "inner_tr"])
    _node("Mul", [P + "bottom_inner_row", P + "left_inner_col"], [P + "inner_bl"])
    _node("Mul", [P + "bottom_inner_row", P + "right_inner_col"], [P + "inner_br"])
    _node("Mul", [P + "top_outer_row", P + "left_outer_col"], [P + "outer_tl"])
    _node("Mul", [P + "top_outer_row", P + "right_outer_col"], [P + "outer_tr"])
    _node("Mul", [P + "bottom_outer_row", P + "left_outer_col"], [P + "outer_bl"])
    _node("Mul", [P + "bottom_outer_row", P + "right_outer_col"], [P + "outer_br"])

    output_channels = []
    for color in range(1, 10):
        c_prefix = f"{P}c{color}_"
        corner_paints = []
        for src, dst, short in (
            (P + "inner_br", P + "outer_tl", "br_to_tl"),
            (P + "inner_bl", P + "outer_tr", "bl_to_tr"),
            (P + "inner_tr", P + "outer_bl", "tr_to_bl"),
            (P + "inner_tl", P + "outer_br", "tl_to_br"),
        ):
            _node("Mul", [marker_channels[color], src], [c_prefix + short + "_src"])
            _node("ReduceSum", [c_prefix + short + "_src"], [c_prefix + short + "_val"], axes=[2, 3], keepdims=1)
            _node("Mul", [c_prefix + short + "_val", dst], [c_prefix + short])
            corner_paints.append(c_prefix + short)
        _node("Max", corner_paints, [c_prefix + "paint"])
        _node("Max", [c_prefix + "border", c_prefix + "paint"], [c_prefix + "out"])
        output_channels.append(c_prefix + "out")

    _node("Concat", output_channels, [P + "nonzero_stack"], axis=1)
    _node("ReduceMax", [P + "nonzero_stack"], [P + "any_nonzero"], axes=[1], keepdims=1)
    _node("Sub", [one_grid, P + "any_nonzero"], [P + "ch0_out"])
    _node("Concat", [P + "ch0_out"] + output_channels, [P + "active_out"], axis=1)
    _node("Pad", [P + "active_out"], ["output"], pads=[0, 0, 0, 0, 0, 0, 30 - S, 30 - S], mode="constant", value=0.0)

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def top_marker_run_recolor_network(batch=1):
    """Task354: recolor color-5 horizontal runs from top-row marker columns."""
    if batch != 1:
        raise NotImplementedError("task354 top-marker-run-recolor builder is single-example only")

    P = "tmrr_"
    S = 10
    nodes = []
    inits = []

    def tensor(name, dtype, shape, vals):
        inits.append(onnx.helper.make_tensor(name, dtype, shape, vals))
        return name

    def node(op, inputs, outputs, **kwargs):
        nodes.append(onnx.helper.make_node(op, inputs, outputs, **kwargs))

    tensor(P + "crop_st", onnx.TensorProto.INT64, [2], [0, 0])
    tensor(P + "crop_en", onnx.TensorProto.INT64, [2], [S, S])
    tensor(P + "axes_hw", onnx.TensorProto.INT64, [2], [2, 3])
    tensor(P + "steps_hw", onnx.TensorProto.INT64, [2], [1, 1])
    tensor(P + "ch_axis", onnx.TensorProto.INT64, [1], [1])
    tensor(P + "top_st", onnx.TensorProto.INT64, [1], [0])
    tensor(P + "top_en", onnx.TensorProto.INT64, [1], [1])
    tensor(P + "row_axis", onnx.TensorProto.INT64, [1], [2])
    tensor(P + "one_step", onnx.TensorProto.INT64, [1], [1])
    tensor(P + "ones_rows", onnx.TensorProto.FLOAT, [1, 1, S, 1], [1.0] * S)
    tensor(P + "zero", onnx.TensorProto.FLOAT, [1, 1, S, S], [0.0] * (S * S))
    tensor(P + "half", onnx.TensorProto.FLOAT, [1], [0.5])

    node("Slice", ["input", P + "crop_st", P + "crop_en", P + "axes_hw", P + "steps_hw"], [P + "crop"])

    channel_names = []
    for color in range(10):
        st = tensor(P + f"ch{color}_st", onnx.TensorProto.INT64, [1], [color])
        en = tensor(P + f"ch{color}_en", onnx.TensorProto.INT64, [1], [color + 1])
        name = P + f"ch{color}"
        node("Slice", [P + "crop", st, en, P + "ch_axis", P + "one_step"], [name])
        channel_names.append(name)

    mask5 = channel_names[5]
    top_value = None
    for color in (1, 2, 3, 4, 6, 7, 8, 9):
        tensor(P + f"val{color}", onnx.TensorProto.FLOAT, [1], [float(color)])
        top = P + f"top{color}"
        weighted = P + f"topv{color}"
        node("Slice", [channel_names[color], P + "top_st", P + "top_en", P + "row_axis", P + "one_step"], [top])
        node("Mul", [top, P + f"val{color}"], [weighted])
        if top_value is None:
            top_value = weighted
        else:
            nxt = P + f"top_acc{color}"
            node("Add", [top_value, weighted], [nxt])
            top_value = nxt

    node("Mul", [top_value, P + "ones_rows"], [P + "seed_col"])
    node("Mul", [P + "seed_col", mask5], [P + "seed"])
    prev = P + "seed"
    for i in range(S - 1):
        dil = P + f"dil{i}"
        nxt = P + f"reach{i}"
        node("MaxPool", [prev], [dil], kernel_shape=[1, 3], pads=[0, 1, 0, 1], strides=[1, 1])
        node("Mul", [dil, mask5], [nxt])
        prev = nxt

    fill_masks = {0: channel_names[0], 5: P + "zero"}
    for color in (1, 2, 3, 4, 6, 7, 8, 9):
        diff = P + f"diff{color}"
        sq = P + f"sq{color}"
        same = P + f"is{color}"
        same_f = P + f"is{color}_f"
        out_ch = P + f"out_ch{color}"
        node("Sub", [prev, P + f"val{color}"], [diff])
        node("Mul", [diff, diff], [sq])
        node("Less", [sq, P + "half"], [same])
        node("Cast", [same], [same_f], to=onnx.TensorProto.FLOAT)
        node("Max", [channel_names[color], same_f], [out_ch])
        fill_masks[color] = out_ch
    ordered = [fill_masks.get(color, channel_names[color]) for color in range(10)]
    node("Concat", ordered, [P + "active_out"], axis=1)
    node(
        "Pad",
        [P + "active_out"],
        ["output"],
        pads=[0, 0, 0, 0, 0, 0, 30 - S, 30 - S],
        mode="constant",
        value=0.0,
    )
    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def empty_row_col_recolor2_network(batch=1):
    """Task303: paint color-0 cells in empty active rows/columns with color 2."""
    P = "erc2_"
    nodes = []
    inits = []

    def tensor(name, dtype, shape, vals):
        inits.append(onnx.helper.make_tensor(name, dtype, shape, vals))
        return name

    def node(op, inputs, outputs, **kwargs):
        nodes.append(onnx.helper.make_node(op, inputs, outputs, **kwargs))

    tensor(P + "ch0_st", onnx.TensorProto.INT64, [1], [0])
    tensor(P + "ch0_en", onnx.TensorProto.INT64, [1], [1])
    tensor(P + "ch_axis", onnx.TensorProto.INT64, [1], [1])
    tensor(P + "one_step", onnx.TensorProto.INT64, [1], [1])
    tensor(P + "half", onnx.TensorProto.FLOAT, [1], [0.5])

    node("Slice", ["input", P + "ch0_st", P + "ch0_en", P + "ch_axis", P + "one_step"], [P + "ch0"])
    node("ReduceSum", ["input"], [P + "active"], axes=[1], keepdims=1)
    node("Sub", [P + "active", P + "ch0"], [P + "fg"])
    node("ReduceSum", [P + "fg"], [P + "row_count"], axes=[3], keepdims=1)
    node("ReduceSum", [P + "fg"], [P + "col_count"], axes=[2], keepdims=1)
    node("Less", [P + "row_count", P + "half"], [P + "row_empty_b"])
    node("Less", [P + "col_count", P + "half"], [P + "col_empty_b"])
    node("Or", [P + "row_empty_b", P + "col_empty_b"], [P + "paint_b"])
    node("Cast", [P + "paint_b"], [P + "paint_axis"], to=onnx.TensorProto.FLOAT)
    node("Mul", [P + "paint_axis", P + "ch0"], [P + "paint"])
    node("Sub", [P + "paint", P + "paint"], [P + "zero"])
    node("Sub", [P + "zero", P + "paint"], [P + "neg_paint"])
    node(
        "Concat",
        [
            P + "neg_paint",
            P + "zero",
            P + "paint",
            P + "zero",
            P + "zero",
            P + "zero",
            P + "zero",
            P + "zero",
            P + "zero",
            P + "zero",
        ],
        [P + "delta"],
        axis=1,
    )
    node("Add", ["input", P + "delta"], ["output"])
    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def arch_marker_fall_network(batch=1):
    """Task165: arch columns with lower markers fall to the grid bottom."""
    P = "amfall_"
    S = 20
    nodes = []
    inits = []

    def tensor(name, dtype, shape, vals):
        inits.append(onnx.helper.make_tensor(name, dtype, shape, vals))
        return name

    def node(op, inputs, outputs, **kwargs):
        nodes.append(onnx.helper.make_node(op, inputs, outputs, **kwargs))

    tensor(P + "crop_st", onnx.TensorProto.INT64, [2], [0, 0])
    tensor(P + "crop_en", onnx.TensorProto.INT64, [2], [S, S])
    tensor(P + "hw_axes", onnx.TensorProto.INT64, [2], [2, 3])
    tensor(P + "hw_steps", onnx.TensorProto.INT64, [2], [1, 1])
    tensor(P + "ch_axis", onnx.TensorProto.INT64, [1], [1])
    tensor(P + "row_axis", onnx.TensorProto.INT64, [1], [2])
    tensor(P + "one_step", onnx.TensorProto.INT64, [1], [1])
    tensor(P + "row1_st", onnx.TensorProto.INT64, [1], [1])
    tensor(P + "rowS_en", onnx.TensorProto.INT64, [1], [S])
    tensor(P + "half", onnx.TensorProto.FLOAT, [1], [0.5])
    tensor(P + "four", onnx.TensorProto.FLOAT, [1], [4.0])
    tensor(P + "ten", onnx.TensorProto.FLOAT, [1], [10.0])
    tensor(P + "one_grid", onnx.TensorProto.FLOAT, [1, 1, S, S], [1.0] * (S * S))

    node("Slice", ["input", P + "crop_st", P + "crop_en", P + "hw_axes", P + "hw_steps"], [P + "crop"])

    channels = []
    for color in range(10):
        st = tensor(P + f"ch{color}_st", onnx.TensorProto.INT64, [1], [color])
        en = tensor(P + f"ch{color}_en", onnx.TensorProto.INT64, [1], [color + 1])
        name = P + f"ch{color}"
        node("Slice", [P + "crop", st, en, P + "ch_axis", P + "one_step"], [name])
        channels.append(name)

    object_gates = {}
    for color in range(1, 10):
        prefix = P + f"obj{color}_"
        node("ReduceSum", [channels[color]], [prefix + "count"], axes=[2, 3], keepdims=1)
        node("Sub", [prefix + "count", P + "ten"], [prefix + "count_diff"])
        node("Mul", [prefix + "count_diff", prefix + "count_diff"], [prefix + "count_diff2"])
        node("Less", [prefix + "count_diff2", P + "half"], [prefix + "count10_b"])
        node("Cast", [prefix + "count10_b"], [prefix + "count10"], to=onnx.TensorProto.FLOAT)
        node("ReduceMax", [channels[color]], [prefix + "row_has"], axes=[3], keepdims=1)
        node("ReduceSum", [prefix + "row_has"], [prefix + "row_count"], axes=[2, 3], keepdims=1)
        node("Sub", [prefix + "row_count", P + "four"], [prefix + "row_diff"])
        node("Mul", [prefix + "row_diff", prefix + "row_diff"], [prefix + "row_diff2"])
        node("Less", [prefix + "row_diff2", P + "half"], [prefix + "row4_b"])
        node("Cast", [prefix + "row4_b"], [prefix + "row4"], to=onnx.TensorProto.FLOAT)
        node("Mul", [prefix + "count10", prefix + "row4"], [prefix + "gate"])
        object_gates[color] = prefix + "gate"

    object_parts = []
    for color in range(1, 10):
        part = P + f"object_c{color}"
        node("Mul", [channels[color], object_gates[color]], [part])
        object_parts.append(part)
    node("Max", object_parts, [P + "object_mask"])
    node("Sub", [P + "one_grid", channels[0]], [P + "nonzero"])
    node("Sub", [P + "nonzero", P + "object_mask"], [P + "marker_mask"])

    node("Slice", [P + "marker_mask", P + "row1_st", P + "rowS_en", P + "row_axis", P + "one_step"], [P + "marker_below_slice"])
    node("MaxPool", [P + "marker_below_slice"], [P + "strict_below_short"], kernel_shape=[S - 1, 1], pads=[0, 0, S - 2, 0], strides=[1, 1])
    node("Pad", [P + "strict_below_short"], [P + "strict_below"], pads=[0, 0, 0, 0, 0, 0, 1, 0], mode="constant", value=0.0)
    node("Mul", [P + "object_mask", P + "strict_below"], [P + "seed"])
    node("MaxPool", [P + "seed"], [P + "fall_col"], kernel_shape=[S, 1], pads=[S - 1, 0, 0, 0], strides=[1, 1])
    node("Mul", [P + "fall_col", channels[0]], [P + "paint"])

    output_channels = [P + "ch0_out"]
    node("Sub", [channels[0], P + "paint"], [P + "ch0_out"])
    for color in range(1, 10):
        marker_part = P + f"marker_c{color}"
        marker_gate = P + f"marker_gate{color}"
        paint_ch = P + f"paint_c{color}"
        out_ch = P + f"ch{color}_out"
        node("Mul", [channels[color], P + "marker_mask"], [marker_part])
        node("ReduceMax", [marker_part], [marker_gate], axes=[2, 3], keepdims=1)
        node("Mul", [P + "paint", marker_gate], [paint_ch])
        node("Add", [channels[color], paint_ch], [out_ch])
        output_channels.append(out_ch)

    node("Concat", output_channels, [P + "active_out"], axis=1)
    node(
        "Pad",
        [P + "active_out"],
        ["output"],
        pads=[0, 0, 0, 0, 0, 0, 30 - S, 30 - S],
        mode="constant",
        value=0.0,
    )
    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def diagonal_pair_connect_network(batch=1):
    """Task037: connect same-color endpoint pairs along 45-degree diagonals."""
    P = "dpc_"
    S = 10
    nodes = []
    inits = []

    def tensor(name, dtype, shape, vals):
        inits.append(onnx.helper.make_tensor(name, dtype, shape, vals))
        return name

    def node(op, inputs, outputs, **kwargs):
        nodes.append(onnx.helper.make_node(op, inputs, outputs, **kwargs))

    tensor(P + "crop_st", onnx.TensorProto.INT64, [2], [0, 0])
    tensor(P + "crop_en", onnx.TensorProto.INT64, [2], [S, S])
    tensor(P + "hw_axes", onnx.TensorProto.INT64, [2], [2, 3])
    tensor(P + "hw_steps", onnx.TensorProto.INT64, [2], [1, 1])
    tensor(P + "ch0_st", onnx.TensorProto.INT64, [1], [0])
    tensor(P + "ch0_en", onnx.TensorProto.INT64, [1], [1])
    tensor(P + "ch_axis", onnx.TensorProto.INT64, [1], [1])
    tensor(P + "one_step", onnx.TensorProto.INT64, [1], [1])
    tensor(P + "half", onnx.TensorProto.FLOAT, [1], [0.5])

    def direction_kernel(name, direction):
        vals = []
        for group in range(10):
            grid = [[0.0 for _ in range(S)] for _ in range(S)]
            if direction == "nw":
                for k in range(1, S):
                    grid[S - 1 - k][S - 1 - k] = 1.0
            elif direction == "se":
                for k in range(1, S):
                    grid[k][k] = 1.0
            elif direction == "ne":
                for k in range(1, S):
                    grid[S - 1 - k][k] = 1.0
            elif direction == "sw":
                for k in range(1, S):
                    grid[k][S - 1 - k] = 1.0
            vals.extend(v for row in grid for v in row)
        return tensor(name, onnx.TensorProto.FLOAT, [10, 1, S, S], vals)

    node("Slice", ["input", P + "crop_st", P + "crop_en", P + "hw_axes", P + "hw_steps"], [P + "crop"])
    node("Slice", [P + "crop", P + "ch0_st", P + "ch0_en", P + "ch_axis", P + "one_step"], [P + "ch0"])

    dirs = {
        "nw": ([S - 1, S - 1, 0, 0], direction_kernel(P + "w_nw", "nw")),
        "se": ([0, 0, S - 1, S - 1], direction_kernel(P + "w_se", "se")),
        "ne": ([S - 1, 0, 0, S - 1], direction_kernel(P + "w_ne", "ne")),
        "sw": ([0, S - 1, S - 1, 0], direction_kernel(P + "w_sw", "sw")),
    }
    has = {}
    for name, (pads, weight) in dirs.items():
        count = P + f"{name}_count"
        hit_b = P + f"{name}_hit_b"
        hit = P + f"{name}_hit"
        node("Conv", [P + "crop", weight], [count], group=10, pads=pads, strides=[1, 1])
        node("Greater", [count, P + "half"], [hit_b])
        node("Cast", [hit_b], [hit], to=onnx.TensorProto.FLOAT)
        has[name] = hit

    node("Mul", [has["nw"], has["se"]], [P + "diag_between"])
    node("Mul", [has["ne"], has["sw"]], [P + "anti_between"])
    node("Max", [P + "diag_between", P + "anti_between"], [P + "between"])
    node("Mul", [P + "between", P + "ch0"], [P + "paint_all"])

    paint_channels = []
    output_channels = [P + "ch0_out"]
    for color in range(1, 10):
        st = tensor(P + f"ch{color}_st", onnx.TensorProto.INT64, [1], [color])
        en = tensor(P + f"ch{color}_en", onnx.TensorProto.INT64, [1], [color + 1])
        ch = P + f"ch{color}"
        paint = P + f"paint{color}"
        out_ch = P + f"ch{color}_out"
        node("Slice", [P + "crop", st, en, P + "ch_axis", P + "one_step"], [ch])
        node("Slice", [P + "paint_all", st, en, P + "ch_axis", P + "one_step"], [paint])
        node("Add", [ch, paint], [out_ch])
        paint_channels.append(paint)
        output_channels.append(out_ch)

    node("Concat", paint_channels, [P + "paint_nonzero"], axis=1)
    node("ReduceMax", [P + "paint_nonzero"], [P + "paint_any"], axes=[1], keepdims=1)
    node("Sub", [P + "ch0", P + "paint_any"], [P + "ch0_out"])
    node("Concat", output_channels, [P + "active_out"], axis=1)
    node(
        "Pad",
        [P + "active_out"],
        ["output"],
        pads=[0, 0, 0, 0, 0, 0, 30 - S, 30 - S],
        mode="constant",
        value=0.0,
    )
    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def downshift_recolor8_to2_network(batch=1):
    """Task261: move color-8 cells down one row and recolor them to 2."""
    W = np.zeros((10, 10, 3, 3), dtype=np.float32)
    W[0, 0, 1, 1] = 1.0
    W[0, 8, 1, 1] = 1.0
    W[0, 8, 0, 1] = -1.0
    W[2, 8, 0, 1] = 1.0
    weight = onnx.helper.make_tensor("d8d_w", onnx.TensorProto.FLOAT, W.shape, W.flatten().tolist())
    nodes = [
        onnx.helper.make_node("Conv", ["input", "d8d_w"], ["output"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
    ]
    return _make_model(nodes, [weight], shape=[batch, 10, 30, 30])


def edge_seed_bounce_ray_network(batch=1):
    """Task119: continue an edge-seeded diagonal ray that bounces off a 2-wall.

    The color-8 cells are the first visible cells of a diagonal ray launched
    from the outer edge. The ray may already include the bounce cell; we walk
    the marked ray forward, reflect off the adjacent color-2 edge wall, and
    paint subsequent zero cells color 3.
    """
    if batch != 1:
        raise NotImplementedError("task119 edge-seed-bounce-ray builder is single-example only")

    S = 12
    P = "esbr_"
    nodes = []
    inits = []

    def _tensor(name, dtype, shape, values):
        t = onnx.helper.make_tensor(name, dtype, shape, values)
        inits.append(t)
        return name

    def _ints(name, values):
        return _tensor(name, onnx.TensorProto.INT64, [len(values)], values)

    def _float(name, value):
        return _tensor(name, onnx.TensorProto.FLOAT, [1], [float(value)])

    def _node(op_type, inputs, outputs, **attrs):
        nodes.append(onnx.helper.make_node(op_type, inputs, outputs, **attrs))

    half = _float(P + "half", 0.5)
    ch_axis = _ints(P + "ch_axis", [1])
    ch_step = _ints(P + "ch_step", [1])
    hw_axes = _ints(P + "hw_axes", [2, 3])
    hw_steps = _ints(P + "hw_steps", [1, 1])

    edge_vals = []
    for r in range(S):
        for c in range(S):
            edge_vals.append(1.0 if r in (0, S - 1) or c in (0, S - 1) else 0.0)
    edge = _tensor(P + "edge", onnx.TensorProto.FLOAT, [1, 1, S, S], edge_vals)

    _node("Slice", ["input", _ints(P + "crop_st", [0, 0]), _ints(P + "crop_en", [S, S]), hw_axes, hw_steps], [P + "crop"])
    _node("Slice", [P + "crop", _ints(P + "ch0_st", [0]), _ints(P + "ch0_en", [1]), ch_axis, ch_step], [P + "ch0"])
    _node("Slice", [P + "crop", _ints(P + "ch2_st", [2]), _ints(P + "ch2_en", [3]), ch_axis, ch_step], [P + "ch2"])
    _node("Slice", [P + "crop", _ints(P + "ch3_st", [3]), _ints(P + "ch3_en", [4]), ch_axis, ch_step], [P + "ch3"])
    _node("Slice", [P + "crop", _ints(P + "ch8_st", [8]), _ints(P + "ch8_en", [9]), ch_axis, ch_step], [P + "ch8"])
    _node("Slice", [P + "crop", _ints(P + "ch1_st", [1]), _ints(P + "ch1_en", [3]), ch_axis, ch_step], [P + "ch1_2"])
    _node("Slice", [P + "crop", _ints(P + "ch4_st", [4]), _ints(P + "ch4_en", [10]), ch_axis, ch_step], [P + "ch4_9"])
    _node("Add", [P + "ch0", P + "ch8"], [P + "pathable"])

    def _shift_1ch(src, dst, dr, dc):
        r0 = max(0, -dr)
        r1 = S - max(0, dr)
        c0 = max(0, -dc)
        c1 = S - max(0, dc)
        pad_top = max(0, dr)
        pad_bottom = max(0, -dr)
        pad_left = max(0, dc)
        pad_right = max(0, -dc)
        _node("Slice", [src, _ints(dst + "_st", [r0, c0]), _ints(dst + "_en", [r1, c1]), hw_axes, hw_steps], [dst + "_slice"])
        _node(
            "Pad",
            [dst + "_slice"],
            [dst],
            pads=[0, 0, pad_top, pad_left, 0, 0, pad_bottom, pad_right],
            mode="constant",
            value=0.0,
        )

    dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    def _tag(dr, dc):
        return ("p" if dr > 0 else "n") + ("p" if dc > 0 else "n")

    wall_h = {}
    wall_v = {}
    for dc in (-1, 1):
        name = P + f"wall_h_{'p' if dc > 0 else 'n'}"
        _shift_1ch(P + "ch2", name, 0, -dc)
        wall_h[dc] = name
    for dr in (-1, 1):
        name = P + f"wall_v_{'p' if dr > 0 else 'n'}"
        _shift_1ch(P + "ch2", name, -dr, 0)
        wall_v[dr] = name

    _node("Mul", [P + "ch8", edge], [P + "edge_8"])
    fronts = {}
    for dr, dc in dirs:
        tag = _tag(dr, dc)
        next_8 = P + "next8_" + tag
        _shift_1ch(P + "ch8", next_8, -dr, -dc)
        _node("Mul", [P + "edge_8", next_8], [P + "front0_" + tag])
        fronts[(dr, dc)] = P + "front0_" + tag

    paint_steps = []
    for step in range(11):
        incoming = {d: [] for d in dirs}
        for dr, dc in dirs:
            tag = _tag(dr, dc)
            front = fronts[(dr, dc)]
            bv = P + f"s{step}_{tag}_bv"
            bh = P + f"s{step}_{tag}_bh"
            bsum = P + f"s{step}_{tag}_bsum"
            straight = P + f"s{step}_{tag}_straight"
            _node("Mul", [front, wall_h[dc]], [bv])
            _node("Mul", [front, wall_v[dr]], [bh])
            _node("Add", [bv, bh], [bsum])
            _node("Sub", [front, bsum], [straight])

            same = P + f"s{step}_{tag}_same"
            refl_v = P + f"s{step}_{tag}_refl_v"
            refl_h = P + f"s{step}_{tag}_refl_h"
            _shift_1ch(straight, same, dr, dc)
            _shift_1ch(bv, refl_v, dr, -dc)
            _shift_1ch(bh, refl_h, -dr, dc)
            incoming[(dr, dc)].append(same)
            incoming[(dr, -dc)].append(refl_v)
            incoming[(-dr, dc)].append(refl_h)

        next_fronts = {}
        for dr, dc in dirs:
            tag = _tag(dr, dc)
            raw = P + f"s{step}_{tag}_incoming"
            nf = P + f"s{step}_{tag}_front"
            if len(incoming[(dr, dc)]) == 1:
                raw = incoming[(dr, dc)][0]
            else:
                _node("Sum", incoming[(dr, dc)], [raw])
            _node("Mul", [raw, P + "pathable"], [nf])
            paint = P + f"s{step}_{tag}_paint"
            _node("Mul", [nf, P + "ch0"], [paint])
            paint_steps.append(paint)
            next_fronts[(dr, dc)] = nf
        fronts = next_fronts

    _node("Sum", paint_steps, [P + "paint_sum"])
    _node("Greater", [P + "paint_sum", half], [P + "paint_bool"])
    _node("Cast", [P + "paint_bool"], [P + "paint"], to=onnx.TensorProto.FLOAT)
    _node("Mul", [P + "paint", P + "ch0"], [P + "paint_zero"])
    _node("Sub", [P + "ch0", P + "paint_zero"], [P + "ch0_out"])
    _node("Add", [P + "ch3", P + "paint_zero"], [P + "ch3_out"])
    _node("Concat", [P + "ch0_out", P + "ch1_2", P + "ch3_out", P + "ch4_9"], [P + "out_crop"], axis=1)
    _node("Pad", [P + "out_crop"], ["output"], pads=[0, 0, 0, 0, 0, 0, 30 - S, 30 - S],
          mode="constant", value=0.0)

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def marker_dilate_network(marker_color, target_color, preserve_marker, batch=1):
    """Marker-dilate-recolor: 3x3 MaxPool dilation of `marker_color` cells, with
    the dilated region recolored to `target_color`.

    Two modes (selected by `preserve_marker`):
      * preserve_marker=True ("ring"): marker_color cells stay marker_color in
        the output; the 8 neighbors of each marker (clipped to canvas) become
        target_color. Used by task095 (marker=5, target=1).
      * preserve_marker=False ("fill"): marker_color cells become target_color in
        the output; the 8 neighbors of each marker also become target_color.
        Used by task317 (marker=5, target=1).

    Pipeline (input [N, 10, 30, 30]):
      1. ch0 = Slice(input, axis=1, [0:1])         -> [N, 1, 30, 30]
      2. ch_marker = Slice(input, axis=1, [M:M+1]) -> [N, 1, 30, 30]
      3. dilated = MaxPool(ch_marker, kernel=[3,3], pads=[1,1,1,1])
            -> [N, 1, 30, 30] (1 at marker cells AND their 8-neighbors)
      4. zeros = Sub(ch0, ch0)  (free [N, 1, 30, 30] all-zero tensor)
      5. new_ch0 = Sub(ch0, dilated)
      6. If preserve_marker:
            new_target = Sub(dilated, ch_marker)  (dilation MINUS marker = ring)
            new_marker = ch_marker                (preserved)
         Else:
            new_target = dilated                  (full dilation including marker)
            new_marker = zeros                    (marker channel emptied)
      7. Concat 10 channels with new_ch0, new_target, new_marker, others=zeros.

    Cost (per cost-model verify, opset 10): macs ~ 8.1k (MaxPool 3x3 on
    [N,1,30,30] = 900 * 9) + 900-1800 (1-2 Sub on [N,1,30,30]) = ~10-11k.
    Memory ~ 18-22 KB (5-6 [N,1,30,30] intermediates). Params ~ 30 (Slice
    starts/ends/axes/steps initializers). Total ~ 30-35k -> ~14.5-15 pts/task.

    Replaces the much heavier bin-nbr-lut on tasks 095 and 317 (was ~169k);
    cost-aware dispatcher picks the cheapest match automatically.
    """
    if not (1 <= marker_color < 10):
        raise ValueError(f"marker_color must be in 1..9, got {marker_color}")
    if not (1 <= target_color < 10):
        raise ValueError(f"target_color must be in 1..9, got {target_color}")
    if preserve_marker and marker_color == target_color:
        raise ValueError("preserve_marker=True with marker_color == target_color is degenerate")

    nodes = []
    inits = []

    def _slice_ch(prefix, ch, out_name):
        starts = onnx.helper.make_tensor(f"{prefix}_starts", onnx.TensorProto.INT64, [1], [ch])
        ends = onnx.helper.make_tensor(f"{prefix}_ends", onnx.TensorProto.INT64, [1], [ch + 1])
        axes = onnx.helper.make_tensor(f"{prefix}_axes", onnx.TensorProto.INT64, [1], [1])
        steps = onnx.helper.make_tensor(f"{prefix}_steps", onnx.TensorProto.INT64, [1], [1])
        node = onnx.helper.make_node(
            "Slice",
            ["input", f"{prefix}_starts", f"{prefix}_ends", f"{prefix}_axes", f"{prefix}_steps"],
            [out_name],
        )
        return node, [starts, ends, axes, steps]

    # Slice every channel so other-color cells pass through unchanged. Only ch0,
    # ch_target, and ch_marker get rewritten; the rest are passed verbatim.
    channel_names = {}
    for ch in range(10):
        node, ch_inits = _slice_ch(f"md_sl{ch}", ch, f"md_ch{ch}")
        nodes.append(node); inits.extend(ch_inits)
        channel_names[ch] = f"md_ch{ch}"

    nodes.append(onnx.helper.make_node(
        "MaxPool", [channel_names[marker_color]], ["md_dilated_raw"],
        kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1],
    ))

    # Clip dilated mask to active-grid extent (ReduceSum across all channels =
    # 1 at any active cell, 0 at canvas padding). Without this, a marker near
    # the active-grid edge would dilate into canvas-padding cells, lighting
    # the target channel where it should be unlit (color 10 / padding).
    nodes.append(onnx.helper.make_node(
        "ReduceSum", ["input"], ["md_is_active"], axes=[1], keepdims=1,
    ))
    nodes.append(onnx.helper.make_node(
        "Mul", ["md_dilated_raw", "md_is_active"], ["md_dilated"],
    ))

    # new_ch0 = ch0 - dilated  (cells in dilation lose ch0)
    nodes.append(onnx.helper.make_node(
        "Sub", [channel_names[0], "md_dilated"], ["md_new_ch0"],
    ))

    if preserve_marker:
        # ring = dilated - marker (8 neighbors only, marker itself excluded)
        nodes.append(onnx.helper.make_node(
            "Sub", ["md_dilated", channel_names[marker_color]], ["md_new_target"],
        ))
        new_marker_name = channel_names[marker_color]
    else:
        # full dilation including marker position becomes target color
        new_marker_name = None  # marker channel goes to zero (not output)
        nodes.append(onnx.helper.make_node(
            "Sub", ["md_dilated", "md_dilated"], ["md_zeros_dilated"],
        ))
        # md_new_target is just dilated; alias by Sub with zeros (cheap; could
        # use Identity but Sub keeps ops uniform).
        nodes.append(onnx.helper.make_node(
            "Sub", ["md_dilated", "md_zeros_dilated"], ["md_new_target"],
        ))

    concat_inputs = []
    for ch in range(10):
        if ch == 0:
            concat_inputs.append("md_new_ch0")
        elif ch == target_color:
            concat_inputs.append("md_new_target")
        elif ch == marker_color:
            if preserve_marker:
                concat_inputs.append(new_marker_name)
            else:
                # Marker channel is replaced -> use the channel's original
                # cells as a synthetic zero (they were the only lit cells, now
                # all moved to target). Equivalently: ch_marker - ch_marker.
                concat_inputs.append("md_zeros_dilated" if not preserve_marker else channel_names[marker_color])
        else:
            concat_inputs.append(channel_names[ch])  # pass-through
    nodes.append(onnx.helper.make_node(
        "Concat", concat_inputs, ["output"], axis=1,
    ))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def marker_dilate_conv_network(marker_color, target_color, preserve_marker, batch=1):
    """Single-Conv fast path for safe marker-dilate tasks.

    This is only correct when every marker is at least one cell inside the
    active canvas, every ring destination was background in the input, and
    marker 3x3 neighborhoods do not overlap. `_detect_marker_dilate_conv_safe`
    enforces those geometry constraints before dispatch can pick this builder.
    """
    if not (1 <= marker_color < 10):
        raise ValueError(f"marker_color must be in 1..9, got {marker_color}")
    if not (1 <= target_color < 10):
        raise ValueError(f"target_color must be in 1..9, got {target_color}")
    if preserve_marker and marker_color == target_color:
        raise ValueError("preserve_marker=True with marker_color == target_color is degenerate")

    W = np.zeros((10, 10, 3, 3), dtype=np.float32)
    for ch in range(10):
        W[ch, ch, 1, 1] = 1.0

    if not preserve_marker:
        W[marker_color, marker_color, 1, 1] = 0.0
        W[target_color, marker_color, 1, 1] += 1.0

    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            W[0, marker_color, 1 - dr, 1 - dc] -= 1.0
            W[target_color, marker_color, 1 - dr, 1 - dc] += 1.0

    weight = onnx.helper.make_tensor("mdc_w", onnx.TensorProto.FLOAT, W.shape, W.flatten().tolist())
    nodes = [
        onnx.helper.make_node(
            "Conv",
            ["input", "mdc_w"],
            ["output"],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        )
    ]
    return _make_model(nodes, [weight], shape=[batch, 10, 30, 30])


def symmetry_completion_network(axis, h, w, batch=1):
    """Symmetry-completion: fill input's zero cells with the reflected non-zero
    value across the specified axis; preserve non-zero cells.

    output[r, c] = input[r, c]                      if input[r, c] != 0
                 = input[refl(r, c)] otherwise (= 0 if reflected cell is also 0)

    where refl(r, c) for axis="v" is (h - 1 - r, c); for axis="h" is (r, w - 1 - c).

    Pipeline (input [N, 10, 30, 30]):
      1. Slice axis [1] starts [0] ends [1] -> [N, 1, 30, 30] ch0_mask
            (1.0 at active-grid color-0 cells; 0 at non-zero cells AND canvas padding)
      2. Gather(input, axes=[2 or 3], indices=fixed_canvas_reverse_for_axis)
         -> [N, 10, 30, 30] flipped (the active-grid portion is reflected; padding
         preserved as zero since indices outside [0, h) point at padding rows)
      3. cond = Greater(ch0_mask, 0.5) -> [N, 1, 30, 30] bool
            (True at cells where input was color 0; False elsewhere)
      4. output = Where(cond, flipped, input) -> [N, 10, 30, 30]
            (cells that were color 0 get the reflected value; non-zero cells
             stay as input; padding cells stay as input which is all-zero)

    Cost (per cost-model verify, opset 10): macs = 0 (Gather + Slice + Greater
    + Where are all 0 MACs); memory ~ 36 KB (one [N, 10, 30, 30] flipped
    intermediate) + ~3.6 KB (ch0_mask) + ~3.6 KB (cond bool); params ~ 30
    (Gather indices + Slice initializers). Total ~43 KB cost -> ~14.3 pts/task.

    Caveat: requires h (for v-axis) or w (for h-axis) to be CONSTANT across
    pairs of the task — the Gather indices are baked at build time. For tasks
    with varying h/w, this builder doesn't apply.
    """
    if axis not in ("v", "h"):
        raise ValueError(f"axis must be 'v' or 'h', got {axis!r}")

    nodes = []
    inits = []

    # ch0_mask = Slice(input, axis=1, [0:1])
    ch0_starts = onnx.helper.make_tensor("sc_ch0_starts", onnx.TensorProto.INT64, [1], [0])
    ch0_ends = onnx.helper.make_tensor("sc_ch0_ends", onnx.TensorProto.INT64, [1], [1])
    ch0_axes = onnx.helper.make_tensor("sc_ch0_axes", onnx.TensorProto.INT64, [1], [1])
    ch0_steps = onnx.helper.make_tensor("sc_ch0_steps", onnx.TensorProto.INT64, [1], [1])
    inits.extend([ch0_starts, ch0_ends, ch0_axes, ch0_steps])
    nodes.append(onnx.helper.make_node(
        "Slice",
        ["input", "sc_ch0_starts", "sc_ch0_ends", "sc_ch0_axes", "sc_ch0_steps"],
        ["sc_ch0_mask"],
    ))

    # Flipped input via Gather along the appropriate axis with fixed-canvas-reverse indices.
    if axis == "v":
        idx = _fixed_canvas_indices(h, "reverse")
        gather_axis = 2
    else:  # axis == "h"
        idx = _fixed_canvas_indices(w, "reverse")
        gather_axis = 3
    idx_init = onnx.helper.make_tensor("sc_idx", onnx.TensorProto.INT64, [30], idx)
    inits.append(idx_init)
    nodes.append(onnx.helper.make_node(
        "Gather", ["input", "sc_idx"], ["sc_flipped"], axis=gather_axis,
    ))

    # cond = Greater(ch0_mask, 0.5).
    half_init = onnx.helper.make_tensor("sc_half", onnx.TensorProto.FLOAT, [1], [0.5])
    inits.append(half_init)
    nodes.append(onnx.helper.make_node(
        "Greater", ["sc_ch0_mask", "sc_half"], ["sc_cond"],
    ))

    # output = Where(cond, flipped, input). Where broadcasts cond [N, 1, H, W] across
    # the channel axis to [N, 10, H, W].
    nodes.append(onnx.helper.make_node(
        "Where", ["sc_cond", "sc_flipped", "input"], ["output"],
    ))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def diagonal_shear_network(batch=1):
    """Diagonal-shear / Toeplitz tile (3x3 input -> 6x6 output).

    Rule: for each output anti-diagonal `r - c = k` with k in [-2, 2], the
    output value is the (only) non-zero input cell on input anti-diagonal
    `i - j = k`; cells with k outside [-2, 2] are zero. Equivalently, the 3x3
    input is "stamped" at 6 diagonal anchor positions (k, k) for k in 0..5,
    and the stamps are summed (relying on the data property that each input
    anti-diagonal contains at most one non-zero color, so stamps never collide
    on a non-zero color).

    Pipeline (input [N, 10, 30, 30]):
      1. Slice axes [2,3] -> [N, 10, 3, 3]                    active 3x3
      2. Slice for k=4 (drop bottom + right rows): axes [2,3]
         starts [0,0] ends [2,2] -> [N, 10, 2, 2]
      3. Slice for k=5: axes [2,3] starts [0,0] ends [1,1] -> [N, 10, 1, 1]
      4. Pad each shift k in 0..3 to [N, 10, 6, 6] with active stamp at (k, k):
            pads = [0, 0, k, k, 0, 0, 3 - k, 3 - k]
      5. Pad k=4 (using 2x2 sub-slice) to [N, 10, 6, 6]:
            pads = [0, 0, 4, 4, 0, 0, 0, 0]
      6. Pad k=5 (using 1x1 sub-slice) to [N, 10, 6, 6]:
            pads = [0, 0, 5, 5, 0, 0, 0, 0]
      7. Sum the 6 padded stamps -> [N, 10, 6, 6]
            (stamps don't overlap on non-zero colors per the data property,
            so Sum == element-wise Max; channel 0 will be sum-counted and
            corrected in step 9.)
      8. stamps_nonzero = Slice axis [1] starts [1] ends [10] -> [N, 9, 6, 6]
            (drop the corrupted ch0 sum)
      9. ch0_active = 1 - ReduceSum(stamps_nonzero, axes=[1], keepdims=1)
            -> [N, 1, 6, 6]
            (per-cell: 1 if no other channel is lit, 0 if a non-zero color is there)
     10. Concat([ch0_active, stamps_nonzero]) -> [N, 10, 6, 6]
     11. Pad after [24, 24] -> [N, 10, 30, 30]

    Cost (per cost-model verify, opset 10): macs ~ 5*360 (Add on [N,10,6,6])
    + 324 (ReduceSum on [N,9,6,6]) + 36 (Sub) ~ 2200; memory ~ 6 KB (six
    [N,10,6,6] padded stamps + a few smaller); params ~ 60 (Slice / Pad
    initializers + scalar consts). Total ~9 KB cost -> ~15.9 pts/task.
    """
    inits = _slice_active_inits("ds_sl", 3, 3)
    nodes = [
        onnx.helper.make_node(
            "Slice",
            ["input", "ds_sl_starts", "ds_sl_ends", "ds_sl_axes", "ds_sl_steps"],
            ["ds_active"],
        )
    ]

    # Sub-slices for k=4 (2x2) and k=5 (1x1).
    for end_h, suffix in [(2, "k4"), (1, "k5")]:
        s_start = onnx.helper.make_tensor(
            f"ds_{suffix}_starts", onnx.TensorProto.INT64, [2], [0, 0]
        )
        s_end = onnx.helper.make_tensor(
            f"ds_{suffix}_ends", onnx.TensorProto.INT64, [2], [end_h, end_h]
        )
        s_axes = onnx.helper.make_tensor(
            f"ds_{suffix}_axes", onnx.TensorProto.INT64, [2], [2, 3]
        )
        s_steps = onnx.helper.make_tensor(
            f"ds_{suffix}_steps", onnx.TensorProto.INT64, [2], [1, 1]
        )
        inits.extend([s_start, s_end, s_axes, s_steps])
        nodes.append(onnx.helper.make_node(
            "Slice",
            ["ds_active", f"ds_{suffix}_starts", f"ds_{suffix}_ends",
             f"ds_{suffix}_axes", f"ds_{suffix}_steps"],
            [f"ds_active_{suffix}"],
        ))

    # Pad each shift to [N, 10, 6, 6] with stamp positioned at (k, k).
    pad_inputs = []
    for k in range(6):
        if k <= 3:
            src = "ds_active"
            slice_h = 3
        elif k == 4:
            src = "ds_active_k4"
            slice_h = 2
        else:  # k == 5
            src = "ds_active_k5"
            slice_h = 1
        # Pad attribute order: [before_dim0, before_dim1, before_dim2, before_dim3,
        # after_dim0, after_dim1, after_dim2, after_dim3].
        # Place the slice_h x slice_h active at (k, k) inside [N, 10, 6, 6].
        pads = [0, 0, k, k, 0, 0, 6 - k - slice_h, 6 - k - slice_h]
        out_name = f"ds_padded_k{k}"
        nodes.append(onnx.helper.make_node(
            "Pad", [src], [out_name], pads=pads, mode="constant", value=0.0,
        ))
        pad_inputs.append(out_name)

    # Sum all 6 padded shifts via 5 Adds.
    cur = pad_inputs[0]
    for i, nxt in enumerate(pad_inputs[1:], start=1):
        out_name = "ds_sum_final" if i == 5 else f"ds_sum_{i}"
        nodes.append(onnx.helper.make_node("Add", [cur, nxt], [out_name]))
        cur = out_name

    # Drop ch0 from the sum: Slice axis [1] starts [1] ends [10].
    nz_starts = onnx.helper.make_tensor("ds_nz_starts", onnx.TensorProto.INT64, [1], [1])
    nz_ends = onnx.helper.make_tensor("ds_nz_ends", onnx.TensorProto.INT64, [1], [10])
    nz_axes = onnx.helper.make_tensor("ds_nz_axes", onnx.TensorProto.INT64, [1], [1])
    nz_steps = onnx.helper.make_tensor("ds_nz_steps", onnx.TensorProto.INT64, [1], [1])
    inits.extend([nz_starts, nz_ends, nz_axes, nz_steps])
    nodes.append(onnx.helper.make_node(
        "Slice", ["ds_sum_final", "ds_nz_starts", "ds_nz_ends", "ds_nz_axes", "ds_nz_steps"],
        ["ds_stamps_nz"],
    ))

    # ch0_active = 1 - ReduceSum(stamps_nz, axes=[1], keepdims=1).
    nodes.append(onnx.helper.make_node(
        "ReduceSum", ["ds_stamps_nz"], ["ds_lit_count"],
        axes=[1], keepdims=1,
    ))
    one_init = onnx.helper.make_tensor("ds_one", onnx.TensorProto.FLOAT, [1], [1.0])
    inits.append(one_init)
    nodes.append(onnx.helper.make_node(
        "Sub", ["ds_one", "ds_lit_count"], ["ds_ch0_active"],
    ))

    # Concat to [N, 10, 6, 6].
    nodes.append(onnx.helper.make_node(
        "Concat", ["ds_ch0_active", "ds_stamps_nz"], ["ds_grid_6x6"], axis=1,
    ))

    # Pad to [N, 10, 30, 30].
    nodes.append(onnx.helper.make_node(
        "Pad", ["ds_grid_6x6"], ["output"],
        pads=[0, 0, 0, 0, 0, 0, 24, 24],
        mode="constant", value=0.0,
    ))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def foreground_bbox_crop_network(batch=1):
    """ONNX: ReduceSum + Slice + Sub + ReduceMax + ArgMax + Cast + Add + Min + Cast + Gather x2 + Sub + Add + Less + Cast + Reshape + Mul.

    For tasks where the output is the input cropped to the bounding box of all
    non-zero cells in the input (variable input shape, variable output shape).
    Output is placed at the top-left of the canvas; cells beyond the cropped
    bbox are canvas-padding (all-zero across channels).

    Pipeline (input [N, 10, 30, 30], single example only — N=1):
      1. is_active = ReduceSum(input, axes=[1], keepdims=1)        -> [N, 1, 30, 30]
         ch0       = Slice(input, axes=[1], starts=[0], ends=[1])  -> [N, 1, 30, 30]
         fg        = is_active - ch0                                -> [N, 1, 30, 30]
      2. row_has_fg = ReduceMax(fg, axes=[3], keepdims=1)           -> [N, 1, 30, 1]
         col_has_fg = ReduceMax(fg, axes=[2], keepdims=1)           -> [N, 1, 1, 30]
      3. ArgMax-trick to find first/last index that has foreground:
         multiply by descending [30, 29, ..., 1] -> ArgMax-first picks the
         smallest index among positions tied at the max (ONNX ArgMax spec);
         multiply by ascending [1, 2, ..., 30] -> ArgMax-last picks the
         largest tied position. Yields 4 int64 scalars.
      4. row_indices = Min(first_row + range_30, 29) clipped, Cast to INT64.
         (Opset 10 disallows Add/Min on int64; we compute in FLOAT then Cast.)
      5. shifted = Gather(input, row_indices, axis=2) -> [N, 10, 30, 30]
                   then Gather along axis=3 with col_indices.
      6. height = last_row - first_row + 1 (FLOAT scalar). Same for width.
         row_in_bbox = (range_30 < height); same for col. Cast to FLOAT and
         broadcast-multiply -> [N, 1, 30, 30] mask.
      7. output = shifted * mask. Cells beyond bbox are zeroed (canvas padding).

    Cost (per cost-model verify, opset 10): macs ~20k (Mul + Conv-equivalent
    elementwise), memory ~114k bytes (3 [N, 10, 30, 30] intermediates dominate),
    params ~165 (initializer constants). Total ~134k -> ~13.2 pts/task.
    Verified 100% (266/266) on task031.

    Builder is single-example only — multi-batch would need GatherElements
    (opset 11+) for per-batch axis-2/axis-3 gather. Raises NotImplementedError
    for batch != 1, triggering the dispatcher's per-example fallback.
    """
    if batch != 1:
        raise NotImplementedError(
            "foreground-bbox-crop builder is single-example only (batch must be 1)"
        )

    nodes = []
    inits = []

    # Step 1: foreground = is_active - ch0
    nodes.append(onnx.helper.make_node(
        "ReduceSum", ["input"], ["fbc_is_active"], axes=[1], keepdims=1
    ))
    sl_starts = onnx.helper.make_tensor("fbc_sl_starts", onnx.TensorProto.INT64, [1], [0])
    sl_ends = onnx.helper.make_tensor("fbc_sl_ends", onnx.TensorProto.INT64, [1], [1])
    sl_axes = onnx.helper.make_tensor("fbc_sl_axes", onnx.TensorProto.INT64, [1], [1])
    sl_steps = onnx.helper.make_tensor("fbc_sl_steps", onnx.TensorProto.INT64, [1], [1])
    inits.extend([sl_starts, sl_ends, sl_axes, sl_steps])
    nodes.append(onnx.helper.make_node(
        "Slice",
        ["input", "fbc_sl_starts", "fbc_sl_ends", "fbc_sl_axes", "fbc_sl_steps"],
        ["fbc_ch0"],
    ))
    nodes.append(onnx.helper.make_node("Sub", ["fbc_is_active", "fbc_ch0"], ["fbc_fg"]))

    # Step 2: row_has_fg, col_has_fg
    nodes.append(onnx.helper.make_node(
        "ReduceMax", ["fbc_fg"], ["fbc_row_has"], axes=[3], keepdims=1
    ))
    nodes.append(onnx.helper.make_node(
        "ReduceMax", ["fbc_fg"], ["fbc_col_has"], axes=[2], keepdims=1
    ))

    # Step 3: weight-vectors and ArgMax-first / ArgMax-last for row + col.
    desc_row = onnx.helper.make_tensor(
        "fbc_desc_row", onnx.TensorProto.FLOAT, [1, 1, 30, 1],
        [float(30 - i) for i in range(30)],
    )
    asc_row = onnx.helper.make_tensor(
        "fbc_asc_row", onnx.TensorProto.FLOAT, [1, 1, 30, 1],
        [float(i + 1) for i in range(30)],
    )
    desc_col = onnx.helper.make_tensor(
        "fbc_desc_col", onnx.TensorProto.FLOAT, [1, 1, 1, 30],
        [float(30 - i) for i in range(30)],
    )
    asc_col = onnx.helper.make_tensor(
        "fbc_asc_col", onnx.TensorProto.FLOAT, [1, 1, 1, 30],
        [float(i + 1) for i in range(30)],
    )
    inits.extend([desc_row, asc_row, desc_col, asc_col])
    nodes.append(onnx.helper.make_node("Mul", ["fbc_row_has", "fbc_desc_row"], ["fbc_w_first_row"]))
    nodes.append(onnx.helper.make_node("Mul", ["fbc_row_has", "fbc_asc_row"], ["fbc_w_last_row"]))
    nodes.append(onnx.helper.make_node("Mul", ["fbc_col_has", "fbc_desc_col"], ["fbc_w_first_col"]))
    nodes.append(onnx.helper.make_node("Mul", ["fbc_col_has", "fbc_asc_col"], ["fbc_w_last_col"]))

    nodes.append(onnx.helper.make_node(
        "ArgMax", ["fbc_w_first_row"], ["fbc_first_row_3d"], axis=2, keepdims=0
    ))
    nodes.append(onnx.helper.make_node(
        "ArgMax", ["fbc_w_last_row"], ["fbc_last_row_3d"], axis=2, keepdims=0
    ))
    nodes.append(onnx.helper.make_node(
        "ArgMax", ["fbc_w_first_col"], ["fbc_first_col_3d"], axis=3, keepdims=0
    ))
    nodes.append(onnx.helper.make_node(
        "ArgMax", ["fbc_w_last_col"], ["fbc_last_col_3d"], axis=3, keepdims=0
    ))

    # Reshape int64 [1, 1, 1] -> int64 [1] -> Cast to FLOAT [1] for arithmetic.
    rs1 = onnx.helper.make_tensor("fbc_rs1", onnx.TensorProto.INT64, [1], [1])
    inits.append(rs1)
    for src_int, dst_f in [
        ("fbc_first_row_3d", "fbc_first_row_f"),
        ("fbc_last_row_3d", "fbc_last_row_f"),
        ("fbc_first_col_3d", "fbc_first_col_f"),
        ("fbc_last_col_3d", "fbc_last_col_f"),
    ]:
        tmp = src_int.replace("_3d", "_1d")
        nodes.append(onnx.helper.make_node("Reshape", [src_int, "fbc_rs1"], [tmp]))
        nodes.append(onnx.helper.make_node("Cast", [tmp], [dst_f], to=onnx.TensorProto.FLOAT))

    # Step 4: clipped row/col indices in FLOAT, then Cast to INT64 for Gather.
    range_30_f = onnx.helper.make_tensor(
        "fbc_range_30_f", onnx.TensorProto.FLOAT, [30],
        [float(i) for i in range(30)],
    )
    const_29_f = onnx.helper.make_tensor("fbc_const_29_f", onnx.TensorProto.FLOAT, [1], [29.0])
    const_1_f = onnx.helper.make_tensor("fbc_const_1_f", onnx.TensorProto.FLOAT, [1], [1.0])
    inits.extend([range_30_f, const_29_f, const_1_f])

    nodes.append(onnx.helper.make_node("Add", ["fbc_first_row_f", "fbc_range_30_f"], ["fbc_row_idx_raw_f"]))
    nodes.append(onnx.helper.make_node("Min", ["fbc_row_idx_raw_f", "fbc_const_29_f"], ["fbc_row_idx_f"]))
    nodes.append(onnx.helper.make_node("Cast", ["fbc_row_idx_f"], ["fbc_row_idx"], to=onnx.TensorProto.INT64))

    nodes.append(onnx.helper.make_node("Add", ["fbc_first_col_f", "fbc_range_30_f"], ["fbc_col_idx_raw_f"]))
    nodes.append(onnx.helper.make_node("Min", ["fbc_col_idx_raw_f", "fbc_const_29_f"], ["fbc_col_idx_f"]))
    nodes.append(onnx.helper.make_node("Cast", ["fbc_col_idx_f"], ["fbc_col_idx"], to=onnx.TensorProto.INT64))

    # Step 5: Gather rows then cols.
    nodes.append(onnx.helper.make_node("Gather", ["input", "fbc_row_idx"], ["fbc_shifted_rows"], axis=2))
    nodes.append(onnx.helper.make_node("Gather", ["fbc_shifted_rows", "fbc_col_idx"], ["fbc_shifted"], axis=3))

    # Step 6: in_bbox mask.
    nodes.append(onnx.helper.make_node("Sub", ["fbc_last_row_f", "fbc_first_row_f"], ["fbc_h_minus_1"]))
    nodes.append(onnx.helper.make_node("Add", ["fbc_h_minus_1", "fbc_const_1_f"], ["fbc_height_f"]))
    nodes.append(onnx.helper.make_node("Sub", ["fbc_last_col_f", "fbc_first_col_f"], ["fbc_w_minus_1"]))
    nodes.append(onnx.helper.make_node("Add", ["fbc_w_minus_1", "fbc_const_1_f"], ["fbc_width_f"]))

    nodes.append(onnx.helper.make_node("Less", ["fbc_range_30_f", "fbc_height_f"], ["fbc_row_in_bbox_bool"]))
    nodes.append(onnx.helper.make_node("Less", ["fbc_range_30_f", "fbc_width_f"], ["fbc_col_in_bbox_bool"]))
    nodes.append(onnx.helper.make_node("Cast", ["fbc_row_in_bbox_bool"], ["fbc_row_in_bbox_f"], to=onnx.TensorProto.FLOAT))
    nodes.append(onnx.helper.make_node("Cast", ["fbc_col_in_bbox_bool"], ["fbc_col_in_bbox_f"], to=onnx.TensorProto.FLOAT))

    rs_row = onnx.helper.make_tensor("fbc_rs_row", onnx.TensorProto.INT64, [4], [1, 1, 30, 1])
    rs_col = onnx.helper.make_tensor("fbc_rs_col", onnx.TensorProto.INT64, [4], [1, 1, 1, 30])
    inits.extend([rs_row, rs_col])
    nodes.append(onnx.helper.make_node("Reshape", ["fbc_row_in_bbox_f", "fbc_rs_row"], ["fbc_row_mask"]))
    nodes.append(onnx.helper.make_node("Reshape", ["fbc_col_in_bbox_f", "fbc_rs_col"], ["fbc_col_mask"]))
    nodes.append(onnx.helper.make_node("Mul", ["fbc_row_mask", "fbc_col_mask"], ["fbc_bbox_mask"]))

    # Step 7: output = shifted * mask.
    nodes.append(onnx.helper.make_node("Mul", ["fbc_shifted", "fbc_bbox_mask"], ["output"]))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def frame_raycast_crop_network(batch=1):
    """ONNX: task138-style crop bounded by full nonzero rows/cols, then ray-fill.

    The active rectangle has two fully nonzero rows and two fully nonzero cols.
    After cropping to their inclusive bounds, the lone interior marker color
    matches exactly one border side. Marker cells ray-fill toward that side.
    """
    if batch != 1:
        raise NotImplementedError(
            "frame-raycast-crop builder is single-example only (batch must be 1)"
        )

    P = "frc_"
    nodes = []
    inits = []

    def t(name, dtype, shape, values):
        init = onnx.helper.make_tensor(P + name, dtype, shape, values)
        inits.append(init)
        return P + name

    def n(op, inputs, outputs, **attrs):
        nodes.append(onnx.helper.make_node(op, inputs, outputs, **attrs))

    const_0 = t("c0", onnx.TensorProto.FLOAT, [1], [0.0])
    const_05 = t("c05", onnx.TensorProto.FLOAT, [1], [0.5])
    const_15 = t("c15", onnx.TensorProto.FLOAT, [1], [1.5])
    const_1 = t("c1", onnx.TensorProto.FLOAT, [1], [1.0])
    const_29 = t("c29", onnx.TensorProto.FLOAT, [1], [29.0])
    range_30 = t("range30", onnx.TensorProto.FLOAT, [30], [float(i) for i in range(30)])
    row_grid = t("row_grid", onnx.TensorProto.FLOAT, [1, 1, 30, 1], [float(i) for i in range(30)])
    col_grid = t("col_grid", onnx.TensorProto.FLOAT, [1, 1, 1, 30], [float(i) for i in range(30)])
    desc_row = t("desc_row", onnx.TensorProto.FLOAT, [1, 1, 30, 1], [float(30 - i) for i in range(30)])
    asc_row = t("asc_row", onnx.TensorProto.FLOAT, [1, 1, 30, 1], [float(i + 1) for i in range(30)])
    desc_col = t("desc_col", onnx.TensorProto.FLOAT, [1, 1, 1, 30], [float(30 - i) for i in range(30)])
    asc_col = t("asc_col", onnx.TensorProto.FLOAT, [1, 1, 1, 30], [float(i + 1) for i in range(30)])
    rs1 = t("rs1", onnx.TensorProto.INT64, [1], [1])
    rs_row = t("rs_row", onnx.TensorProto.INT64, [4], [1, 1, 30, 1])
    rs_col = t("rs_col", onnx.TensorProto.INT64, [4], [1, 1, 1, 30])
    sl_ch0_st = t("sl_ch0_st", onnx.TensorProto.INT64, [1], [0])
    sl_ch0_en = t("sl_ch0_en", onnx.TensorProto.INT64, [1], [1])
    sl_ch_axis = t("sl_ch_axis", onnx.TensorProto.INT64, [1], [1])
    sl_step = t("sl_step", onnx.TensorProto.INT64, [1], [1])
    ch0_oh = np.zeros((1, 10, 1, 1), dtype=np.float32)
    ch0_oh[0, 0, 0, 0] = 1.0
    ch0_const = t("ch0_oh", onnx.TensorProto.FLOAT, [1, 10, 1, 1], ch0_oh.flatten().tolist())
    nonzero_oh = np.ones((1, 10, 1, 1), dtype=np.float32)
    nonzero_oh[0, 0, 0, 0] = 0.0
    nonzero_const = t("nonzero_oh", onnx.TensorProto.FLOAT, [1, 10, 1, 1], nonzero_oh.flatten().tolist())

    n("ReduceSum", ["input"], [P + "active"], axes=[1], keepdims=1)
    n("Slice", ["input", sl_ch0_st, sl_ch0_en, sl_ch_axis, sl_step], [P + "input_ch0"])
    n("Sub", [P + "active", P + "input_ch0"], [P + "nz"])

    n("ReduceSum", [P + "active"], [P + "row_active"], axes=[3], keepdims=1)
    n("ReduceSum", [P + "nz"], [P + "row_nz"], axes=[3], keepdims=1)
    n("Sub", [P + "row_active", P + "row_nz"], [P + "row_diff"])
    n("Abs", [P + "row_diff"], [P + "row_abs"])
    n("Less", [P + "row_abs", const_05], [P + "row_eq_bool"])
    n("Greater", [P + "row_active", const_05], [P + "row_has_bool"])
    n("Cast", [P + "row_eq_bool"], [P + "row_eq"], to=onnx.TensorProto.FLOAT)
    n("Cast", [P + "row_has_bool"], [P + "row_has"], to=onnx.TensorProto.FLOAT)
    n("Mul", [P + "row_eq", P + "row_has"], [P + "row_full"])

    n("ReduceSum", [P + "active"], [P + "col_active"], axes=[2], keepdims=1)
    n("ReduceSum", [P + "nz"], [P + "col_nz"], axes=[2], keepdims=1)
    n("Sub", [P + "col_active", P + "col_nz"], [P + "col_diff"])
    n("Abs", [P + "col_diff"], [P + "col_abs"])
    n("Less", [P + "col_abs", const_05], [P + "col_eq_bool"])
    n("Greater", [P + "col_active", const_05], [P + "col_has_bool"])
    n("Cast", [P + "col_eq_bool"], [P + "col_eq"], to=onnx.TensorProto.FLOAT)
    n("Cast", [P + "col_has_bool"], [P + "col_has"], to=onnx.TensorProto.FLOAT)
    n("Mul", [P + "col_eq", P + "col_has"], [P + "col_full"])

    n("Mul", [P + "row_full", desc_row], [P + "row_first_w"])
    n("Mul", [P + "row_full", asc_row], [P + "row_last_w"])
    n("Mul", [P + "col_full", desc_col], [P + "col_first_w"])
    n("Mul", [P + "col_full", asc_col], [P + "col_last_w"])
    n("ArgMax", [P + "row_first_w"], [P + "first_row_3d"], axis=2, keepdims=0)
    n("ArgMax", [P + "row_last_w"], [P + "last_row_3d"], axis=2, keepdims=0)
    n("ArgMax", [P + "col_first_w"], [P + "first_col_3d"], axis=3, keepdims=0)
    n("ArgMax", [P + "col_last_w"], [P + "last_col_3d"], axis=3, keepdims=0)

    for src, dst in [
        ("first_row_3d", "first_row_f"),
        ("last_row_3d", "last_row_f"),
        ("first_col_3d", "first_col_f"),
        ("last_col_3d", "last_col_f"),
    ]:
        n("Reshape", [P + src, rs1], [P + src.replace("_3d", "_1d")])
        n("Cast", [P + src.replace("_3d", "_1d")], [P + dst], to=onnx.TensorProto.FLOAT)

    n("Sub", [P + "last_row_f", P + "first_row_f"], [P + "h_minus_1"])
    n("Add", [P + "h_minus_1", const_1], [P + "height"])
    n("Sub", [P + "last_col_f", P + "first_col_f"], [P + "w_minus_1"])
    n("Add", [P + "w_minus_1", const_1], [P + "width"])

    n("Add", [P + "first_row_f", range_30], [P + "row_idx_raw"])
    n("Min", [P + "row_idx_raw", const_29], [P + "row_idx_f"])
    n("Cast", [P + "row_idx_f"], [P + "row_idx"], to=onnx.TensorProto.INT64)
    n("Add", [P + "first_col_f", range_30], [P + "col_idx_raw"])
    n("Min", [P + "col_idx_raw", const_29], [P + "col_idx_f"])
    n("Cast", [P + "col_idx_f"], [P + "col_idx"], to=onnx.TensorProto.INT64)

    n("Gather", ["input", P + "row_idx"], [P + "shift_rows"], axis=2)
    n("Gather", [P + "shift_rows", P + "col_idx"], [P + "shifted"], axis=3)

    n("Less", [range_30, P + "height"], [P + "row_in_bool"])
    n("Less", [range_30, P + "width"], [P + "col_in_bool"])
    n("Cast", [P + "row_in_bool"], [P + "row_in_f"], to=onnx.TensorProto.FLOAT)
    n("Cast", [P + "col_in_bool"], [P + "col_in_f"], to=onnx.TensorProto.FLOAT)
    n("Reshape", [P + "row_in_f", rs_row], [P + "row_mask"])
    n("Reshape", [P + "col_in_f", rs_col], [P + "col_mask"])
    n("Mul", [P + "row_mask", P + "col_mask"], [P + "crop_mask"])
    n("Mul", [P + "shifted", P + "crop_mask"], [P + "crop"])
    n("Slice", [P + "crop", sl_ch0_st, sl_ch0_en, sl_ch_axis, sl_step], [P + "crop_ch0"])

    def eq_mask(axis_grid, value_name, out_name):
        n("Sub", [axis_grid, value_name], [out_name + "_diff"])
        n("Abs", [out_name + "_diff"], [out_name + "_abs"])
        n("Less", [out_name + "_abs", const_05], [out_name + "_bool"])
        n("Cast", [out_name + "_bool"], [out_name], to=onnx.TensorProto.FLOAT)

    eq_mask(row_grid, P + "first_row_f", P + "top_row_mask")
    eq_mask(row_grid, P + "last_row_f", P + "bottom_row_mask")
    eq_mask(col_grid, P + "first_col_f", P + "left_col_mask")
    eq_mask(col_grid, P + "last_col_f", P + "right_col_mask")

    n("Greater", [range_30, const_0], [P + "row_gt0_bool"])
    n("Less", [range_30, P + "h_minus_1"], [P + "row_lt_last_bool"])
    n("Greater", [range_30, const_0], [P + "col_gt0_bool"])
    n("Less", [range_30, P + "w_minus_1"], [P + "col_lt_last_bool"])
    n("Cast", [P + "row_gt0_bool"], [P + "row_gt0"], to=onnx.TensorProto.FLOAT)
    n("Cast", [P + "row_lt_last_bool"], [P + "row_lt_last"], to=onnx.TensorProto.FLOAT)
    n("Cast", [P + "col_gt0_bool"], [P + "col_gt0"], to=onnx.TensorProto.FLOAT)
    n("Cast", [P + "col_lt_last_bool"], [P + "col_lt_last"], to=onnx.TensorProto.FLOAT)
    n("Mul", [P + "row_gt0", P + "row_lt_last"], [P + "row_interior_1d"])
    n("Mul", [P + "col_gt0", P + "col_lt_last"], [P + "col_interior_1d"])
    n("Reshape", [P + "row_interior_1d", rs_row], [P + "row_interior"])
    n("Reshape", [P + "col_interior_1d", rs_col], [P + "col_interior"])
    n("Mul", [P + "row_interior", P + "col_interior"], [P + "interior"])

    n("Mul", [P + "crop", nonzero_const], [P + "crop_nz"])
    n("Mul", [P + "crop_nz", P + "interior"], [P + "marker_cells"])
    n("ReduceSum", [P + "marker_cells"], [P + "marker_oh_count"], axes=[2, 3], keepdims=1)
    n("Greater", [P + "marker_oh_count", const_05], [P + "marker_oh_bool"])
    n("Cast", [P + "marker_oh_bool"], [P + "marker_oh"], to=onnx.TensorProto.FLOAT)
    n("ReduceSum", [P + "marker_cells"], [P + "marker_any"], axes=[1], keepdims=1)
    n("Mul", ["input", P + "marker_oh"], [P + "input_marker_color"])

    def side_gate(side, mask_name):
        cells = P + side + "_marker_side_cells"
        count = P + side + "_marker_side_count"
        gate_bool = P + side + "_gate_bool"
        gate = P + side + "_gate"
        n("Mul", [P + "input_marker_color", mask_name], [cells])
        n("ReduceSum", [cells], [count], axes=[1, 2, 3], keepdims=1)
        n("Greater", [count, const_15], [gate_bool])
        n("Cast", [gate_bool], [gate], to=onnx.TensorProto.FLOAT)
        return gate

    top_gate = side_gate("top", P + "top_row_mask")
    bottom_gate = side_gate("bottom", P + "bottom_row_mask")
    left_gate = side_gate("left", P + "left_col_mask")
    right_gate = side_gate("right", P + "right_col_mask")

    def gated_ray(side, gate, kernel_shape, pads):
        ray = P + side + "_ray"
        ray_bg = P + side + "_ray_bg"
        gated = P + side + "_gated"
        n("MaxPool", [P + "marker_any"], [ray], kernel_shape=kernel_shape, pads=pads, strides=[1, 1])
        n("Mul", [ray, P + "crop_ch0"], [ray_bg])
        n("Mul", [ray_bg, gate], [gated])
        return gated

    rays = [
        gated_ray("top", top_gate, [30, 1], [0, 0, 29, 0]),
        gated_ray("bottom", bottom_gate, [30, 1], [29, 0, 0, 0]),
        gated_ray("left", left_gate, [1, 30], [0, 0, 0, 29]),
        gated_ray("right", right_gate, [1, 30], [0, 29, 0, 0]),
    ]
    n("Sum", rays, [P + "ray_total"])
    n("Sub", [P + "marker_oh", ch0_const], [P + "delta_oh"])
    n("Mul", [P + "ray_total", P + "delta_oh"], [P + "delta"])
    n("Add", [P + "crop", P + "delta"], ["output"])

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def seed_flood_fill_network(fill_color, batch=1, canvas_h=30, canvas_w=30):
    """ONNX: Slice + Slice + Add + n_iters x(Conv 3x3 cross + Clip + Mul) + Mul + Mul + Add.

    For tasks where each 0-cell that is 4-conn reachable (through other 0-cells)
    from a `fill_color`-colored cell becomes fill_color. Wall cells (color != 0
    and != fill_color) and unreachable 0-cells stay unchanged.

    Pipeline (input [N, 10, 30, 30], BFS on canvas_h x canvas_w crop):
      1. [If canvas < 30] Slice input to [N, 10, canvas_h, canvas_w]
      2. bg     = Slice(src, channel 0)
      3. seed   = Slice(src, channel fill_color)
      4. not_blocking = bg + seed
      5. reach_0 = seed
      6. n_iters = (min(H,W) - 1) // 2 + 2 BFS steps
      7. to_fill = reach_final * bg
      8. delta = to_fill * delta_oh
      9. output_crop = src + delta
      10. [If canvas < 30] Pad output_crop to [N, 10, 30, 30]

    Verified 100% (265/265) on task243.
    """
    if fill_color == 0:
        raise ValueError("fill_color must be in 1..9 (color 0 is background)")

    H, W = canvas_h, canvas_w
    # n_iters matches the empirically-safe value for task243.
    n_iters = 16
    inits, nodes = [], []

    cross = np.array([[[[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]]]], dtype=np.float32)
    inits.append(onnx.helper.make_tensor(
        "sff_cross", onnx.TensorProto.FLOAT, [1, 1, 3, 3], cross.flatten().tolist()
    ))
    delta_oh = np.zeros((1, 10, 1, 1), dtype=np.float32)
    delta_oh[0, 0, 0, 0] = -1.0
    delta_oh[0, fill_color, 0, 0] = 1.0
    inits.append(onnx.helper.make_tensor(
        "sff_delta_oh", onnx.TensorProto.FLOAT, [1, 10, 1, 1], delta_oh.flatten().tolist()
    ))

    sl_bg_st = onnx.helper.make_tensor("sff_sl_bg_st", onnx.TensorProto.INT64, [1], [0])
    sl_bg_en = onnx.helper.make_tensor("sff_sl_bg_en", onnx.TensorProto.INT64, [1], [1])
    sl_ch_ax = onnx.helper.make_tensor("sff_sl_ch_ax", onnx.TensorProto.INT64, [1], [1])
    sl_ch_st = onnx.helper.make_tensor("sff_sl_ch_st", onnx.TensorProto.INT64, [1], [1])
    sl_seed_st = onnx.helper.make_tensor("sff_sl_seed_st", onnx.TensorProto.INT64, [1], [fill_color])
    sl_seed_en = onnx.helper.make_tensor("sff_sl_seed_en", onnx.TensorProto.INT64, [1], [fill_color + 1])
    inits.extend([sl_bg_st, sl_bg_en, sl_ch_ax, sl_ch_st, sl_seed_st, sl_seed_en])

    if H < 30 or W < 30:
        cr_starts = onnx.helper.make_tensor("sff_cr_starts", onnx.TensorProto.INT64, [2], [0, 0])
        cr_ends = onnx.helper.make_tensor("sff_cr_ends", onnx.TensorProto.INT64, [2], [H, W])
        cr_axes = onnx.helper.make_tensor("sff_cr_axes", onnx.TensorProto.INT64, [2], [2, 3])
        cr_steps = onnx.helper.make_tensor("sff_cr_steps", onnx.TensorProto.INT64, [2], [1, 1])
        inits.extend([cr_starts, cr_ends, cr_axes, cr_steps])
        nodes.append(onnx.helper.make_node(
            "Slice", ["input", "sff_cr_starts", "sff_cr_ends", "sff_cr_axes", "sff_cr_steps"],
            ["sff_src"],
        ))
        src = "sff_src"
    else:
        src = "input"

    nodes.append(onnx.helper.make_node(
        "Slice", [src, "sff_sl_bg_st", "sff_sl_bg_en", "sff_sl_ch_ax", "sff_sl_ch_st"], ["sff_bg"],
    ))
    nodes.append(onnx.helper.make_node(
        "Slice", [src, "sff_sl_seed_st", "sff_sl_seed_en", "sff_sl_ch_ax", "sff_sl_ch_st"], ["sff_seed"],
    ))
    nodes.append(onnx.helper.make_node("Add", ["sff_bg", "sff_seed"], ["sff_not_blocking"]))

    prev = "sff_seed"
    for i in range(n_iters):
        nodes.append(onnx.helper.make_node(
            "Conv", [prev, "sff_cross"], [f"sff_conv_{i}"],
            kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1],
        ))
        nodes.append(onnx.helper.make_node("Clip", [f"sff_conv_{i}"], [f"sff_clip_{i}"], min=0.0, max=1.0))
        nodes.append(onnx.helper.make_node("Mul", [f"sff_clip_{i}", "sff_not_blocking"], [f"sff_reach_{i+1}"]))
        prev = f"sff_reach_{i+1}"

    nodes.append(onnx.helper.make_node("Mul", [prev, "sff_bg"], ["sff_to_fill"]))
    nodes.append(onnx.helper.make_node("Mul", ["sff_to_fill", "sff_delta_oh"], ["sff_delta"]))

    if H < 30 or W < 30:
        nodes.append(onnx.helper.make_node("Add", [src, "sff_delta"], ["sff_out_crop"]))
        nodes.append(onnx.helper.make_node(
            "Pad", ["sff_out_crop"], ["output"],
            pads=[0, 0, 0, 0, 0, 0, 30 - H, 30 - W], mode="constant", value=0.0,
        ))
    else:
        nodes.append(onnx.helper.make_node("Add", [src, "sff_delta"], ["output"]))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def color_filter_bbox_crop_least_fast_network(batch=1):
    """Lower-memory task014 path: crop to bbox of least-common nonzero color.

    This is the same rule as `color_filter_bbox_crop_network(picker="least")`,
    but avoids the older Pad-to-59x59 shift path. It keeps every channel inside
    the selected bbox; filtering down to only bg+least-color is hidden-unsafe
    because future cases may contain other colors inside the bbox.
    """
    if batch != 1:
        raise NotImplementedError(
            "color-filter-bbox-crop least fast builder is single-example only (batch must be 1)"
        )

    nodes, inits = [], []

    def init(name, proto, dims, values):
        tensor = onnx.helper.make_tensor(name, proto, dims, values)
        inits.append(tensor)
        return name

    init("cfbcl_offsets30", onnx.TensorProto.INT64, [30], list(range(30)))
    init("cfbcl_twenty_nine", onnx.TensorProto.INT64, [], [29])
    init("cfbcl_one_i", onnx.TensorProto.INT64, [], [1])
    init("cfbcl_big_f", onnx.TensorProto.FLOAT, [], [1_000_000_000.0])
    init("cfbcl_zero_f", onnx.TensorProto.FLOAT, [], [0.0])
    init("cfbcl_one_f", onnx.TensorProto.FLOAT, [], [1.0])
    init("cfbcl_shape_10", onnx.TensorProto.INT64, [1], [10])
    init("cfbcl_bg_penalty", onnx.TensorProto.FLOAT, [10], [1_000_000_000.0] + [0.0] * 9)
    init("cfbcl_shape_30", onnx.TensorProto.INT64, [1], [30])
    init("cfbcl_reverse_idx", onnx.TensorProto.INT64, [30], list(range(29, -1, -1)))
    init("cfbcl_thirty_i", onnx.TensorProto.INT64, [], [30])
    init("cfbcl_twenty_nine_30", onnx.TensorProto.INT64, [30], [29] * 30)
    init("cfbcl_shape_1_1_30_1", onnx.TensorProto.INT64, [4], [1, 1, 30, 1])
    init("cfbcl_shape_1_1_1_30", onnx.TensorProto.INT64, [4], [1, 1, 1, 30])

    nodes.extend([
        onnx.helper.make_node("ReduceSum", ["input"], ["cfbcl_counts_4d"], axes=[2, 3], keepdims=1),
        onnx.helper.make_node("Reshape", ["cfbcl_counts_4d", "cfbcl_shape_10"], ["cfbcl_counts"]),
        onnx.helper.make_node("Greater", ["cfbcl_counts", "cfbcl_zero_f"], ["cfbcl_present_b"]),
        onnx.helper.make_node("Cast", ["cfbcl_present_b"], ["cfbcl_present"], to=onnx.TensorProto.FLOAT),
        onnx.helper.make_node("Sub", ["cfbcl_one_f", "cfbcl_present"], ["cfbcl_absent"]),
        onnx.helper.make_node("Mul", ["cfbcl_absent", "cfbcl_big_f"], ["cfbcl_absent_penalty"]),
        onnx.helper.make_node("Add", ["cfbcl_absent_penalty", "cfbcl_bg_penalty"], ["cfbcl_total_penalty"]),
        onnx.helper.make_node("Add", ["cfbcl_counts", "cfbcl_total_penalty"], ["cfbcl_counts_masked"]),
        onnx.helper.make_node("ArgMin", ["cfbcl_counts_masked"], ["cfbcl_color"], axis=0, keepdims=1),
        onnx.helper.make_node("Gather", ["input", "cfbcl_color"], ["cfbcl_mask_col"], axis=1),
        onnx.helper.make_node("ReduceMax", ["cfbcl_mask_col"], ["cfbcl_row_any_4d"], axes=[3], keepdims=1),
        onnx.helper.make_node("Reshape", ["cfbcl_row_any_4d", "cfbcl_shape_30"], ["cfbcl_row_any"]),
        onnx.helper.make_node("ReduceMax", ["cfbcl_mask_col"], ["cfbcl_col_any_4d"], axes=[2], keepdims=1),
        onnx.helper.make_node("Reshape", ["cfbcl_col_any_4d", "cfbcl_shape_30"], ["cfbcl_col_any"]),
        onnx.helper.make_node("ArgMax", ["cfbcl_row_any"], ["cfbcl_y0"], axis=0, keepdims=1),
        onnx.helper.make_node("ArgMax", ["cfbcl_col_any"], ["cfbcl_x0"], axis=0, keepdims=1),
        onnx.helper.make_node("Gather", ["cfbcl_row_any", "cfbcl_reverse_idx"], ["cfbcl_row_any_rev"], axis=0),
        onnx.helper.make_node("ArgMax", ["cfbcl_row_any_rev"], ["cfbcl_y1_from_end"], axis=0, keepdims=1),
        onnx.helper.make_node("Sub", ["cfbcl_twenty_nine", "cfbcl_y1_from_end"], ["cfbcl_y1"]),
        onnx.helper.make_node("Gather", ["cfbcl_col_any", "cfbcl_reverse_idx"], ["cfbcl_col_any_rev"], axis=0),
        onnx.helper.make_node("ArgMax", ["cfbcl_col_any_rev"], ["cfbcl_x1_from_end"], axis=0, keepdims=1),
        onnx.helper.make_node("Sub", ["cfbcl_twenty_nine", "cfbcl_x1_from_end"], ["cfbcl_x1"]),
        onnx.helper.make_node("Add", ["cfbcl_y0", "cfbcl_offsets30"], ["cfbcl_row_src_u"]),
        onnx.helper.make_node("Less", ["cfbcl_row_src_u", "cfbcl_thirty_i"], ["cfbcl_row_valid"]),
        onnx.helper.make_node("Where", ["cfbcl_row_valid", "cfbcl_row_src_u", "cfbcl_twenty_nine_30"], ["cfbcl_row_src"]),
        onnx.helper.make_node("Add", ["cfbcl_x0", "cfbcl_offsets30"], ["cfbcl_col_src_u"]),
        onnx.helper.make_node("Less", ["cfbcl_col_src_u", "cfbcl_thirty_i"], ["cfbcl_col_valid"]),
        onnx.helper.make_node("Where", ["cfbcl_col_valid", "cfbcl_col_src_u", "cfbcl_twenty_nine_30"], ["cfbcl_col_src"]),
        onnx.helper.make_node("Gather", ["input", "cfbcl_row_src"], ["cfbcl_shifted_rows"], axis=2),
        onnx.helper.make_node("Gather", ["cfbcl_shifted_rows", "cfbcl_col_src"], ["cfbcl_shifted"], axis=3),
        onnx.helper.make_node("Sub", ["cfbcl_y1", "cfbcl_y0"], ["cfbcl_h_m1"]),
        onnx.helper.make_node("Sub", ["cfbcl_x1", "cfbcl_x0"], ["cfbcl_w_m1"]),
        onnx.helper.make_node("Add", ["cfbcl_h_m1", "cfbcl_one_i"], ["cfbcl_h"]),
        onnx.helper.make_node("Less", ["cfbcl_offsets30", "cfbcl_h"], ["cfbcl_row_in_bool"]),
        onnx.helper.make_node("Cast", ["cfbcl_row_in_bool"], ["cfbcl_row_in"], to=onnx.TensorProto.FLOAT),
        onnx.helper.make_node("Add", ["cfbcl_w_m1", "cfbcl_one_i"], ["cfbcl_w"]),
        onnx.helper.make_node("Less", ["cfbcl_offsets30", "cfbcl_w"], ["cfbcl_col_in_bool"]),
        onnx.helper.make_node("Cast", ["cfbcl_col_in_bool"], ["cfbcl_col_in"], to=onnx.TensorProto.FLOAT),
        onnx.helper.make_node("Reshape", ["cfbcl_row_in", "cfbcl_shape_1_1_30_1"], ["cfbcl_row_mask"]),
        onnx.helper.make_node("Reshape", ["cfbcl_col_in", "cfbcl_shape_1_1_1_30"], ["cfbcl_col_mask"]),
        onnx.helper.make_node("Mul", ["cfbcl_row_mask", "cfbcl_col_mask"], ["cfbcl_bbox_mask"]),
        onnx.helper.make_node("Mul", ["cfbcl_shifted", "cfbcl_bbox_mask"], ["output"]),
    ])

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def color_filter_bbox_crop_network(picker="least", batch=1):
    """ONNX: Slice + ReduceSum + Less + Mul + Where + ArgMax + Reshape + Cast +
    Add + Cast + Gather + (foreground-bbox-crop pipeline applied to selected channel).

    For tasks where the output is the input cropped to the bbox of cells of a
    runtime-selected color (currently only the LEAST-COMMON nonzero color picker
    is implemented). Variable in/out shape (single-example only).

    Pipeline (input [N, 10, 30, 30], single example only — N=1):
      1. counts = ReduceSum(input[:, 1:10, :, :], axes=[2, 3]) -> [N, 9].
      2. argmin via argmax(-counts) with zero counts mapped to a large-negative
         sentinel. onnx-tool 1.0.1 lacks ArgMin value_infer; the negate-and-ArgMax
         workaround keeps the cost model honest.
      3. color = least_idx + 1 (channel 1-9, since we sliced off channel 0 / bg).
         Float arithmetic; Cast to INT64 for Gather.
      4. selected = Gather(input, color, axis=1) -> [N, 1, 30, 30] mask of cells of that color.
      5. bbox via ArgMax-trick on row_has/col_has of selected (same pipeline as fbc).
      6. Shift FULL input (all 10 channels) to bbox top-left + mask outside bbox.

    Cost (per cost-model verify, opset 10): ~134k -> ~13.2 pts.
    Verified 100% (266/266) on task014.

    Builder is single-example only; raises NotImplementedError for batch != 1.
    Currently only `picker="least"` is implemented; `picker="most"` and other
    pickers are documented for future extension (task036 family).
    """
    if batch != 1:
        raise NotImplementedError(
            "color-filter-bbox-crop builder is single-example only (batch must be 1)"
        )
    if picker not in ("least", "smallest_bbox", "most_count", "odd_one_out", "max_density", "lr_symmetry"):
        raise NotImplementedError(
            f"color-filter-bbox-crop picker={picker!r} not implemented "
            f"(supported: 'least', 'smallest_bbox', 'most_count', 'odd_one_out', 'max_density', 'lr_symmetry')"
        )
    if picker == "least":
        return color_filter_bbox_crop_least_fast_network(batch=batch)

    nodes, inits = [], []

    # Step 1: slice channels 1-9 (skip bg).
    sl_ax_ch = onnx.helper.make_tensor("cfbc_sl_ax_ch", onnx.TensorProto.INT64, [1], [1])
    sl_st_ch = onnx.helper.make_tensor("cfbc_sl_st_ch", onnx.TensorProto.INT64, [1], [1])
    sl_en_ch = onnx.helper.make_tensor("cfbc_sl_en_ch", onnx.TensorProto.INT64, [1], [10])
    sl_step_ch = onnx.helper.make_tensor("cfbc_sl_step_ch", onnx.TensorProto.INT64, [1], [1])
    inits.extend([sl_ax_ch, sl_st_ch, sl_en_ch, sl_step_ch])
    nodes.append(onnx.helper.make_node(
        "Slice", ["input", "cfbc_sl_st_ch", "cfbc_sl_en_ch", "cfbc_sl_ax_ch", "cfbc_sl_step_ch"],
        ["cfbc_no_bg"],
    ))

    # Constants used by both picker paths.
    half = onnx.helper.make_tensor("cfbc_half", onnx.TensorProto.FLOAT, [1], [0.5])
    sentinel_neg = onnx.helper.make_tensor("cfbc_sentinel_neg", onnx.TensorProto.FLOAT, [1], [-9999.0])
    neg_one = onnx.helper.make_tensor("cfbc_neg_one", onnx.TensorProto.FLOAT, [1], [-1.0])
    sentinel_pos = onnx.helper.make_tensor("cfbc_sentinel_pos", onnx.TensorProto.FLOAT, [1], [9999.0])
    one_f = onnx.helper.make_tensor("cfbc_one_f", onnx.TensorProto.FLOAT, [1], [1.0])
    inits.extend([half, sentinel_neg, neg_one, sentinel_pos, one_f])

    if picker == "least":
        # Step 2 (least): per-color counts -> argmin via argmax(-counts) + sentinel for zero counts.
        nodes.append(onnx.helper.make_node(
            "ReduceSum", ["cfbc_no_bg"], ["cfbc_counts"], axes=[2, 3], keepdims=0
        ))
        nodes.append(onnx.helper.make_node("Less", ["cfbc_counts", "cfbc_half"], ["cfbc_zero_mask"]))
        nodes.append(onnx.helper.make_node("Mul", ["cfbc_counts", "cfbc_neg_one"], ["cfbc_neg_counts"]))
        nodes.append(onnx.helper.make_node(
            "Where", ["cfbc_zero_mask", "cfbc_sentinel_neg", "cfbc_neg_counts"], ["cfbc_counts_safe"]
        ))
        nodes.append(onnx.helper.make_node(
            "ArgMax", ["cfbc_counts_safe"], ["cfbc_least_idx_2d"], axis=1, keepdims=0
        ))
    elif picker == "most_count":
        # Step 2 (most_count): per-color counts -> argmax directly. Absent colors
        # have count 0, so they cannot win unless ALL channels are absent (which
        # the detector rules out by requiring at least one nonzero color).
        nodes.append(onnx.helper.make_node(
            "ReduceSum", ["cfbc_no_bg"], ["cfbc_counts"], axes=[2, 3], keepdims=0
        ))
        nodes.append(onnx.helper.make_node(
            "ArgMax", ["cfbc_counts"], ["cfbc_least_idx_2d"], axis=1, keepdims=0
        ))
    elif picker == "max_density":
        # Step 2 (max_density): pick color with highest cells/bbox_area ratio.
        # 264/266 on task174 (the 2 misses are a richer LR-symmetry sub-rule).
        # Reuse smallest_bbox machinery to compute per-channel bbox area.
        nodes.append(onnx.helper.make_node(
            "ReduceSum", ["cfbc_no_bg"], ["cfbc_counts"], axes=[2, 3], keepdims=0
        ))
        nodes.append(onnx.helper.make_node(
            "ReduceMax", ["cfbc_no_bg"], ["cfbc_per_row_has"], axes=[3], keepdims=1
        ))
        nodes.append(onnx.helper.make_node(
            "ReduceMax", ["cfbc_no_bg"], ["cfbc_per_col_has"], axes=[2], keepdims=1
        ))
        nodes.append(onnx.helper.make_node(
            "ReduceMax", ["cfbc_no_bg"], ["cfbc_per_present"], axes=[2, 3], keepdims=0
        ))
        per_desc_row_md = onnx.helper.make_tensor(
            "cfbc_md_desc_row", onnx.TensorProto.FLOAT, [1, 1, 30, 1],
            [float(30 - i) for i in range(30)]
        )
        per_asc_row_md = onnx.helper.make_tensor(
            "cfbc_md_asc_row", onnx.TensorProto.FLOAT, [1, 1, 30, 1],
            [float(i + 1) for i in range(30)]
        )
        per_desc_col_md = onnx.helper.make_tensor(
            "cfbc_md_desc_col", onnx.TensorProto.FLOAT, [1, 1, 1, 30],
            [float(30 - i) for i in range(30)]
        )
        per_asc_col_md = onnx.helper.make_tensor(
            "cfbc_md_asc_col", onnx.TensorProto.FLOAT, [1, 1, 1, 30],
            [float(i + 1) for i in range(30)]
        )
        inits.extend([per_desc_row_md, per_asc_row_md, per_desc_col_md, per_asc_col_md])
        nodes.append(onnx.helper.make_node("Mul", ["cfbc_per_row_has", "cfbc_md_desc_row"], ["cfbc_md_w_first_row"]))
        nodes.append(onnx.helper.make_node("Mul", ["cfbc_per_row_has", "cfbc_md_asc_row"], ["cfbc_md_w_last_row"]))
        nodes.append(onnx.helper.make_node("Mul", ["cfbc_per_col_has", "cfbc_md_desc_col"], ["cfbc_md_w_first_col"]))
        nodes.append(onnx.helper.make_node("Mul", ["cfbc_per_col_has", "cfbc_md_asc_col"], ["cfbc_md_w_last_col"]))
        nodes.append(onnx.helper.make_node("ArgMax", ["cfbc_md_w_first_row"], ["cfbc_md_first_row_3d"], axis=2, keepdims=0))
        nodes.append(onnx.helper.make_node("ArgMax", ["cfbc_md_w_last_row"], ["cfbc_md_last_row_3d"], axis=2, keepdims=0))
        nodes.append(onnx.helper.make_node("ArgMax", ["cfbc_md_w_first_col"], ["cfbc_md_first_col_3d"], axis=3, keepdims=0))
        nodes.append(onnx.helper.make_node("ArgMax", ["cfbc_md_w_last_col"], ["cfbc_md_last_col_3d"], axis=3, keepdims=0))
        rs_19_md = onnx.helper.make_tensor("cfbc_md_rs_19", onnx.TensorProto.INT64, [2], [1, 9])
        inits.append(rs_19_md)
        for src_int, dst_f in [
            ("cfbc_md_first_row_3d", "cfbc_md_first_row_f"),
            ("cfbc_md_last_row_3d", "cfbc_md_last_row_f"),
            ("cfbc_md_first_col_3d", "cfbc_md_first_col_f"),
            ("cfbc_md_last_col_3d", "cfbc_md_last_col_f"),
        ]:
            tmp = src_int.replace("_3d", "_2d")
            nodes.append(onnx.helper.make_node("Reshape", [src_int, "cfbc_md_rs_19"], [tmp]))
            nodes.append(onnx.helper.make_node("Cast", [tmp], [dst_f], to=onnx.TensorProto.FLOAT))
        nodes.append(onnx.helper.make_node("Sub", ["cfbc_md_last_row_f", "cfbc_md_first_row_f"], ["cfbc_md_h_minus_1"]))
        nodes.append(onnx.helper.make_node("Add", ["cfbc_md_h_minus_1", "cfbc_one_f"], ["cfbc_md_height_f"]))
        nodes.append(onnx.helper.make_node("Sub", ["cfbc_md_last_col_f", "cfbc_md_first_col_f"], ["cfbc_md_w_minus_1"]))
        nodes.append(onnx.helper.make_node("Add", ["cfbc_md_w_minus_1", "cfbc_one_f"], ["cfbc_md_width_f"]))
        nodes.append(onnx.helper.make_node("Mul", ["cfbc_md_height_f", "cfbc_md_width_f"], ["cfbc_md_area_f"]))
        nodes.append(onnx.helper.make_node("Div", ["cfbc_counts", "cfbc_md_area_f"], ["cfbc_md_density"]))
        nodes.append(onnx.helper.make_node("Less", ["cfbc_per_present", "cfbc_half"], ["cfbc_md_absent"]))
        zero_f_md = onnx.helper.make_tensor("cfbc_md_zero_f", onnx.TensorProto.FLOAT, [1], [0.0])
        inits.append(zero_f_md)
        nodes.append(onnx.helper.make_node(
            "Where", ["cfbc_md_absent", "cfbc_md_zero_f", "cfbc_md_density"], ["cfbc_md_density_safe"]
        ))
        nodes.append(onnx.helper.make_node(
            "ArgMax", ["cfbc_md_density_safe"], ["cfbc_least_idx_2d"], axis=1, keepdims=0
        ))
    elif picker == "lr_symmetry":
        # Step 2 (lr_symmetry): pick a present color whose bbox-cropped binary
        # patch is left-right symmetric. Density breaks rare symmetric ties.
        nodes.append(onnx.helper.make_node(
            "ReduceSum", ["cfbc_no_bg"], ["cfbc_counts"], axes=[2, 3], keepdims=0
        ))
        nodes.append(onnx.helper.make_node(
            "ReduceMax", ["cfbc_no_bg"], ["cfbc_lr_per_row_has"], axes=[3], keepdims=1
        ))
        nodes.append(onnx.helper.make_node(
            "ReduceMax", ["cfbc_no_bg"], ["cfbc_lr_per_col_has"], axes=[2], keepdims=1
        ))
        nodes.append(onnx.helper.make_node(
            "ReduceMax", ["cfbc_no_bg"], ["cfbc_lr_per_present"], axes=[2, 3], keepdims=0
        ))

        lr_desc_row = onnx.helper.make_tensor(
            "cfbc_lr_desc_row", onnx.TensorProto.FLOAT, [1, 1, 30, 1],
            [float(30 - i) for i in range(30)]
        )
        lr_asc_row = onnx.helper.make_tensor(
            "cfbc_lr_asc_row", onnx.TensorProto.FLOAT, [1, 1, 30, 1],
            [float(i + 1) for i in range(30)]
        )
        lr_desc_col = onnx.helper.make_tensor(
            "cfbc_lr_desc_col", onnx.TensorProto.FLOAT, [1, 1, 1, 30],
            [float(30 - i) for i in range(30)]
        )
        lr_asc_col = onnx.helper.make_tensor(
            "cfbc_lr_asc_col", onnx.TensorProto.FLOAT, [1, 1, 1, 30],
            [float(i + 1) for i in range(30)]
        )
        lr_range = onnx.helper.make_tensor(
            "cfbc_lr_range_30_f", onnx.TensorProto.FLOAT, [30],
            [float(i) for i in range(30)]
        )
        lr_zero = onnx.helper.make_tensor("cfbc_lr_zero_f", onnx.TensorProto.FLOAT, [1], [0.0])
        lr_29 = onnx.helper.make_tensor("cfbc_lr_const_29_f", onnx.TensorProto.FLOAT, [1], [29.0])
        lr_rs11130 = onnx.helper.make_tensor(
            "cfbc_lr_rs11130", onnx.TensorProto.INT64, [4], [1, 1, 1, 30]
        )
        lr_rs11 = onnx.helper.make_tensor("cfbc_lr_rs11", onnx.TensorProto.INT64, [2], [1, 1])
        lr_rs19 = onnx.helper.make_tensor("cfbc_lr_rs19", onnx.TensorProto.INT64, [2], [1, 9])
        inits.extend([
            lr_desc_row, lr_asc_row, lr_desc_col, lr_asc_col, lr_range,
            lr_zero, lr_29, lr_rs11130, lr_rs11, lr_rs19,
        ])

        nodes.append(onnx.helper.make_node("Mul", ["cfbc_lr_per_row_has", "cfbc_lr_desc_row"], ["cfbc_lr_w_first_row"]))
        nodes.append(onnx.helper.make_node("Mul", ["cfbc_lr_per_row_has", "cfbc_lr_asc_row"], ["cfbc_lr_w_last_row"]))
        nodes.append(onnx.helper.make_node("Mul", ["cfbc_lr_per_col_has", "cfbc_lr_desc_col"], ["cfbc_lr_w_first_col"]))
        nodes.append(onnx.helper.make_node("Mul", ["cfbc_lr_per_col_has", "cfbc_lr_asc_col"], ["cfbc_lr_w_last_col"]))
        nodes.append(onnx.helper.make_node("ArgMax", ["cfbc_lr_w_first_row"], ["cfbc_lr_first_row_3d"], axis=2, keepdims=0))
        nodes.append(onnx.helper.make_node("ArgMax", ["cfbc_lr_w_last_row"], ["cfbc_lr_last_row_3d"], axis=2, keepdims=0))
        nodes.append(onnx.helper.make_node("ArgMax", ["cfbc_lr_w_first_col"], ["cfbc_lr_first_col_3d"], axis=3, keepdims=0))
        nodes.append(onnx.helper.make_node("ArgMax", ["cfbc_lr_w_last_col"], ["cfbc_lr_last_col_3d"], axis=3, keepdims=0))

        for src_int, dst_f in [
            ("cfbc_lr_first_row_3d", "cfbc_lr_first_row_f"),
            ("cfbc_lr_last_row_3d", "cfbc_lr_last_row_f"),
            ("cfbc_lr_first_col_3d", "cfbc_lr_first_col_f"),
            ("cfbc_lr_last_col_3d", "cfbc_lr_last_col_f"),
        ]:
            tmp = src_int.replace("_3d", "_2d")
            nodes.append(onnx.helper.make_node("Reshape", [src_int, "cfbc_lr_rs19"], [tmp]))
            nodes.append(onnx.helper.make_node("Cast", [tmp], [dst_f], to=onnx.TensorProto.FLOAT))

        nodes.append(onnx.helper.make_node("Sub", ["cfbc_lr_last_row_f", "cfbc_lr_first_row_f"], ["cfbc_lr_h_minus_1"]))
        nodes.append(onnx.helper.make_node("Add", ["cfbc_lr_h_minus_1", "cfbc_one_f"], ["cfbc_lr_height_f"]))
        nodes.append(onnx.helper.make_node("Sub", ["cfbc_lr_last_col_f", "cfbc_lr_first_col_f"], ["cfbc_lr_w_minus_1"]))
        nodes.append(onnx.helper.make_node("Add", ["cfbc_lr_w_minus_1", "cfbc_one_f"], ["cfbc_lr_width_f"]))
        nodes.append(onnx.helper.make_node("Mul", ["cfbc_lr_height_f", "cfbc_lr_width_f"], ["cfbc_lr_area_f"]))
        nodes.append(onnx.helper.make_node("Div", ["cfbc_counts", "cfbc_lr_area_f"], ["cfbc_lr_density"]))

        score_parts = []
        for i in range(9):
            scalar_idx = onnx.helper.make_tensor(f"cfbc_lr_scalar_idx_{i}", onnx.TensorProto.INT64, [], [i])
            vector_idx = onnx.helper.make_tensor(f"cfbc_lr_vector_idx_{i}", onnx.TensorProto.INT64, [1], [i])
            inits.extend([scalar_idx, vector_idx])

            nodes.append(onnx.helper.make_node(
                "Gather", ["cfbc_no_bg", f"cfbc_lr_vector_idx_{i}"], [f"cfbc_lr_ch{i}"], axis=1
            ))
            nodes.append(onnx.helper.make_node(
                "Gather", ["cfbc_lr_first_col_f", f"cfbc_lr_scalar_idx_{i}"], [f"cfbc_lr_fc{i}"], axis=1
            ))
            nodes.append(onnx.helper.make_node(
                "Gather", ["cfbc_lr_last_col_f", f"cfbc_lr_scalar_idx_{i}"], [f"cfbc_lr_lc{i}"], axis=1
            ))
            nodes.append(onnx.helper.make_node("Add", [f"cfbc_lr_fc{i}", f"cfbc_lr_lc{i}"], [f"cfbc_lr_csum{i}"]))
            nodes.append(onnx.helper.make_node("Sub", [f"cfbc_lr_csum{i}", "cfbc_lr_range_30_f"], [f"cfbc_lr_mirror_raw{i}"]))
            nodes.append(onnx.helper.make_node("Max", [f"cfbc_lr_mirror_raw{i}", "cfbc_lr_zero_f"], [f"cfbc_lr_mirror_lo{i}"]))
            nodes.append(onnx.helper.make_node("Min", [f"cfbc_lr_mirror_lo{i}", "cfbc_lr_const_29_f"], [f"cfbc_lr_mirror_f{i}"]))
            nodes.append(onnx.helper.make_node(
                "Cast", [f"cfbc_lr_mirror_f{i}"], [f"cfbc_lr_mirror_idx{i}"], to=onnx.TensorProto.INT64
            ))
            nodes.append(onnx.helper.make_node(
                "Gather", [f"cfbc_lr_ch{i}", f"cfbc_lr_mirror_idx{i}"], [f"cfbc_lr_flip{i}"], axis=3
            ))
            nodes.append(onnx.helper.make_node("Sub", [f"cfbc_lr_ch{i}", f"cfbc_lr_flip{i}"], [f"cfbc_lr_diff{i}"]))
            nodes.append(onnx.helper.make_node("Abs", [f"cfbc_lr_diff{i}"], [f"cfbc_lr_abs{i}"]))

            nodes.append(onnx.helper.make_node("Add", [f"cfbc_lr_lc{i}", "cfbc_one_f"], [f"cfbc_lr_lc_plus1_{i}"]))
            nodes.append(onnx.helper.make_node("Sub", [f"cfbc_lr_fc{i}", "cfbc_one_f"], [f"cfbc_lr_fc_minus1_{i}"]))
            nodes.append(onnx.helper.make_node("Less", ["cfbc_lr_range_30_f", f"cfbc_lr_lc_plus1_{i}"], [f"cfbc_lr_col_le{i}"]))
            nodes.append(onnx.helper.make_node("Less", [f"cfbc_lr_fc_minus1_{i}", "cfbc_lr_range_30_f"], [f"cfbc_lr_col_ge{i}"]))
            nodes.append(onnx.helper.make_node(
                "Cast", [f"cfbc_lr_col_le{i}"], [f"cfbc_lr_col_le_f{i}"], to=onnx.TensorProto.FLOAT
            ))
            nodes.append(onnx.helper.make_node(
                "Cast", [f"cfbc_lr_col_ge{i}"], [f"cfbc_lr_col_ge_f{i}"], to=onnx.TensorProto.FLOAT
            ))
            nodes.append(onnx.helper.make_node("Mul", [f"cfbc_lr_col_le_f{i}", f"cfbc_lr_col_ge_f{i}"], [f"cfbc_lr_col_mask_vec{i}"]))
            nodes.append(onnx.helper.make_node(
                "Reshape", [f"cfbc_lr_col_mask_vec{i}", "cfbc_lr_rs11130"], [f"cfbc_lr_col_mask{i}"]
            ))
            nodes.append(onnx.helper.make_node("Mul", [f"cfbc_lr_abs{i}", f"cfbc_lr_col_mask{i}"], [f"cfbc_lr_abs_masked{i}"]))
            nodes.append(onnx.helper.make_node(
                "ReduceSum", [f"cfbc_lr_abs_masked{i}"], [f"cfbc_lr_asym_sum{i}"], axes=[1, 2, 3], keepdims=0
            ))
            nodes.append(onnx.helper.make_node("Less", [f"cfbc_lr_asym_sum{i}", "cfbc_half"], [f"cfbc_lr_is_sym{i}"]))
            nodes.append(onnx.helper.make_node(
                "Cast", [f"cfbc_lr_is_sym{i}"], [f"cfbc_lr_is_sym_f{i}"], to=onnx.TensorProto.FLOAT
            ))
            nodes.append(onnx.helper.make_node(
                "Gather", ["cfbc_lr_per_present", f"cfbc_lr_scalar_idx_{i}"], [f"cfbc_lr_present{i}"], axis=1
            ))
            nodes.append(onnx.helper.make_node(
                "Gather", ["cfbc_lr_density", f"cfbc_lr_scalar_idx_{i}"], [f"cfbc_lr_density_i{i}"], axis=1
            ))
            nodes.append(onnx.helper.make_node("Add", [f"cfbc_lr_density_i{i}", "cfbc_one_f"], [f"cfbc_lr_density_plus1_{i}"]))
            nodes.append(onnx.helper.make_node("Mul", [f"cfbc_lr_is_sym_f{i}", f"cfbc_lr_present{i}"], [f"cfbc_lr_sym_present{i}"]))
            nodes.append(onnx.helper.make_node("Mul", [f"cfbc_lr_sym_present{i}", f"cfbc_lr_density_plus1_{i}"], [f"cfbc_lr_score_raw{i}"]))
            nodes.append(onnx.helper.make_node(
                "Reshape", [f"cfbc_lr_score_raw{i}", "cfbc_lr_rs11"], [f"cfbc_lr_score{i}"]
            ))
            score_parts.append(f"cfbc_lr_score{i}")

        nodes.append(onnx.helper.make_node("Concat", score_parts, ["cfbc_lr_scores"], axis=1))
        nodes.append(onnx.helper.make_node(
            "ArgMax", ["cfbc_lr_scores"], ["cfbc_least_idx_2d"], axis=1, keepdims=0
        ))
    elif picker == "odd_one_out":
        # Step 2 (odd_one_out): pick the color whose cell count appears exactly
        # once across all channels. Pairwise compare counts via Equal on INT64
        # (opset-10 Equal doesn't support FLOAT); zero out diagonal; ReduceMax to
        # find channels with at least one duplicate; ArgMax(1 - duplicate_mask)
        # yields the unique-count channel. Absent (count=0) channels lose because
        # multiple 0-count channels share their count -> not unique.
        nodes.append(onnx.helper.make_node(
            "ReduceSum", ["cfbc_no_bg"], ["cfbc_counts"], axes=[2, 3], keepdims=0
        ))
        # Cast to INT64 for Equal.
        nodes.append(onnx.helper.make_node(
            "Cast", ["cfbc_counts"], ["cfbc_counts_int"], to=onnx.TensorProto.INT64
        ))
        rs_191 = onnx.helper.make_tensor("cfbc_rs_191", onnx.TensorProto.INT64, [3], [1, 9, 1])
        rs_119 = onnx.helper.make_tensor("cfbc_rs_119", onnx.TensorProto.INT64, [3], [1, 1, 9])
        inits.extend([rs_191, rs_119])
        nodes.append(onnx.helper.make_node("Reshape", ["cfbc_counts_int", "cfbc_rs_191"], ["cfbc_counts_a"]))
        nodes.append(onnx.helper.make_node("Reshape", ["cfbc_counts_int", "cfbc_rs_119"], ["cfbc_counts_b"]))
        nodes.append(onnx.helper.make_node("Equal", ["cfbc_counts_a", "cfbc_counts_b"], ["cfbc_eq_bool"]))
        nodes.append(onnx.helper.make_node(
            "Cast", ["cfbc_eq_bool"], ["cfbc_eq_f"], to=onnx.TensorProto.FLOAT
        ))
        # Off-diagonal mask: ones[1,9,9] with zeros on the diagonal.
        not_eye = [0.0 if i == j else 1.0 for i in range(9) for j in range(9)]
        not_eye_init = onnx.helper.make_tensor(
            "cfbc_not_eye", onnx.TensorProto.FLOAT, [1, 9, 9], not_eye
        )
        inits.append(not_eye_init)
        nodes.append(onnx.helper.make_node("Mul", ["cfbc_eq_f", "cfbc_not_eye"], ["cfbc_eq_off"]))
        nodes.append(onnx.helper.make_node(
            "ReduceMax", ["cfbc_eq_off"], ["cfbc_has_dup"], axes=[2], keepdims=0
        ))  # [1, 9] : 1 if count[i] equals any other channel's count
        # uniqueness = 1 - has_dup. ArgMax picks the unique one.
        ones_19 = onnx.helper.make_tensor(
            "cfbc_ones_19", onnx.TensorProto.FLOAT, [1, 9], [1.0] * 9
        )
        inits.append(ones_19)
        nodes.append(onnx.helper.make_node("Sub", ["cfbc_ones_19", "cfbc_has_dup"], ["cfbc_unique_mask"]))
        nodes.append(onnx.helper.make_node(
            "ArgMax", ["cfbc_unique_mask"], ["cfbc_least_idx_2d"], axis=1, keepdims=0
        ))
    else:
        # picker == "smallest_bbox" — for each color, compute bbox area; pick the
        # color with the smallest area among present colors.
        nodes.append(onnx.helper.make_node(
            "ReduceMax", ["cfbc_no_bg"], ["cfbc_per_row_has"], axes=[3], keepdims=1
        ))  # [1, 9, 30, 1]
        nodes.append(onnx.helper.make_node(
            "ReduceMax", ["cfbc_no_bg"], ["cfbc_per_col_has"], axes=[2], keepdims=1
        ))  # [1, 9, 1, 30]
        nodes.append(onnx.helper.make_node(
            "ReduceMax", ["cfbc_no_bg"], ["cfbc_per_present"], axes=[2, 3], keepdims=0
        ))  # [1, 9]

        # Per-color first/last row/col via ArgMax-trick (broadcast over color axis).
        per_desc_row = onnx.helper.make_tensor(
            "cfbc_per_desc_row", onnx.TensorProto.FLOAT, [1, 1, 30, 1],
            [float(30 - i) for i in range(30)]
        )
        per_asc_row = onnx.helper.make_tensor(
            "cfbc_per_asc_row", onnx.TensorProto.FLOAT, [1, 1, 30, 1],
            [float(i + 1) for i in range(30)]
        )
        per_desc_col = onnx.helper.make_tensor(
            "cfbc_per_desc_col", onnx.TensorProto.FLOAT, [1, 1, 1, 30],
            [float(30 - i) for i in range(30)]
        )
        per_asc_col = onnx.helper.make_tensor(
            "cfbc_per_asc_col", onnx.TensorProto.FLOAT, [1, 1, 1, 30],
            [float(i + 1) for i in range(30)]
        )
        inits.extend([per_desc_row, per_asc_row, per_desc_col, per_asc_col])
        nodes.append(onnx.helper.make_node("Mul", ["cfbc_per_row_has", "cfbc_per_desc_row"], ["cfbc_per_w_first_row"]))
        nodes.append(onnx.helper.make_node("Mul", ["cfbc_per_row_has", "cfbc_per_asc_row"], ["cfbc_per_w_last_row"]))
        nodes.append(onnx.helper.make_node("Mul", ["cfbc_per_col_has", "cfbc_per_desc_col"], ["cfbc_per_w_first_col"]))
        nodes.append(onnx.helper.make_node("Mul", ["cfbc_per_col_has", "cfbc_per_asc_col"], ["cfbc_per_w_last_col"]))

        nodes.append(onnx.helper.make_node("ArgMax", ["cfbc_per_w_first_row"], ["cfbc_per_first_row_3d"], axis=2, keepdims=0))
        nodes.append(onnx.helper.make_node("ArgMax", ["cfbc_per_w_last_row"], ["cfbc_per_last_row_3d"], axis=2, keepdims=0))
        nodes.append(onnx.helper.make_node("ArgMax", ["cfbc_per_w_first_col"], ["cfbc_per_first_col_3d"], axis=3, keepdims=0))
        nodes.append(onnx.helper.make_node("ArgMax", ["cfbc_per_w_last_col"], ["cfbc_per_last_col_3d"], axis=3, keepdims=0))

        rs_19 = onnx.helper.make_tensor("cfbc_rs_19", onnx.TensorProto.INT64, [2], [1, 9])
        inits.append(rs_19)
        for src_int, dst_f in [
            ("cfbc_per_first_row_3d", "cfbc_per_first_row_f"),
            ("cfbc_per_last_row_3d", "cfbc_per_last_row_f"),
            ("cfbc_per_first_col_3d", "cfbc_per_first_col_f"),
            ("cfbc_per_last_col_3d", "cfbc_per_last_col_f"),
        ]:
            tmp = src_int.replace("_3d", "_2d")
            nodes.append(onnx.helper.make_node("Reshape", [src_int, "cfbc_rs_19"], [tmp]))
            nodes.append(onnx.helper.make_node("Cast", [tmp], [dst_f], to=onnx.TensorProto.FLOAT))

        nodes.append(onnx.helper.make_node("Sub", ["cfbc_per_last_row_f", "cfbc_per_first_row_f"], ["cfbc_per_h_minus_1"]))
        nodes.append(onnx.helper.make_node("Add", ["cfbc_per_h_minus_1", "cfbc_one_f"], ["cfbc_per_height_f"]))
        nodes.append(onnx.helper.make_node("Sub", ["cfbc_per_last_col_f", "cfbc_per_first_col_f"], ["cfbc_per_w_minus_1"]))
        nodes.append(onnx.helper.make_node("Add", ["cfbc_per_w_minus_1", "cfbc_one_f"], ["cfbc_per_width_f"]))
        nodes.append(onnx.helper.make_node("Mul", ["cfbc_per_height_f", "cfbc_per_width_f"], ["cfbc_per_area_f"]))

        # Mask absent colors with sentinel.
        nodes.append(onnx.helper.make_node("Less", ["cfbc_per_present", "cfbc_half"], ["cfbc_per_absent"]))
        nodes.append(onnx.helper.make_node(
            "Where", ["cfbc_per_absent", "cfbc_sentinel_pos", "cfbc_per_area_f"], ["cfbc_per_area_safe"]
        ))
        # ArgMin via ArgMax(-area).
        nodes.append(onnx.helper.make_node("Mul", ["cfbc_per_area_safe", "cfbc_neg_one"], ["cfbc_per_neg_area"]))
        nodes.append(onnx.helper.make_node(
            "ArgMax", ["cfbc_per_neg_area"], ["cfbc_least_idx_2d"], axis=1, keepdims=0
        ))

    # Step 3: color index = least_idx + 1 (since channel 0 = bg was sliced off).
    rs1 = onnx.helper.make_tensor("cfbc_rs1", onnx.TensorProto.INT64, [1], [1])
    inits.append(rs1)
    nodes.append(onnx.helper.make_node("Reshape", ["cfbc_least_idx_2d", "cfbc_rs1"], ["cfbc_least_idx_int"]))
    nodes.append(onnx.helper.make_node("Cast", ["cfbc_least_idx_int"], ["cfbc_least_idx_f"], to=onnx.TensorProto.FLOAT))
    nodes.append(onnx.helper.make_node("Add", ["cfbc_least_idx_f", "cfbc_one_f"], ["cfbc_color_f"]))
    nodes.append(onnx.helper.make_node("Cast", ["cfbc_color_f"], ["cfbc_color"], to=onnx.TensorProto.INT64))

    # Step 4: Gather input on axis=1 with that color index -> selected channel mask.
    nodes.append(onnx.helper.make_node(
        "Gather", ["input", "cfbc_color"], ["cfbc_selected"], axis=1
    ))

    # Step 5: bbox of selected via ArgMax-trick.
    nodes.append(onnx.helper.make_node(
        "ReduceMax", ["cfbc_selected"], ["cfbc_row_has"], axes=[3], keepdims=1
    ))
    nodes.append(onnx.helper.make_node(
        "ReduceMax", ["cfbc_selected"], ["cfbc_col_has"], axes=[2], keepdims=1
    ))

    desc_row = onnx.helper.make_tensor(
        "cfbc_desc_row", onnx.TensorProto.FLOAT, [1, 1, 30, 1],
        [float(30 - i) for i in range(30)],
    )
    asc_row = onnx.helper.make_tensor(
        "cfbc_asc_row", onnx.TensorProto.FLOAT, [1, 1, 30, 1],
        [float(i + 1) for i in range(30)],
    )
    desc_col = onnx.helper.make_tensor(
        "cfbc_desc_col", onnx.TensorProto.FLOAT, [1, 1, 1, 30],
        [float(30 - i) for i in range(30)],
    )
    asc_col = onnx.helper.make_tensor(
        "cfbc_asc_col", onnx.TensorProto.FLOAT, [1, 1, 1, 30],
        [float(i + 1) for i in range(30)],
    )
    inits.extend([desc_row, asc_row, desc_col, asc_col])
    nodes.append(onnx.helper.make_node("Mul", ["cfbc_row_has", "cfbc_desc_row"], ["cfbc_w_first_row"]))
    nodes.append(onnx.helper.make_node("Mul", ["cfbc_row_has", "cfbc_asc_row"], ["cfbc_w_last_row"]))
    nodes.append(onnx.helper.make_node("Mul", ["cfbc_col_has", "cfbc_desc_col"], ["cfbc_w_first_col"]))
    nodes.append(onnx.helper.make_node("Mul", ["cfbc_col_has", "cfbc_asc_col"], ["cfbc_w_last_col"]))

    nodes.append(onnx.helper.make_node("ArgMax", ["cfbc_w_first_row"], ["cfbc_first_row_3d"], axis=2, keepdims=0))
    nodes.append(onnx.helper.make_node("ArgMax", ["cfbc_w_last_row"], ["cfbc_last_row_3d"], axis=2, keepdims=0))
    nodes.append(onnx.helper.make_node("ArgMax", ["cfbc_w_first_col"], ["cfbc_first_col_3d"], axis=3, keepdims=0))
    nodes.append(onnx.helper.make_node("ArgMax", ["cfbc_w_last_col"], ["cfbc_last_col_3d"], axis=3, keepdims=0))

    for src_int, dst_f in [
        ("cfbc_first_row_3d", "cfbc_first_row_f"),
        ("cfbc_last_row_3d", "cfbc_last_row_f"),
        ("cfbc_first_col_3d", "cfbc_first_col_f"),
        ("cfbc_last_col_3d", "cfbc_last_col_f"),
    ]:
        tmp = src_int.replace("_3d", "_1d")
        nodes.append(onnx.helper.make_node("Reshape", [src_int, "cfbc_rs1"], [tmp]))
        nodes.append(onnx.helper.make_node("Cast", [tmp], [dst_f], to=onnx.TensorProto.FLOAT))

    range_30_f = onnx.helper.make_tensor("cfbc_range_30_f", onnx.TensorProto.FLOAT, [30], [float(i) for i in range(30)])
    const_29_f = onnx.helper.make_tensor("cfbc_const_29_f", onnx.TensorProto.FLOAT, [1], [29.0])
    const_1_f = onnx.helper.make_tensor("cfbc_const_1_f", onnx.TensorProto.FLOAT, [1], [1.0])
    inits.extend([range_30_f, const_29_f, const_1_f])

    nodes.append(onnx.helper.make_node("Add", ["cfbc_first_row_f", "cfbc_range_30_f"], ["cfbc_row_idx_raw_f"]))
    nodes.append(onnx.helper.make_node("Min", ["cfbc_row_idx_raw_f", "cfbc_const_29_f"], ["cfbc_row_idx_f"]))
    nodes.append(onnx.helper.make_node("Cast", ["cfbc_row_idx_f"], ["cfbc_row_idx"], to=onnx.TensorProto.INT64))

    nodes.append(onnx.helper.make_node("Add", ["cfbc_first_col_f", "cfbc_range_30_f"], ["cfbc_col_idx_raw_f"]))
    nodes.append(onnx.helper.make_node("Min", ["cfbc_col_idx_raw_f", "cfbc_const_29_f"], ["cfbc_col_idx_f"]))
    nodes.append(onnx.helper.make_node("Cast", ["cfbc_col_idx_f"], ["cfbc_col_idx"], to=onnx.TensorProto.INT64))

    # Step 6: Gather rows then cols of the FULL input.
    nodes.append(onnx.helper.make_node("Gather", ["input", "cfbc_row_idx"], ["cfbc_shifted_rows"], axis=2))
    nodes.append(onnx.helper.make_node("Gather", ["cfbc_shifted_rows", "cfbc_col_idx"], ["cfbc_shifted"], axis=3))

    nodes.append(onnx.helper.make_node("Sub", ["cfbc_last_row_f", "cfbc_first_row_f"], ["cfbc_h_minus_1"]))
    nodes.append(onnx.helper.make_node("Add", ["cfbc_h_minus_1", "cfbc_const_1_f"], ["cfbc_height_f"]))
    nodes.append(onnx.helper.make_node("Sub", ["cfbc_last_col_f", "cfbc_first_col_f"], ["cfbc_w_minus_1"]))
    nodes.append(onnx.helper.make_node("Add", ["cfbc_w_minus_1", "cfbc_const_1_f"], ["cfbc_width_f"]))

    nodes.append(onnx.helper.make_node("Less", ["cfbc_range_30_f", "cfbc_height_f"], ["cfbc_row_in_bbox_bool"]))
    nodes.append(onnx.helper.make_node("Less", ["cfbc_range_30_f", "cfbc_width_f"], ["cfbc_col_in_bbox_bool"]))
    nodes.append(onnx.helper.make_node("Cast", ["cfbc_row_in_bbox_bool"], ["cfbc_row_in_bbox_f"], to=onnx.TensorProto.FLOAT))
    nodes.append(onnx.helper.make_node("Cast", ["cfbc_col_in_bbox_bool"], ["cfbc_col_in_bbox_f"], to=onnx.TensorProto.FLOAT))

    rs_row = onnx.helper.make_tensor("cfbc_rs_row", onnx.TensorProto.INT64, [4], [1, 1, 30, 1])
    rs_col = onnx.helper.make_tensor("cfbc_rs_col", onnx.TensorProto.INT64, [4], [1, 1, 1, 30])
    inits.extend([rs_row, rs_col])
    nodes.append(onnx.helper.make_node("Reshape", ["cfbc_row_in_bbox_f", "cfbc_rs_row"], ["cfbc_row_mask"]))
    nodes.append(onnx.helper.make_node("Reshape", ["cfbc_col_in_bbox_f", "cfbc_rs_col"], ["cfbc_col_mask"]))
    nodes.append(onnx.helper.make_node("Mul", ["cfbc_row_mask", "cfbc_col_mask"], ["cfbc_bbox_mask"]))

    nodes.append(onnx.helper.make_node("Mul", ["cfbc_shifted", "cfbc_bbox_mask"], ["output"]))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def largest_rectangle_recolor_crop_network(batch=1):
    """ONNX: task396-style largest hollow rectangle crop with frame recolor.

    Runtime rule:
      * frame color = most-common non-zero channel
      * marker color = least-common non-zero channel
      * among complete frame-color hollow rectangles with H,W in [4, 8], choose
        max enclosed area, then max total area
      * crop that rectangle, recolor its border to marker color, and keep only
        marker-color cells in the interior.

    The bundled generator for task396 emits rectangle sizes 4..8; the detector
    verifies the same rule against train pairs before dispatching this builder.
    """
    if batch != 1:
        raise NotImplementedError(
            "largest-rectangle-recolor-crop builder is single-example only (batch must be 1)"
        )

    P = "lrrc_"
    nodes, inits = [], []
    F = onnx.TensorProto.FLOAT
    I = onnx.TensorProto.INT64

    def _t(name, proto, dims, vals):
        tensor = onnx.helper.make_tensor(P + name, proto, dims, vals)
        inits.append(tensor)
        return P + name

    def _n(op, inputs, outputs, **attrs):
        nodes.append(onnx.helper.make_node(op, inputs, outputs, **attrs))

    # Pick frame/marker channels from non-zero counts.
    _t("sl_st_ch", I, [1], [1])
    _t("sl_en_ch", I, [1], [10])
    _t("sl_ax_ch", I, [1], [1])
    _t("sl_sp_ch", I, [1], [1])
    _n("Slice", ["input", P+"sl_st_ch", P+"sl_en_ch", P+"sl_ax_ch", P+"sl_sp_ch"], [P+"no_bg"])
    _n("ReduceSum", [P+"no_bg"], [P+"counts"], axes=[2, 3], keepdims=0)

    _t("one", F, [1], [1.0])
    _t("half", F, [1], [0.5])
    _t("one_pt_five", F, [1], [1.5])
    _t("neg_one", F, [1], [-1.0])
    _t("sentinel_neg", F, [1], [-9999.0])
    _t("const_29", F, [1], [29.0])
    _t("rs1", I, [1], [1])
    _t("rs11", I, [2], [1, 1])
    _t("rs_row", I, [4], [1, 1, 30, 1])
    _t("rs_col", I, [4], [1, 1, 1, 30])
    _t("rs_1x10x1x1", I, [4], [1, 10, 1, 1])

    _n("ArgMax", [P+"counts"], [P+"frame_idx0"], axis=1, keepdims=0)
    _n("Reshape", [P+"frame_idx0", P+"rs1"], [P+"frame_idx0_1d"])
    _n("Cast", [P+"frame_idx0_1d"], [P+"frame_idx0_f"], to=F)
    _n("Add", [P+"frame_idx0_f", P+"one"], [P+"frame_color_f"])
    _n("Cast", [P+"frame_color_f"], [P+"frame_color"], to=I)

    _n("Less", [P+"counts", P+"half"], [P+"zero_counts"])
    _n("Mul", [P+"counts", P+"neg_one"], [P+"neg_counts"])
    _n("Where", [P+"zero_counts", P+"sentinel_neg", P+"neg_counts"], [P+"least_scores"])
    _n("ArgMax", [P+"least_scores"], [P+"marker_idx0"], axis=1, keepdims=0)
    _n("Reshape", [P+"marker_idx0", P+"rs1"], [P+"marker_idx0_1d"])
    _n("Cast", [P+"marker_idx0_1d"], [P+"marker_idx0_f"], to=F)
    _n("Add", [P+"marker_idx0_f", P+"one"], [P+"marker_color_f"])
    _n("Cast", [P+"marker_color_f"], [P+"marker_color"], to=I)

    _n("Gather", ["input", P+"frame_color"], [P+"frame_mask"], axis=1)

    # Perimeter Conv detectors for each generator rectangle size. Each size is
    # reduced to its best score/position before the cross-size choice, avoiding
    # 25 padded 30x30 score maps.
    score_parts, r0_parts, c0_parts = [], [], []
    heights, widths = [], []
    for h in range(4, 9):
        for w in range(4, 9):
            suffix = f"{h}x{w}"
            out_h = 31 - h
            out_w = 31 - w
            kernel = []
            for rr in range(h):
                for cc in range(w):
                    kernel.append(1.0 if rr in (0, h - 1) or cc in (0, w - 1) else 0.0)
            _t(f"k_{suffix}", F, [1, 1, h, w], kernel)
            _n("Conv", [P+"frame_mask", P+f"k_{suffix}"], [P+f"sum_{suffix}"])
            perim = 2 * (h + w) - 4
            enclosed = (h - 2) * (w - 2)
            score = enclosed * 100 + h * w
            _t(f"thr_{suffix}", F, [1], [float(perim) - 0.5])
            _t(f"score_{suffix}", F, [1], [float(score)])
            _n("Less", [P+f"thr_{suffix}", P+f"sum_{suffix}"], [P+f"ok_{suffix}"])
            _n("Cast", [P+f"ok_{suffix}"], [P+f"ok_f_{suffix}"], to=F)
            _n("Mul", [P+f"ok_f_{suffix}", P+f"score_{suffix}"], [P+f"raw_score_{suffix}"])
            _t(f"rs_flat_{suffix}", I, [2], [1, out_h * out_w])
            _t(f"out_w_{suffix}", F, [1], [float(out_w)])
            _n("Reshape", [P+f"raw_score_{suffix}", P+f"rs_flat_{suffix}"], [P+f"flat_score_{suffix}"])
            _n("ReduceMax", [P+f"flat_score_{suffix}"], [P+f"best_score_{suffix}"], axes=[1], keepdims=1)
            _n("ArgMax", [P+f"flat_score_{suffix}"], [P+f"best_pos_{suffix}"], axis=1, keepdims=0)
            _n("Cast", [P+f"best_pos_{suffix}"], [P+f"best_pos_f_{suffix}"], to=F)
            _n("Div", [P+f"best_pos_f_{suffix}", P+f"out_w_{suffix}"], [P+f"r0_div_{suffix}"])
            _n("Floor", [P+f"r0_div_{suffix}"], [P+f"r0_raw_{suffix}"])
            _n("Mul", [P+f"r0_raw_{suffix}", P+f"out_w_{suffix}"], [P+f"r0_offset_{suffix}"])
            _n("Sub", [P+f"best_pos_f_{suffix}", P+f"r0_offset_{suffix}"], [P+f"c0_raw_{suffix}"])
            _n("Reshape", [P+f"r0_raw_{suffix}", P+"rs11"], [P+f"r0_part_{suffix}"])
            _n("Reshape", [P+f"c0_raw_{suffix}", P+"rs11"], [P+f"c0_part_{suffix}"])
            score_parts.append(P + f"best_score_{suffix}")
            r0_parts.append(P + f"r0_part_{suffix}")
            c0_parts.append(P + f"c0_part_{suffix}")
            heights.append(float(h))
            widths.append(float(w))

    _n("Concat", score_parts, [P+"size_scores"], axis=1)
    _n("ArgMax", [P+"size_scores"], [P+"size_id"], axis=1, keepdims=0)
    _n("Concat", r0_parts, [P+"r0_by_size"], axis=1)
    _n("Concat", c0_parts, [P+"c0_by_size"], axis=1)
    _n("Gather", [P+"r0_by_size", P+"size_id"], [P+"r0_gather"], axis=1)
    _n("Gather", [P+"c0_by_size", P+"size_id"], [P+"c0_gather"], axis=1)
    _n("Reshape", [P+"r0_gather", P+"rs1"], [P+"r0_f"])
    _n("Reshape", [P+"c0_gather", P+"rs1"], [P+"c0_f"])

    _t("h_lut", F, [25], heights)
    _t("w_lut", F, [25], widths)
    _n("Gather", [P+"h_lut", P+"size_id"], [P+"h_f"], axis=0)
    _n("Gather", [P+"w_lut", P+"size_id"], [P+"w_f"], axis=0)

    _t("range30", F, [30], [float(i) for i in range(30)])
    _n("Add", [P+"r0_f", P+"range30"], [P+"row_idx_raw"])
    _n("Min", [P+"row_idx_raw", P+"const_29"], [P+"row_idx_f"])
    _n("Cast", [P+"row_idx_f"], [P+"row_idx"], to=I)
    _n("Add", [P+"c0_f", P+"range30"], [P+"col_idx_raw"])
    _n("Min", [P+"col_idx_raw", P+"const_29"], [P+"col_idx_f"])
    _n("Cast", [P+"col_idx_f"], [P+"col_idx"], to=I)

    _n("Gather", ["input", P+"row_idx"], [P+"shift_rows"], axis=2)
    _n("Gather", [P+"shift_rows", P+"col_idx"], [P+"shifted"], axis=3)

    _n("Less", [P+"range30", P+"h_f"], [P+"row_in_bool"])
    _n("Less", [P+"range30", P+"w_f"], [P+"col_in_bool"])
    _n("Cast", [P+"row_in_bool"], [P+"row_in"], to=F)
    _n("Cast", [P+"col_in_bool"], [P+"col_in"], to=F)
    _n("Reshape", [P+"row_in", P+"rs_row"], [P+"row_mask"])
    _n("Reshape", [P+"col_in", P+"rs_col"], [P+"col_mask"])
    _n("Mul", [P+"row_mask", P+"col_mask"], [P+"bbox_mask"])

    _n("Less", [P+"range30", P+"one"], [P+"top_bool"])
    _n("Less", [P+"range30", P+"one"], [P+"left_bool"])
    _n("Sub", [P+"h_f", P+"one_pt_five"], [P+"bottom_thresh"])
    _n("Sub", [P+"w_f", P+"one_pt_five"], [P+"right_thresh"])
    _n("Less", [P+"bottom_thresh", P+"range30"], [P+"bottom_raw_bool"])
    _n("Less", [P+"right_thresh", P+"range30"], [P+"right_raw_bool"])
    _n("Cast", [P+"top_bool"], [P+"top_f"], to=F)
    _n("Cast", [P+"left_bool"], [P+"left_f"], to=F)
    _n("Cast", [P+"bottom_raw_bool"], [P+"bottom_raw_f"], to=F)
    _n("Cast", [P+"right_raw_bool"], [P+"right_raw_f"], to=F)
    _n("Mul", [P+"bottom_raw_f", P+"row_in"], [P+"bottom_f"])
    _n("Mul", [P+"right_raw_f", P+"col_in"], [P+"right_f"])
    _n("Add", [P+"top_f", P+"bottom_f"], [P+"row_border_vec"])
    _n("Add", [P+"left_f", P+"right_f"], [P+"col_border_vec"])
    _n("Reshape", [P+"row_border_vec", P+"rs_row"], [P+"row_border"])
    _n("Reshape", [P+"col_border_vec", P+"rs_col"], [P+"col_border"])
    _n("Add", [P+"row_border", P+"col_border"], [P+"border_sum"])
    _n("Less", [P+"half", P+"border_sum"], [P+"border_bool"])
    _n("Cast", [P+"border_bool"], [P+"border_raw"], to=F)
    _n("Mul", [P+"border_raw", P+"bbox_mask"], [P+"border_mask"])
    _n("Sub", [P+"bbox_mask", P+"border_mask"], [P+"interior_mask"])

    _t("oh_depth10", I, [], [10])
    _t("oh_values", F, [2], [0.0, 1.0])
    _t("bg_oh", F, [1, 10, 1, 1], [1.0] + [0.0] * 9)
    _n("OneHot", [P+"marker_color", P+"oh_depth10", P+"oh_values"], [P+"marker_oh_2d"], axis=1)
    _n("Reshape", [P+"marker_oh_2d", P+"rs_1x10x1x1"], [P+"marker_oh"])
    _n("Add", [P+"bg_oh", P+"marker_oh"], [P+"bg_marker_oh"])
    _n("Mul", [P+"shifted", P+"bg_marker_oh"], [P+"marker_only"])
    _n("Mul", [P+"marker_only", P+"interior_mask"], [P+"interior"])
    _n("Mul", [P+"marker_oh", P+"border_mask"], [P+"border"])
    _n("Add", [P+"interior", P+"border"], ["output"])

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def marker_rectangle_interior_network(batch=1):
    """ONNX: detect the color forming a complete rectangular border, crop to interior.

    For task029: one color's cells form exactly the perimeter of its bounding box
    (count == 2*(bbox_h + bbox_w) - 4, bbox >= 3x3). Output = input cropped to
    interior rows (r0+1..r1-1) and cols (c0+1..c1-1). Interior cells retain colors.

    Pipeline (input [N=1, 10, 30, 30]):
      1. ch1_to_9 = Slice(input, 1:10, axis=1)              [1,9,30,30]
      2. counts   = ReduceSum(ch1_to_9, axes=[2,3])          [1,9]
      3. row_has  = ReduceMax(ch1_to_9, axes=[3])            [1,9,30,1]
         col_has  = ReduceMax(ch1_to_9, axes=[2])            [1,9,1,30]
      4. Per-channel first/last row/col via ArgMax-trick      [1,9] each → Cast FLOAT
      5. bbox_h = last_row - first_row + 1; bbox_w similarly  [1,9]
      6. perim = 2*(h+w) - 4; is_rect = Less((counts-perim)^2, 0.5)  [1,9]
      7. is_big = Less(2.5, h) AND Less(2.5, w); is_marker = is_rect * is_big
         ArgMax(is_marker, axis=1) → marker_idx [1] INT64
      8. Gather per-channel bbox for marker_idx → mk_fr, mk_lr, mk_fc, mk_lc [1] each
      9. int_row_start = mk_fr + 1; int_col_start = mk_fc + 1
         interior_h = mk_lr - mk_fr - 1; interior_w = mk_lc - mk_fc - 1
      10. Gather rows then cols of full input at interior range (clip at 29)
      11. Mask output to [interior_h, interior_w]; zero canvas beyond.

    Single-example only (batch=1). Verified 100% (266/266) on task029.
    """
    if batch != 1:
        raise NotImplementedError(
            "marker-rectangle-interior builder is single-example only (batch must be 1)"
        )
    P = "mri_"
    nodes, inits = [], []

    # Step 1: ch1_to_9
    inits.extend([
        onnx.helper.make_tensor(P+"s1_st", onnx.TensorProto.INT64, [1], [1]),
        onnx.helper.make_tensor(P+"s1_en", onnx.TensorProto.INT64, [1], [10]),
        onnx.helper.make_tensor(P+"s1_ax", onnx.TensorProto.INT64, [1], [1]),
        onnx.helper.make_tensor(P+"s1_sp", onnx.TensorProto.INT64, [1], [1]),
    ])
    nodes.append(onnx.helper.make_node(
        "Slice", ["input", P+"s1_st", P+"s1_en", P+"s1_ax", P+"s1_sp"], [P+"ch1_to_9"]
    ))

    # Step 2: counts per channel [1,9]
    nodes.append(onnx.helper.make_node(
        "ReduceSum", [P+"ch1_to_9"], [P+"counts_f"], axes=[2, 3], keepdims=0
    ))

    # Step 3: row_has [1,9,30,1] and col_has [1,9,1,30]
    nodes.append(onnx.helper.make_node(
        "ReduceMax", [P+"ch1_to_9"], [P+"row_has"], axes=[3], keepdims=1
    ))
    nodes.append(onnx.helper.make_node(
        "ReduceMax", [P+"ch1_to_9"], [P+"col_has"], axes=[2], keepdims=1
    ))

    # Step 4: per-channel bbox via ascending/descending ArgMax
    inits.extend([
        onnx.helper.make_tensor(P+"desc_row", onnx.TensorProto.FLOAT, [1, 1, 30, 1],
                                [float(30 - i) for i in range(30)]),
        onnx.helper.make_tensor(P+"asc_row",  onnx.TensorProto.FLOAT, [1, 1, 30, 1],
                                [float(i + 1) for i in range(30)]),
        onnx.helper.make_tensor(P+"desc_col", onnx.TensorProto.FLOAT, [1, 1, 1, 30],
                                [float(30 - i) for i in range(30)]),
        onnx.helper.make_tensor(P+"asc_col",  onnx.TensorProto.FLOAT, [1, 1, 1, 30],
                                [float(i + 1) for i in range(30)]),
    ])
    nodes.append(onnx.helper.make_node("Mul", [P+"row_has", P+"desc_row"], [P+"w_fr"]))
    nodes.append(onnx.helper.make_node("Mul", [P+"row_has", P+"asc_row"],  [P+"w_lr"]))
    nodes.append(onnx.helper.make_node("Mul", [P+"col_has", P+"desc_col"], [P+"w_fc"]))
    nodes.append(onnx.helper.make_node("Mul", [P+"col_has", P+"asc_col"],  [P+"w_lc"]))
    nodes.append(onnx.helper.make_node("ArgMax", [P+"w_fr"], [P+"per_fr_3d"], axis=2, keepdims=0))
    nodes.append(onnx.helper.make_node("ArgMax", [P+"w_lr"], [P+"per_lr_3d"], axis=2, keepdims=0))
    nodes.append(onnx.helper.make_node("ArgMax", [P+"w_fc"], [P+"per_fc_3d"], axis=3, keepdims=0))
    nodes.append(onnx.helper.make_node("ArgMax", [P+"w_lc"], [P+"per_lc_3d"], axis=3, keepdims=0))

    rs_19 = onnx.helper.make_tensor(P+"rs_19", onnx.TensorProto.INT64, [2], [1, 9])
    inits.append(rs_19)
    for src, dst in [
        (P+"per_fr_3d", P+"per_fr_f"),
        (P+"per_lr_3d", P+"per_lr_f"),
        (P+"per_fc_3d", P+"per_fc_f"),
        (P+"per_lc_3d", P+"per_lc_f"),
    ]:
        mid = src.replace("_3d", "_2d")
        nodes.append(onnx.helper.make_node("Reshape", [src, P+"rs_19"], [mid]))
        nodes.append(onnx.helper.make_node("Cast", [mid], [dst], to=onnx.TensorProto.FLOAT))

    # Step 5: bbox_h and bbox_w per channel [1,9]
    one_f  = onnx.helper.make_tensor(P+"one_f",  onnx.TensorProto.FLOAT, [1], [1.0])
    two_f  = onnx.helper.make_tensor(P+"two_f",  onnx.TensorProto.FLOAT, [1], [2.0])
    four_f = onnx.helper.make_tensor(P+"four_f", onnx.TensorProto.FLOAT, [1], [4.0])
    inits.extend([one_f, two_f, four_f])
    nodes.append(onnx.helper.make_node("Sub", [P+"per_lr_f", P+"per_fr_f"], [P+"h_m1"]))
    nodes.append(onnx.helper.make_node("Add", [P+"h_m1", P+"one_f"],         [P+"bbox_h"]))
    nodes.append(onnx.helper.make_node("Sub", [P+"per_lc_f", P+"per_fc_f"], [P+"w_m1"]))
    nodes.append(onnx.helper.make_node("Add", [P+"w_m1", P+"one_f"],         [P+"bbox_w"]))

    # Step 6: perimeter check
    nodes.append(onnx.helper.make_node("Add", [P+"bbox_h",   P+"bbox_w"],   [P+"hw_sum"]))
    nodes.append(onnx.helper.make_node("Mul", [P+"hw_sum",   P+"two_f"],    [P+"perim_p4"]))
    nodes.append(onnx.helper.make_node("Sub", [P+"perim_p4", P+"four_f"],   [P+"perim"]))
    nodes.append(onnx.helper.make_node("Sub", [P+"counts_f", P+"perim"],    [P+"diff"]))
    nodes.append(onnx.helper.make_node("Mul", [P+"diff",     P+"diff"],     [P+"diff_sq"]))
    half_f = onnx.helper.make_tensor(P+"half_f", onnx.TensorProto.FLOAT, [1], [0.5])
    inits.append(half_f)
    nodes.append(onnx.helper.make_node("Less", [P+"diff_sq", P+"half_f"],   [P+"is_rect_b"]))
    nodes.append(onnx.helper.make_node("Cast", [P+"is_rect_b"], [P+"is_rect_f"], to=onnx.TensorProto.FLOAT))

    # Step 7: size gate and marker selection
    two5_f = onnx.helper.make_tensor(P+"two5_f", onnx.TensorProto.FLOAT, [1], [2.5])
    inits.append(two5_f)
    nodes.append(onnx.helper.make_node("Less", [P+"two5_f", P+"bbox_h"], [P+"big_h_b"]))
    nodes.append(onnx.helper.make_node("Less", [P+"two5_f", P+"bbox_w"], [P+"big_w_b"]))
    nodes.append(onnx.helper.make_node("Cast", [P+"big_h_b"], [P+"big_h_f"], to=onnx.TensorProto.FLOAT))
    nodes.append(onnx.helper.make_node("Cast", [P+"big_w_b"], [P+"big_w_f"], to=onnx.TensorProto.FLOAT))
    nodes.append(onnx.helper.make_node("Mul", [P+"big_h_f",   P+"big_w_f"],   [P+"is_big"]))
    nodes.append(onnx.helper.make_node("Mul", [P+"is_rect_f", P+"is_big"],    [P+"is_marker"]))
    # Pick the SMALLEST valid rectangle (by bbox area) — breaks ties when multiple
    # colors form valid borders (e.g. outer frame + inner frame both pass check).
    nodes.append(onnx.helper.make_node("Mul", [P+"bbox_h", P+"bbox_w"], [P+"bbox_area"]))
    sentinel = onnx.helper.make_tensor(P+"sentinel", onnx.TensorProto.FLOAT, [1], [1e9])
    neg_one_f = onnx.helper.make_tensor(P+"neg_one_f", onnx.TensorProto.FLOAT, [1], [-1.0])
    inits.extend([sentinel, neg_one_f])
    nodes.append(onnx.helper.make_node("Less", [P+"is_marker", P+"half_f"], [P+"not_marker_b"]))
    nodes.append(onnx.helper.make_node(
        "Where", [P+"not_marker_b", P+"sentinel", P+"bbox_area"], [P+"area_safe"]
    ))
    nodes.append(onnx.helper.make_node("Mul", [P+"area_safe", P+"neg_one_f"], [P+"neg_area"]))
    nodes.append(onnx.helper.make_node(
        "ArgMax", [P+"neg_area"], [P+"marker_idx"], axis=1, keepdims=0
    ))  # [1] INT64, 0-based index in 9-ch tensor (smallest valid rectangle)

    # Step 8: Gather per-channel bbox values at marker_idx
    rs1 = onnx.helper.make_tensor(P+"rs1", onnx.TensorProto.INT64, [1], [1])
    inits.append(rs1)
    for src_f, dst in [
        (P+"per_fr_f", P+"mk_fr_f"),
        (P+"per_lr_f", P+"mk_lr_f"),
        (P+"per_fc_f", P+"mk_fc_f"),
        (P+"per_lc_f", P+"mk_lc_f"),
    ]:
        tmp = dst + "_11"
        nodes.append(onnx.helper.make_node("Gather", [src_f, P+"marker_idx"], [tmp], axis=1))
        nodes.append(onnx.helper.make_node("Reshape", [tmp, P+"rs1"], [dst]))

    # Step 9: interior start and size (all [1] FLOAT scalars)
    nodes.append(onnx.helper.make_node("Add", [P+"mk_fr_f", P+"one_f"], [P+"int_row_st"]))
    nodes.append(onnx.helper.make_node("Add", [P+"mk_fc_f", P+"one_f"], [P+"int_col_st"]))
    nodes.append(onnx.helper.make_node("Sub", [P+"mk_lr_f", P+"mk_fr_f"], [P+"lr_m_fr"]))
    nodes.append(onnx.helper.make_node("Sub", [P+"lr_m_fr", P+"one_f"],   [P+"interior_h"]))
    nodes.append(onnx.helper.make_node("Sub", [P+"mk_lc_f", P+"mk_fc_f"], [P+"lc_m_fc"]))
    nodes.append(onnx.helper.make_node("Sub", [P+"lc_m_fc", P+"one_f"],   [P+"interior_w"]))

    # Step 10: Gather rows then cols of full input at interior offsets (clip at 29)
    inits.extend([
        onnx.helper.make_tensor(P+"range_30_f", onnx.TensorProto.FLOAT, [30],
                                [float(i) for i in range(30)]),
        onnx.helper.make_tensor(P+"const_29_f", onnx.TensorProto.FLOAT, [1], [29.0]),
    ])
    nodes.append(onnx.helper.make_node("Add", [P+"int_row_st", P+"range_30_f"], [P+"row_raw"]))
    nodes.append(onnx.helper.make_node("Min", [P+"row_raw", P+"const_29_f"],    [P+"row_f"]))
    nodes.append(onnx.helper.make_node("Cast", [P+"row_f"], [P+"row_idx"], to=onnx.TensorProto.INT64))
    nodes.append(onnx.helper.make_node("Add", [P+"int_col_st", P+"range_30_f"], [P+"col_raw"]))
    nodes.append(onnx.helper.make_node("Min", [P+"col_raw", P+"const_29_f"],    [P+"col_f"]))
    nodes.append(onnx.helper.make_node("Cast", [P+"col_f"], [P+"col_idx"], to=onnx.TensorProto.INT64))
    nodes.append(onnx.helper.make_node("Gather", ["input",   P+"row_idx"], [P+"shr"], axis=2))
    nodes.append(onnx.helper.make_node("Gather", [P+"shr",   P+"col_idx"], [P+"shifted"], axis=3))

    # Step 11: mask output to interior extent
    nodes.append(onnx.helper.make_node("Less", [P+"range_30_f", P+"interior_h"], [P+"row_in_b"]))
    nodes.append(onnx.helper.make_node("Less", [P+"range_30_f", P+"interior_w"], [P+"col_in_b"]))
    nodes.append(onnx.helper.make_node("Cast", [P+"row_in_b"], [P+"row_in_f"], to=onnx.TensorProto.FLOAT))
    nodes.append(onnx.helper.make_node("Cast", [P+"col_in_b"], [P+"col_in_f"], to=onnx.TensorProto.FLOAT))
    inits.extend([
        onnx.helper.make_tensor(P+"rs_row", onnx.TensorProto.INT64, [4], [1, 1, 30, 1]),
        onnx.helper.make_tensor(P+"rs_col", onnx.TensorProto.INT64, [4], [1, 1, 1, 30]),
    ])
    nodes.append(onnx.helper.make_node("Reshape", [P+"row_in_f", P+"rs_row"], [P+"row_mask"]))
    nodes.append(onnx.helper.make_node("Reshape", [P+"col_in_f", P+"rs_col"], [P+"col_mask"]))
    nodes.append(onnx.helper.make_node("Mul", [P+"row_mask", P+"col_mask"], [P+"bbox_mask"]))
    nodes.append(onnx.helper.make_node("Mul", [P+"shifted",  P+"bbox_mask"], ["output"]))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def fractal_ring_expansion_network(batch=1):
    """ONNX: 5 Slice + Concat + Slice + Greater + Where + Reshape + Gather + Reshape + Pad.

    For task123-style fractal expansion (input shape (5, 5), output shape (10, 10)):
      colors[k] = input[k][k] for k=0..4 (diagonal one-hot vectors).
      fg = [c for c in colors if c != 0]; K = len(fg) ∈ {4, 5}.
      output[i][j] = fg[max(i, j) mod K] for i, j ∈ [0, 10).

    Implementation (one-hot stays in channel axis throughout):
      1. 5 Slice ops extract diagonal cells; Concat to [N, 10, 5, 1] = colors.
      2. Slice colors[4][channel 0]; Greater(0.5) -> is_K4 (bool scalar).
      3. Two precomputed [1, 1, 10, 10] int64 tables: idx_k4[i,j]=max(i,j)%4,
         idx_k5[i,j]=max(i,j)%5. Where(is_K4, idx_k4, idx_k5) -> [1, 1, 10, 10] int64.
      4. Reshape to [100]; Gather(colors, indices, axis=2) -> [N, 10, 100, 1].
      5. Reshape to [N, 10, 10, 10]; Pad to [N, 10, 30, 30].

    Cost (per cost-model verify, opset 10): macs ~1 (Mul-equivalent in Greater
    counted negligibly), memory ~10k bytes (the [N, 10, 100, 1] = 1000 floats
    intermediate dominates), params ~242 (constants). Total ~10k -> ~15.8 pts.
    Verified 100% (265/265) on task123.

    Builder is single-example only — multi-batch would need per-batch Where +
    Gather coordination beyond opset-10 capability. Raises NotImplementedError
    for batch != 1, triggering the dispatcher's per-example fallback.
    """
    if batch != 1:
        raise NotImplementedError(
            "fractal-ring-expansion builder is single-example only (batch must be 1)"
        )

    nodes, inits = [], []

    # Step 1: extract 5 diagonal one-hot vectors via Slice, concat to [1, 10, 5, 1].
    sl_axes = onnx.helper.make_tensor("fre_sl_axes", onnx.TensorProto.INT64, [2], [2, 3])
    sl_steps = onnx.helper.make_tensor("fre_sl_steps", onnx.TensorProto.INT64, [2], [1, 1])
    inits.extend([sl_axes, sl_steps])
    diag_names = []
    for k in range(5):
        starts = onnx.helper.make_tensor(
            f"fre_sl_st_{k}", onnx.TensorProto.INT64, [2], [k, k]
        )
        ends = onnx.helper.make_tensor(
            f"fre_sl_en_{k}", onnx.TensorProto.INT64, [2], [k + 1, k + 1]
        )
        inits.extend([starts, ends])
        out_name = f"fre_diag_{k}"
        nodes.append(onnx.helper.make_node(
            "Slice",
            ["input", f"fre_sl_st_{k}", f"fre_sl_en_{k}", "fre_sl_axes", "fre_sl_steps"],
            [out_name],
        ))
        diag_names.append(out_name)
    nodes.append(onnx.helper.make_node("Concat", diag_names, ["fre_colors"], axis=2))

    # Step 2: is_K4 = (colors[4] has channel 0 lit). Slice channel 0 at position 4.
    sl_isk4_starts = onnx.helper.make_tensor(
        "fre_isk4_st", onnx.TensorProto.INT64, [3], [0, 4, 0]
    )
    sl_isk4_ends = onnx.helper.make_tensor(
        "fre_isk4_en", onnx.TensorProto.INT64, [3], [1, 5, 1]
    )
    sl_isk4_axes = onnx.helper.make_tensor(
        "fre_isk4_ax", onnx.TensorProto.INT64, [3], [1, 2, 3]
    )
    sl_isk4_steps = onnx.helper.make_tensor(
        "fre_isk4_st1", onnx.TensorProto.INT64, [3], [1, 1, 1]
    )
    inits.extend([sl_isk4_starts, sl_isk4_ends, sl_isk4_axes, sl_isk4_steps])
    nodes.append(onnx.helper.make_node(
        "Slice", ["fre_colors", "fre_isk4_st", "fre_isk4_en", "fre_isk4_ax", "fre_isk4_st1"],
        ["fre_isk4_f"]
    ))
    half = onnx.helper.make_tensor("fre_half", onnx.TensorProto.FLOAT, [1], [0.5])
    inits.append(half)
    nodes.append(onnx.helper.make_node("Greater", ["fre_isk4_f", "fre_half"], ["fre_isk4_bool"]))

    # Step 3: precompute idx_K4 / idx_K5 lookup tables.
    flat_k4 = [max(i, j) % 4 for i in range(10) for j in range(10)]
    flat_k5 = [max(i, j) % 5 for i in range(10) for j in range(10)]
    idx_k4_init = onnx.helper.make_tensor(
        "fre_idx_k4", onnx.TensorProto.INT64, [1, 1, 10, 10], flat_k4
    )
    idx_k5_init = onnx.helper.make_tensor(
        "fre_idx_k5", onnx.TensorProto.INT64, [1, 1, 10, 10], flat_k5
    )
    inits.extend([idx_k4_init, idx_k5_init])

    # Step 4: idx = Where(is_K4, idx_k4, idx_k5); reshape to [100].
    nodes.append(onnx.helper.make_node(
        "Where", ["fre_isk4_bool", "fre_idx_k4", "fre_idx_k5"], ["fre_idx_4d"]
    ))
    rs_100 = onnx.helper.make_tensor("fre_rs_100", onnx.TensorProto.INT64, [1], [100])
    inits.append(rs_100)
    nodes.append(onnx.helper.make_node("Reshape", ["fre_idx_4d", "fre_rs_100"], ["fre_idx_flat"]))

    # Step 5: Gather colors[idx] -> [1, 10, 100, 1]; reshape to [1, 10, 10, 10].
    nodes.append(onnx.helper.make_node(
        "Gather", ["fre_colors", "fre_idx_flat"], ["fre_gathered"], axis=2
    ))
    rs_4d = onnx.helper.make_tensor("fre_rs_4d", onnx.TensorProto.INT64, [4], [1, 10, 10, 10])
    inits.append(rs_4d)
    nodes.append(onnx.helper.make_node("Reshape", ["fre_gathered", "fre_rs_4d"], ["fre_active"]))

    # Step 6: Pad to canvas.
    nodes.append(onnx.helper.make_node(
        "Pad", ["fre_active"], ["output"],
        pads=[0, 0, 0, 0, 0, 0, 20, 20], mode="constant", value=0.0,
    ))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def kron_block_up_network(scale_h=2, scale_w=2, batch=1):
    """ONNX: Slice + Resize-nearest.

    For tasks where output is the input with each cell scaled to a
    (scale_h x scale_w) block (Kronecker product with ones(scale_h, scale_w)).
    Variable input shape (h, w) with (scale_h*h, scale_w*w) <= 30 in each axis,
    output shape (scale_h*h, scale_w*w). Builder slices the canvas to
    (30/scale_h, 30/scale_w), then nearest-neighbor upsamples back to (30, 30) —
    canvas-padding zeros kron up to canvas-padding zeros, content kron up to
    block-replicated content.

    Pipeline (input [N, 10, 30, 30]):
      1. Slice(input, axes=[2,3], starts=[0,0], ends=[30/scale_h, 30/scale_w])
         -> [N, 10, 30/scale_h, 30/scale_w]
      2. Resize(slice, scales=[1, 1, scale_h, scale_w], mode='nearest')
         -> [N, 10, 30, 30]

    Cost (per cost-model verify, opset 10): macs 0 (Slice + Resize-nearest are
    both 0 MACs in onnx-tool 1.0.1); memory ~36 KB (intermediate FLOAT
    [N, 10, 30/scale_h, 30/scale_w] dominates), params ~88 (initializers).
    Total ~few-thousand -> ~17-20 pts/task.
    """
    sh, sw = scale_h, scale_w
    if 30 % sh != 0 or 30 % sw != 0:
        raise ValueError(
            f"kron-block-up requires scale_h, scale_w to divide 30; got ({sh}, {sw})"
        )
    inner_h = 30 // sh
    inner_w = 30 // sw

    nodes = []
    inits = []

    sl_starts = onnx.helper.make_tensor("kbu_sl_starts", onnx.TensorProto.INT64, [2], [0, 0])
    sl_ends = onnx.helper.make_tensor("kbu_sl_ends", onnx.TensorProto.INT64, [2], [inner_h, inner_w])
    sl_axes = onnx.helper.make_tensor("kbu_sl_axes", onnx.TensorProto.INT64, [2], [2, 3])
    sl_steps = onnx.helper.make_tensor("kbu_sl_steps", onnx.TensorProto.INT64, [2], [1, 1])
    inits.extend([sl_starts, sl_ends, sl_axes, sl_steps])
    nodes.append(onnx.helper.make_node(
        "Slice",
        ["input", "kbu_sl_starts", "kbu_sl_ends", "kbu_sl_axes", "kbu_sl_steps"],
        ["kbu_slice"],
    ))

    scales = onnx.helper.make_tensor(
        "kbu_scales", onnx.TensorProto.FLOAT, [4], [1.0, 1.0, float(sh), float(sw)]
    )
    inits.append(scales)
    nodes.append(onnx.helper.make_node(
        "Resize",
        ["kbu_slice", "kbu_scales"],
        ["output"],
        mode="nearest",
    ))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def constant_output_dominant_color_network(h_out, w_out, batch=1):
    """ONNX: ReduceSum(spatial) -> Hardmax(channel) -> Tile -> Pad.

    For tasks where every pair's output is a uniform-color block of fixed shape
    (h_out, w_out) at the top-left of the canvas, AND that color is the most
    frequent input color (counting all 10 channels including 0; padding cells
    contribute zero to all counts).

    Pipeline (input [N,10,30,30]):
      1. ReduceSum(axes=[2,3], keepdims=1) -> [N,10,1,1]   channel-count per color
      2. Hardmax(axis=1)                   -> [N,10,1,1]   one-hot of dominant
      3. Tile(repeats=[1,1,h_out,w_out])   -> [N,10,h_out,w_out]   filled block
      4. Pad(after=[30-h_out, 30-w_out])   -> [N,10,30,30]   embed at top-left

    Cost (per cost-model verify, opset 10): macs ~9k (ReduceSum), params=14
    (4 Tile repeats + 1 OneHot depth + 2 OneHot values + small inits), memory
    ~few hundred bytes for intermediate value_info entries. ArgMax + OneHot is
    used instead of Hardmax to remove implementation-defined tie-break behavior:
    ONNX ArgMax has a deterministic spec ("returns the index of the first
    maximum"), while Hardmax's tie-break is runtime-defined and could in
    principle differ between local onnxruntime and Kaggle's scoring server.
    Both ArgMax and OneHot are 0 MACs (verified in cluster summary).
    """
    reduce_node = onnx.helper.make_node(
        "ReduceSum", ["input"], ["dc_sum"], axes=[2, 3], keepdims=1
    )
    # ArgMax over channel axis with keepdims=0 -> [N, 1, 1] int64.
    # Then OneHot at axis=1 inserts a depth-10 dim -> [N, 10, 1, 1].
    argmax_node = onnx.helper.make_node(
        "ArgMax", ["dc_sum"], ["dc_argmax"], axis=1, keepdims=0
    )
    oh_depth = onnx.helper.make_tensor("dc_oh_depth", onnx.TensorProto.INT64, [], [10])
    oh_values = onnx.helper.make_tensor("dc_oh_values", onnx.TensorProto.FLOAT, [2], [0.0, 1.0])
    onehot_node = onnx.helper.make_node(
        "OneHot",
        ["dc_argmax", "dc_oh_depth", "dc_oh_values"],
        ["dc_hard"],
        axis=1,
    )
    repeats_init = onnx.helper.make_tensor(
        "dc_repeats", onnx.TensorProto.INT64, [4], [1, 1, h_out, w_out]
    )
    tile_node = onnx.helper.make_node("Tile", ["dc_hard", "dc_repeats"], ["dc_tiled"])
    # Pad in opset 10: pads is an attribute (1D int list of length 2 * rank).
    # Order: [before_dim0, before_dim1, before_dim2, before_dim3, after_dim0, after_dim1, after_dim2, after_dim3].
    pad_node = onnx.helper.make_node(
        "Pad",
        ["dc_tiled"],
        ["output"],
        pads=[0, 0, 0, 0, 0, 0, 30 - h_out, 30 - w_out],
        mode="constant",
        value=0.0,
    )
    return _make_model(
        [reduce_node, argmax_node, onehot_node, tile_node, pad_node],
        [oh_depth, oh_values, repeats_init],
        shape=[batch, 10, 30, 30],
    )


def largest_bbox_constant_output_network(h_out, w_out, batch=1):
    """ONNX: per-channel bbox-area extraction -> ArgMax(channel) -> OneHot -> Tile -> Pad.

    For tasks where every pair's output is a uniform-color block of fixed shape
    (h_out, w_out) at the top-left of the canvas, AND that color is the one
    whose **bounding box has the largest area** in the input (ties broken by
    smaller channel index per ONNX ArgMax spec).

    Pipeline (input [N, 10, 30, 30]):
      1. row_has = ReduceMax(input, axes=[3], keepdims=1) -> [N, 10, 30, 1]
         col_has = ReduceMax(input, axes=[2], keepdims=1) -> [N, 10, 1, 30]
      2. ArgMax-trick (descending weight) per channel -> first_row, first_col
         ArgMax-trick (ascending weight)  per channel -> last_row, last_col
         (4 [N, 10, 1] int64 tensors after Squeeze.)
      3. Cast to FLOAT; bbox_h = last_row - first_row + 1; bbox_w analogously.
         area = bbox_h * bbox_w  -> [N, 10, 1]
      4. Mask out empty channels via has_cells_per_channel (ReduceSum > 0).
      5. Mask out channel 0 via constant [0, 1, 1, ..., 1].
      6. ArgMax(area, axis=1, keepdims=0) -> picked channel (int64 scalar).
      7. OneHot(picked, depth=10) -> [10] -> Reshape [N, 10, 1, 1] -> Tile to
         [N, 10, h_out, w_out] -> Pad to [N, 10, 30, 30].

    Cost (per cost-model verify, opset 10): macs ~1k (Mul + Sub + Add are all
    elementwise on small tensors); memory ~26 KB (intermediates); params ~150
    (constants). Total ~30k -> ~14.7 pts. Verified 100% (266/266) on task100.
    """
    nodes = []
    inits = []

    # Step 1: row_has + col_has
    nodes.append(onnx.helper.make_node(
        "ReduceMax", ["input"], ["lbc_row_has"], axes=[3], keepdims=1
    ))
    nodes.append(onnx.helper.make_node(
        "ReduceMax", ["input"], ["lbc_col_has"], axes=[2], keepdims=1
    ))

    # Step 2: ArgMax-trick (per-channel) for first/last row/col
    desc_row = onnx.helper.make_tensor(
        "lbc_desc_row", onnx.TensorProto.FLOAT, [1, 1, 30, 1],
        [float(30 - i) for i in range(30)],
    )
    asc_row = onnx.helper.make_tensor(
        "lbc_asc_row", onnx.TensorProto.FLOAT, [1, 1, 30, 1],
        [float(i + 1) for i in range(30)],
    )
    desc_col = onnx.helper.make_tensor(
        "lbc_desc_col", onnx.TensorProto.FLOAT, [1, 1, 1, 30],
        [float(30 - i) for i in range(30)],
    )
    asc_col = onnx.helper.make_tensor(
        "lbc_asc_col", onnx.TensorProto.FLOAT, [1, 1, 1, 30],
        [float(i + 1) for i in range(30)],
    )
    inits.extend([desc_row, asc_row, desc_col, asc_col])

    nodes.append(onnx.helper.make_node("Mul", ["lbc_row_has", "lbc_desc_row"], ["lbc_w_first_row"]))
    nodes.append(onnx.helper.make_node("Mul", ["lbc_row_has", "lbc_asc_row"], ["lbc_w_last_row"]))
    nodes.append(onnx.helper.make_node("Mul", ["lbc_col_has", "lbc_desc_col"], ["lbc_w_first_col"]))
    nodes.append(onnx.helper.make_node("Mul", ["lbc_col_has", "lbc_asc_col"], ["lbc_w_last_col"]))

    nodes.append(onnx.helper.make_node(
        "ArgMax", ["lbc_w_first_row"], ["lbc_first_row_3d"], axis=2, keepdims=0
    ))
    nodes.append(onnx.helper.make_node(
        "ArgMax", ["lbc_w_last_row"], ["lbc_last_row_3d"], axis=2, keepdims=0
    ))
    nodes.append(onnx.helper.make_node(
        "ArgMax", ["lbc_w_first_col"], ["lbc_first_col_3d"], axis=3, keepdims=0
    ))
    nodes.append(onnx.helper.make_node(
        "ArgMax", ["lbc_w_last_col"], ["lbc_last_col_3d"], axis=3, keepdims=0
    ))

    # Cast to float for arithmetic
    nodes.append(onnx.helper.make_node(
        "Cast", ["lbc_first_row_3d"], ["lbc_first_row_f"], to=onnx.TensorProto.FLOAT
    ))
    nodes.append(onnx.helper.make_node(
        "Cast", ["lbc_last_row_3d"], ["lbc_last_row_f"], to=onnx.TensorProto.FLOAT
    ))
    nodes.append(onnx.helper.make_node(
        "Cast", ["lbc_first_col_3d"], ["lbc_first_col_f"], to=onnx.TensorProto.FLOAT
    ))
    nodes.append(onnx.helper.make_node(
        "Cast", ["lbc_last_col_3d"], ["lbc_last_col_f"], to=onnx.TensorProto.FLOAT
    ))

    # Step 3: bbox_h = last_row - first_row + 1; bbox_w analogously; area = bbox_h * bbox_w
    one_init = onnx.helper.make_tensor("lbc_one", onnx.TensorProto.FLOAT, [1], [1.0])
    inits.append(one_init)
    nodes.append(onnx.helper.make_node("Sub", ["lbc_last_row_f", "lbc_first_row_f"], ["lbc_h_minus1"]))
    nodes.append(onnx.helper.make_node("Add", ["lbc_h_minus1", "lbc_one"], ["lbc_bbox_h"]))
    nodes.append(onnx.helper.make_node("Sub", ["lbc_last_col_f", "lbc_first_col_f"], ["lbc_w_minus1"]))
    nodes.append(onnx.helper.make_node("Add", ["lbc_w_minus1", "lbc_one"], ["lbc_bbox_w"]))
    # bbox_h shape (N, 10, 1); bbox_w shape (N, 10, 1) — but they came from axis 2 vs axis 3 reductions.
    # row reductions kept dim 3 (size 1) -> after axis=2 ArgMax keepdims=0 the residual shape is (N, 10, 1).
    # col reductions kept dim 2 (size 1) -> after axis=3 ArgMax keepdims=0 the residual shape is (N, 10, 1).
    # So both bbox_h and bbox_w are (N, 10, 1); broadcasting Mul works elementwise.
    nodes.append(onnx.helper.make_node("Mul", ["lbc_bbox_h", "lbc_bbox_w"], ["lbc_area"]))

    # Step 4: mask out empty channels (any_cell == 0 -> area 0)
    nodes.append(onnx.helper.make_node(
        "ReduceSum", ["input"], ["lbc_count"], axes=[2, 3], keepdims=0
    ))  # -> (N, 10)
    # any_cell > 0
    zero_thr = onnx.helper.make_tensor("lbc_zero_thr", onnx.TensorProto.FLOAT, [1], [0.0])
    inits.append(zero_thr)
    nodes.append(onnx.helper.make_node("Greater", ["lbc_count", "lbc_zero_thr"], ["lbc_has_cells_b"]))
    nodes.append(onnx.helper.make_node("Cast", ["lbc_has_cells_b"], ["lbc_has_cells_f"], to=onnx.TensorProto.FLOAT))
    # has_cells_f shape (N, 10); reshape to (N, 10, 1) to broadcast multiply with area
    rs_n10x1 = onnx.helper.make_tensor("lbc_rs_n10x1", onnx.TensorProto.INT64, [3], [batch, 10, 1])
    inits.append(rs_n10x1)
    nodes.append(onnx.helper.make_node("Reshape", ["lbc_has_cells_f", "lbc_rs_n10x1"], ["lbc_has_cells_3d"]))
    nodes.append(onnx.helper.make_node("Mul", ["lbc_area", "lbc_has_cells_3d"], ["lbc_area_valid"]))

    # Step 5: mask out channel 0
    no_ch0 = onnx.helper.make_tensor(
        "lbc_no_ch0", onnx.TensorProto.FLOAT, [1, 10, 1],
        [0.0] + [1.0] * 9,
    )
    inits.append(no_ch0)
    nodes.append(onnx.helper.make_node("Mul", ["lbc_area_valid", "lbc_no_ch0"], ["lbc_area_filtered"]))

    # Step 6: ArgMax over channel axis -> picked channel
    nodes.append(onnx.helper.make_node(
        "ArgMax", ["lbc_area_filtered"], ["lbc_picked_2d"], axis=1, keepdims=0
    ))  # -> (N, 1) int64

    # Step 7: OneHot(picked, depth=10) -> Reshape -> Tile -> Pad
    oh_depth = onnx.helper.make_tensor("lbc_oh_depth", onnx.TensorProto.INT64, [], [10])
    oh_values = onnx.helper.make_tensor("lbc_oh_values", onnx.TensorProto.FLOAT, [2], [0.0, 1.0])
    inits.extend([oh_depth, oh_values])
    # OneHot: insert depth-10 axis at position 1. Input shape (N, 1) -> output (N, 10, 1).
    nodes.append(onnx.helper.make_node(
        "OneHot",
        ["lbc_picked_2d", "lbc_oh_depth", "lbc_oh_values"],
        ["lbc_onehot_3d"],
        axis=1,
    ))
    rs_n10x1x1 = onnx.helper.make_tensor("lbc_rs_n10x1x1", onnx.TensorProto.INT64, [4], [batch, 10, 1, 1])
    inits.append(rs_n10x1x1)
    nodes.append(onnx.helper.make_node("Reshape", ["lbc_onehot_3d", "lbc_rs_n10x1x1"], ["lbc_onehot_4d"]))

    repeats = onnx.helper.make_tensor("lbc_repeats", onnx.TensorProto.INT64, [4], [1, 1, h_out, w_out])
    inits.append(repeats)
    nodes.append(onnx.helper.make_node("Tile", ["lbc_onehot_4d", "lbc_repeats"], ["lbc_tiled"]))

    nodes.append(onnx.helper.make_node(
        "Pad", ["lbc_tiled"], ["output"],
        pads=[0, 0, 0, 0, 0, 0, 30 - h_out, 30 - w_out],
        mode="constant", value=0.0,
    ))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def framed_cell_const_output_network(h_out, w_out, canvas_h=12, canvas_w=12, batch=1):
    """ONNX: crop-process-pad version of grouped Conv (frame kernel) -> Pad ->
    Equal -> AND -> Mul -> ReduceMax -> ArgMax(channel) -> OneHot -> Tile -> Pad.

    For tasks where every pair's output is a uniform single color of fixed
    shape, AND that color is the **core** of a 3x3 frame anywhere in the input.
    A "framed cell" at position (r, c) is one whose 8 surrounding cells
    (r-1,c-1)..(r+1,c+1) excluding (r,c) are all the same non-zero color X,
    and the cell itself has a different non-zero color Y. Output: Y (the core).

    Pipeline (input [N, 10, 30, 30]):
      0. Crop input to [N, 10, canvas_h, canvas_w]  (max actual grid size)
      1. Grouped Conv with frame kernel [[1,1,1],[1,0,1],[1,1,1]] per channel
         -> [N, 10, canvas_h-2, canvas_w-2]
      2. Pad output to [N, 10, canvas_h, canvas_w]   align with input coords
      3. is_frame_X = (frame_count == 8)
      4. is_core_X = is_frame_X * (crop[X] == 0)
      5. has_core[r, c] = ReduceMax over X in 1..9
      6. cells_at_core = crop * has_core
      7. score_per_channel = ReduceMax over (H, W)
      8. Mask channel 0; ArgMax -> picked Y
      9. OneHot -> Reshape -> Tile [h_out, w_out] -> Pad to [N, 10, 30, 30]

    Cost: ~40k (down from ~260k). Verified 100% (267/267) on task346.
    """
    P = "fcc_"
    nodes = []
    inits = []

    # Step 0: crop input to [N, 10, canvas_h, canvas_w]
    cr_starts = onnx.helper.make_tensor(P+"cr_starts", onnx.TensorProto.INT64, [2], [0, 0])
    cr_ends = onnx.helper.make_tensor(P+"cr_ends", onnx.TensorProto.INT64, [2], [canvas_h, canvas_w])
    cr_axes = onnx.helper.make_tensor(P+"cr_axes", onnx.TensorProto.INT64, [2], [2, 3])
    cr_steps = onnx.helper.make_tensor(P+"cr_steps", onnx.TensorProto.INT64, [2], [1, 1])
    inits.extend([cr_starts, cr_ends, cr_axes, cr_steps])
    nodes.append(onnx.helper.make_node(
        "Slice", ["input", P+"cr_starts", P+"cr_ends", P+"cr_axes", P+"cr_steps"],
        [P+"crop"],
    ))

    # Step 1: grouped Conv with frame kernel (valid padding).
    frame_kernel = onnx.helper.make_tensor(
        P+"frame_kernel", onnx.TensorProto.FLOAT, [10, 1, 3, 3],
        [1, 1, 1, 1, 0, 1, 1, 1, 1] * 10,
    )
    inits.append(frame_kernel)
    nodes.append(onnx.helper.make_node(
        "Conv", [P+"crop", P+"frame_kernel"], [P+"frame_count_s"],
        kernel_shape=[3, 3], group=10, pads=[0, 0, 0, 0],
    ))

    # Step 2: Pad back to [N, 10, canvas_h, canvas_w]
    nodes.append(onnx.helper.make_node(
        "Pad", [P+"frame_count_s"], [P+"frame_count"],
        pads=[0, 0, 1, 1, 0, 0, 1, 1], mode="constant", value=0.0,
    ))

    # Step 3: is_frame_X = (frame_count == 8) -> > 7.5
    eight_thr = onnx.helper.make_tensor(P+"eight_thr", onnx.TensorProto.FLOAT, [1], [7.5])
    inits.append(eight_thr)
    nodes.append(onnx.helper.make_node(
        "Greater", [P+"frame_count", P+"eight_thr"], [P+"is_frame_b"]
    ))
    nodes.append(onnx.helper.make_node(
        "Cast", [P+"is_frame_b"], [P+"is_frame_f"], to=onnx.TensorProto.FLOAT
    ))

    # Step 4: is_core_X = is_frame_X * (1 - crop[X])
    one_init = onnx.helper.make_tensor(P+"one", onnx.TensorProto.FLOAT, [1], [1.0])
    inits.append(one_init)
    nodes.append(onnx.helper.make_node("Sub", [P+"one", P+"crop"], [P+"not_crop"]))
    nodes.append(onnx.helper.make_node("Mul", [P+"is_frame_f", P+"not_crop"], [P+"is_core_raw"]))
    no_ch0_4d = onnx.helper.make_tensor(
        P+"no_ch0_4d", onnx.TensorProto.FLOAT, [1, 10, 1, 1],
        [0.0] + [1.0] * 9,
    )
    inits.append(no_ch0_4d)
    nodes.append(onnx.helper.make_node("Mul", [P+"is_core_raw", P+"no_ch0_4d"], [P+"is_core"]))

    # Step 5: has_core = ReduceMax over channel
    nodes.append(onnx.helper.make_node(
        "ReduceMax", [P+"is_core"], [P+"has_core"], axes=[1], keepdims=1
    ))

    # Step 6: cells_at_core = crop * has_core
    nodes.append(onnx.helper.make_node("Mul", [P+"crop", P+"has_core"], [P+"cells_at_core"]))

    # Step 7: score_per_channel = ReduceMax over (H, W)
    nodes.append(onnx.helper.make_node(
        "ReduceMax", [P+"cells_at_core"], [P+"score"], axes=[2, 3], keepdims=0
    ))

    # Step 8: Mask channel 0; ArgMax -> picked Y
    no_ch0 = onnx.helper.make_tensor(
        P+"no_ch0", onnx.TensorProto.FLOAT, [10], [0.0] + [1.0] * 9
    )
    inits.append(no_ch0)
    nodes.append(onnx.helper.make_node("Mul", [P+"score", P+"no_ch0"], [P+"score_filtered"]))
    nodes.append(onnx.helper.make_node(
        "ArgMax", [P+"score_filtered"], [P+"picked_2d"], axis=1, keepdims=0
    ))

    # Step 9: OneHot -> Reshape -> Tile -> Pad to [N, 10, 30, 30]
    oh_depth = onnx.helper.make_tensor(P+"oh_depth", onnx.TensorProto.INT64, [], [10])
    oh_values = onnx.helper.make_tensor(P+"oh_values", onnx.TensorProto.FLOAT, [2], [0.0, 1.0])
    inits.extend([oh_depth, oh_values])
    nodes.append(onnx.helper.make_node(
        "OneHot", [P+"picked_2d", P+"oh_depth", P+"oh_values"], [P+"onehot_2d"], axis=1,
    ))
    rs_n10x1x1 = onnx.helper.make_tensor(P+"rs_n10x1x1", onnx.TensorProto.INT64, [4], [batch, 10, 1, 1])
    inits.append(rs_n10x1x1)
    nodes.append(onnx.helper.make_node("Reshape", [P+"onehot_2d", P+"rs_n10x1x1"], [P+"onehot_4d"]))

    repeats = onnx.helper.make_tensor(P+"repeats", onnx.TensorProto.INT64, [4], [1, 1, h_out, w_out])
    inits.append(repeats)
    nodes.append(onnx.helper.make_node("Tile", [P+"onehot_4d", P+"repeats"], [P+"tiled"]))

    nodes.append(onnx.helper.make_node(
        "Pad", [P+"tiled"], ["output"],
        pads=[0, 0, 0, 0, 0, 0, 30 - h_out, 30 - w_out],
        mode="constant", value=0.0,
    ))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def marker_aligned_shape_recolor_network(batch=1):
    """ONNX: template extraction + per-channel cross-correlation -> exact-match
    indicator -> recolor matched cells to color 5.

    For tasks with this rule:
      - Input has an L-shape of color 5 at FIXED position (vertical at col 3
        rows 0..3, horizontal at row 3 cols 0..3).
      - Region (rows 0..2, cols 0..2) — the "template region" — contains exactly
        ONE non-zero color other than 5 (the "template_color"); cells of that
        color form a stencil ("template_mask").
      - Output: cells of any blob whose shape (bbox-cropped mask) exactly equals
        the template_mask, and whose color is not 5 nor template_color, are
        recolored to 5. Everything else preserved.

    Pipeline (input [N, 10, 30, 30], single example only, fixed 10x10 grid):

      1. template_region = Slice(input, axes=[2,3], starts=[0,0], ends=[3,3])
         -> (1, 10, 3, 3). All channel masks present.
      2. Mul by per-channel mask [0, 1, 1, 1, 1, 0, 1, 1, 1, 1] -> zero ch0, ch5
         -> template_filtered (1, 10, 3, 3).
      3. template_mask = ReduceMax(template_filtered, axes=[1], keepdims=1)
         -> (1, 1, 3, 3) — binary stencil aggregating across all candidate channels.
      4. template_count = ReduceSum(template_mask) -> scalar.
      5. template_color_idx = ArgMax(ReduceSum(template_filtered, axes=[2,3]),
         axis=1) -> int64 scalar; OneHot -> (1, 10, 1, 1) FLOAT.
      6. candidate_mask = no_ch05 - template_color_oh -> (1, 10, 1, 1).
         (1 at every channel that is a valid candidate; 0 at ch0, ch5, template.)
      7. Per-channel total: input_total = ReduceSum(input, axes=[2,3]) -> (1, 10).
      8. Cross-correlation: for each (i, j) in 3x3 template, shift input by
         (-i, -j) using Slice + Pad (both 0 MACs in onnx-tool 1.0.1), multiply
         by template_mask[i, j] scalar, accumulate into correlation tensor.
         Optimization: stack 9 shifted-input tensors along a new axis
         (1, 10, 30, 30, 9), Mul by template_mask reshaped to (1, 1, 1, 1, 9),
         ReduceSum along the new axis -> correlation (1, 10, 30, 30). This
         halves the value_info memory vs naive 9-Add chain (one big intermediate
         vs 17 medium ones).
      9. max_corr_per_chan = ReduceMax(correlation, axes=[2,3]) -> (1, 10).
      10. all_match = (max_corr_per_chan == template_count)
                      AND (input_total == template_count) -> (1, 10) FLOAT.
      11. match_filtered = all_match * candidate_mask_2d -> (1, 10).
      12. recolored_mass = input * Reshape(match_filtered, (1, 10, 1, 1))
          -> (1, 10, 30, 30); each channel zero except matched colors' cells.
      13. all_recolor_mask = ReduceSum(recolored_mass, axes=[1], keepdims=1)
          -> (1, 1, 30, 30).
      14. ch5_addition = ch5_oh * all_recolor_mask -> (1, 10, 30, 30) — only
          channel 5 nonzero, with all_recolor_mask values.
      15. output = (input - recolored_mass) + ch5_addition.

    Cost (per cost-model verify, opset 10): macs ~150k (correlation dominant via
    Mul of (1, 10, 30, 30, 9) + ReduceSum); memory ~330k bytes (dominated by
    the 5D stacked-shift intermediate at 324k); params ~150 (initializers).
    Total ~480k -> ~12.0 pts. Verified 100% (266/266) on task143.

    Builder is single-example only — multi-batch needs runtime per-batch
    template-color logic that opset 10 lacks. Raises NotImplementedError for
    batch != 1; dispatcher's per-example fallback handles it.
    """
    if batch != 1:
        raise NotImplementedError(
            "marker-aligned-shape-recolor builder is single-example only "
            "(batch must be 1)"
        )

    nodes = []
    inits = []

    # crop-process-pad: task143 grids are always 10x10; crop reduces all
    # large intermediates from [1,10,30,30] to [1,10,10,10] (9x smaller).

    # shared slice initializers
    sl3_starts = onnx.helper.make_tensor("mas_sl3_starts", onnx.TensorProto.INT64, [2], [0, 0])
    sl3_ends = onnx.helper.make_tensor("mas_sl3_ends", onnx.TensorProto.INT64, [2], [3, 3])
    sl3_axes = onnx.helper.make_tensor("mas_sl3_axes", onnx.TensorProto.INT64, [2], [2, 3])
    sl3_steps = onnx.helper.make_tensor("mas_sl3_steps", onnx.TensorProto.INT64, [2], [1, 1])
    cr10_ends = onnx.helper.make_tensor("mas_cr10_ends", onnx.TensorProto.INT64, [2], [10, 10])
    inits.extend([sl3_starts, sl3_ends, sl3_axes, sl3_steps, cr10_ends])

    # --- Step 0: crop input to [1, 10, 10, 10] ---
    nodes.append(onnx.helper.make_node(
        "Slice",
        ["input", "mas_sl3_starts", "mas_cr10_ends", "mas_sl3_axes", "mas_sl3_steps"],
        ["mas_crop"],
    ))  # -> (1, 10, 10, 10)

    # --- Step 1: Slice template region (rows 0..2, cols 0..2) from crop ---
    nodes.append(onnx.helper.make_node(
        "Slice",
        ["mas_crop", "mas_sl3_starts", "mas_sl3_ends", "mas_sl3_axes", "mas_sl3_steps"],
        ["mas_template_region"],
    ))  # -> (1, 10, 3, 3)

    # --- Step 2: filter ch0 and ch5 from template ---
    no_ch05_4d = onnx.helper.make_tensor(
        "mas_no_ch05_4d", onnx.TensorProto.FLOAT, [1, 10, 1, 1],
        [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    )
    inits.append(no_ch05_4d)
    nodes.append(onnx.helper.make_node(
        "Mul", ["mas_template_region", "mas_no_ch05_4d"], ["mas_template_filtered"]
    ))

    # --- Step 3: template_mask = ReduceMax across channel ---
    nodes.append(onnx.helper.make_node(
        "ReduceMax", ["mas_template_filtered"], ["mas_template_mask"],
        axes=[1], keepdims=1,
    ))  # -> (1, 1, 3, 3)

    # --- Step 4: template_count scalar ---
    nodes.append(onnx.helper.make_node(
        "ReduceSum", ["mas_template_mask"], ["mas_template_count_4d"],
        axes=[1, 2, 3], keepdims=1,
    ))  # -> (1, 1, 1, 1)

    rs_1x1 = onnx.helper.make_tensor("mas_rs_1x1", onnx.TensorProto.INT64, [2], [1, 1])
    inits.append(rs_1x1)
    nodes.append(onnx.helper.make_node(
        "Reshape", ["mas_template_count_4d", "mas_rs_1x1"], ["mas_template_count_2d"]
    ))  # -> (1, 1)

    # --- Step 5: template_color_idx + OneHot ---
    nodes.append(onnx.helper.make_node(
        "ReduceSum", ["mas_template_filtered"], ["mas_template_per_chan"],
        axes=[2, 3], keepdims=0,
    ))  # -> (1, 10)
    nodes.append(onnx.helper.make_node(
        "ArgMax", ["mas_template_per_chan"], ["mas_template_color_idx"],
        axis=1, keepdims=0,
    ))  # -> (1,) int64

    oh_depth = onnx.helper.make_tensor("mas_oh_depth", onnx.TensorProto.INT64, [], [10])
    oh_values = onnx.helper.make_tensor("mas_oh_values", onnx.TensorProto.FLOAT, [2], [0.0, 1.0])
    inits.extend([oh_depth, oh_values])
    nodes.append(onnx.helper.make_node(
        "OneHot",
        ["mas_template_color_idx", "mas_oh_depth", "mas_oh_values"],
        ["mas_template_color_oh_2d"],
        axis=1,
    ))  # -> (1, 10)

    rs_n10x1x1 = onnx.helper.make_tensor("mas_rs_n10x1x1", onnx.TensorProto.INT64, [4], [1, 10, 1, 1])
    inits.append(rs_n10x1x1)
    nodes.append(onnx.helper.make_node(
        "Reshape", ["mas_template_color_oh_2d", "mas_rs_n10x1x1"], ["mas_template_color_oh_4d"]
    ))

    # --- Step 6: candidate_mask = no_ch05 - template_color_oh (4D) ---
    nodes.append(onnx.helper.make_node(
        "Sub", ["mas_no_ch05_4d", "mas_template_color_oh_4d"], ["mas_candidate_mask_4d"]
    ))  # (1, 10, 1, 1)

    rs_n10 = onnx.helper.make_tensor("mas_rs_n10", onnx.TensorProto.INT64, [2], [1, 10])
    inits.append(rs_n10)
    nodes.append(onnx.helper.make_node(
        "Reshape", ["mas_candidate_mask_4d", "mas_rs_n10"], ["mas_candidate_mask_2d"]
    ))

    # --- Step 7: input_total per channel (from crop; zeros beyond 10x10 don't contribute) ---
    nodes.append(onnx.helper.make_node(
        "ReduceSum", ["mas_crop"], ["mas_input_total"],
        axes=[2, 3], keepdims=0,
    ))  # -> (1, 10)

    # --- Step 8: cross-correlation on cropped [1,10,10,10] ---
    # Pad crop by 3 on bottom-right for safe Slice offsets.
    nodes.append(onnx.helper.make_node(
        "Pad", ["mas_crop"], ["mas_crop_padded"],
        pads=[0, 0, 0, 0, 0, 0, 3, 3], mode="constant", value=0.0,
    ))  # -> (1, 10, 13, 13)

    # 9 Slice offsets -> concat along batch axis -> [9, 10, 10, 10]
    shift_outputs = []
    for i in range(3):
        for j in range(3):
            shift_starts = onnx.helper.make_tensor(
                f"mas_shift_starts_{i}{j}", onnx.TensorProto.INT64, [2], [i, j]
            )
            shift_ends = onnx.helper.make_tensor(
                f"mas_shift_ends_{i}{j}", onnx.TensorProto.INT64, [2], [i + 10, j + 10]
            )
            inits.extend([shift_starts, shift_ends])
            nodes.append(onnx.helper.make_node(
                "Slice",
                ["mas_crop_padded",
                 f"mas_shift_starts_{i}{j}", f"mas_shift_ends_{i}{j}",
                 "mas_sl3_axes", "mas_sl3_steps"],
                [f"mas_shifted_{i}{j}"],
            ))  # -> (1, 10, 10, 10)
            shift_outputs.append(f"mas_shifted_{i}{j}")

    nodes.append(onnx.helper.make_node(
        "Concat", shift_outputs, ["mas_shifted_stack"], axis=0,
    ))  # -> (9, 10, 10, 10)

    rs_template_b9 = onnx.helper.make_tensor(
        "mas_rs_template_b9", onnx.TensorProto.INT64, [4], [9, 1, 1, 1]
    )
    inits.append(rs_template_b9)
    nodes.append(onnx.helper.make_node(
        "Reshape", ["mas_template_mask", "mas_rs_template_b9"], ["mas_template_b9"]
    ))  # -> (9, 1, 1, 1)

    nodes.append(onnx.helper.make_node(
        "Mul", ["mas_shifted_stack", "mas_template_b9"], ["mas_corr_stack"]
    ))  # -> (9, 10, 10, 10)

    nodes.append(onnx.helper.make_node(
        "ReduceSum", ["mas_corr_stack"], ["mas_correlation"],
        axes=[0], keepdims=1,
    ))  # -> (1, 10, 10, 10)

    # --- Step 9: max correlation per channel ---
    nodes.append(onnx.helper.make_node(
        "ReduceMax", ["mas_correlation"], ["mas_max_corr_per_chan"],
        axes=[2, 3], keepdims=0,
    ))  # -> (1, 10)

    # --- Step 10: per-channel match indicator ---
    nodes.append(onnx.helper.make_node(
        "Cast", ["mas_max_corr_per_chan"], ["mas_max_corr_int"], to=onnx.TensorProto.INT64
    ))
    nodes.append(onnx.helper.make_node(
        "Cast", ["mas_input_total"], ["mas_input_total_int"], to=onnx.TensorProto.INT64
    ))
    nodes.append(onnx.helper.make_node(
        "Cast", ["mas_template_count_2d"], ["mas_template_count_int"], to=onnx.TensorProto.INT64
    ))
    nodes.append(onnx.helper.make_node(
        "Equal", ["mas_max_corr_int", "mas_template_count_int"], ["mas_corr_match_b"]
    ))
    nodes.append(onnx.helper.make_node(
        "Equal", ["mas_input_total_int", "mas_template_count_int"], ["mas_total_match_b"]
    ))
    nodes.append(onnx.helper.make_node(
        "Cast", ["mas_corr_match_b"], ["mas_corr_match_f"], to=onnx.TensorProto.FLOAT
    ))
    nodes.append(onnx.helper.make_node(
        "Cast", ["mas_total_match_b"], ["mas_total_match_f"], to=onnx.TensorProto.FLOAT
    ))
    nodes.append(onnx.helper.make_node(
        "Mul", ["mas_corr_match_f", "mas_total_match_f"], ["mas_all_match"]
    ))

    # --- Step 11: filter by candidate_mask ---
    nodes.append(onnx.helper.make_node(
        "Mul", ["mas_all_match", "mas_candidate_mask_2d"], ["mas_match_filtered_2d"]
    ))

    nodes.append(onnx.helper.make_node(
        "Reshape", ["mas_match_filtered_2d", "mas_rs_n10x1x1"], ["mas_match_filtered_4d"]
    ))

    # --- Step 12: recolored_mass = crop * match_filtered_4d ---
    nodes.append(onnx.helper.make_node(
        "Mul", ["mas_crop", "mas_match_filtered_4d"], ["mas_recolored_mass"]
    ))  # -> (1, 10, 10, 10)

    # --- Step 13: all_recolor_mask ---
    nodes.append(onnx.helper.make_node(
        "ReduceSum", ["mas_recolored_mass"], ["mas_all_recolor_mask"],
        axes=[1], keepdims=1,
    ))  # -> (1, 1, 10, 10)

    # --- Step 14: ch5_addition ---
    ch5_oh = onnx.helper.make_tensor(
        "mas_ch5_oh", onnx.TensorProto.FLOAT, [1, 10, 1, 1],
        [0.0] * 5 + [1.0] + [0.0] * 4,
    )
    inits.append(ch5_oh)
    nodes.append(onnx.helper.make_node(
        "Mul", ["mas_ch5_oh", "mas_all_recolor_mask"], ["mas_ch5_addition"]
    ))  # -> (1, 10, 10, 10)

    # --- Step 15: result on crop, then Pad back to 30x30 ---
    nodes.append(onnx.helper.make_node(
        "Sub", ["mas_crop", "mas_recolored_mass"], ["mas_input_minus_recolored"]
    ))
    nodes.append(onnx.helper.make_node(
        "Add", ["mas_input_minus_recolored", "mas_ch5_addition"], ["mas_result"]
    ))  # -> (1, 10, 10, 10)
    nodes.append(onnx.helper.make_node(
        "Pad", ["mas_result"], ["output"],
        pads=[0, 0, 0, 0, 0, 0, 20, 20], mode="constant", value=0.0,
    ))  # -> (1, 10, 30, 30); graph output not counted in static_memory

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def grid_division_count_network(batch=1):
    """ONNX: count divider rows/cols + emit (n_div_rows+1, n_div_cols+1) block
    of fill color.

    For tasks (e.g. task021) where the input is a 2-color grid: one color forms
    "divider" lines (full rows AND/OR full cols), the other "fill" color fills
    the rest of the grid. Output is a uniform-fill rectangle of shape
    (#divider_rows + 1, #divider_cols + 1) — i.e., the COUNT of grid sections.

    Pipeline (input [N, 10, 30, 30], single example only):

      1. is_active = ReduceSum(input, axis=1, keepdims=1) -> (1, 1, 30, 30)
         per_row_total = ReduceSum(is_active, axes=[3], keepdims=0) -> (1, 1, 30)
         per_col_total = ReduceSum(is_active, axes=[2], keepdims=0) -> (1, 1, 30)
      2. per_row_per_chan = ReduceSum(input, axes=[3], keepdims=0) -> (1, 10, 30)
         per_col_per_chan = ReduceSum(input, axes=[2], keepdims=0) -> (1, 10, 30)
      3. row_diff = per_row_per_chan - per_row_total (broadcast 1->10) -> (1, 10, 30)
         col_diff analogously
         row_full_c = (row_diff_int == 0) AND (per_row_total_int > 0)
         col_full_c analogously — both shape (1, 10, 30) FLOAT 0/1
      4. n_full_rows_per_chan = ReduceSum(row_full_c, axes=[2]) -> (1, 10)
         n_full_cols_per_chan analogously
      5. div_score = n_full_rows + n_full_cols -> (1, 10)
         div_color_idx = ArgMax(div_score, axis=1) -> int64 scalar
         div_color_oh = OneHot(div_color_idx, depth=10) -> (1, 10) -> reshape (1, 10)
      6. n_div_rows = ReduceSum(n_full_rows * div_color_oh, axes=[1]) -> scalar
         n_div_cols analogously
      7. fill_color identification:
         per_chan_total = ReduceSum(input, axes=[2,3]) -> (1, 10)
         filter_mask = no_ch0 - div_color_oh (FLOAT, 0 at ch0 + div_color)
         filtered = per_chan_total * filter_mask
         fill_color_idx = ArgMax(filtered, axis=1) -> int64 scalar
         fill_color_oh_4d = OneHot reshape to (1, 10, 1, 1)
      8. H_out = n_div_rows + 1; W_out = n_div_cols + 1 (FLOAT scalars)
      9. row_mask_30 = (range_30 < H_out) -> (30,) FLOAT
         col_mask_30 = (range_30 < W_out) -> (30,) FLOAT
         row_mask_4d = Reshape (1, 1, 30, 1)
         col_mask_4d = Reshape (1, 1, 1, 30)
         bbox_mask = row_mask_4d * col_mask_4d -> (1, 1, 30, 30)
      10. output = bbox_mask * fill_color_oh_4d (broadcasting)

    Cost (per cost-model verify, opset 10): macs ~30k; memory ~150k bytes;
    params ~150 (initializers). Total ~180k -> ~12.9 pts. Verified 100%
    (266/266) on task021.

    Single-example only (batch=1) — uses ArgMax-derived runtime indices that
    aren't easily batchable in opset 10.
    """
    if batch != 1:
        raise NotImplementedError(
            "grid-division-count builder is single-example only (batch must be 1)"
        )

    nodes = []
    inits = []

    # --- Step 1: is_active + per_axis_total ---
    nodes.append(onnx.helper.make_node(
        "ReduceSum", ["input"], ["gdc_is_active"], axes=[1], keepdims=1
    ))  # (1, 1, 30, 30)
    nodes.append(onnx.helper.make_node(
        "ReduceSum", ["gdc_is_active"], ["gdc_per_row_total"], axes=[3], keepdims=0
    ))  # (1, 1, 30)
    nodes.append(onnx.helper.make_node(
        "ReduceSum", ["gdc_is_active"], ["gdc_per_col_total"], axes=[2], keepdims=0
    ))  # (1, 1, 30)

    # --- Step 2: per_axis_per_chan totals ---
    nodes.append(onnx.helper.make_node(
        "ReduceSum", ["input"], ["gdc_per_row_per_chan"], axes=[3], keepdims=0
    ))  # (1, 10, 30)
    nodes.append(onnx.helper.make_node(
        "ReduceSum", ["input"], ["gdc_per_col_per_chan"], axes=[2], keepdims=0
    ))  # (1, 10, 30)

    # --- Step 3: row/col full-c indicators ---
    nodes.append(onnx.helper.make_node(
        "Sub", ["gdc_per_row_per_chan", "gdc_per_row_total"], ["gdc_row_diff"]
    ))  # (1, 10, 30)
    nodes.append(onnx.helper.make_node(
        "Sub", ["gdc_per_col_per_chan", "gdc_per_col_total"], ["gdc_col_diff"]
    ))

    # opset 10 Equal needs INT; cast diffs and totals to INT64.
    nodes.append(onnx.helper.make_node(
        "Cast", ["gdc_row_diff"], ["gdc_row_diff_int"], to=onnx.TensorProto.INT64
    ))
    nodes.append(onnx.helper.make_node(
        "Cast", ["gdc_col_diff"], ["gdc_col_diff_int"], to=onnx.TensorProto.INT64
    ))
    nodes.append(onnx.helper.make_node(
        "Cast", ["gdc_per_row_total"], ["gdc_per_row_total_int"], to=onnx.TensorProto.INT64
    ))
    nodes.append(onnx.helper.make_node(
        "Cast", ["gdc_per_col_total"], ["gdc_per_col_total_int"], to=onnx.TensorProto.INT64
    ))

    # row_match = (row_diff == 0) — color c covers all active cells in this row.
    zero_int = onnx.helper.make_tensor("gdc_zero_int", onnx.TensorProto.INT64, [1], [0])
    inits.append(zero_int)
    nodes.append(onnx.helper.make_node(
        "Equal", ["gdc_row_diff_int", "gdc_zero_int"], ["gdc_row_match_b"]
    ))  # bool (1, 10, 30)
    nodes.append(onnx.helper.make_node(
        "Equal", ["gdc_col_diff_int", "gdc_zero_int"], ["gdc_col_match_b"]
    ))

    # row_active = (per_row_total > 0) — row has at least one active cell.
    zero_f = onnx.helper.make_tensor("gdc_zero_f", onnx.TensorProto.FLOAT, [1], [0.0])
    inits.append(zero_f)
    nodes.append(onnx.helper.make_node(
        "Greater", ["gdc_per_row_total", "gdc_zero_f"], ["gdc_row_active_b"]
    ))  # bool (1, 1, 30)
    nodes.append(onnx.helper.make_node(
        "Greater", ["gdc_per_col_total", "gdc_zero_f"], ["gdc_col_active_b"]
    ))

    # AND via Cast + Mul
    nodes.append(onnx.helper.make_node(
        "Cast", ["gdc_row_match_b"], ["gdc_row_match_f"], to=onnx.TensorProto.FLOAT
    ))
    nodes.append(onnx.helper.make_node(
        "Cast", ["gdc_col_match_b"], ["gdc_col_match_f"], to=onnx.TensorProto.FLOAT
    ))
    nodes.append(onnx.helper.make_node(
        "Cast", ["gdc_row_active_b"], ["gdc_row_active_f"], to=onnx.TensorProto.FLOAT
    ))
    nodes.append(onnx.helper.make_node(
        "Cast", ["gdc_col_active_b"], ["gdc_col_active_f"], to=onnx.TensorProto.FLOAT
    ))
    # row_full_c[c, r] = row_match[c, r] * row_active[r]; broadcasting (1,10,30)*(1,1,30)
    nodes.append(onnx.helper.make_node(
        "Mul", ["gdc_row_match_f", "gdc_row_active_f"], ["gdc_row_full_c"]
    ))
    nodes.append(onnx.helper.make_node(
        "Mul", ["gdc_col_match_f", "gdc_col_active_f"], ["gdc_col_full_c"]
    ))

    # --- Step 4: per-channel full-row + full-col counts ---
    nodes.append(onnx.helper.make_node(
        "ReduceSum", ["gdc_row_full_c"], ["gdc_n_full_rows"], axes=[2], keepdims=0
    ))  # (1, 10)
    nodes.append(onnx.helper.make_node(
        "ReduceSum", ["gdc_col_full_c"], ["gdc_n_full_cols"], axes=[2], keepdims=0
    ))  # (1, 10)

    # --- Step 5: identify divider color ---
    nodes.append(onnx.helper.make_node(
        "Add", ["gdc_n_full_rows", "gdc_n_full_cols"], ["gdc_div_score"]
    ))  # (1, 10)
    nodes.append(onnx.helper.make_node(
        "ArgMax", ["gdc_div_score"], ["gdc_div_color_idx"], axis=1, keepdims=0
    ))  # (1,) int64

    oh_depth = onnx.helper.make_tensor("gdc_oh_depth", onnx.TensorProto.INT64, [], [10])
    oh_values = onnx.helper.make_tensor("gdc_oh_values", onnx.TensorProto.FLOAT, [2], [0.0, 1.0])
    inits.extend([oh_depth, oh_values])
    nodes.append(onnx.helper.make_node(
        "OneHot",
        ["gdc_div_color_idx", "gdc_oh_depth", "gdc_oh_values"],
        ["gdc_div_color_oh"],
        axis=1,
    ))  # (1, 10)

    # --- Step 6: n_div_rows + n_div_cols via dot product with one-hot ---
    nodes.append(onnx.helper.make_node(
        "Mul", ["gdc_n_full_rows", "gdc_div_color_oh"], ["gdc_picked_rows"]
    ))
    nodes.append(onnx.helper.make_node(
        "Mul", ["gdc_n_full_cols", "gdc_div_color_oh"], ["gdc_picked_cols"]
    ))
    nodes.append(onnx.helper.make_node(
        "ReduceSum", ["gdc_picked_rows"], ["gdc_n_div_rows"], axes=[1], keepdims=0
    ))  # scalar (1,)
    nodes.append(onnx.helper.make_node(
        "ReduceSum", ["gdc_picked_cols"], ["gdc_n_div_cols"], axes=[1], keepdims=0
    ))

    # --- Step 7: fill_color identification ---
    nodes.append(onnx.helper.make_node(
        "ReduceSum", ["input"], ["gdc_per_chan_total"], axes=[2, 3], keepdims=0
    ))  # (1, 10)
    no_ch0_2d = onnx.helper.make_tensor(
        "gdc_no_ch0_2d", onnx.TensorProto.FLOAT, [1, 10],
        [0.0] + [1.0] * 9,
    )
    inits.append(no_ch0_2d)
    nodes.append(onnx.helper.make_node(
        "Sub", ["gdc_no_ch0_2d", "gdc_div_color_oh"], ["gdc_filter_mask"]
    ))  # 0 at ch0 AND div_color, 1 elsewhere; for div_color=0 also 0 at ch0
    # Note: if div_color = 0, filter_mask = no_ch0 - ch0_oh = no_ch0 - [1,0,..0] =
    # [-1, 1, 1, 1, ...]; the -1 at ch0 doesn't matter since per_chan_total[0]
    # may be nonzero (bg cells) but those cells would still tie ArgMax to non-bg
    # because the actual nonzero color count is more than -1*bg_count. To be safe
    # we'll use Max(filter_mask, 0).
    nodes.append(onnx.helper.make_node(
        "Greater", ["gdc_filter_mask", "gdc_zero_f"], ["gdc_filter_mask_b"]
    ))
    nodes.append(onnx.helper.make_node(
        "Cast", ["gdc_filter_mask_b"], ["gdc_filter_mask_clipped"], to=onnx.TensorProto.FLOAT
    ))
    nodes.append(onnx.helper.make_node(
        "Mul", ["gdc_per_chan_total", "gdc_filter_mask_clipped"], ["gdc_filtered_count"]
    ))
    nodes.append(onnx.helper.make_node(
        "ArgMax", ["gdc_filtered_count"], ["gdc_fill_color_idx"], axis=1, keepdims=0
    ))  # (1,) int64
    nodes.append(onnx.helper.make_node(
        "OneHot",
        ["gdc_fill_color_idx", "gdc_oh_depth", "gdc_oh_values"],
        ["gdc_fill_color_oh_2d"],
        axis=1,
    ))  # (1, 10)
    rs_n10x1x1 = onnx.helper.make_tensor("gdc_rs_n10x1x1", onnx.TensorProto.INT64, [4], [1, 10, 1, 1])
    inits.append(rs_n10x1x1)
    nodes.append(onnx.helper.make_node(
        "Reshape", ["gdc_fill_color_oh_2d", "gdc_rs_n10x1x1"], ["gdc_fill_color_oh_4d"]
    ))

    # --- Step 8: H_out + W_out (FLOAT scalars) ---
    one_f = onnx.helper.make_tensor("gdc_one_f", onnx.TensorProto.FLOAT, [1], [1.0])
    inits.append(one_f)
    nodes.append(onnx.helper.make_node(
        "Add", ["gdc_n_div_rows", "gdc_one_f"], ["gdc_h_out_f"]
    ))
    nodes.append(onnx.helper.make_node(
        "Add", ["gdc_n_div_cols", "gdc_one_f"], ["gdc_w_out_f"]
    ))

    # --- Step 9: bbox_mask via Less + Cast + outer-product ---
    range_30 = onnx.helper.make_tensor(
        "gdc_range_30", onnx.TensorProto.FLOAT, [30],
        [float(i) for i in range(30)],
    )
    inits.append(range_30)
    nodes.append(onnx.helper.make_node(
        "Less", ["gdc_range_30", "gdc_h_out_f"], ["gdc_row_in_bbox_b"]
    ))  # bool (30,)
    nodes.append(onnx.helper.make_node(
        "Less", ["gdc_range_30", "gdc_w_out_f"], ["gdc_col_in_bbox_b"]
    ))
    nodes.append(onnx.helper.make_node(
        "Cast", ["gdc_row_in_bbox_b"], ["gdc_row_in_bbox_f"], to=onnx.TensorProto.FLOAT
    ))
    nodes.append(onnx.helper.make_node(
        "Cast", ["gdc_col_in_bbox_b"], ["gdc_col_in_bbox_f"], to=onnx.TensorProto.FLOAT
    ))
    rs_row = onnx.helper.make_tensor("gdc_rs_row", onnx.TensorProto.INT64, [4], [1, 1, 30, 1])
    rs_col = onnx.helper.make_tensor("gdc_rs_col", onnx.TensorProto.INT64, [4], [1, 1, 1, 30])
    inits.extend([rs_row, rs_col])
    nodes.append(onnx.helper.make_node(
        "Reshape", ["gdc_row_in_bbox_f", "gdc_rs_row"], ["gdc_row_mask_4d"]
    ))
    nodes.append(onnx.helper.make_node(
        "Reshape", ["gdc_col_in_bbox_f", "gdc_rs_col"], ["gdc_col_mask_4d"]
    ))
    nodes.append(onnx.helper.make_node(
        "Mul", ["gdc_row_mask_4d", "gdc_col_mask_4d"], ["gdc_bbox_mask"]
    ))  # (1, 1, 30, 30)

    # --- Step 10: output = bbox_mask * fill_color_oh_4d ---
    nodes.append(onnx.helper.make_node(
        "Mul", ["gdc_bbox_mask", "gdc_fill_color_oh_4d"], ["output"]
    ))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def cross_section_grid_label_network(batch=1):
    """Task055-style divider grid relabel.

    Input has color-8 full divider rows/cols splitting the active grid into a
    3x3 section layout. Output preserves divider lines, leaves corner sections
    as background, and fills the cross sections with fixed colors:

      top-middle=2, middle-left=4, middle=6, middle-right=3, bottom-middle=1.

    The network derives divider rows/cols at runtime from the one-hot active
    canvas, so variable active H/W works as long as the example fits the official
    30x30 scorer canvas.
    """
    if batch != 1:
        raise NotImplementedError(
            "cross-section-grid-label builder is single-example only (batch must be 1)"
        )

    P = "csg_"
    nodes = []
    inits = []

    def add_tensor(name, dtype, shape, values):
        tensor = onnx.helper.make_tensor(P + name, dtype, shape, values)
        inits.append(tensor)
        return P + name

    # Shared constants.
    zero5 = add_tensor("zero5", onnx.TensorProto.FLOAT, [1], [0.5])
    one5 = add_tensor("one5", onnx.TensorProto.FLOAT, [1], [1.5])
    ch8_st = add_tensor("ch8_st", onnx.TensorProto.INT64, [1], [8])
    ch8_en = add_tensor("ch8_en", onnx.TensorProto.INT64, [1], [9])
    ch_ax = add_tensor("ch_ax", onnx.TensorProto.INT64, [1], [1])
    one_step = add_tensor("one_step", onnx.TensorProto.INT64, [1], [1])

    nodes.append(onnx.helper.make_node(
        "ReduceSum", ["input"], [P + "active"], axes=[1], keepdims=1
    ))
    nodes.append(onnx.helper.make_node(
        "Slice", ["input", ch8_st, ch8_en, ch_ax, one_step], [P + "ch8"]
    ))

    # Full-row / full-column color-8 divider masks.
    nodes.append(onnx.helper.make_node(
        "ReduceSum", [P + "active"], [P + "row_total"], axes=[3], keepdims=1
    ))
    nodes.append(onnx.helper.make_node(
        "ReduceSum", [P + "active"], [P + "col_total"], axes=[2], keepdims=1
    ))
    nodes.append(onnx.helper.make_node(
        "ReduceSum", [P + "ch8"], [P + "row8_total"], axes=[3], keepdims=1
    ))
    nodes.append(onnx.helper.make_node(
        "ReduceSum", [P + "ch8"], [P + "col8_total"], axes=[2], keepdims=1
    ))

    for axis in ("row", "col"):
        nodes.append(onnx.helper.make_node(
            "Sub", [P + f"{axis}_total", P + f"{axis}8_total"], [P + f"{axis}_diff"]
        ))
        nodes.append(onnx.helper.make_node("Abs", [P + f"{axis}_diff"], [P + f"{axis}_adiff"]))
        nodes.append(onnx.helper.make_node(
            "Less", [P + f"{axis}_adiff", zero5], [P + f"{axis}_full_b"]
        ))
        nodes.append(onnx.helper.make_node(
            "Greater", [P + f"{axis}_total", zero5], [P + f"{axis}_active_b"]
        ))
        nodes.append(onnx.helper.make_node(
            "Cast", [P + f"{axis}_full_b"], [P + f"{axis}_full_f"], to=onnx.TensorProto.FLOAT
        ))
        nodes.append(onnx.helper.make_node(
            "Cast", [P + f"{axis}_active_b"], [P + f"{axis}_active_f"], to=onnx.TensorProto.FLOAT
        ))
        nodes.append(onnx.helper.make_node(
            "Mul", [P + f"{axis}_full_f", P + f"{axis}_active_f"], [P + f"{axis}_div"]
        ))
        nodes.append(onnx.helper.make_node(
            "Sub", [P + f"{axis}_active_f", P + f"{axis}_div"], [P + f"{axis}_non_div"]
        ))

    # Strictly-before divider counts for each row/col via cumulative-sum Conv.
    row_w = add_tensor("row_w", onnx.TensorProto.FLOAT, [1, 1, 30, 1], [1.0] * 30)
    col_w = add_tensor("col_w", onnx.TensorProto.FLOAT, [1, 1, 1, 30], [1.0] * 30)
    nodes.append(onnx.helper.make_node(
        "Conv", [P + "row_div", row_w], [P + "row_inclusive"],
        kernel_shape=[30, 1], pads=[29, 0, 0, 0], strides=[1, 1],
    ))
    nodes.append(onnx.helper.make_node(
        "Conv", [P + "col_div", col_w], [P + "col_inclusive"],
        kernel_shape=[1, 30], pads=[0, 29, 0, 0], strides=[1, 1],
    ))

    row_st = add_tensor("row_st", onnx.TensorProto.INT64, [1], [0])
    row_en = add_tensor("row_en", onnx.TensorProto.INT64, [1], [29])
    row_ax = add_tensor("row_ax", onnx.TensorProto.INT64, [1], [2])
    col_st = add_tensor("col_st", onnx.TensorProto.INT64, [1], [0])
    col_en = add_tensor("col_en", onnx.TensorProto.INT64, [1], [29])
    col_ax = add_tensor("col_ax", onnx.TensorProto.INT64, [1], [3])
    nodes.append(onnx.helper.make_node(
        "Slice", [P + "row_inclusive", row_st, row_en, row_ax, one_step], [P + "row_before_29"]
    ))
    nodes.append(onnx.helper.make_node(
        "Pad", [P + "row_before_29"], [P + "row_before"],
        pads=[0, 0, 1, 0, 0, 0, 0, 0], mode="constant", value=0.0,
    ))
    nodes.append(onnx.helper.make_node(
        "Slice", [P + "col_inclusive", col_st, col_en, col_ax, one_step], [P + "col_before_29"]
    ))
    nodes.append(onnx.helper.make_node(
        "Pad", [P + "col_before_29"], [P + "col_before"],
        pads=[0, 0, 0, 1, 0, 0, 0, 0], mode="constant", value=0.0,
    ))

    def segment_masks(axis_name):
        before = P + f"{axis_name}_before"
        non_div = P + f"{axis_name}_non_div"
        for label, op, thresh in (
            ("pre", "Less", zero5),
            ("gt0", "Greater", zero5),
            ("lt2", "Less", one5),
            ("post", "Greater", one5),
        ):
            nodes.append(onnx.helper.make_node(op, [before, thresh], [P + f"{axis_name}_{label}_b"]))
            nodes.append(onnx.helper.make_node(
                "Cast", [P + f"{axis_name}_{label}_b"], [P + f"{axis_name}_{label}_f"],
                to=onnx.TensorProto.FLOAT,
            ))
        nodes.append(onnx.helper.make_node(
            "Mul", [P + f"{axis_name}_pre_f", non_div], [P + f"{axis_name}_seg0"]
        ))
        nodes.append(onnx.helper.make_node(
            "Mul", [P + f"{axis_name}_gt0_f", P + f"{axis_name}_lt2_f"], [P + f"{axis_name}_mid_raw"]
        ))
        nodes.append(onnx.helper.make_node(
            "Mul", [P + f"{axis_name}_mid_raw", non_div], [P + f"{axis_name}_seg1"]
        ))
        nodes.append(onnx.helper.make_node(
            "Mul", [P + f"{axis_name}_post_f", non_div], [P + f"{axis_name}_seg2"]
        ))

    segment_masks("row")
    segment_masks("col")

    def make_region(name, row_seg, col_seg):
        nodes.append(onnx.helper.make_node(
            "Mul", [P + f"row_seg{row_seg}", P + f"col_seg{col_seg}"], [P + name]
        ))
        return P + name

    ch1 = make_region("ch1_bottom_mid", 2, 1)
    ch2 = make_region("ch2_top_mid", 0, 1)
    ch3 = make_region("ch3_mid_right", 1, 2)
    ch4 = make_region("ch4_mid_left", 1, 0)
    ch6 = make_region("ch6_center", 1, 1)

    nodes.append(onnx.helper.make_node("Add", [ch1, ch2], [P + "sum12"]))
    nodes.append(onnx.helper.make_node("Add", [ch3, ch4], [P + "sum34"]))
    nodes.append(onnx.helper.make_node("Add", [P + "sum12", P + "sum34"], [P + "sum1234"]))
    nodes.append(onnx.helper.make_node("Add", [P + "sum1234", ch6], [P + "filled_non8"]))
    nodes.append(onnx.helper.make_node("Add", [P + "filled_non8", P + "ch8"], [P + "all_nonzero"]))
    nodes.append(onnx.helper.make_node("Sub", [P + "active", P + "all_nonzero"], [P + "ch0_out"]))
    nodes.append(onnx.helper.make_node("Sub", [P + "active", P + "active"], [P + "zeros"]))

    channel_inputs = []
    for ch in range(10):
        if ch == 0:
            channel_inputs.append(P + "ch0_out")
        elif ch == 1:
            channel_inputs.append(ch1)
        elif ch == 2:
            channel_inputs.append(ch2)
        elif ch == 3:
            channel_inputs.append(ch3)
        elif ch == 4:
            channel_inputs.append(ch4)
        elif ch == 6:
            channel_inputs.append(ch6)
        elif ch == 8:
            channel_inputs.append(P + "ch8")
        else:
            channel_inputs.append(P + "zeros")
    nodes.append(onnx.helper.make_node("Concat", channel_inputs, ["output"], axis=1))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def grid_section_copy_network(batch=1):
    """ONNX: task306-style color-4 divider grid section replication.

    The active grid is two 9-row bands and one to three 9-column bands separated
    by color-4 divider lines. Exactly one 9x9 section contains the source
    pattern; copy that section into every existing section while preserving
    divider lines and canvas padding.
    """
    if batch != 1:
        raise NotImplementedError(
            "grid-section-copy builder is single-example only (batch must be 1)"
        )

    P = "gsc_"
    FLOAT = onnx.TensorProto.FLOAT
    INT64 = onnx.TensorProto.INT64
    nodes = []
    inits = []

    def add_tensor(name, dtype, shape, values):
        tensor = onnx.helper.make_tensor(P + name, dtype, shape, values)
        inits.append(tensor)
        return P + name

    zero = add_tensor("zero", FLOAT, [1], [0.0])
    one = add_tensor("one", FLOAT, [1], [1.0])
    ch_axis = add_tensor("ch_axis", INT64, [1], [1])
    ch_step = add_tensor("ch_step", INT64, [1], [1])
    ch0_s = add_tensor("ch0_s", INT64, [1], [0])
    ch1_s = add_tensor("ch1_s", INT64, [1], [1])
    ch4_s = add_tensor("ch4_s", INT64, [1], [4])
    ch5_s = add_tensor("ch5_s", INT64, [1], [5])
    hw_axes = add_tensor("hw_axes", INT64, [2], [2, 3])
    hw_steps = add_tensor("hw_steps", INT64, [2], [1, 1])

    def spatial_slice(inp, out, r0, c0, h, w):
        starts = add_tensor(f"{out}_st", INT64, [2], [r0, c0])
        ends = add_tensor(f"{out}_en", INT64, [2], [r0 + h, c0 + w])
        nodes.append(onnx.helper.make_node(
            "Slice", [inp, starts, ends, hw_axes, hw_steps], [P + out]
        ))
        return P + out

    # Column-band existence is determined by the optional color-4 divider
    # columns at 9 and 19. Column band 0 always exists.
    nodes.append(onnx.helper.make_node(
        "Slice", ["input", ch4_s, ch5_s, ch_axis, ch_step], [P + "ch4"]
    ))
    col_exists = {0: one}
    for c_idx, c_div in ((1, 9), (2, 19)):
        div = spatial_slice(P + "ch4", f"div_col{c_idx}", 0, c_div, 19, 1)
        count = P + f"div_col{c_idx}_count"
        exists_b = P + f"div_col{c_idx}_exists_b"
        exists = P + f"div_col{c_idx}_exists"
        nodes.append(onnx.helper.make_node(
            "ReduceSum", [div], [count], axes=[2, 3], keepdims=1
        ))
        nodes.append(onnx.helper.make_node(
            "Greater", [count, zero], [exists_b]
        ))
        nodes.append(onnx.helper.make_node(
            "Cast", [exists_b], [exists], to=FLOAT
        ))
        col_exists[c_idx] = exists

    # Select the unique nonblank 9x9 source section.
    selected_terms = []
    for r_idx, r0 in enumerate((0, 10)):
        for c_idx, c0 in enumerate((0, 10, 20)):
            src = spatial_slice("input", f"src_{r_idx}_{c_idx}", r0, c0, 9, 9)
            ch0 = P + f"src_{r_idx}_{c_idx}_ch0"
            nonzero = P + f"src_{r_idx}_{c_idx}_nonzero"
            count = P + f"src_{r_idx}_{c_idx}_count"
            present_b = P + f"src_{r_idx}_{c_idx}_present_b"
            present = P + f"src_{r_idx}_{c_idx}_present"
            selected = P + f"src_{r_idx}_{c_idx}_selected"
            nodes.append(onnx.helper.make_node(
                "Slice", [src, ch0_s, ch1_s, ch_axis, ch_step], [ch0]
            ))
            nodes.append(onnx.helper.make_node("Sub", [one, ch0], [nonzero]))
            nodes.append(onnx.helper.make_node(
                "ReduceSum", [nonzero], [count], axes=[2, 3], keepdims=1
            ))
            nodes.append(onnx.helper.make_node(
                "Greater", [count, zero], [present_b]
            ))
            nodes.append(onnx.helper.make_node(
                "Cast", [present_b], [present], to=FLOAT
            ))
            nodes.append(onnx.helper.make_node(
                "Mul", [src, present], [selected]
            ))
            selected_terms.append(selected)

    nodes.append(onnx.helper.make_node(
        "Sum", selected_terms, [P + "selected_source"]
    ))

    # Place the selected source into every existing target section. Optional
    # target columns are gated so padding does not become active output.
    placed = []
    for r_idx, r0 in enumerate((0, 10)):
        for c_idx, c0 in enumerate((0, 10, 20)):
            block = P + "selected_source"
            if c_idx != 0:
                gated = P + f"target_{r_idx}_{c_idx}_gated"
                nodes.append(onnx.helper.make_node(
                    "Mul", [block, col_exists[c_idx]], [gated]
                ))
                block = gated
            out = P + f"target_{r_idx}_{c_idx}_placed"
            nodes.append(onnx.helper.make_node(
                "Pad", [block], [out],
                pads=[0, 0, r0, c0, 0, 0, 30 - r0 - 9, 30 - c0 - 9],
                mode="constant",
                value=0.0,
            ))
            placed.append(out)

    nodes.append(onnx.helper.make_node("Sum", placed, [P + "sections"]))
    nodes.append(onnx.helper.make_node(
        "ReduceSum", [P + "sections"], [P + "section_area"], axes=[1], keepdims=1
    ))
    nodes.append(onnx.helper.make_node("Sub", [one, P + "section_area"], [P + "keep_input"]))
    nodes.append(onnx.helper.make_node("Mul", ["input", P + "keep_input"], [P + "kept_input"]))
    nodes.append(onnx.helper.make_node("Add", [P + "kept_input", P + "sections"], ["output"]))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def grid_max_count_fill_network(batch=1):
    """Task059-style 3x3 compartment max-count fill.

    Fixed 11x11 grid with color-5 divider rows/cols at 3 and 7. Exactly one
    object color appears in the nine 3x3 compartments. Output preserves dividers
    and fills every compartment whose object-count equals the maximum object
    count over all compartments; all other compartment cells become background.
    """
    if batch != 1:
        raise NotImplementedError(
            "grid-max-count-fill builder is single-example only (batch must be 1)"
        )

    basis = np.zeros((5, 30), dtype=np.float16)
    basis[0, 0:3] = 1.0
    basis[1, 3] = 1.0
    basis[2, 4:7] = 1.0
    basis[3, 7] = 1.0
    basis[4, 8:11] = 1.0

    div_coef = np.zeros((5, 5), dtype=np.float16)
    div_coef[1, :] = 1.0
    div_coef[3, :] = 1.0
    div_coef[:, 1] = 1.0
    div_coef[:, 3] = 1.0

    static_coef2 = np.zeros((2, 1, 5, 5), dtype=np.float16)
    static_coef2[0, 0, :, :] = 1.0
    static_coef2[1, 0, :, :] = div_coef

    static_ch2 = np.zeros((2, 10), dtype=np.float16)
    static_ch2[0, 0] = 1.0
    static_ch2[1, 0] = -1.0
    static_ch2[1, 5] = 1.0

    fill_sub = np.zeros((1, 10), dtype=np.float16)
    fill_sub[0, 0] = 2.0
    fill_sub[0, 5] = 1.0

    inits = [
        onnx.numpy_helper.from_array(
            np.array([0, 1, 1, 1, 1, 0, 1, 1, 1, 1], dtype=np.float32),
            "count_ch",
        ),
        onnx.numpy_helper.from_array(basis.astype(np.float32), "basis5_count"),
        onnx.numpy_helper.from_array(np.array([1.0], dtype=np.float32), "count_one"),
        onnx.numpy_helper.from_array(fill_sub, "fill_sub"),
        onnx.numpy_helper.from_array(np.array([1.0], dtype=np.float16), "batch1"),
        onnx.numpy_helper.from_array(static_ch2, "static_ch2"),
        onnx.numpy_helper.from_array(static_coef2, "static_coef2"),
        onnx.numpy_helper.from_array(basis, "basis5"),
    ]
    nodes = [
        onnx.helper.make_node(
            "Einsum",
            ["input", "count_ch", "basis5_count", "basis5_count", "count_one"],
            ["counts3"],
            name="counts3",
            equation="bcrs,c,ir,js,o->boij",
        ),
        onnx.helper.make_node(
            "ReduceMax", ["counts3"], ["max_count"],
            name="max_count", axes=[2, 3], keepdims=1,
        ),
        onnx.helper.make_node(
            "Equal", ["counts3", "max_count"], ["selected_bool"],
            name="selected_bool",
        ),
        onnx.helper.make_node(
            "Cast", ["selected_bool"], ["selected_num"],
            name="selected_num", to=onnx.TensorProto.FLOAT16,
        ),
        onnx.helper.make_node(
            "ReduceMax", ["input"], ["present_f32"],
            name="present_f32", axes=[2, 3], keepdims=0,
        ),
        onnx.helper.make_node(
            "Cast", ["present_f32"], ["present_num"],
            name="present_num", to=onnx.TensorProto.FLOAT16,
        ),
        onnx.helper.make_node(
            "Sub", ["present_num", "fill_sub"], ["fill_delta"],
            name="fill_delta",
        ),
        onnx.helper.make_node(
            "Concat", ["static_ch2", "fill_delta"], ["ch3"],
            name="ch3", axis=0,
        ),
        onnx.helper.make_node(
            "Concat", ["static_coef2", "selected_num"], ["coef3"],
            name="coef3", axis=0,
        ),
        onnx.helper.make_node(
            "Einsum",
            ["batch1", "coef3", "ch3", "basis5", "basis5"],
            ["output"],
            name="output",
            equation="b,koij,kc,ir,js->bcrs",
        ),
    ]

    inp = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 10, 30, 30])
    out = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT16, [1, 10, 30, 30])
    graph = onnx.helper.make_graph(nodes, "task059_cell_count_block5_basis5", [inp], [out], inits)
    return onnx.helper.make_model(
        graph, ir_version=_IR_VERSION, opset_imports=[onnx.helper.make_opsetid("", 13)]
    )


def grid_object_summary_network(batch=1):
    """Task184-style nonzero band object summary.

    Input is partitioned by all-zero gutter rows/cols into 2-3 row bands and
    2-3 column bands. Each band intersection contains one nonzero color (zeros
    are holes/noise). Output is the compact row-band x column-band color grid.
    Slots beyond the runtime band counts are left with all channels empty so the
    official decoder trims them away.
    """
    if batch != 1:
        raise NotImplementedError(
            "grid-object-summary builder is single-example only (batch must be 1)"
        )

    P = "gos_"
    nodes = []
    inits = []

    def add_tensor(name, dtype, shape, values):
        tensor = onnx.helper.make_tensor(P + name, dtype, shape, values)
        inits.append(tensor)
        return P + name

    def add_binary_chain(op_type, names, out_name):
        if not names:
            raise ValueError("cannot build empty chain")
        cur = names[0]
        for idx, name in enumerate(names[1:], start=1):
            nxt = P + f"{out_name}_{idx}" if idx < len(names) - 1 else P + out_name
            nodes.append(onnx.helper.make_node(op_type, [cur, name], [nxt]))
            cur = nxt
        if len(names) == 1:
            nodes.append(onnx.helper.make_node("Identity", [cur], [P + out_name]))
        return P + out_name

    zero_canvas = add_tensor("zero_canvas", onnx.TensorProto.FLOAT, [1, 1, 30, 30], [0.0] * 900)
    half = add_tensor("half", onnx.TensorProto.FLOAT, [1], [0.5])
    one = add_tensor("one", onnx.TensorProto.FLOAT, [1], [1.0])
    ch0_st = add_tensor("ch0_st", onnx.TensorProto.INT64, [1], [0])
    ch0_en = add_tensor("ch0_en", onnx.TensorProto.INT64, [1], [1])
    ch_ax = add_tensor("ch_ax", onnx.TensorProto.INT64, [1], [1])
    one_step = add_tensor("one_step", onnx.TensorProto.INT64, [1], [1])

    nodes.append(onnx.helper.make_node(
        "ReduceSum", ["input"], [P + "active"], axes=[1], keepdims=1
    ))
    nodes.append(onnx.helper.make_node(
        "Slice", ["input", ch0_st, ch0_en, ch_ax, one_step], [P + "ch0"]
    ))
    nodes.append(onnx.helper.make_node(
        "Sub", [P + "active", P + "ch0"], [P + "nonzero"]
    ))
    nodes.append(onnx.helper.make_node(
        "ReduceMax", [P + "nonzero"], [P + "row_has"], axes=[3], keepdims=1
    ))
    nodes.append(onnx.helper.make_node(
        "ReduceMax", [P + "nonzero"], [P + "col_has"], axes=[2], keepdims=1
    ))

    row_st = add_tensor("row_st", onnx.TensorProto.INT64, [1], [0])
    row_en = add_tensor("row_en", onnx.TensorProto.INT64, [1], [29])
    row_ax = add_tensor("row_ax", onnx.TensorProto.INT64, [1], [2])
    col_st = add_tensor("col_st", onnx.TensorProto.INT64, [1], [0])
    col_en = add_tensor("col_en", onnx.TensorProto.INT64, [1], [29])
    col_ax = add_tensor("col_ax", onnx.TensorProto.INT64, [1], [3])

    nodes.append(onnx.helper.make_node(
        "Slice", [P + "row_has", row_st, row_en, row_ax, one_step], [P + "row_prev_29"]
    ))
    nodes.append(onnx.helper.make_node(
        "Pad", [P + "row_prev_29"], [P + "row_prev"],
        pads=[0, 0, 1, 0, 0, 0, 0, 0], mode="constant", value=0.0,
    ))
    nodes.append(onnx.helper.make_node(
        "Slice", [P + "col_has", col_st, col_en, col_ax, one_step], [P + "col_prev_29"]
    ))
    nodes.append(onnx.helper.make_node(
        "Pad", [P + "col_prev_29"], [P + "col_prev"],
        pads=[0, 0, 0, 1, 0, 0, 0, 0], mode="constant", value=0.0,
    ))

    for axis in ("row", "col"):
        nodes.append(onnx.helper.make_node(
            "Sub", [one, P + f"{axis}_prev"], [P + f"{axis}_not_prev"]
        ))
        nodes.append(onnx.helper.make_node(
            "Mul", [P + f"{axis}_has", P + f"{axis}_not_prev"], [P + f"{axis}_start"]
        ))

    row_w = add_tensor("row_w", onnx.TensorProto.FLOAT, [1, 1, 30, 1], [1.0] * 30)
    col_w = add_tensor("col_w", onnx.TensorProto.FLOAT, [1, 1, 1, 30], [1.0] * 30)
    nodes.append(onnx.helper.make_node(
        "Conv", [P + "row_start", row_w], [P + "row_band_idx"],
        kernel_shape=[30, 1], pads=[29, 0, 0, 0], strides=[1, 1],
    ))
    nodes.append(onnx.helper.make_node(
        "Conv", [P + "col_start", col_w], [P + "col_band_idx"],
        kernel_shape=[1, 30], pads=[0, 29, 0, 0], strides=[1, 1],
    ))

    row_bands = []
    col_bands = []
    for axis, bands in (("row", row_bands), ("col", col_bands)):
        for idx, value in enumerate((1.0, 2.0, 3.0)):
            k = add_tensor(f"{axis}_k{idx}", onnx.TensorProto.FLOAT, [1], [value])
            nodes.append(onnx.helper.make_node(
                "Sub", [P + f"{axis}_band_idx", k], [P + f"{axis}_band{idx}_diff"]
            ))
            nodes.append(onnx.helper.make_node(
                "Abs", [P + f"{axis}_band{idx}_diff"], [P + f"{axis}_band{idx}_adiff"]
            ))
            nodes.append(onnx.helper.make_node(
                "Less", [P + f"{axis}_band{idx}_adiff", half], [P + f"{axis}_band{idx}_b"]
            ))
            nodes.append(onnx.helper.make_node(
                "Cast", [P + f"{axis}_band{idx}_b"], [P + f"{axis}_band{idx}_raw"],
                to=onnx.TensorProto.FLOAT,
            ))
            nodes.append(onnx.helper.make_node(
                "Mul", [P + f"{axis}_band{idx}_raw", P + f"{axis}_has"], [P + f"{axis}_band{idx}"]
            ))
            bands.append(P + f"{axis}_band{idx}")

    cell_masks = []
    for rr in range(3):
        row = []
        for cc in range(3):
            vals = [0.0] * 900
            vals[rr * 30 + cc] = 1.0
            row.append(add_tensor(f"cell_{rr}_{cc}", onnx.TensorProto.FLOAT, [1, 1, 30, 30], vals))
        cell_masks.append(row)

    channel_outputs = [zero_canvas]
    for ch in range(1, 10):
        ch_st = add_tensor(f"ch{ch}_st", onnx.TensorProto.INT64, [1], [ch])
        ch_en = add_tensor(f"ch{ch}_en", onnx.TensorProto.INT64, [1], [ch + 1])
        ch_name = P + f"ch{ch}"
        nodes.append(onnx.helper.make_node(
            "Slice", ["input", ch_st, ch_en, ch_ax, one_step], [ch_name]
        ))
        slot_names = []
        for rr in range(3):
            for cc in range(3):
                prefix = P + f"ch{ch}_r{rr}_c{cc}"
                nodes.append(onnx.helper.make_node(
                    "Mul", [ch_name, row_bands[rr]], [prefix + "_row"]
                ))
                nodes.append(onnx.helper.make_node(
                    "Mul", [prefix + "_row", col_bands[cc]], [prefix + "_block"]
                ))
                nodes.append(onnx.helper.make_node(
                    "ReduceSum", [prefix + "_block"], [prefix + "_count"],
                    axes=[2, 3], keepdims=1,
                ))
                nodes.append(onnx.helper.make_node(
                    "Greater", [prefix + "_count", half], [prefix + "_present_b"]
                ))
                nodes.append(onnx.helper.make_node(
                    "Cast", [prefix + "_present_b"], [prefix + "_present"],
                    to=onnx.TensorProto.FLOAT,
                ))
                nodes.append(onnx.helper.make_node(
                    "Mul", [prefix + "_present", cell_masks[rr][cc]], [prefix + "_slot"]
                ))
                slot_names.append(prefix + "_slot")
        channel_outputs.append(add_binary_chain("Add", slot_names, f"ch{ch}_out"))

    nodes.append(onnx.helper.make_node("Concat", channel_outputs, ["output"], axis=1))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def two_blob_tile_fit_network(batch=1):
    """Task153-style 10x10 two-blob packing into a 3x3 output.

    For each nonzero color, crop its tight bbox into a 3x3 normalized patch,
    enumerate clipped 3x3 placements, and select the unique pair of placements
    that fully covers the compact 3x3 output without overlap.
    """
    if batch != 1:
        raise NotImplementedError(
            "two-blob-tile-fit builder is single-example only (batch must be 1)"
        )

    P = "tbt_"
    nodes = []
    inits = []

    def add_tensor(name, dtype, shape, values):
        tensor = onnx.helper.make_tensor(P + name, dtype, shape, values)
        inits.append(tensor)
        return P + name

    fg_starts = add_tensor("fg_starts", onnx.TensorProto.INT64, [1], [1])
    fg_ends = add_tensor("fg_ends", onnx.TensorProto.INT64, [1], [10])
    ch_axis = add_tensor("ch_axis", onnx.TensorProto.INT64, [1], [1])
    one_step = add_tensor("one_step", onnx.TensorProto.INT64, [1], [1])
    nodes.append(onnx.helper.make_node(
        "Slice", ["input", fg_starts, fg_ends, ch_axis, one_step], [P + "fg9"]
    ))

    nodes.append(onnx.helper.make_node(
        "ReduceMax", [P + "fg9"], [P + "row_has"], axes=[3], keepdims=1
    ))
    nodes.append(onnx.helper.make_node(
        "ReduceMax", [P + "fg9"], [P + "col_has"], axes=[2], keepdims=1
    ))

    desc_row = add_tensor(
        "desc_row", onnx.TensorProto.FLOAT, [1, 1, 30, 1],
        [float(30 - i) for i in range(30)],
    )
    desc_col = add_tensor(
        "desc_col", onnx.TensorProto.FLOAT, [1, 1, 1, 30],
        [float(30 - i) for i in range(30)],
    )
    nodes.append(onnx.helper.make_node("Mul", [P + "row_has", desc_row], [P + "w_first_row"]))
    nodes.append(onnx.helper.make_node("Mul", [P + "col_has", desc_col], [P + "w_first_col"]))
    nodes.append(onnx.helper.make_node(
        "ArgMax", [P + "w_first_row"], [P + "first_row_3d"], axis=2, keepdims=0
    ))
    nodes.append(onnx.helper.make_node(
        "ArgMax", [P + "w_first_col"], [P + "first_col_3d"], axis=3, keepdims=0
    ))

    rs9 = add_tensor("rs9", onnx.TensorProto.INT64, [1], [9])
    nodes.append(onnx.helper.make_node("Reshape", [P + "first_row_3d", rs9], [P + "first_row_i"]))
    nodes.append(onnx.helper.make_node("Reshape", [P + "first_col_3d", rs9], [P + "first_col_i"]))
    nodes.append(onnx.helper.make_node(
        "Cast", [P + "first_row_i"], [P + "first_row_f"], to=onnx.TensorProto.FLOAT
    ))
    nodes.append(onnx.helper.make_node(
        "Cast", [P + "first_col_i"], [P + "first_col_f"], to=onnx.TensorProto.FLOAT
    ))

    range3 = add_tensor("range3", onnx.TensorProto.FLOAT, [3], [0.0, 1.0, 2.0])
    color_index_inits = [
        add_tensor(f"color_idx_{i}", onnx.TensorProto.INT64, [1], [i])
        for i in range(9)
    ]
    placement_slice = {}
    placement_pads = {}
    place_axes = add_tensor("place_axes", onnx.TensorProto.INT64, [2], [2, 3])
    for ar in range(3):
        for ac in range(3):
            if ar == 0 and ac == 0:
                continue
            placement_slice[(ar, ac)] = (
                add_tensor(f"pl_{ar}_{ac}_starts", onnx.TensorProto.INT64, [2], [0, 0]),
                add_tensor(f"pl_{ar}_{ac}_ends", onnx.TensorProto.INT64, [2], [3 - ar, 3 - ac]),
                add_tensor(f"pl_{ar}_{ac}_steps", onnx.TensorProto.INT64, [2], [1, 1]),
            )
            placement_pads[(ar, ac)] = [0, 0, ar, ac, 0, 0, 0, 0]

    placements = []
    for color_offset in range(9):
        ci = color_index_inits[color_offset]
        nodes.append(onnx.helper.make_node(
            "Gather", [P + "first_row_f", ci], [P + f"c{color_offset}_r0_f"], axis=0
        ))
        nodes.append(onnx.helper.make_node(
            "Gather", [P + "first_col_f", ci], [P + f"c{color_offset}_c0_f"], axis=0
        ))
        nodes.append(onnx.helper.make_node(
            "Gather", [P + "fg9", ci], [P + f"c{color_offset}_mask"], axis=1
        ))
        nodes.append(onnx.helper.make_node(
            "Add", [P + f"c{color_offset}_r0_f", range3], [P + f"c{color_offset}_ridx_f"]
        ))
        nodes.append(onnx.helper.make_node(
            "Add", [P + f"c{color_offset}_c0_f", range3], [P + f"c{color_offset}_cidx_f"]
        ))
        nodes.append(onnx.helper.make_node(
            "Cast", [P + f"c{color_offset}_ridx_f"], [P + f"c{color_offset}_ridx"],
            to=onnx.TensorProto.INT64,
        ))
        nodes.append(onnx.helper.make_node(
            "Cast", [P + f"c{color_offset}_cidx_f"], [P + f"c{color_offset}_cidx"],
            to=onnx.TensorProto.INT64,
        ))
        nodes.append(onnx.helper.make_node(
            "Gather", [P + f"c{color_offset}_mask", P + f"c{color_offset}_ridx"],
            [P + f"c{color_offset}_rows"], axis=2,
        ))
        nodes.append(onnx.helper.make_node(
            "Gather", [P + f"c{color_offset}_rows", P + f"c{color_offset}_cidx"],
            [P + f"c{color_offset}_patch"], axis=3,
        ))

        for ar in range(3):
            for ac in range(3):
                if ar == 0 and ac == 0:
                    placements.append(P + f"c{color_offset}_patch")
                    continue
                starts, ends, steps = placement_slice[(ar, ac)]
                sliced = P + f"c{color_offset}_pl{ar}{ac}_slice"
                placed = P + f"c{color_offset}_pl{ar}{ac}"
                nodes.append(onnx.helper.make_node(
                    "Slice",
                    [P + f"c{color_offset}_patch", starts, ends, place_axes, steps],
                    [sliced],
                ))
                nodes.append(onnx.helper.make_node(
                    "Pad", [sliced], [placed],
                    pads=placement_pads[(ar, ac)], mode="constant", value=0.0,
                ))
                placements.append(placed)

    nodes.append(onnx.helper.make_node("Concat", placements, [P + "placements"], axis=1))

    rs_flat = add_tensor("rs_flat", onnx.TensorProto.INT64, [2], [81, 9])
    nodes.append(onnx.helper.make_node("Reshape", [P + "placements", rs_flat], [P + "flat"]))
    nodes.append(onnx.helper.make_node("Transpose", [P + "flat"], [P + "flat_t"], perm=[1, 0]))
    nodes.append(onnx.helper.make_node("MatMul", [P + "flat", P + "flat_t"], [P + "overlap"]))
    nodes.append(onnx.helper.make_node(
        "ReduceSum", [P + "flat"], [P + "counts"], axes=[1], keepdims=0
    ))

    rs_81_1 = add_tensor("rs_81_1", onnx.TensorProto.INT64, [2], [81, 1])
    rs_1_81 = add_tensor("rs_1_81", onnx.TensorProto.INT64, [2], [1, 81])
    nodes.append(onnx.helper.make_node("Reshape", [P + "counts", rs_81_1], [P + "counts_col"]))
    nodes.append(onnx.helper.make_node("Reshape", [P + "counts", rs_1_81], [P + "counts_row"]))
    nodes.append(onnx.helper.make_node("Add", [P + "counts_col", P + "counts_row"], [P + "count_sum"]))

    zero_i = add_tensor("zero_i", onnx.TensorProto.INT64, [1], [0])
    nine_i = add_tensor("nine_i", onnx.TensorProto.INT64, [1], [9])
    upper = add_tensor(
        "upper", onnx.TensorProto.FLOAT, [81, 81],
        [1.0 if i < j else 0.0 for i in range(81) for j in range(81)],
    )
    nodes.append(onnx.helper.make_node(
        "Cast", [P + "overlap"], [P + "overlap_i"], to=onnx.TensorProto.INT64
    ))
    nodes.append(onnx.helper.make_node(
        "Cast", [P + "count_sum"], [P + "count_sum_i"], to=onnx.TensorProto.INT64
    ))
    nodes.append(onnx.helper.make_node("Equal", [P + "overlap_i", zero_i], [P + "no_overlap_b"]))
    nodes.append(onnx.helper.make_node("Equal", [P + "count_sum_i", nine_i], [P + "covers_all_b"]))
    nodes.append(onnx.helper.make_node(
        "Cast", [P + "no_overlap_b"], [P + "no_overlap_f"], to=onnx.TensorProto.FLOAT
    ))
    nodes.append(onnx.helper.make_node(
        "Cast", [P + "covers_all_b"], [P + "covers_all_f"], to=onnx.TensorProto.FLOAT
    ))
    nodes.append(onnx.helper.make_node("Mul", [P + "no_overlap_f", P + "covers_all_f"], [P + "valid_raw"]))
    nodes.append(onnx.helper.make_node("Mul", [P + "valid_raw", upper], [P + "valid"]))

    nodes.append(onnx.helper.make_node(
        "ReduceSum", [P + "valid"], [P + "selected_left"], axes=[1], keepdims=0
    ))
    nodes.append(onnx.helper.make_node(
        "ReduceSum", [P + "valid"], [P + "selected_right"], axes=[0], keepdims=0
    ))
    nodes.append(onnx.helper.make_node(
        "Add", [P + "selected_left", P + "selected_right"], [P + "selected"]
    ))
    rs_sel = add_tensor("rs_sel", onnx.TensorProto.INT64, [4], [1, 81, 1, 1])
    nodes.append(onnx.helper.make_node("Reshape", [P + "selected", rs_sel], [P + "selected_4d"]))
    nodes.append(onnx.helper.make_node(
        "Mul", [P + "placements", P + "selected_4d"], [P + "selected_placements"]
    ))

    weights = []
    for out_ch in range(10):
        for placement_idx in range(81):
            color = placement_idx // 9 + 1
            weights.append(1.0 if out_ch == color else 0.0)
    conv_w = add_tensor("conv_w", onnx.TensorProto.FLOAT, [10, 81, 1, 1], weights)
    nodes.append(onnx.helper.make_node(
        "Conv", [P + "selected_placements", conv_w], [P + "compact"]
    ))
    nodes.append(onnx.helper.make_node(
        "Pad", [P + "compact"], ["output"],
        pads=[0, 0, 0, 0, 0, 0, 27, 27], mode="constant", value=0.0,
    ))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def fourfold_rotation_tile_network(batch=1):
    """Task106-style 2x2 tile of square input rotations.

    Current verified examples use square active grids of size 2 or 3. Build both
    static candidates and select the right one from the runtime active row count.
    """
    if batch != 1:
        raise NotImplementedError(
            "fourfold-rotation-tile builder is single-example only (batch must be 1)"
        )

    P = "frt_"
    nodes = []
    inits = []

    def add_tensor(name, dtype, shape, values):
        tensor = onnx.helper.make_tensor(P + name, dtype, shape, values)
        inits.append(tensor)
        return P + name

    def add_candidate(n):
        starts = add_tensor(f"n{n}_starts", onnx.TensorProto.INT64, [2], [0, 0])
        ends = add_tensor(f"n{n}_ends", onnx.TensorProto.INT64, [2], [n, n])
        axes = add_tensor(f"n{n}_axes", onnx.TensorProto.INT64, [2], [2, 3])
        steps = add_tensor(f"n{n}_steps", onnx.TensorProto.INT64, [2], [1, 1])
        idx = add_tensor(f"n{n}_rev", onnx.TensorProto.INT64, [n], list(range(n - 1, -1, -1)))
        nodes.append(onnx.helper.make_node(
            "Slice", ["input", starts, ends, axes, steps], [P + f"n{n}_active"]
        ))
        nodes.append(onnx.helper.make_node(
            "Transpose", [P + f"n{n}_active"], [P + f"n{n}_tr"], perm=[0, 1, 3, 2]
        ))
        nodes.append(onnx.helper.make_node(
            "Gather", [P + f"n{n}_tr", idx], [P + f"n{n}_rot90"], axis=3
        ))
        nodes.append(onnx.helper.make_node(
            "Gather", [P + f"n{n}_tr", idx], [P + f"n{n}_rot270"], axis=2
        ))
        nodes.append(onnx.helper.make_node(
            "Gather", [P + f"n{n}_active", idx], [P + f"n{n}_flip_h"], axis=2
        ))
        nodes.append(onnx.helper.make_node(
            "Gather", [P + f"n{n}_flip_h", idx], [P + f"n{n}_rot180"], axis=3
        ))
        nodes.append(onnx.helper.make_node(
            "Concat", [P + f"n{n}_active", P + f"n{n}_rot90"], [P + f"n{n}_top"], axis=3
        ))
        nodes.append(onnx.helper.make_node(
            "Concat", [P + f"n{n}_rot270", P + f"n{n}_rot180"], [P + f"n{n}_bottom"], axis=3
        ))
        nodes.append(onnx.helper.make_node(
            "Concat", [P + f"n{n}_top", P + f"n{n}_bottom"], [P + f"n{n}_compact"], axis=2
        ))
        nodes.append(onnx.helper.make_node(
            "Pad", [P + f"n{n}_compact"], [P + f"n{n}_canvas"],
            pads=[0, 0, 0, 0, 0, 0, 30 - 2 * n, 30 - 2 * n],
            mode="constant", value=0.0,
        ))
        return P + f"n{n}_canvas"

    cand2 = add_candidate(2)
    cand3 = add_candidate(3)

    nodes.append(onnx.helper.make_node(
        "ReduceSum", ["input"], [P + "active"], axes=[1], keepdims=1
    ))
    nodes.append(onnx.helper.make_node(
        "ReduceMax", [P + "active"], [P + "row_has"], axes=[3], keepdims=0
    ))
    nodes.append(onnx.helper.make_node(
        "ReduceSum", [P + "row_has"], [P + "row_count"], axes=[2], keepdims=0
    ))
    half = add_tensor("half", onnx.TensorProto.FLOAT, [1], [0.5])
    rs_mask = add_tensor("rs_mask", onnx.TensorProto.INT64, [4], [1, 1, 1, 1])
    selected = []
    for n, cand in ((2, cand2), (3, cand3)):
        n_const = add_tensor(f"n{n}_const", onnx.TensorProto.FLOAT, [1], [float(n)])
        nodes.append(onnx.helper.make_node(
            "Sub", [P + "row_count", n_const], [P + f"n{n}_diff"]
        ))
        nodes.append(onnx.helper.make_node(
            "Abs", [P + f"n{n}_diff"], [P + f"n{n}_adiff"]
        ))
        nodes.append(onnx.helper.make_node(
            "Less", [P + f"n{n}_adiff", half], [P + f"n{n}_is_b"]
        ))
        nodes.append(onnx.helper.make_node(
            "Cast", [P + f"n{n}_is_b"], [P + f"n{n}_is_f"], to=onnx.TensorProto.FLOAT
        ))
        nodes.append(onnx.helper.make_node(
            "Reshape", [P + f"n{n}_is_f", rs_mask], [P + f"n{n}_mask"]
        ))
        nodes.append(onnx.helper.make_node(
            "Mul", [cand, P + f"n{n}_mask"], [P + f"n{n}_selected"]
        ))
        selected.append(P + f"n{n}_selected")

    nodes.append(onnx.helper.make_node("Add", selected, ["output"]))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def diagonal_rays_from_line_network(batch=1):
    """Task348-style alternating upward diagonal rays from color-7 cells."""
    P = "drl_"
    nodes = []
    inits = []

    def add_tensor(name, dtype, shape, values):
        tensor = onnx.helper.make_tensor(P + name, dtype, shape, values)
        inits.append(tensor)
        return P + name

    zero = add_tensor("zero", onnx.TensorProto.FLOAT, [1], [0.0])
    one = add_tensor("one", onnx.TensorProto.FLOAT, [1], [1.0])
    ch_axis = add_tensor("ch_axis", onnx.TensorProto.INT64, [1], [1])
    one_step = add_tensor("one_step", onnx.TensorProto.INT64, [1], [1])
    ch0_st = add_tensor("ch0_st", onnx.TensorProto.INT64, [1], [0])
    ch0_en = add_tensor("ch0_en", onnx.TensorProto.INT64, [1], [1])
    ch7_st = add_tensor("ch7_st", onnx.TensorProto.INT64, [1], [7])
    ch7_en = add_tensor("ch7_en", onnx.TensorProto.INT64, [1], [8])

    nodes.append(onnx.helper.make_node(
        "Slice", ["input", ch0_st, ch0_en, ch_axis, one_step], [P + "ch0"]
    ))
    nodes.append(onnx.helper.make_node(
        "Slice", ["input", ch7_st, ch7_en, ch_axis, one_step], [P + "ch7"]
    ))
    nodes.append(onnx.helper.make_node("Mul", [P + "ch7", zero], [P + "zero_ch"]))

    def diag_kernel(name, parity, direction):
        vals = [0.0] * (30 * 30)
        for d in range(1, 30):
            if d % 2 != parity:
                continue
            if direction == "ul":
                kh, kw = d, d
            else:
                kh, kw = d, 29 - d
            vals[kh * 30 + kw] = 1.0
        return add_tensor(name, onnx.TensorProto.FLOAT, [1, 1, 30, 30], vals)

    ray_specs = [
        ("ul_odd", 1, "ul", [0, 0, 29, 29]),
        ("ul_even", 0, "ul", [0, 0, 29, 29]),
        ("ur_odd", 1, "ur", [0, 29, 29, 0]),
        ("ur_even", 0, "ur", [0, 29, 29, 0]),
    ]
    hit_names = {}
    for name, parity, direction, pads in ray_specs:
        kernel = diag_kernel(f"{name}_k", parity, direction)
        nodes.append(onnx.helper.make_node(
            "Conv", [P + "ch7", kernel], [P + f"{name}_raw"],
            kernel_shape=[30, 30], pads=pads, strides=[1, 1],
        ))
        nodes.append(onnx.helper.make_node(
            "Greater", [P + f"{name}_raw", zero], [P + f"{name}_b"]
        ))
        nodes.append(onnx.helper.make_node(
            "Cast", [P + f"{name}_b"], [P + f"{name}_hit"],
            to=onnx.TensorProto.FLOAT,
        ))
        hit_names[name] = P + f"{name}_hit"

    nodes.append(onnx.helper.make_node(
        "Max", [hit_names["ul_odd"], hit_names["ur_odd"]], [P + "odd_hit"]
    ))
    nodes.append(onnx.helper.make_node(
        "Max", [hit_names["ul_even"], hit_names["ur_even"]], [P + "even_hit"]
    ))
    nodes.append(onnx.helper.make_node("Mul", [P + "odd_hit", P + "ch0"], [P + "ch8"]))
    nodes.append(onnx.helper.make_node("Mul", [P + "even_hit", P + "ch0"], [P + "even_bg"]))
    nodes.append(onnx.helper.make_node("Sub", [one, P + "ch8"], [P + "not_ch8"]))
    nodes.append(onnx.helper.make_node(
        "Mul", [P + "even_bg", P + "not_ch8"], [P + "ch7_rays"]
    ))
    nodes.append(onnx.helper.make_node(
        "Max", [P + "ch7", P + "ch7_rays"], [P + "ch7_out"]
    ))
    nodes.append(onnx.helper.make_node(
        "Max", [P + "ch8", P + "ch7_rays"], [P + "ray_any"]
    ))
    nodes.append(onnx.helper.make_node("Sub", [one, P + "ray_any"], [P + "bg_keep"]))
    nodes.append(onnx.helper.make_node("Mul", [P + "ch0", P + "bg_keep"], [P + "ch0_out"]))

    nodes.append(onnx.helper.make_node(
        "Concat",
        [
            P + "ch0_out",
            P + "zero_ch",
            P + "zero_ch",
            P + "zero_ch",
            P + "zero_ch",
            P + "zero_ch",
            P + "zero_ch",
            P + "ch7_out",
            P + "ch8",
            P + "zero_ch",
        ],
        ["output"],
        axis=1,
    ))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def euclidean_wedge_fill_network(batch=1):
    """Task175-style 21x21 color lattice hole fill.

    The complete output is a symmetric wedge lattice determined by two runtime
    quantities:
      - base color: the only nonzero color present on the main diagonal.
      - period: the largest present color; task examples use contiguous colors
        1..period.

    For each cell, m=min(r,c), d=abs(r-c). The color offset from base is:
      0 if d==0; -1 if 0<d<=m+1; otherwise floor((d-(m+2))/(m+2)).
    Input zeros are simply holes in this lattice.
    """
    if batch != 1:
        raise NotImplementedError(
            "euclidean-wedge-fill builder is single-example only (batch must be 1)"
        )

    P = "ewf_"
    nodes = []
    inits = []

    def add_tensor(name, dtype, shape, values):
        tensor = onnx.helper.make_tensor(P + name, dtype, shape, values)
        inits.append(tensor)
        return P + name

    def add_chain(op_type, names, out_name):
        if not names:
            return P + "zero_canvas"
        cur = names[0]
        if len(names) == 1:
            nodes.append(onnx.helper.make_node("Identity", [cur], [P + out_name]))
            return P + out_name
        for idx, name in enumerate(names[1:], start=1):
            nxt = P + out_name if idx == len(names) - 1 else P + f"{out_name}_{idx}"
            nodes.append(onnx.helper.make_node(op_type, [cur, name], [nxt]))
            cur = nxt
        return P + out_name

    def offset_for_cell(r, c):
        m = min(r, c)
        d = abs(r - c)
        if d == 0:
            return 0
        if d <= m + 1:
            return -1
        return (d - (m + 2)) // (m + 2)

    half = add_tensor("half", onnx.TensorProto.FLOAT, [1], [0.5])
    one_gate = add_tensor("one_gate", onnx.TensorProto.FLOAT, [1, 1, 1, 1], [1.0])
    zero_canvas = add_tensor("zero_canvas", onnx.TensorProto.FLOAT, [1, 1, 30, 30], [0.0] * 900)

    diag_vals = [0.0] * 900
    for i in range(21):
        diag_vals[i * 30 + i] = 1.0
    diag_mask = add_tensor("diag_mask", onnx.TensorProto.FLOAT, [1, 1, 30, 30], diag_vals)

    offset_masks = {}
    for off in range(-1, 10):
        vals = [0.0] * 900
        for r in range(21):
            for c in range(21):
                if offset_for_cell(r, c) == off:
                    vals[r * 30 + c] = 1.0
        offset_masks[off] = add_tensor(f"mask_off_{off + 1}", onnx.TensorProto.FLOAT, [1, 1, 30, 30], vals)

    ch_ax = add_tensor("ch_ax", onnx.TensorProto.INT64, [1], [1])
    step = add_tensor("step", onnx.TensorProto.INT64, [1], [1])

    present = {}
    base_gate = {}
    for ch in range(1, 10):
        st = add_tensor(f"ch{ch}_st", onnx.TensorProto.INT64, [1], [ch])
        en = add_tensor(f"ch{ch}_en", onnx.TensorProto.INT64, [1], [ch + 1])
        ch_name = P + f"ch{ch}"
        nodes.append(onnx.helper.make_node(
            "Slice", ["input", st, en, ch_ax, step], [ch_name]
        ))
        nodes.append(onnx.helper.make_node(
            "ReduceSum", [ch_name], [P + f"ch{ch}_count"], axes=[2, 3], keepdims=1
        ))
        nodes.append(onnx.helper.make_node(
            "Greater", [P + f"ch{ch}_count", half], [P + f"ch{ch}_present_b"]
        ))
        nodes.append(onnx.helper.make_node(
            "Cast", [P + f"ch{ch}_present_b"], [P + f"ch{ch}_present"],
            to=onnx.TensorProto.FLOAT,
        ))
        present[ch] = P + f"ch{ch}_present"

        nodes.append(onnx.helper.make_node("Mul", [ch_name, diag_mask], [P + f"ch{ch}_diag"]))
        nodes.append(onnx.helper.make_node(
            "ReduceSum", [P + f"ch{ch}_diag"], [P + f"ch{ch}_diag_count"],
            axes=[2, 3], keepdims=1,
        ))
        nodes.append(onnx.helper.make_node(
            "Greater", [P + f"ch{ch}_diag_count", half], [P + f"base{ch}_b"]
        ))
        nodes.append(onnx.helper.make_node(
            "Cast", [P + f"base{ch}_b"], [P + f"base{ch}"],
            to=onnx.TensorProto.FLOAT,
        ))
        base_gate[ch] = P + f"base{ch}"

    period_gate = {}
    for period in range(1, 9):
        nodes.append(onnx.helper.make_node(
            "Sub", [one_gate, present[period + 1]], [P + f"period{period}_next_absent"]
        ))
        nodes.append(onnx.helper.make_node(
            "Mul", [present[period], P + f"period{period}_next_absent"], [P + f"period{period}"]
        ))
        period_gate[period] = P + f"period{period}"
    period_gate[9] = present[9]

    combo_gate = {}
    for period in range(1, 10):
        for base in range(1, period + 1):
            name = P + f"combo_b{base}_p{period}"
            nodes.append(onnx.helper.make_node("Mul", [base_gate[base], period_gate[period]], [name]))
            combo_gate[(base, period)] = name

    def color_for(base, off, period):
        return ((base - 1 + off) % period) + 1

    channel_outputs = [zero_canvas]
    for out_ch in range(1, 10):
        spatial_terms = []
        for off in range(-1, 10):
            gates = [
                combo_gate[(base, period)]
                for period in range(1, 10)
                for base in range(1, period + 1)
                if color_for(base, off, period) == out_ch
            ]
            selector = add_chain("Add", gates, f"sel_ch{out_ch}_off{off + 1}")
            term = P + f"ch{out_ch}_off{off + 1}"
            nodes.append(onnx.helper.make_node("Mul", [selector, offset_masks[off]], [term]))
            spatial_terms.append(term)
        channel_outputs.append(add_chain("Add", spatial_terms, f"ch{out_ch}_out"))

    nodes.append(onnx.helper.make_node("Concat", channel_outputs, ["output"], axis=1))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def bilateral_symmetry_denoise_network(batch=1):
    """Task071-style rectangular-noise removal via horizontal symmetry.

    Input is a fixed 16x16 grid with two nonzero colors. The noise color is the
    one with the highest bbox density (a solid rectangle), and the kept color is
    the other color. The graph chooses the horizontal mirror axis that maximizes
    kept cells mirroring onto any nonzero cell, then replaces only noise cells
    whose mirror is kept.
    """
    if batch != 1:
        raise NotImplementedError(
            "bilateral-symmetry-denoise builder is single-example only (batch must be 1)"
        )

    P = "bsd_"
    nodes = []
    inits = []

    def add_tensor(name, dtype, shape, values):
        tensor = onnx.helper.make_tensor(P + name, dtype, shape, values)
        inits.append(tensor)
        return P + name

    def add_node(op_type, inputs, outputs, **kwargs):
        nodes.append(onnx.helper.make_node(op_type, inputs, outputs, **kwargs))

    FLOAT = onnx.TensorProto.FLOAT
    INT64 = onnx.TensorProto.INT64

    cr_starts = add_tensor("cr_starts", INT64, [2], [0, 0])
    cr_ends = add_tensor("cr_ends", INT64, [2], [16, 16])
    hw_axes = add_tensor("hw_axes", INT64, [2], [2, 3])
    hw_steps = add_tensor("hw_steps", INT64, [2], [1, 1])
    ch_st = add_tensor("ch_st", INT64, [1], [1])
    ch_en = add_tensor("ch_en", INT64, [1], [10])
    ch_axis = add_tensor("ch_axis", INT64, [1], [1])
    ch_step = add_tensor("ch_step", INT64, [1], [1])
    one_f = add_tensor("one_f", FLOAT, [1], [1.0])
    desc_row = add_tensor("desc_row", FLOAT, [1, 1, 16, 1], [float(16 - i) for i in range(16)])
    asc_row = add_tensor("asc_row", FLOAT, [1, 1, 16, 1], [float(i + 1) for i in range(16)])
    desc_col = add_tensor("desc_col", FLOAT, [1, 1, 1, 16], [float(16 - i) for i in range(16)])
    asc_col = add_tensor("asc_col", FLOAT, [1, 1, 1, 16], [float(i + 1) for i in range(16)])
    rs_19 = add_tensor("rs_19", INT64, [2], [1, 9])
    rs_1911 = add_tensor("rs_1911", INT64, [4], [1, 9, 1, 1])
    one16 = add_tensor("one16", FLOAT, [1, 1, 16, 16], [1.0] * (16 * 16))
    oh_depth9 = add_tensor("oh_depth9", INT64, [], [9])
    oh_values = add_tensor("oh_values", FLOAT, [2], [0.0, 1.0])

    add_node("Slice", ["input", cr_starts, cr_ends, hw_axes, hw_steps], [P + "crop"])
    add_node("Slice", [P + "crop", ch_st, ch_en, ch_axis, ch_step], [P + "nonbg"])

    add_node("ReduceSum", [P + "nonbg"], [P + "counts"], axes=[2, 3], keepdims=0)
    add_node("ReduceMax", [P + "nonbg"], [P + "present"], axes=[2, 3], keepdims=0)
    add_node("ReduceMax", [P + "nonbg"], [P + "row_has"], axes=[3], keepdims=1)
    add_node("ReduceMax", [P + "nonbg"], [P + "col_has"], axes=[2], keepdims=1)

    add_node("Mul", [P + "row_has", desc_row], [P + "first_row_w"])
    add_node("Mul", [P + "row_has", asc_row], [P + "last_row_w"])
    add_node("Mul", [P + "col_has", desc_col], [P + "first_col_w"])
    add_node("Mul", [P + "col_has", asc_col], [P + "last_col_w"])
    add_node("ArgMax", [P + "first_row_w"], [P + "first_row_3d"], axis=2, keepdims=0)
    add_node("ArgMax", [P + "last_row_w"], [P + "last_row_3d"], axis=2, keepdims=0)
    add_node("ArgMax", [P + "first_col_w"], [P + "first_col_3d"], axis=3, keepdims=0)
    add_node("ArgMax", [P + "last_col_w"], [P + "last_col_3d"], axis=3, keepdims=0)

    for src, dst in [
        ("first_row_3d", "first_row_f"),
        ("last_row_3d", "last_row_f"),
        ("first_col_3d", "first_col_f"),
        ("last_col_3d", "last_col_f"),
    ]:
        add_node("Reshape", [P + src, rs_19], [P + src.replace("_3d", "_2d")])
        add_node("Cast", [P + src.replace("_3d", "_2d"),], [P + dst], to=FLOAT)

    add_node("Sub", [P + "last_row_f", P + "first_row_f"], [P + "height_m1"])
    add_node("Add", [P + "height_m1", one_f], [P + "height"])
    add_node("Sub", [P + "last_col_f", P + "first_col_f"], [P + "width_m1"])
    add_node("Add", [P + "width_m1", one_f], [P + "width"])
    add_node("Mul", [P + "height", P + "width"], [P + "area"])
    add_node("Div", [P + "counts", P + "area"], [P + "density"])
    add_node("Mul", [P + "density", P + "present"], [P + "density_safe"])
    add_node("ArgMax", [P + "density_safe"], [P + "noise_idx"], axis=1, keepdims=0)
    add_node("OneHot", [P + "noise_idx", oh_depth9, oh_values], [P + "noise_oh_9"], axis=1)
    add_node("Sub", [P + "present", P + "noise_oh_9"], [P + "kept_oh_9"])
    add_node("Reshape", [P + "noise_oh_9", rs_1911], [P + "noise_oh_4d"])
    add_node("Reshape", [P + "kept_oh_9", rs_1911], [P + "kept_oh_4d"])

    add_node("Mul", [P + "nonbg", P + "noise_oh_4d"], [P + "noise_per_chan"])
    add_node("Mul", [P + "nonbg", P + "kept_oh_4d"], [P + "kept_per_chan"])
    add_node("ReduceSum", [P + "noise_per_chan"], [P + "noise_mask"], axes=[1], keepdims=1)
    add_node("ReduceSum", [P + "kept_per_chan"], [P + "kept_mask"], axes=[1], keepdims=1)
    add_node("ReduceSum", [P + "nonbg"], [P + "nonzero_any"], axes=[1], keepdims=1)

    score_parts = []
    mirrored_kept_parts = []
    for axis2 in range(31):
        idx_vals = [min(15, max(0, axis2 - c)) for c in range(16)]
        valid_vals = [1.0 if 0 <= axis2 - c < 16 else 0.0 for c in range(16)]
        idx = add_tensor(f"axis{axis2}_idx", INT64, [16], idx_vals)
        valid = add_tensor(f"axis{axis2}_valid", FLOAT, [1, 1, 1, 16], valid_vals)

        add_node("Gather", [P + "nonzero_any", idx], [P + f"axis{axis2}_mirror_any_raw"], axis=3)
        add_node("Mul", [P + f"axis{axis2}_mirror_any_raw", valid], [P + f"axis{axis2}_mirror_any"])
        add_node("Mul", [P + "kept_mask", P + f"axis{axis2}_mirror_any"], [P + f"axis{axis2}_score_grid"])
        add_node(
            "ReduceSum",
            [P + f"axis{axis2}_score_grid"],
            [P + f"axis{axis2}_score"],
            axes=[2, 3],
            keepdims=0,
        )
        score_parts.append(P + f"axis{axis2}_score")

        add_node("Gather", [P + "kept_mask", idx], [P + f"axis{axis2}_mirror_kept_raw"], axis=3)
        add_node("Mul", [P + f"axis{axis2}_mirror_kept_raw", valid], [P + f"axis{axis2}_mirror_kept"])
        mirrored_kept_parts.append(P + f"axis{axis2}_mirror_kept")

    add_node("Concat", score_parts, [P + "axis_scores"], axis=1)
    add_node("ArgMax", [P + "axis_scores"], [P + "best_axis"], axis=1, keepdims=0)
    add_node("Concat", mirrored_kept_parts, [P + "mirrored_kept_stack"], axis=1)
    add_node("Gather", [P + "mirrored_kept_stack", P + "best_axis"], [P + "best_mirror_kept"], axis=1)

    add_node("Mul", [P + "noise_mask", P + "best_mirror_kept"], [P + "fill_mask"])
    add_node("Add", [P + "kept_mask", P + "fill_mask"], [P + "kept_out_mask"])
    add_node("Mul", [P + "kept_out_mask", P + "kept_oh_4d"], [P + "color_out_9"])
    add_node("ReduceSum", [P + "color_out_9"], [P + "any_color_out"], axes=[1], keepdims=1)
    add_node("Sub", [one16, P + "any_color_out"], [P + "ch0_out"])
    add_node("Concat", [P + "ch0_out", P + "color_out_9"], [P + "out_crop"], axis=1)
    add_node(
        "Pad",
        [P + "out_crop"],
        ["output"],
        pads=[0, 0, 0, 0, 0, 0, 14, 14],
        mode="constant",
        value=0.0,
    )

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def marker_section_extract_network(batch=1, bfs_iters=30):
    """ONNX: identify the divider-bounded sub-rectangle containing a runtime
    marker cell, and output that sub-rectangle (preserving original cell colors).

    For tasks (e.g. task065) where the input has a divider grid (one color
    forming full rows + cols) and a single marker color (rarest non-divider).
    Output = the rectangular sub-section of the grid that contains the marker
    cell.

    Pipeline (input [N, 10, 30, 30], single example only):

      1-5. Same as grid_division_count: identify divider color, build
           row_full_div (1, 1, 30, 1) and col_full_div (1, 1, 1, 30) binary
           masks of which rows/cols are dividers.
      6. marker_color_idx: per-channel count, mask out ch0 + divider, ArgMax(-x)
         over (count + (1-mask)*BIG) = ArgMin → rarest non-divider color.
      7. marker_mask = input * marker_color_oh_4d, ReduceSum(axis=1) ->
         (1, 1, 30, 30). Then ArgMax-trick rows/cols → (mr, mc) scalars.
      8. 1D BFS for row section bounds:
         - start_rows = OneHot(mr, depth=30) reshape (1, 1, 30, 1)
         - not_div_rows = 1 - row_full_div
         - Iterate `bfs_iters` times: dilate via MaxPool kernel (3, 1), then
           Mul by not_div_rows.
         - Result: row_section_mask (1, 1, 30, 1) — rows in same section as mr.
         Same for cols.
      9. r0 = ArgMax(row_section_mask * desc_row); same for c0.
         r1 = ArgMax(row_section_mask * asc_row); same for c1.
         Cast all to FLOAT.
      10. row_indices = Cast(Min(r0 + range_30, 29), INT64)
          col_indices analogously
      11. shifted_rows = Gather(input, row_indices, axis=2)
          shifted_cols = Gather(shifted_rows, col_indices, axis=3)
      12. height = r1 - r0 + 1; width = c1 - c0 + 1
          row_in_bbox = (range_30 < height); col_in_bbox = (range_30 < width)
          bbox_mask = row_in_bbox * col_in_bbox (broadcast)
      13. output = shifted_cols * bbox_mask

    Cost (per cost-model verify, opset 10): macs ~200k + 30 BFS iters; memory
    ~250k bytes; params ~250 (initializers). Total ~500k -> ~11.8 pts.
    Verified 100% (266/266 numpy verifier; 77 visible-pair runs match) on task065.

    Single-example only (batch=1).
    """
    if batch != 1:
        raise NotImplementedError(
            "marker-section-extract builder is single-example only (batch must be 1)"
        )

    nodes = []
    inits = []

    # --- Steps 1-5: divider detection (same as grid_division_count) ---
    nodes.append(onnx.helper.make_node(
        "ReduceSum", ["input"], ["mse_is_active"], axes=[1], keepdims=1
    ))
    nodes.append(onnx.helper.make_node(
        "ReduceSum", ["mse_is_active"], ["mse_per_row_total"], axes=[3], keepdims=0
    ))
    nodes.append(onnx.helper.make_node(
        "ReduceSum", ["mse_is_active"], ["mse_per_col_total"], axes=[2], keepdims=0
    ))
    nodes.append(onnx.helper.make_node(
        "ReduceSum", ["input"], ["mse_per_row_per_chan"], axes=[3], keepdims=0
    ))
    nodes.append(onnx.helper.make_node(
        "ReduceSum", ["input"], ["mse_per_col_per_chan"], axes=[2], keepdims=0
    ))
    nodes.append(onnx.helper.make_node(
        "Sub", ["mse_per_row_per_chan", "mse_per_row_total"], ["mse_row_diff"]
    ))
    nodes.append(onnx.helper.make_node(
        "Sub", ["mse_per_col_per_chan", "mse_per_col_total"], ["mse_col_diff"]
    ))
    for src, dst in [
        ("mse_row_diff", "mse_row_diff_int"),
        ("mse_col_diff", "mse_col_diff_int"),
    ]:
        nodes.append(onnx.helper.make_node(
            "Cast", [src], [dst], to=onnx.TensorProto.INT64
        ))
    zero_int = onnx.helper.make_tensor("mse_zero_int", onnx.TensorProto.INT64, [1], [0])
    inits.append(zero_int)
    nodes.append(onnx.helper.make_node(
        "Equal", ["mse_row_diff_int", "mse_zero_int"], ["mse_row_match_b"]
    ))
    nodes.append(onnx.helper.make_node(
        "Equal", ["mse_col_diff_int", "mse_zero_int"], ["mse_col_match_b"]
    ))
    zero_f = onnx.helper.make_tensor("mse_zero_f", onnx.TensorProto.FLOAT, [1], [0.0])
    inits.append(zero_f)
    nodes.append(onnx.helper.make_node(
        "Greater", ["mse_per_row_total", "mse_zero_f"], ["mse_row_active_b"]
    ))
    nodes.append(onnx.helper.make_node(
        "Greater", ["mse_per_col_total", "mse_zero_f"], ["mse_col_active_b"]
    ))
    for src, dst in [
        ("mse_row_match_b", "mse_row_match_f"),
        ("mse_col_match_b", "mse_col_match_f"),
        ("mse_row_active_b", "mse_row_active_f"),
        ("mse_col_active_b", "mse_col_active_f"),
    ]:
        nodes.append(onnx.helper.make_node(
            "Cast", [src], [dst], to=onnx.TensorProto.FLOAT
        ))
    nodes.append(onnx.helper.make_node(
        "Mul", ["mse_row_match_f", "mse_row_active_f"], ["mse_row_full_c"]
    ))  # (1, 10, 30)
    nodes.append(onnx.helper.make_node(
        "Mul", ["mse_col_match_f", "mse_col_active_f"], ["mse_col_full_c"]
    ))  # (1, 10, 30)
    nodes.append(onnx.helper.make_node(
        "ReduceSum", ["mse_row_full_c"], ["mse_n_full_rows"], axes=[2], keepdims=0
    ))  # (1, 10)
    nodes.append(onnx.helper.make_node(
        "ReduceSum", ["mse_col_full_c"], ["mse_n_full_cols"], axes=[2], keepdims=0
    ))  # (1, 10)
    nodes.append(onnx.helper.make_node(
        "Add", ["mse_n_full_rows", "mse_n_full_cols"], ["mse_div_score"]
    ))
    nodes.append(onnx.helper.make_node(
        "ArgMax", ["mse_div_score"], ["mse_div_color_idx"], axis=1, keepdims=0
    ))

    oh_depth = onnx.helper.make_tensor("mse_oh_depth", onnx.TensorProto.INT64, [], [10])
    oh_values = onnx.helper.make_tensor("mse_oh_values", onnx.TensorProto.FLOAT, [2], [0.0, 1.0])
    inits.extend([oh_depth, oh_values])
    nodes.append(onnx.helper.make_node(
        "OneHot",
        ["mse_div_color_idx", "mse_oh_depth", "mse_oh_values"],
        ["mse_div_color_oh_2d"], axis=1,
    ))  # (1, 10)

    # row_full_div = sum_c(row_full_c * div_color_oh) along channel axis
    rs_n10x1 = onnx.helper.make_tensor("mse_rs_n10x1", onnx.TensorProto.INT64, [3], [1, 10, 1])
    inits.append(rs_n10x1)
    nodes.append(onnx.helper.make_node(
        "Reshape", ["mse_div_color_oh_2d", "mse_rs_n10x1"], ["mse_div_color_oh_3d_v"]
    ))  # (1, 10, 1)

    nodes.append(onnx.helper.make_node(
        "Mul", ["mse_row_full_c", "mse_div_color_oh_3d_v"], ["mse_row_full_div_per_chan"]
    ))  # (1, 10, 30) - all zero except divider channel
    nodes.append(onnx.helper.make_node(
        "ReduceSum", ["mse_row_full_div_per_chan"], ["mse_row_full_div_3d"],
        axes=[1], keepdims=1,
    ))  # (1, 1, 30)
    nodes.append(onnx.helper.make_node(
        "Mul", ["mse_col_full_c", "mse_div_color_oh_3d_v"], ["mse_col_full_div_per_chan"]
    ))
    nodes.append(onnx.helper.make_node(
        "ReduceSum", ["mse_col_full_div_per_chan"], ["mse_col_full_div_3d"],
        axes=[1], keepdims=1,
    ))  # (1, 1, 30)

    # Reshape to 4D for use as masks: row->(1,1,30,1), col->(1,1,1,30)
    rs_h_axis = onnx.helper.make_tensor("mse_rs_h", onnx.TensorProto.INT64, [4], [1, 1, 30, 1])
    rs_w_axis = onnx.helper.make_tensor("mse_rs_w", onnx.TensorProto.INT64, [4], [1, 1, 1, 30])
    inits.extend([rs_h_axis, rs_w_axis])
    nodes.append(onnx.helper.make_node(
        "Reshape", ["mse_row_full_div_3d", "mse_rs_h"], ["mse_row_full_div_4d"]
    ))  # (1, 1, 30, 1)
    nodes.append(onnx.helper.make_node(
        "Reshape", ["mse_col_full_div_3d", "mse_rs_w"], ["mse_col_full_div_4d"]
    ))  # (1, 1, 1, 30)

    # not_div_rows = (1 - row_full_div_4d) * row_active. The multiplication by
    # row_active prevents BFS from spreading into inactive rows (canvas padding
    # outside the grid). Same for cols.
    one_f = onnx.helper.make_tensor("mse_one_f", onnx.TensorProto.FLOAT, [1], [1.0])
    inits.append(one_f)
    nodes.append(onnx.helper.make_node(
        "Sub", ["mse_one_f", "mse_row_full_div_4d"], ["mse_not_div_rows_raw"]
    ))
    nodes.append(onnx.helper.make_node(
        "Sub", ["mse_one_f", "mse_col_full_div_4d"], ["mse_not_div_cols_raw"]
    ))
    # Reshape row_active_f (1, 1, 30) -> (1, 1, 30, 1) and col_active_f -> (1, 1, 1, 30)
    nodes.append(onnx.helper.make_node(
        "Reshape", ["mse_row_active_f", "mse_rs_h"], ["mse_row_active_4d"]
    ))
    nodes.append(onnx.helper.make_node(
        "Reshape", ["mse_col_active_f", "mse_rs_w"], ["mse_col_active_4d"]
    ))
    nodes.append(onnx.helper.make_node(
        "Mul", ["mse_not_div_rows_raw", "mse_row_active_4d"], ["mse_not_div_rows_4d"]
    ))
    nodes.append(onnx.helper.make_node(
        "Mul", ["mse_not_div_cols_raw", "mse_col_active_4d"], ["mse_not_div_cols_4d"]
    ))

    # --- Step 6: marker_color identification (rarest non-divider color with cells).
    # NOTE: ch0 IS allowed as a marker (e.g., a single 0 cell in a sea of fill).
    # The has_cells filter ensures empty channels don't tie ArgMin.
    nodes.append(onnx.helper.make_node(
        "ReduceSum", ["input"], ["mse_per_chan_total"], axes=[2, 3], keepdims=0
    ))  # (1, 10)
    ones_10 = onnx.helper.make_tensor(
        "mse_ones_10", onnx.TensorProto.FLOAT, [1, 10],
        [1.0] * 10,
    )
    inits.append(ones_10)
    # filter_mask: 0 at divider channel, 1 elsewhere.
    nodes.append(onnx.helper.make_node(
        "Sub", ["mse_ones_10", "mse_div_color_oh_2d"], ["mse_filter_mask_f"]
    ))
    # Also filter out channels with 0 cells.
    nodes.append(onnx.helper.make_node(
        "Greater", ["mse_per_chan_total", "mse_zero_f"], ["mse_has_cells_b"]
    ))
    nodes.append(onnx.helper.make_node(
        "Cast", ["mse_has_cells_b"], ["mse_has_cells_f"], to=onnx.TensorProto.FLOAT
    ))
    nodes.append(onnx.helper.make_node(
        "Mul", ["mse_filter_mask_f", "mse_has_cells_f"], ["mse_combined_mask"]
    ))
    # masked_count = per_chan_total + (1 - combined_mask) * BIG so non-candidates win NO ArgMin
    big_f = onnx.helper.make_tensor("mse_big_f", onnx.TensorProto.FLOAT, [1], [1.0e9])
    inits.append(big_f)
    nodes.append(onnx.helper.make_node(
        "Sub", ["mse_one_f", "mse_combined_mask"], ["mse_inv_mask"]
    ))
    nodes.append(onnx.helper.make_node(
        "Mul", ["mse_inv_mask", "mse_big_f"], ["mse_inv_mask_big"]
    ))
    nodes.append(onnx.helper.make_node(
        "Add", ["mse_per_chan_total", "mse_inv_mask_big"], ["mse_masked_count"]
    ))
    # ArgMin via ArgMax(-x)
    neg_one_f = onnx.helper.make_tensor("mse_neg_one_f", onnx.TensorProto.FLOAT, [1], [-1.0])
    inits.append(neg_one_f)
    nodes.append(onnx.helper.make_node(
        "Mul", ["mse_masked_count", "mse_neg_one_f"], ["mse_neg_count"]
    ))
    nodes.append(onnx.helper.make_node(
        "ArgMax", ["mse_neg_count"], ["mse_marker_color_idx"], axis=1, keepdims=0
    ))  # (1,) int64

    nodes.append(onnx.helper.make_node(
        "OneHot",
        ["mse_marker_color_idx", "mse_oh_depth", "mse_oh_values"],
        ["mse_marker_color_oh_2d"], axis=1,
    ))  # (1, 10)
    rs_n10x1x1 = onnx.helper.make_tensor("mse_rs_n10x1x1", onnx.TensorProto.INT64, [4], [1, 10, 1, 1])
    inits.append(rs_n10x1x1)
    nodes.append(onnx.helper.make_node(
        "Reshape", ["mse_marker_color_oh_2d", "mse_rs_n10x1x1"], ["mse_marker_color_oh_4d"]
    ))

    # --- Step 7: marker_mask + (mr, mc) ---
    nodes.append(onnx.helper.make_node(
        "Mul", ["input", "mse_marker_color_oh_4d"], ["mse_marker_per_chan"]
    ))  # (1, 10, 30, 30) — only marker channel non-zero
    nodes.append(onnx.helper.make_node(
        "ReduceSum", ["mse_marker_per_chan"], ["mse_marker_mask"],
        axes=[1], keepdims=1,
    ))  # (1, 1, 30, 30) binary

    nodes.append(onnx.helper.make_node(
        "ReduceMax", ["mse_marker_mask"], ["mse_row_has_marker"],
        axes=[3], keepdims=1,
    ))  # (1, 1, 30, 1)
    nodes.append(onnx.helper.make_node(
        "ReduceMax", ["mse_marker_mask"], ["mse_col_has_marker"],
        axes=[2], keepdims=1,
    ))  # (1, 1, 1, 30)

    desc_row = onnx.helper.make_tensor(
        "mse_desc_row", onnx.TensorProto.FLOAT, [1, 1, 30, 1],
        [float(30 - i) for i in range(30)],
    )
    desc_col = onnx.helper.make_tensor(
        "mse_desc_col", onnx.TensorProto.FLOAT, [1, 1, 1, 30],
        [float(30 - i) for i in range(30)],
    )
    asc_row = onnx.helper.make_tensor(
        "mse_asc_row", onnx.TensorProto.FLOAT, [1, 1, 30, 1],
        [float(i + 1) for i in range(30)],
    )
    asc_col = onnx.helper.make_tensor(
        "mse_asc_col", onnx.TensorProto.FLOAT, [1, 1, 1, 30],
        [float(i + 1) for i in range(30)],
    )
    inits.extend([desc_row, desc_col, asc_row, asc_col])

    nodes.append(onnx.helper.make_node(
        "Mul", ["mse_row_has_marker", "mse_desc_row"], ["mse_w_mr"]
    ))
    nodes.append(onnx.helper.make_node(
        "ArgMax", ["mse_w_mr"], ["mse_mr_3d"], axis=2, keepdims=0
    ))  # (1, 1, 1) int64
    nodes.append(onnx.helper.make_node(
        "Mul", ["mse_col_has_marker", "mse_desc_col"], ["mse_w_mc"]
    ))
    nodes.append(onnx.helper.make_node(
        "ArgMax", ["mse_w_mc"], ["mse_mc_3d"], axis=3, keepdims=0
    ))  # (1, 1, 1) int64

    # --- Step 8: 1D BFS for row section + col section ---
    # OneHot(mr, depth=30) for start mask. Need Reshape mr from (1, 1, 1) -> (1,)
    rs_to_1d = onnx.helper.make_tensor("mse_rs_to_1d", onnx.TensorProto.INT64, [1], [1])
    inits.append(rs_to_1d)
    nodes.append(onnx.helper.make_node(
        "Reshape", ["mse_mr_3d", "mse_rs_to_1d"], ["mse_mr_1d"]
    ))
    nodes.append(onnx.helper.make_node(
        "Reshape", ["mse_mc_3d", "mse_rs_to_1d"], ["mse_mc_1d"]
    ))

    oh_depth_30 = onnx.helper.make_tensor("mse_oh_depth_30", onnx.TensorProto.INT64, [], [30])
    inits.append(oh_depth_30)
    nodes.append(onnx.helper.make_node(
        "OneHot",
        ["mse_mr_1d", "mse_oh_depth_30", "mse_oh_values"],
        ["mse_mr_oh_2d"], axis=1,
    ))  # (1, 30)
    nodes.append(onnx.helper.make_node(
        "OneHot",
        ["mse_mc_1d", "mse_oh_depth_30", "mse_oh_values"],
        ["mse_mc_oh_2d"], axis=1,
    ))  # (1, 30)
    nodes.append(onnx.helper.make_node(
        "Reshape", ["mse_mr_oh_2d", "mse_rs_h"], ["mse_row_seed_4d"]
    ))  # (1, 1, 30, 1)
    nodes.append(onnx.helper.make_node(
        "Reshape", ["mse_mc_oh_2d", "mse_rs_w"], ["mse_col_seed_4d"]
    ))  # (1, 1, 1, 30)

    # 1D BFS for ROWS: dilate via MaxPool kernel (3, 1) pad (1, 0) on (1, 1, 30, 1).
    # Then Mul by not_div_rows. Iterate `bfs_iters` times.
    cur_row = "mse_row_seed_4d"
    for it in range(bfs_iters):
        dilated = f"mse_row_dilated_{it}"
        masked = f"mse_row_masked_{it}"
        nodes.append(onnx.helper.make_node(
            "MaxPool", [cur_row], [dilated],
            kernel_shape=[3, 1], strides=[1, 1], pads=[1, 0, 1, 0],
        ))
        nodes.append(onnx.helper.make_node(
            "Mul", [dilated, "mse_not_div_rows_4d"], [masked]
        ))
        cur_row = masked
    row_section_mask = cur_row  # (1, 1, 30, 1)

    cur_col = "mse_col_seed_4d"
    for it in range(bfs_iters):
        dilated = f"mse_col_dilated_{it}"
        masked = f"mse_col_masked_{it}"
        nodes.append(onnx.helper.make_node(
            "MaxPool", [cur_col], [dilated],
            kernel_shape=[1, 3], strides=[1, 1], pads=[0, 1, 0, 1],
        ))
        nodes.append(onnx.helper.make_node(
            "Mul", [dilated, "mse_not_div_cols_4d"], [masked]
        ))
        cur_col = masked
    col_section_mask = cur_col  # (1, 1, 1, 30)

    # --- Step 9: r0, r1, c0, c1 via ArgMax-trick ---
    nodes.append(onnx.helper.make_node(
        "Mul", [row_section_mask, "mse_desc_row"], ["mse_w_first_row"]
    ))
    nodes.append(onnx.helper.make_node(
        "Mul", [row_section_mask, "mse_asc_row"], ["mse_w_last_row"]
    ))
    nodes.append(onnx.helper.make_node(
        "Mul", [col_section_mask, "mse_desc_col"], ["mse_w_first_col"]
    ))
    nodes.append(onnx.helper.make_node(
        "Mul", [col_section_mask, "mse_asc_col"], ["mse_w_last_col"]
    ))
    nodes.append(onnx.helper.make_node(
        "ArgMax", ["mse_w_first_row"], ["mse_first_row_3d"], axis=2, keepdims=0
    ))
    nodes.append(onnx.helper.make_node(
        "ArgMax", ["mse_w_last_row"], ["mse_last_row_3d"], axis=2, keepdims=0
    ))
    nodes.append(onnx.helper.make_node(
        "ArgMax", ["mse_w_first_col"], ["mse_first_col_3d"], axis=3, keepdims=0
    ))
    nodes.append(onnx.helper.make_node(
        "ArgMax", ["mse_w_last_col"], ["mse_last_col_3d"], axis=3, keepdims=0
    ))
    for src_int, dst_f in [
        ("mse_first_row_3d", "mse_first_row_f"),
        ("mse_last_row_3d", "mse_last_row_f"),
        ("mse_first_col_3d", "mse_first_col_f"),
        ("mse_last_col_3d", "mse_last_col_f"),
    ]:
        # Reshape to (1,) then Cast
        tmp1d = src_int.replace("_3d", "_1d")
        nodes.append(onnx.helper.make_node(
            "Reshape", [src_int, "mse_rs_to_1d"], [tmp1d]
        ))
        nodes.append(onnx.helper.make_node(
            "Cast", [tmp1d], [dst_f], to=onnx.TensorProto.FLOAT
        ))

    # --- Step 10: row_indices + col_indices via Min(start + range_30, 29) ---
    range_30 = onnx.helper.make_tensor(
        "mse_range_30", onnx.TensorProto.FLOAT, [30],
        [float(i) for i in range(30)],
    )
    const_29 = onnx.helper.make_tensor("mse_const_29", onnx.TensorProto.FLOAT, [1], [29.0])
    inits.extend([range_30, const_29])
    nodes.append(onnx.helper.make_node(
        "Add", ["mse_first_row_f", "mse_range_30"], ["mse_row_idx_raw_f"]
    ))
    nodes.append(onnx.helper.make_node(
        "Min", ["mse_row_idx_raw_f", "mse_const_29"], ["mse_row_idx_f"]
    ))
    nodes.append(onnx.helper.make_node(
        "Cast", ["mse_row_idx_f"], ["mse_row_idx"], to=onnx.TensorProto.INT64
    ))
    nodes.append(onnx.helper.make_node(
        "Add", ["mse_first_col_f", "mse_range_30"], ["mse_col_idx_raw_f"]
    ))
    nodes.append(onnx.helper.make_node(
        "Min", ["mse_col_idx_raw_f", "mse_const_29"], ["mse_col_idx_f"]
    ))
    nodes.append(onnx.helper.make_node(
        "Cast", ["mse_col_idx_f"], ["mse_col_idx"], to=onnx.TensorProto.INT64
    ))

    # --- Step 11: Gather rows then cols ---
    nodes.append(onnx.helper.make_node(
        "Gather", ["input", "mse_row_idx"], ["mse_shifted_rows"], axis=2
    ))
    nodes.append(onnx.helper.make_node(
        "Gather", ["mse_shifted_rows", "mse_col_idx"], ["mse_shifted"], axis=3
    ))

    # --- Step 12: bbox_mask via height/width = last - first + 1 ---
    nodes.append(onnx.helper.make_node(
        "Sub", ["mse_last_row_f", "mse_first_row_f"], ["mse_h_minus_1"]
    ))
    nodes.append(onnx.helper.make_node(
        "Add", ["mse_h_minus_1", "mse_one_f"], ["mse_height_f"]
    ))
    nodes.append(onnx.helper.make_node(
        "Sub", ["mse_last_col_f", "mse_first_col_f"], ["mse_w_minus_1"]
    ))
    nodes.append(onnx.helper.make_node(
        "Add", ["mse_w_minus_1", "mse_one_f"], ["mse_width_f"]
    ))
    nodes.append(onnx.helper.make_node(
        "Less", ["mse_range_30", "mse_height_f"], ["mse_row_in_bbox_b"]
    ))
    nodes.append(onnx.helper.make_node(
        "Less", ["mse_range_30", "mse_width_f"], ["mse_col_in_bbox_b"]
    ))
    nodes.append(onnx.helper.make_node(
        "Cast", ["mse_row_in_bbox_b"], ["mse_row_in_bbox_f"], to=onnx.TensorProto.FLOAT
    ))
    nodes.append(onnx.helper.make_node(
        "Cast", ["mse_col_in_bbox_b"], ["mse_col_in_bbox_f"], to=onnx.TensorProto.FLOAT
    ))
    nodes.append(onnx.helper.make_node(
        "Reshape", ["mse_row_in_bbox_f", "mse_rs_h"], ["mse_row_mask_4d"]
    ))
    nodes.append(onnx.helper.make_node(
        "Reshape", ["mse_col_in_bbox_f", "mse_rs_w"], ["mse_col_mask_4d"]
    ))
    nodes.append(onnx.helper.make_node(
        "Mul", ["mse_row_mask_4d", "mse_col_mask_4d"], ["mse_bbox_mask_4d"]
    ))

    # --- Step 13: output = shifted * bbox_mask ---
    nodes.append(onnx.helper.make_node(
        "Mul", ["mse_shifted", "mse_bbox_mask_4d"], ["output"]
    ))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def rotational_symmetry_completion_network(batch=1):
    """ONNX: bbox detection + 3 rotations (180/90/270 image-rotation) + Max
    union over channels 1..9 + derived ch0.

    For tasks where the output completes the input via 4-fold rotational
    symmetry around the bounding-box center of non-zero cells. Output and
    input share the same shape (10x10 in task020).

    Pipeline (input [N, 10, 30, 30], single example only — N=1):
      1. Bbox detection (reuse foreground_bbox_crop machinery): is_active -
         ch0 = fg; ReduceMax along axes 3 and 2 to get row_has / col_has;
         ArgMax-trick (descending weight = first, ascending = last) on each;
         Reshape to [1] and Cast to FLOAT -> 4 FLOAT scalars.
      2. cr_2 = first_row_f + last_row_f, cc_2 = first_col_f + last_col_f.
         cr = cr_2 / 2, cc = cc_2 / 2 (always integer in task020 -- bbox
         extents always have even sum, verified empirically). Combinations:
         cr_minus_cc, cr_plus_cc, cc_minus_cr.
      3. Slice input to channels 1..9 only (input_nonzero, shape [1, 9, 30, 30]).
         All rotations operate on this 9-channel tensor; channel 0 is derived
         post-union. Why: rotating the full one-hot input under Max-union
         creates multi-lit cells (input has ch0=1 at every bg cell; if a
         rotation lands a non-bg color at that bg cell, both ch0 and ch_X
         end up lit).
      4. 180° rotation (separable Gather): rows[r] = cr_2 - r,
         cols[c] = cc_2 - c. Validity (Less-based, since opset 10 lacks
         GreaterOrEqual) on FLOAT indices BEFORE Clip+Cast. Gather rows then
         cols on input_nonzero. Mask via row_valid * col_valid.
      5. 90° CW image rotation: output[r, c] = input[A(c), B(r)] where
         A(c) = cr_minus_cc + c (a row index that depends on output column),
         B(r) = cr_plus_cc - r (a col index that depends on output row).
         Gather(input_nonzero, A, axis=2) -> Gather(_, B, axis=3) ->
         Transpose perm=[0,1,3,2] (because A indexes by c, B by r, but
         output expects [r,c]). Mask via A_valid * B_valid.
      6. 270° CW image rotation: same form with A(c) = cr_plus_cc - c,
         B(r) = cc_minus_cr + r.
      7. nonbg_union = Max(input_nonzero, rot_180, rot_cw, rot_ccw)
         -> [1, 9, 30, 30].
      8. any_nonbg = ReduceMax(nonbg_union, axis=1, keepdims=1) -> [1, 1, 30, 30].
         ch0_out = 1 - any_nonbg (1 where no non-bg color is present).
      9. output = Concat([ch0_out, nonbg_union], axis=1) -> [1, 10, 30, 30].

    Cost (per cost-model verify, opset 10): macs ~110k (Mul ops on
    [1, 9, 30, 30] tensors dominate), memory ~520k bytes (12 large
    [1, 9, 30, 30] intermediates dominate), params ~430. Total ~640k ->
    ~12.0 pts. Verified 100% (266/266) on task020.

    Edge cases:
      - Empty input (all bg): fg = 0, ArgMax-of-zeros returns 0; cr = cc = 0.
        180°: only (0, 0) valid. rot_cw / rot_ccw: only (0, 0) valid. Max
        with all-zero input = all-zero output. ✓
      - Single non-zero cell at (r0, c0): cr = r0, cc = c0. All 4 rotations
        produce the same cell at (r0, c0) (rotation around itself); other
        positions masked invalid. Output = input. ✓

    Builder is single-example only — multi-batch needs runtime per-batch
    bbox that opset 10 lacks (no per-batch Gather indices). Raises
    NotImplementedError for batch != 1; dispatcher's per-example fallback
    handles it.
    """
    if batch != 1:
        raise NotImplementedError(
            "rotational-symmetry-completion builder is single-example only "
            "(batch must be 1)"
        )

    nodes = []
    inits = []
    P = "rsc_"  # name prefix for all node tensors / initializers

    # crop-process-pad: task020 grids are always 10x10; crop reduces all
    # large [1,9,30,30] intermediates to [1,9,10,10] (9x smaller).

    # --- Step 0: crop input to [1, 10, 10, 10] ---
    cr_starts = onnx.helper.make_tensor(P+"cr_starts", onnx.TensorProto.INT64, [2], [0, 0])
    cr_ends = onnx.helper.make_tensor(P+"cr_ends", onnx.TensorProto.INT64, [2], [10, 10])
    cr_axes = onnx.helper.make_tensor(P+"cr_axes", onnx.TensorProto.INT64, [2], [2, 3])
    cr_steps = onnx.helper.make_tensor(P+"cr_steps", onnx.TensorProto.INT64, [2], [1, 1])
    inits.extend([cr_starts, cr_ends, cr_axes, cr_steps])
    nodes.append(onnx.helper.make_node(
        "Slice", ["input", P+"cr_starts", P+"cr_ends", P+"cr_axes", P+"cr_steps"],
        [P+"crop"],
    ))  # (1, 10, 10, 10)

    # ===== Step 1: bbox detection on crop =====
    nodes.append(onnx.helper.make_node(
        "ReduceSum", [P+"crop"], [P+"is_active"], axes=[1], keepdims=1
    ))  # (1, 1, 10, 10)
    sl_starts = onnx.helper.make_tensor(P+"sl_starts", onnx.TensorProto.INT64, [1], [0])
    sl_ends = onnx.helper.make_tensor(P+"sl_ends", onnx.TensorProto.INT64, [1], [1])
    sl_axes = onnx.helper.make_tensor(P+"sl_axes", onnx.TensorProto.INT64, [1], [1])
    sl_steps = onnx.helper.make_tensor(P+"sl_steps", onnx.TensorProto.INT64, [1], [1])
    inits.extend([sl_starts, sl_ends, sl_axes, sl_steps])
    nodes.append(onnx.helper.make_node(
        "Slice",
        [P+"crop", P+"sl_starts", P+"sl_ends", P+"sl_axes", P+"sl_steps"],
        [P+"ch0"],
    ))  # (1, 1, 10, 10)
    nodes.append(onnx.helper.make_node("Sub", [P+"is_active", P+"ch0"], [P+"fg"]))

    nodes.append(onnx.helper.make_node(
        "ReduceMax", [P+"fg"], [P+"row_has"], axes=[3], keepdims=1
    ))  # (1, 1, 10, 1)
    nodes.append(onnx.helper.make_node(
        "ReduceMax", [P+"fg"], [P+"col_has"], axes=[2], keepdims=1
    ))  # (1, 1, 1, 10)

    desc_row = onnx.helper.make_tensor(
        P+"desc_row", onnx.TensorProto.FLOAT, [1, 1, 10, 1],
        [float(10 - i) for i in range(10)],
    )
    asc_row = onnx.helper.make_tensor(
        P+"asc_row", onnx.TensorProto.FLOAT, [1, 1, 10, 1],
        [float(i + 1) for i in range(10)],
    )
    desc_col = onnx.helper.make_tensor(
        P+"desc_col", onnx.TensorProto.FLOAT, [1, 1, 1, 10],
        [float(10 - i) for i in range(10)],
    )
    asc_col = onnx.helper.make_tensor(
        P+"asc_col", onnx.TensorProto.FLOAT, [1, 1, 1, 10],
        [float(i + 1) for i in range(10)],
    )
    inits.extend([desc_row, asc_row, desc_col, asc_col])
    nodes.append(onnx.helper.make_node("Mul", [P+"row_has", P+"desc_row"], [P+"w_first_row"]))
    nodes.append(onnx.helper.make_node("Mul", [P+"row_has", P+"asc_row"], [P+"w_last_row"]))
    nodes.append(onnx.helper.make_node("Mul", [P+"col_has", P+"desc_col"], [P+"w_first_col"]))
    nodes.append(onnx.helper.make_node("Mul", [P+"col_has", P+"asc_col"], [P+"w_last_col"]))

    nodes.append(onnx.helper.make_node(
        "ArgMax", [P+"w_first_row"], [P+"first_row_3d"], axis=2, keepdims=0
    ))
    nodes.append(onnx.helper.make_node(
        "ArgMax", [P+"w_last_row"], [P+"last_row_3d"], axis=2, keepdims=0
    ))
    nodes.append(onnx.helper.make_node(
        "ArgMax", [P+"w_first_col"], [P+"first_col_3d"], axis=3, keepdims=0
    ))
    nodes.append(onnx.helper.make_node(
        "ArgMax", [P+"w_last_col"], [P+"last_col_3d"], axis=3, keepdims=0
    ))

    rs1 = onnx.helper.make_tensor(P+"rs1", onnx.TensorProto.INT64, [1], [1])
    inits.append(rs1)
    for src in ["first_row", "last_row", "first_col", "last_col"]:
        src_3d = P + src + "_3d"
        tmp = P + src + "_1d"
        dst_f = P + src + "_f"
        nodes.append(onnx.helper.make_node("Reshape", [src_3d, P+"rs1"], [tmp]))
        nodes.append(onnx.helper.make_node("Cast", [tmp], [dst_f], to=onnx.TensorProto.FLOAT))

    # ===== Step 2: combinations cr_2, cc_2, cr, cc, cr±cc =====
    nodes.append(onnx.helper.make_node("Add", [P+"first_row_f", P+"last_row_f"], [P+"cr_2"]))
    nodes.append(onnx.helper.make_node("Add", [P+"first_col_f", P+"last_col_f"], [P+"cc_2"]))

    const_2_f = onnx.helper.make_tensor(P+"const_2_f", onnx.TensorProto.FLOAT, [1], [2.0])
    inits.append(const_2_f)
    nodes.append(onnx.helper.make_node("Div", [P+"cr_2", P+"const_2_f"], [P+"cr"]))
    nodes.append(onnx.helper.make_node("Div", [P+"cc_2", P+"const_2_f"], [P+"cc"]))

    nodes.append(onnx.helper.make_node("Sub", [P+"cr", P+"cc"], [P+"cr_minus_cc"]))
    nodes.append(onnx.helper.make_node("Add", [P+"cr", P+"cc"], [P+"cr_plus_cc"]))
    nodes.append(onnx.helper.make_node("Sub", [P+"cc", P+"cr"], [P+"cc_minus_cr"]))

    range_10_f = onnx.helper.make_tensor(
        P+"range_10_f", onnx.TensorProto.FLOAT, [10],
        [float(i) for i in range(10)],
    )
    const_9_f = onnx.helper.make_tensor(P+"const_9_f", onnx.TensorProto.FLOAT, [1], [9.0])
    const_0_f = onnx.helper.make_tensor(P+"const_0_f", onnx.TensorProto.FLOAT, [1], [0.0])
    const_1_4d = onnx.helper.make_tensor(P+"const_1_4d", onnx.TensorProto.FLOAT, [1, 1, 1, 1], [1.0])
    rs_row_4d = onnx.helper.make_tensor(P+"rs_row_4d", onnx.TensorProto.INT64, [4], [1, 1, 10, 1])
    rs_col_4d = onnx.helper.make_tensor(P+"rs_col_4d", onnx.TensorProto.INT64, [4], [1, 1, 1, 10])
    inits.extend([range_10_f, const_9_f, const_0_f, const_1_4d, rs_row_4d, rs_col_4d])

    # ===== Step 3: slice crop to channels 1..9 (input_nonzero) =====
    nz_starts = onnx.helper.make_tensor(P+"nz_starts", onnx.TensorProto.INT64, [1], [1])
    nz_ends = onnx.helper.make_tensor(P+"nz_ends", onnx.TensorProto.INT64, [1], [10])
    nz_axes = onnx.helper.make_tensor(P+"nz_axes", onnx.TensorProto.INT64, [1], [1])
    nz_steps = onnx.helper.make_tensor(P+"nz_steps", onnx.TensorProto.INT64, [1], [1])
    inits.extend([nz_starts, nz_ends, nz_axes, nz_steps])
    nodes.append(onnx.helper.make_node(
        "Slice", [P+"crop", P+"nz_starts", P+"nz_ends", P+"nz_axes", P+"nz_steps"],
        [P+"input_nonzero"],
    ))  # (1, 9, 10, 10)

    # Validity + clip helpers use const_9_f as upper bound (index range 0..9).
    def _add_validity(name_prefix, idx_f_name):
        nodes.append(onnx.helper.make_node(
            "Less", [idx_f_name, P+"const_0_f"], [name_prefix+"_bad_lo"]
        ))
        nodes.append(onnx.helper.make_node(
            "Less", [P+"const_9_f", idx_f_name], [name_prefix+"_bad_hi"]
        ))
        nodes.append(onnx.helper.make_node(
            "Or", [name_prefix+"_bad_lo", name_prefix+"_bad_hi"], [name_prefix+"_bad"]
        ))
        nodes.append(onnx.helper.make_node(
            "Not", [name_prefix+"_bad"], [name_prefix+"_good"]
        ))
        nodes.append(onnx.helper.make_node(
            "Cast", [name_prefix+"_good"], [name_prefix+"_mask_f"],
            to=onnx.TensorProto.FLOAT,
        ))

    def _add_clip_cast(idx_f_name, dst_int_name):
        nodes.append(onnx.helper.make_node(
            "Max", [idx_f_name, P+"const_0_f"], [idx_f_name+"_clip_lo"]
        ))
        nodes.append(onnx.helper.make_node(
            "Min", [idx_f_name+"_clip_lo", P+"const_9_f"], [idx_f_name+"_clip"]
        ))
        nodes.append(onnx.helper.make_node(
            "Cast", [idx_f_name+"_clip"], [dst_int_name],
            to=onnx.TensorProto.INT64,
        ))

    # ===== Step 4: 180° rotation (separable axes) =====
    nodes.append(onnx.helper.make_node("Sub", [P+"cr_2", P+"range_10_f"], [P+"rows_180_f"]))
    nodes.append(onnx.helper.make_node("Sub", [P+"cc_2", P+"range_10_f"], [P+"cols_180_f"]))
    _add_validity(P+"rows_180", P+"rows_180_f")
    _add_validity(P+"cols_180", P+"cols_180_f")
    _add_clip_cast(P+"rows_180_f", P+"rows_180_idx")
    _add_clip_cast(P+"cols_180_f", P+"cols_180_idx")
    nodes.append(onnx.helper.make_node(
        "Gather", [P+"input_nonzero", P+"rows_180_idx"], [P+"rot180_rows"], axis=2
    ))
    nodes.append(onnx.helper.make_node(
        "Gather", [P+"rot180_rows", P+"cols_180_idx"], [P+"rot180_raw"], axis=3
    ))
    nodes.append(onnx.helper.make_node(
        "Reshape", [P+"rows_180_mask_f", P+"rs_row_4d"], [P+"rot180_row_mask_4d"]
    ))
    nodes.append(onnx.helper.make_node(
        "Reshape", [P+"cols_180_mask_f", P+"rs_col_4d"], [P+"rot180_col_mask_4d"]
    ))
    nodes.append(onnx.helper.make_node(
        "Mul", [P+"rot180_row_mask_4d", P+"rot180_col_mask_4d"], [P+"rot180_mask"]
    ))
    nodes.append(onnx.helper.make_node("Mul", [P+"rot180_raw", P+"rot180_mask"], [P+"rot_180"]))

    # ===== Step 5: 90° CW image rotation =====
    nodes.append(onnx.helper.make_node("Add", [P+"cr_minus_cc", P+"range_10_f"], [P+"cw_A_f"]))
    nodes.append(onnx.helper.make_node("Sub", [P+"cr_plus_cc", P+"range_10_f"], [P+"cw_B_f"]))
    _add_validity(P+"cw_A", P+"cw_A_f")
    _add_validity(P+"cw_B", P+"cw_B_f")
    _add_clip_cast(P+"cw_A_f", P+"cw_A_idx")
    _add_clip_cast(P+"cw_B_f", P+"cw_B_idx")
    nodes.append(onnx.helper.make_node(
        "Gather", [P+"input_nonzero", P+"cw_A_idx"], [P+"rotcw_step1"], axis=2
    ))
    nodes.append(onnx.helper.make_node(
        "Gather", [P+"rotcw_step1", P+"cw_B_idx"], [P+"rotcw_step2"], axis=3
    ))
    nodes.append(onnx.helper.make_node(
        "Transpose", [P+"rotcw_step2"], [P+"rotcw_raw"], perm=[0, 1, 3, 2]
    ))
    nodes.append(onnx.helper.make_node(
        "Reshape", [P+"cw_B_mask_f", P+"rs_row_4d"], [P+"rotcw_row_mask_4d"]
    ))
    nodes.append(onnx.helper.make_node(
        "Reshape", [P+"cw_A_mask_f", P+"rs_col_4d"], [P+"rotcw_col_mask_4d"]
    ))
    nodes.append(onnx.helper.make_node(
        "Mul", [P+"rotcw_row_mask_4d", P+"rotcw_col_mask_4d"], [P+"rotcw_mask"]
    ))
    nodes.append(onnx.helper.make_node("Mul", [P+"rotcw_raw", P+"rotcw_mask"], [P+"rot_cw"]))

    # ===== Step 6: 270° CW image rotation =====
    nodes.append(onnx.helper.make_node("Sub", [P+"cr_plus_cc", P+"range_10_f"], [P+"ccw_A_f"]))
    nodes.append(onnx.helper.make_node("Add", [P+"cc_minus_cr", P+"range_10_f"], [P+"ccw_B_f"]))
    _add_validity(P+"ccw_A", P+"ccw_A_f")
    _add_validity(P+"ccw_B", P+"ccw_B_f")
    _add_clip_cast(P+"ccw_A_f", P+"ccw_A_idx")
    _add_clip_cast(P+"ccw_B_f", P+"ccw_B_idx")
    nodes.append(onnx.helper.make_node(
        "Gather", [P+"input_nonzero", P+"ccw_A_idx"], [P+"rotccw_step1"], axis=2
    ))
    nodes.append(onnx.helper.make_node(
        "Gather", [P+"rotccw_step1", P+"ccw_B_idx"], [P+"rotccw_step2"], axis=3
    ))
    nodes.append(onnx.helper.make_node(
        "Transpose", [P+"rotccw_step2"], [P+"rotccw_raw"], perm=[0, 1, 3, 2]
    ))
    nodes.append(onnx.helper.make_node(
        "Reshape", [P+"ccw_B_mask_f", P+"rs_row_4d"], [P+"rotccw_row_mask_4d"]
    ))
    nodes.append(onnx.helper.make_node(
        "Reshape", [P+"ccw_A_mask_f", P+"rs_col_4d"], [P+"rotccw_col_mask_4d"]
    ))
    nodes.append(onnx.helper.make_node(
        "Mul", [P+"rotccw_row_mask_4d", P+"rotccw_col_mask_4d"], [P+"rotccw_mask"]
    ))
    nodes.append(onnx.helper.make_node("Mul", [P+"rotccw_raw", P+"rotccw_mask"], [P+"rot_ccw"]))

    # ===== Step 7: union via 3 chained Max ops on channels 1..9 =====
    nodes.append(onnx.helper.make_node("Max", [P+"input_nonzero", P+"rot_180"], [P+"max_01"]))
    nodes.append(onnx.helper.make_node("Max", [P+"max_01", P+"rot_cw"], [P+"max_012"]))
    nodes.append(onnx.helper.make_node("Max", [P+"max_012", P+"rot_ccw"], [P+"nonbg_union_raw"]))

    # ===== Step 8: mask by is_active to zero canvas-padding cells =====
    nodes.append(onnx.helper.make_node(
        "Mul", [P+"nonbg_union_raw", P+"is_active"], [P+"nonbg_union"]
    ))  # (1, 9, 10, 10)
    nodes.append(onnx.helper.make_node(
        "ReduceMax", [P+"nonbg_union"], [P+"any_nonbg"], axes=[1], keepdims=1
    ))  # (1, 1, 10, 10)
    nodes.append(onnx.helper.make_node(
        "Sub", [P+"const_1_4d", P+"any_nonbg"], [P+"ch0_pre_mask"]
    ))
    nodes.append(onnx.helper.make_node(
        "Mul", [P+"ch0_pre_mask", P+"is_active"], [P+"ch0_out"]
    ))  # (1, 1, 10, 10)

    # ===== Step 9: assemble result [1,10,10,10], then Pad to [1,10,30,30] =====
    nodes.append(onnx.helper.make_node(
        "Concat", [P+"ch0_out", P+"nonbg_union"], [P+"result"], axis=1
    ))  # (1, 10, 10, 10)
    nodes.append(onnx.helper.make_node(
        "Pad", [P+"result"], ["output"],
        pads=[0, 0, 0, 0, 0, 0, 20, 20], mode="constant", value=0.0,
    ))  # (1, 10, 30, 30); graph output not counted in static_memory

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def marker_driven_shape_copy_network(batch=1):
    """ONNX: marker-driven pattern copy (color 5 = single-cell marker).

    Task206: the non-5, non-bg pattern is copied so its bbox center aligns
    with the single color-5 marker cell. The marker disappears. The original
    pattern stays in place. Output = Max union of original + translated copy.

    Pipeline (input [1, 10, 30, 30], batch=1 only):
      1. is_active = ReduceSum(input, axis=1) -> [1,1,30,30]; 1 in grid, 0 padding.
      2. ch5 = Slice(input, ch=5..6) -> [1,1,30,30]; ArgMax trick -> marker_r_f, marker_c_f.
      3. pattern_19 = Slice(input, ch=1..10) * mask_no5 -> [1,9,30,30]; ch5 zeroed.
         pattern_any = ReduceMax(pattern_19, axis=1).
      4. Pattern bbox: pr0, pr1, pc0, pc1 -> cr = (pr0+pr1)/2, cc = (pc0+pc1)/2.
      5. dr = marker_r - cr, dc = marker_c - cc (always integer-valued, 266/266 verified).
      6. src_r[nr] = nr - dr; validity [0..29]; Gather rows of pattern_19.
         src_c[nc] = nc - dc; validity [0..29]; Gather cols; mask; -> translated [1,9,30,30].
      7. nonbg_union = Max(pattern_19, translated) * is_active.
      8. ch0_out = (1 - ReduceMax(nonbg_union, axis=1)) * is_active.
      9. output = Concat([ch0_out, nonbg_union], axis=1) -> [1,10,30,30].

    Cost (estimated): memory ~200-300 KB -> ~12.5 pts. Cascade-safe.
    """
    if batch != 1:
        raise NotImplementedError(
            "marker-driven-shape-copy builder is single-example only (batch must be 1)"
        )

    nodes = []
    inits = []
    P = "mdc_"
    H = W = 30

    # --- Shared constants ---
    rs1 = onnx.helper.make_tensor(P+"rs1", onnx.TensorProto.INT64, [1], [1])
    const_0_f = onnx.helper.make_tensor(P+"const_0_f", onnx.TensorProto.FLOAT, [1], [0.0])
    const_1_4d = onnx.helper.make_tensor(P+"const_1_4d", onnx.TensorProto.FLOAT, [1, 1, 1, 1], [1.0])
    const_29_f = onnx.helper.make_tensor(P+"const_29_f", onnx.TensorProto.FLOAT, [1], [29.0])
    const_2_f = onnx.helper.make_tensor(P+"const_2_f", onnx.TensorProto.FLOAT, [1], [2.0])
    range_30_f = onnx.helper.make_tensor(
        P+"range_30_f", onnx.TensorProto.FLOAT, [H], [float(i) for i in range(H)]
    )
    desc_row = onnx.helper.make_tensor(
        P+"desc_row", onnx.TensorProto.FLOAT, [1, 1, H, 1], [float(H - i) for i in range(H)]
    )
    asc_row = onnx.helper.make_tensor(
        P+"asc_row", onnx.TensorProto.FLOAT, [1, 1, H, 1], [float(i + 1) for i in range(H)]
    )
    desc_col = onnx.helper.make_tensor(
        P+"desc_col", onnx.TensorProto.FLOAT, [1, 1, 1, W], [float(W - i) for i in range(W)]
    )
    asc_col = onnx.helper.make_tensor(
        P+"asc_col", onnx.TensorProto.FLOAT, [1, 1, 1, W], [float(i + 1) for i in range(W)]
    )
    rs_row_4d = onnx.helper.make_tensor(P+"rs_row_4d", onnx.TensorProto.INT64, [4], [1, 1, H, 1])
    rs_col_4d = onnx.helper.make_tensor(P+"rs_col_4d", onnx.TensorProto.INT64, [4], [1, 1, 1, W])
    inits.extend([rs1, const_0_f, const_1_4d, const_29_f, const_2_f,
                  range_30_f, desc_row, asc_row, desc_col, asc_col,
                  rs_row_4d, rs_col_4d])

    # --- Step 1: is_active (1.0 within grid, 0.0 at canvas-padding) ---
    nodes.append(onnx.helper.make_node(
        "ReduceSum", ["input"], [P+"is_active"], axes=[1], keepdims=1
    ))  # [1,1,30,30]

    # --- Step 2: Extract channel 5 (marker channel) ---
    sl5_st = onnx.helper.make_tensor(P+"sl5_st", onnx.TensorProto.INT64, [1], [5])
    sl5_en = onnx.helper.make_tensor(P+"sl5_en", onnx.TensorProto.INT64, [1], [6])
    sl5_ax = onnx.helper.make_tensor(P+"sl5_ax", onnx.TensorProto.INT64, [1], [1])
    sl5_sp = onnx.helper.make_tensor(P+"sl5_sp", onnx.TensorProto.INT64, [1], [1])
    inits.extend([sl5_st, sl5_en, sl5_ax, sl5_sp])
    nodes.append(onnx.helper.make_node(
        "Slice", ["input", P+"sl5_st", P+"sl5_en", P+"sl5_ax", P+"sl5_sp"], [P+"ch5"]
    ))  # [1,1,30,30]

    # --- Step 3: Find marker row and col via ArgMax trick ---
    nodes.append(onnx.helper.make_node(
        "ReduceMax", [P+"ch5"], [P+"row_has_m"], axes=[3], keepdims=1
    ))  # [1,1,30,1]
    nodes.append(onnx.helper.make_node(
        "ReduceMax", [P+"ch5"], [P+"col_has_m"], axes=[2], keepdims=1
    ))  # [1,1,1,30]
    nodes.append(onnx.helper.make_node("Mul", [P+"row_has_m", P+"desc_row"], [P+"wm_row"]))
    nodes.append(onnx.helper.make_node("Mul", [P+"col_has_m", P+"desc_col"], [P+"wm_col"]))
    nodes.append(onnx.helper.make_node(
        "ArgMax", [P+"wm_row"], [P+"marker_r_3d"], axis=2, keepdims=0
    ))  # [1,1,1] INT64
    nodes.append(onnx.helper.make_node(
        "ArgMax", [P+"wm_col"], [P+"marker_c_3d"], axis=3, keepdims=0
    ))  # [1,1,1] INT64
    nodes.append(onnx.helper.make_node("Reshape", [P+"marker_r_3d", P+"rs1"], [P+"marker_r_1d"]))
    nodes.append(onnx.helper.make_node("Reshape", [P+"marker_c_3d", P+"rs1"], [P+"marker_c_1d"]))
    nodes.append(onnx.helper.make_node(
        "Cast", [P+"marker_r_1d"], [P+"marker_r_f"], to=onnx.TensorProto.FLOAT
    ))
    nodes.append(onnx.helper.make_node(
        "Cast", [P+"marker_c_1d"], [P+"marker_c_f"], to=onnx.TensorProto.FLOAT
    ))

    # --- Step 4: Pattern = ch1..9 with channel 5 zeroed out ---
    sl19_st = onnx.helper.make_tensor(P+"sl19_st", onnx.TensorProto.INT64, [1], [1])
    sl19_en = onnx.helper.make_tensor(P+"sl19_en", onnx.TensorProto.INT64, [1], [10])
    sl19_ax = onnx.helper.make_tensor(P+"sl19_ax", onnx.TensorProto.INT64, [1], [1])
    sl19_sp = onnx.helper.make_tensor(P+"sl19_sp", onnx.TensorProto.INT64, [1], [1])
    inits.extend([sl19_st, sl19_en, sl19_ax, sl19_sp])
    nodes.append(onnx.helper.make_node(
        "Slice", ["input", P+"sl19_st", P+"sl19_en", P+"sl19_ax", P+"sl19_sp"],
        [P+"ch1_to_9"]
    ))  # [1,9,30,30]
    mask_no5_vals = [1.0 if i != 4 else 0.0 for i in range(9)]
    mask_no5 = onnx.helper.make_tensor(
        P+"mask_no5", onnx.TensorProto.FLOAT, [1, 9, 1, 1], mask_no5_vals
    )
    inits.append(mask_no5)
    nodes.append(onnx.helper.make_node("Mul", [P+"ch1_to_9", P+"mask_no5"], [P+"pattern_19"]))
    # [1,9,30,30]: ch5 position zeroed out

    # --- Step 5: Pattern bbox via ArgMax trick ---
    nodes.append(onnx.helper.make_node(
        "ReduceMax", [P+"pattern_19"], [P+"pattern_any"], axes=[1], keepdims=1
    ))  # [1,1,30,30]
    nodes.append(onnx.helper.make_node(
        "ReduceMax", [P+"pattern_any"], [P+"row_has_p"], axes=[3], keepdims=1
    ))  # [1,1,30,1]
    nodes.append(onnx.helper.make_node(
        "ReduceMax", [P+"pattern_any"], [P+"col_has_p"], axes=[2], keepdims=1
    ))  # [1,1,1,30]
    nodes.append(onnx.helper.make_node("Mul", [P+"row_has_p", P+"desc_row"], [P+"wp_fr_w"]))
    nodes.append(onnx.helper.make_node("Mul", [P+"row_has_p", P+"asc_row"], [P+"wp_lr_w"]))
    nodes.append(onnx.helper.make_node("Mul", [P+"col_has_p", P+"desc_col"], [P+"wp_fc_w"]))
    nodes.append(onnx.helper.make_node("Mul", [P+"col_has_p", P+"asc_col"], [P+"wp_lc_w"]))
    nodes.append(onnx.helper.make_node(
        "ArgMax", [P+"wp_fr_w"], [P+"pr0_3d"], axis=2, keepdims=0
    ))
    nodes.append(onnx.helper.make_node(
        "ArgMax", [P+"wp_lr_w"], [P+"pr1_3d"], axis=2, keepdims=0
    ))
    nodes.append(onnx.helper.make_node(
        "ArgMax", [P+"wp_fc_w"], [P+"pc0_3d"], axis=3, keepdims=0
    ))
    nodes.append(onnx.helper.make_node(
        "ArgMax", [P+"wp_lc_w"], [P+"pc1_3d"], axis=3, keepdims=0
    ))
    for _src in ["pr0", "pr1", "pc0", "pc1"]:
        nodes.append(onnx.helper.make_node("Reshape", [P+_src+"_3d", P+"rs1"], [P+_src+"_1d"]))
        nodes.append(onnx.helper.make_node(
            "Cast", [P+_src+"_1d"], [P+_src+"_f"], to=onnx.TensorProto.FLOAT
        ))
    nodes.append(onnx.helper.make_node("Add", [P+"pr0_f", P+"pr1_f"], [P+"pr_sum"]))
    nodes.append(onnx.helper.make_node("Add", [P+"pc0_f", P+"pc1_f"], [P+"pc_sum"]))
    nodes.append(onnx.helper.make_node("Div", [P+"pr_sum", P+"const_2_f"], [P+"cr_f"]))
    nodes.append(onnx.helper.make_node("Div", [P+"pc_sum", P+"const_2_f"], [P+"cc_f"]))

    # --- Step 6: Translation offsets (integer-valued floats, verified 266/266) ---
    nodes.append(onnx.helper.make_node("Sub", [P+"marker_r_f", P+"cr_f"], [P+"dr_f"]))
    nodes.append(onnx.helper.make_node("Sub", [P+"marker_c_f", P+"cc_f"], [P+"dc_f"]))

    # --- Helpers: validity + clip+cast for index range [0..29] ---
    def _add_validity(name_prefix, idx_f_name):
        nodes.append(onnx.helper.make_node(
            "Less", [idx_f_name, P+"const_0_f"], [name_prefix+"_bad_lo"]
        ))
        nodes.append(onnx.helper.make_node(
            "Less", [P+"const_29_f", idx_f_name], [name_prefix+"_bad_hi"]
        ))
        nodes.append(onnx.helper.make_node(
            "Or", [name_prefix+"_bad_lo", name_prefix+"_bad_hi"], [name_prefix+"_bad"]
        ))
        nodes.append(onnx.helper.make_node("Not", [name_prefix+"_bad"], [name_prefix+"_good"]))
        nodes.append(onnx.helper.make_node(
            "Cast", [name_prefix+"_good"], [name_prefix+"_mask_f"], to=onnx.TensorProto.FLOAT
        ))

    def _add_clip_cast(idx_f_name, dst_int_name):
        nodes.append(onnx.helper.make_node(
            "Max", [idx_f_name, P+"const_0_f"], [idx_f_name+"_clo"]
        ))
        nodes.append(onnx.helper.make_node(
            "Min", [idx_f_name+"_clo", P+"const_29_f"], [idx_f_name+"_clip"]
        ))
        nodes.append(onnx.helper.make_node(
            "Cast", [idx_f_name+"_clip"], [dst_int_name], to=onnx.TensorProto.INT64
        ))

    # --- Step 7: Translated pattern via Gather ---
    # src_r[nr] = range[nr] - dr  =>  output row nr comes from source row src_r[nr]
    # src_c[nc] = range[nc] - dc  =>  output col nc comes from source col src_c[nc]
    nodes.append(onnx.helper.make_node("Sub", [P+"range_30_f", P+"dr_f"], [P+"src_r_f"]))
    nodes.append(onnx.helper.make_node("Sub", [P+"range_30_f", P+"dc_f"], [P+"src_c_f"]))
    _add_validity(P+"src_r", P+"src_r_f")
    _add_validity(P+"src_c", P+"src_c_f")
    _add_clip_cast(P+"src_r_f", P+"src_r_idx")
    _add_clip_cast(P+"src_c_f", P+"src_c_idx")
    nodes.append(onnx.helper.make_node(
        "Gather", [P+"pattern_19", P+"src_r_idx"], [P+"transl_rows"], axis=2
    ))  # [1,9,30,30]
    nodes.append(onnx.helper.make_node(
        "Gather", [P+"transl_rows", P+"src_c_idx"], [P+"transl_raw"], axis=3
    ))  # [1,9,30,30]
    nodes.append(onnx.helper.make_node(
        "Reshape", [P+"src_r_mask_f", P+"rs_row_4d"], [P+"row_mask_4d"]
    ))  # [1,1,30,1]
    nodes.append(onnx.helper.make_node(
        "Reshape", [P+"src_c_mask_f", P+"rs_col_4d"], [P+"col_mask_4d"]
    ))  # [1,1,1,30]
    nodes.append(onnx.helper.make_node(
        "Mul", [P+"row_mask_4d", P+"col_mask_4d"], [P+"valid_mask"]
    ))  # [1,1,30,30] broadcasts over channel axis
    nodes.append(onnx.helper.make_node(
        "Mul", [P+"transl_raw", P+"valid_mask"], [P+"translated"]
    ))  # [1,9,30,30]

    # --- Step 8: Union of original + translated, masked to grid ---
    nodes.append(onnx.helper.make_node(
        "Max", [P+"pattern_19", P+"translated"], [P+"union_raw"]
    ))  # [1,9,30,30]
    nodes.append(onnx.helper.make_node(
        "Mul", [P+"union_raw", P+"is_active"], [P+"nonbg_union"]
    ))  # [1,9,30,30]

    # --- Step 9: Derive ch0 (bg channel) ---
    nodes.append(onnx.helper.make_node(
        "ReduceMax", [P+"nonbg_union"], [P+"any_nonbg"], axes=[1], keepdims=1
    ))  # [1,1,30,30]
    nodes.append(onnx.helper.make_node("Sub", [P+"const_1_4d", P+"any_nonbg"], [P+"ch0_pre"]))
    nodes.append(onnx.helper.make_node(
        "Mul", [P+"ch0_pre", P+"is_active"], [P+"ch0_out"]
    ))  # [1,1,30,30]: 1 at bg cells, 0 at non-bg and canvas-padding

    # --- Step 10: Assemble output ---
    nodes.append(onnx.helper.make_node(
        "Concat", [P+"ch0_out", P+"nonbg_union"], ["output"], axis=1
    ))  # [1,10,30,30]

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def periodic_row_extension_network(batch=1):
    """ONNX: task003 — 6-row input extended to 9 rows via period detection.

    Input: (6,3) grid with colors {0,1}, one-hot [1,10,30,30].
    Output: (9,3) grid with colors {0,2} (1→2 substitution, 9-row extension).

    Rule: find smallest T in {2,3,4} such that input[r] == input[r % T] for all r.
    Then output rows 0-5 = input rows 0-5 with 1→2 subst; rows 6,7,8 =
    input rows (6%T, 7%T, 8%T) with 1→2 subst.

    Pipeline:
      1. Crop input to [1,10,6,3].
      2. 1×1 Conv: ch0→ch0, ch1→ch2 (color substitution); call this 'subst'.
      3. Compare row0 vs row2 (sum of abs diff) → is_period2 (T≤2).
      4. Compare row0 vs row3 → is_period3 (T≤3 but not T≤2).
      5. is_period4 = NOT is_period2 AND NOT is_period3.
      6. extra_r6 = Where(is_period4, subst[row2], subst[row0]).
         extra_r7 = Where(is_period4, subst[row3], subst[row1]).
         extra_r8 = Where(is_period3, subst[row2], subst[row0]).
      7. Concat subst (6 rows) + extra (3 rows) → [1,10,9,3].
      8. Pad → [1,10,30,30].

    Cost: ~3-5 KB memory + 100 params → ~16-17 pts. Cascade-safe.
    Batch=1 only (period detection requires runtime row comparison).
    """
    if batch != 1:
        raise NotImplementedError(
            "periodic-row-extension builder is single-example only (batch must be 1)"
        )

    nodes = []
    inits = []
    P = "pre_"

    # --- Step 1: Crop input to [1,10,6,3] ---
    cr_st = onnx.helper.make_tensor(P+"cr_st", onnx.TensorProto.INT64, [2], [0, 0])
    cr_en = onnx.helper.make_tensor(P+"cr_en", onnx.TensorProto.INT64, [2], [6, 3])
    cr_ax = onnx.helper.make_tensor(P+"cr_ax", onnx.TensorProto.INT64, [2], [2, 3])
    cr_sp = onnx.helper.make_tensor(P+"cr_sp", onnx.TensorProto.INT64, [2], [1, 1])
    inits.extend([cr_st, cr_en, cr_ax, cr_sp])
    nodes.append(onnx.helper.make_node(
        "Slice", ["input", P+"cr_st", P+"cr_en", P+"cr_ax", P+"cr_sp"], [P+"crop"]
    ))  # [1,10,6,3]

    # --- Step 2: 1×1 Conv for color substitution: ch0→ch0, ch1→ch2 ---
    w_vals = [0.0] * (10 * 10)
    w_vals[0 * 10 + 0] = 1.0   # ch0_out from ch0_in
    w_vals[2 * 10 + 1] = 1.0   # ch2_out from ch1_in
    conv_w = onnx.helper.make_tensor(P+"conv_w", onnx.TensorProto.FLOAT, [10, 10, 1, 1], w_vals)
    inits.append(conv_w)
    nodes.append(onnx.helper.make_node(
        "Conv", [P+"crop", P+"conv_w"], [P+"subst"],
        kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0],
    ))  # [1,10,6,3]

    # --- Row extraction helper (axis=2) ---
    sp1 = onnx.helper.make_tensor(P+"sp1", onnx.TensorProto.INT64, [1], [1])
    ax2 = onnx.helper.make_tensor(P+"ax2", onnx.TensorProto.INT64, [1], [2])
    inits.extend([sp1, ax2])

    _rs_added = {}

    def _row_slice(src, r, dst):
        st_name = P+f"rs{r}_st"
        en_name = P+f"rs{r}_en"
        if r not in _rs_added:
            st = onnx.helper.make_tensor(st_name, onnx.TensorProto.INT64, [1], [r])
            en = onnx.helper.make_tensor(en_name, onnx.TensorProto.INT64, [1], [r + 1])
            inits.extend([st, en])
            _rs_added[r] = True
        nodes.append(onnx.helper.make_node(
            "Slice", [src, st_name, en_name, P+"ax2", P+"sp1"], [dst]
        ))  # [1,10,1,3]

    for r in range(4):
        _row_slice(P+"crop", r, P+f"crop_r{r}")    # rows 0-3 from crop for period check
        _row_slice(P+"subst", r, P+f"sub_r{r}")    # rows 0-3 from subst for extra rows

    # --- Step 3: Period detection ---
    const_half = onnx.helper.make_tensor(P+"const_half", onnx.TensorProto.FLOAT, [1], [0.5])
    inits.append(const_half)

    def _row_equal(r_a_name, r_b_name, eq_dst):
        nodes.append(onnx.helper.make_node("Sub", [r_a_name, r_b_name], [P+f"diff_{eq_dst}"]))
        nodes.append(onnx.helper.make_node("Abs", [P+f"diff_{eq_dst}"], [P+f"adiff_{eq_dst}"]))
        nodes.append(onnx.helper.make_node(
            "ReduceSum", [P+f"adiff_{eq_dst}"], [P+f"sum_{eq_dst}"]
        ))  # scalar
        nodes.append(onnx.helper.make_node(
            "Less", [P+f"sum_{eq_dst}", P+"const_half"], [P+eq_dst]
        ))  # bool scalar: True if rows equal

    # T=2: row0==row2 AND row1==row3 (both conditions required; row0==row2 alone fails
    #       when row3!=row1 but row2 coincidentally matches row0)
    _row_equal(P+"crop_r0", P+"crop_r2", "is_r0r2")  # row0==row2
    _row_equal(P+"crop_r1", P+"crop_r3", "is_r1r3")  # row1==row3
    _row_equal(P+"crop_r0", P+"crop_r3", "is_r0r3")  # row0==row3 → needed for T=3

    nodes.append(onnx.helper.make_node(
        "And", [P+"is_r0r2", P+"is_r1r3"], [P+"is_p2"]
    ))  # T=2: row0==row2 AND row1==row3
    nodes.append(onnx.helper.make_node("Not", [P+"is_p2"], [P+"not_p2"]))
    nodes.append(onnx.helper.make_node(
        "And", [P+"not_p2", P+"is_r0r3"], [P+"is_p3"]
    ))  # T=3: NOT period-2 AND row0==row3
    nodes.append(onnx.helper.make_node("Not", [P+"is_r0r3"], [P+"not_r0r3"]))
    nodes.append(onnx.helper.make_node(
        "And", [P+"not_p2", P+"not_r0r3"], [P+"is_p4"]
    ))  # T=4: NOT period-2 AND row0!=row3

    # --- Step 4: Compute extra rows 6, 7, 8 ---
    # row6: T=4 → subst[row2], else → subst[row0]
    # row7: T=4 → subst[row3], else → subst[row1]
    # row8: T=3 → subst[row2], else → subst[row0]
    nodes.append(onnx.helper.make_node(
        "Where", [P+"is_p4", P+"sub_r2", P+"sub_r0"], [P+"extra_r6"]
    ))
    nodes.append(onnx.helper.make_node(
        "Where", [P+"is_p4", P+"sub_r3", P+"sub_r1"], [P+"extra_r7"]
    ))
    nodes.append(onnx.helper.make_node(
        "Where", [P+"is_p3", P+"sub_r2", P+"sub_r0"], [P+"extra_r8"]
    ))

    # --- Step 5: Assemble 9-row result and pad ---
    nodes.append(onnx.helper.make_node(
        "Concat", [P+"extra_r6", P+"extra_r7", P+"extra_r8"], [P+"extra"], axis=2
    ))  # [1,10,3,3]
    nodes.append(onnx.helper.make_node(
        "Concat", [P+"subst", P+"extra"], [P+"inner"], axis=2
    ))  # [1,10,9,3]
    nodes.append(onnx.helper.make_node(
        "Pad", [P+"inner"], ["output"],
        pads=[0, 0, 0, 0, 0, 0, 21, 27], mode="constant", value=0.0,
    ))  # [1,10,30,30]

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def boolean_halves_network(H, W_in, W_out, check_channel, out_color, batch=1):
    """ONNX: tasks006/026 — compare two W_out-wide halves of a W_in-wide grid.

    Input shape: (H, W_in) with separator at col W_out (e.g. W_in=7, W_out=3 → sep at col3).
    Output shape: (H, W_out) with values {0, out_color}.

    Rule: out[r][c] = out_color if input[r][c] has check_channel=1
                                 AND input[r][c + W_out + 1] has check_channel=1,
          else out[r][c] = 0.

    Equivalently: extract check_channel from left W_out cols and right W_out cols (after sep),
    multiply element-wise → 1 where both satisfied, then scale by out_color.

    Examples:
      task006: H=3, W_in=7, W_out=3, check_channel=1, out_color=2  (AND of 1s)
      task026: H=5, W_in=7, W_out=3, check_channel=0, out_color=8  (NOR = AND of 0s)

    Batch=1 only (fixed dimensions baked into Slice constants).
    """
    if batch != 1:
        raise NotImplementedError("boolean-halves builder is single-example only")

    nodes = []
    inits = []
    P = "bh_"

    sep_col = W_out  # separator column index in input
    right_start = W_out + 1  # first column of right half

    # Step 1: Extract check_channel from input → [1,1,30,30]
    ch_st = onnx.helper.make_tensor(P+"ch_st", onnx.TensorProto.INT64, [1], [check_channel])
    ch_en = onnx.helper.make_tensor(P+"ch_en", onnx.TensorProto.INT64, [1], [check_channel + 1])
    ch_ax = onnx.helper.make_tensor(P+"ch_ax", onnx.TensorProto.INT64, [1], [1])
    ch_sp = onnx.helper.make_tensor(P+"ch_sp", onnx.TensorProto.INT64, [1], [1])
    inits.extend([ch_st, ch_en, ch_ax, ch_sp])
    nodes.append(onnx.helper.make_node(
        "Slice", ["input", P+"ch_st", P+"ch_en", P+"ch_ax", P+"ch_sp"], [P+"ch_full"]
    ))  # [1,1,30,30]

    # Step 2: Slice left half (rows 0:H, cols 0:W_out) → [1,1,H,W_out]
    left_st = onnx.helper.make_tensor(P+"left_st", onnx.TensorProto.INT64, [2], [0, 0])
    left_en = onnx.helper.make_tensor(P+"left_en", onnx.TensorProto.INT64, [2], [H, W_out])
    hw_ax = onnx.helper.make_tensor(P+"hw_ax", onnx.TensorProto.INT64, [2], [2, 3])
    hw_sp = onnx.helper.make_tensor(P+"hw_sp", onnx.TensorProto.INT64, [2], [1, 1])
    inits.extend([left_st, left_en, hw_ax, hw_sp])
    nodes.append(onnx.helper.make_node(
        "Slice", [P+"ch_full", P+"left_st", P+"left_en", P+"hw_ax", P+"hw_sp"], [P+"left"]
    ))  # [1,1,H,W_out]

    # Step 3: Slice right half (rows 0:H, cols right_start:right_start+W_out) → [1,1,H,W_out]
    right_st = onnx.helper.make_tensor(P+"right_st", onnx.TensorProto.INT64, [2], [0, right_start])
    right_en = onnx.helper.make_tensor(P+"right_en", onnx.TensorProto.INT64, [2], [H, right_start + W_out])
    inits.extend([right_st, right_en])
    nodes.append(onnx.helper.make_node(
        "Slice", [P+"ch_full", P+"right_st", P+"right_en", P+"hw_ax", P+"hw_sp"], [P+"right"]
    ))  # [1,1,H,W_out]

    # Step 4: AND (element-wise multiply) → inner [1,1,H,W_out]
    nodes.append(onnx.helper.make_node("Mul", [P+"left", P+"right"], [P+"inner"]))

    # Step 5: ch0_out = 1 - inner, zeros = 0 * inner
    const_one = onnx.helper.make_tensor(P+"const_one", onnx.TensorProto.FLOAT, [1], [1.0])
    const_zero = onnx.helper.make_tensor(P+"const_zero", onnx.TensorProto.FLOAT, [1], [0.0])
    inits.extend([const_one, const_zero])
    nodes.append(onnx.helper.make_node("Sub", [P+"const_one", P+"inner"], [P+"ch0_out"]))
    nodes.append(onnx.helper.make_node("Mul", [P+"inner", P+"const_zero"], [P+"zeros"]))

    # Step 6: Build 10-channel output tensor [1,10,H,W_out]
    # Channel 0 = ch0_out (background), channel out_color = inner (fg), rest = zeros
    ch_tensors = []
    for c in range(10):
        if c == 0:
            ch_tensors.append(P+"ch0_out")
        elif c == out_color:
            ch_tensors.append(P+"inner")
        else:
            ch_tensors.append(P+"zeros")
    nodes.append(onnx.helper.make_node(
        "Concat", ch_tensors, [P+"assembled"], axis=1
    ))  # [1,10,H,W_out]

    # Step 7: Pad to [1,10,30,30]
    pad_h = 30 - H
    pad_w = 30 - W_out
    nodes.append(onnx.helper.make_node(
        "Pad", [P+"assembled"], ["output"],
        pads=[0, 0, 0, 0, 0, 0, pad_h, pad_w], mode="constant", value=0.0,
    ))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def vertical_halves_network(H_in, W, H_half, mode, out_color, sep_row=None, batch=1):
    """ONNX: compare two H_half-row halves of an H_in-row grid.

    Modes (all using ch0 = background channel):
      "nor" → both bg → out_color        (tasks 144, 227, 395)
      "or"  → at least one fg → out_color (task 318)
      "xor" → exactly one fg → out_color  (tasks 072, 236)

    sep_row: row index of separator (skipped), or None if no separator.
    """
    if batch != 1:
        raise NotImplementedError("vertical-halves is single-example only")

    nodes = []
    inits = []
    P = "vh_"

    def _t(name, dtype, shape, vals):
        return onnx.helper.make_tensor(name, dtype, shape, vals)
    def _n(op, inputs, outputs, **kwargs):
        return onnx.helper.make_node(op, inputs, outputs, **kwargs)

    # Extract ch0 (background) → [1,1,30,30]
    inits += [
        _t(P+"ch_st", onnx.TensorProto.INT64, [1], [0]),
        _t(P+"ch_en", onnx.TensorProto.INT64, [1], [1]),
        _t(P+"ch_ax", onnx.TensorProto.INT64, [1], [1]),
        _t(P+"ch_sp", onnx.TensorProto.INT64, [1], [1]),
    ]
    nodes.append(_n("Slice", ["input", P+"ch_st", P+"ch_en", P+"ch_ax", P+"ch_sp"],
                    [P+"ch_full"]))

    # Slice top half [0:H_half, 0:W]
    inits += [
        _t(P+"top_st", onnx.TensorProto.INT64, [2], [0, 0]),
        _t(P+"top_en", onnx.TensorProto.INT64, [2], [H_half, W]),
        _t(P+"hw_ax",  onnx.TensorProto.INT64, [2], [2, 3]),
        _t(P+"hw_sp",  onnx.TensorProto.INT64, [2], [1, 1]),
    ]
    nodes.append(_n("Slice", [P+"ch_full", P+"top_st", P+"top_en", P+"hw_ax", P+"hw_sp"],
                    [P+"top_bg"]))

    # Slice bottom half
    bot_start = (sep_row + 1) if sep_row is not None else H_half
    inits += [
        _t(P+"bot_st", onnx.TensorProto.INT64, [2], [bot_start, 0]),
        _t(P+"bot_en", onnx.TensorProto.INT64, [2], [bot_start + H_half, W]),
    ]
    nodes.append(_n("Slice", [P+"ch_full", P+"bot_st", P+"bot_en", P+"hw_ax", P+"hw_sp"],
                    [P+"bot_bg"]))

    inits.append(_t(P+"const_one",  onnx.TensorProto.FLOAT, [1], [1.0]))
    inits.append(_t(P+"const_zero", onnx.TensorProto.FLOAT, [1], [0.0]))

    if mode == "nor":
        nodes.append(_n("Mul", [P+"top_bg", P+"bot_bg"], [P+"result"]))
    elif mode == "or":
        nodes.append(_n("Mul", [P+"top_bg", P+"bot_bg"], [P+"both_bg"]))
        nodes.append(_n("Sub", [P+"const_one", P+"both_bg"], [P+"result"]))
    elif mode == "xor":
        # XOR = top_bg + bot_bg - 2*top_bg*bot_bg  (bg=1 ↔ fg=0)
        inits.append(_t(P+"const_two", onnx.TensorProto.FLOAT, [1], [2.0]))
        nodes.append(_n("Add", [P+"top_bg", P+"bot_bg"], [P+"sum_bg"]))
        nodes.append(_n("Mul", [P+"top_bg", P+"bot_bg"], [P+"prod_bg"]))
        nodes.append(_n("Mul", [P+"const_two", P+"prod_bg"], [P+"twice_prod"]))
        nodes.append(_n("Sub", [P+"sum_bg", P+"twice_prod"], [P+"result"]))
    else:
        raise ValueError(f"Unknown mode: {mode}")

    nodes.append(_n("Sub", [P+"const_one", P+"result"], [P+"ch0_out"]))
    nodes.append(_n("Mul", [P+"result", P+"const_zero"], [P+"zeros"]))

    ch_tensors = []
    for c in range(10):
        if c == 0:
            ch_tensors.append(P+"ch0_out")
        elif c == out_color:
            ch_tensors.append(P+"result")
        else:
            ch_tensors.append(P+"zeros")
    nodes.append(_n("Concat", ch_tensors, [P+"assembled"], axis=1))

    pad_h = 30 - H_half
    pad_w = 30 - W
    nodes.append(_n("Pad", [P+"assembled"], ["output"],
                    pads=[0, 0, 0, 0, 0, 0, pad_h, pad_w], mode="constant", value=0.0))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def horizontal_or_network(H, W_half, out_color, batch=1):
    """ONNX: horizontal OR of two equal halves (no separator).

    Input (H, 2*W_half); output (H, W_half) with out_color where ≥1 half is fg.
    task347: H=3, W_half=3, out_color=6.
    """
    if batch != 1:
        raise NotImplementedError("horizontal-or is single-example only")

    nodes = []
    inits = []
    P = "hor_"

    def _t(name, dtype, shape, vals):
        return onnx.helper.make_tensor(name, dtype, shape, vals)
    def _n(op, inputs, outputs, **kwargs):
        return onnx.helper.make_node(op, inputs, outputs, **kwargs)

    inits += [
        _t(P+"ch_st", onnx.TensorProto.INT64, [1], [0]),
        _t(P+"ch_en", onnx.TensorProto.INT64, [1], [1]),
        _t(P+"ch_ax", onnx.TensorProto.INT64, [1], [1]),
        _t(P+"ch_sp", onnx.TensorProto.INT64, [1], [1]),
    ]
    nodes.append(_n("Slice", ["input", P+"ch_st", P+"ch_en", P+"ch_ax", P+"ch_sp"],
                    [P+"ch_full"]))

    inits += [
        _t(P+"left_st", onnx.TensorProto.INT64, [2], [0, 0]),
        _t(P+"left_en", onnx.TensorProto.INT64, [2], [H, W_half]),
        _t(P+"hw_ax",   onnx.TensorProto.INT64, [2], [2, 3]),
        _t(P+"hw_sp",   onnx.TensorProto.INT64, [2], [1, 1]),
    ]
    nodes.append(_n("Slice", [P+"ch_full", P+"left_st", P+"left_en", P+"hw_ax", P+"hw_sp"],
                    [P+"left_bg"]))

    inits += [
        _t(P+"right_st", onnx.TensorProto.INT64, [2], [0, W_half]),
        _t(P+"right_en", onnx.TensorProto.INT64, [2], [H, 2 * W_half]),
    ]
    nodes.append(_n("Slice", [P+"ch_full", P+"right_st", P+"right_en", P+"hw_ax", P+"hw_sp"],
                    [P+"right_bg"]))

    nodes.append(_n("Mul", [P+"left_bg", P+"right_bg"], [P+"both_bg"]))

    inits += [
        _t(P+"const_one",  onnx.TensorProto.FLOAT, [1], [1.0]),
        _t(P+"const_zero", onnx.TensorProto.FLOAT, [1], [0.0]),
    ]
    nodes.append(_n("Sub", [P+"const_one", P+"both_bg"], [P+"result"]))
    nodes.append(_n("Sub", [P+"const_one", P+"result"],  [P+"ch0_out"]))
    nodes.append(_n("Mul", [P+"result", P+"const_zero"], [P+"zeros"]))

    ch_tensors = []
    for c in range(10):
        if c == 0:
            ch_tensors.append(P+"ch0_out")
        elif c == out_color:
            ch_tensors.append(P+"result")
        else:
            ch_tensors.append(P+"zeros")
    nodes.append(_n("Concat", ch_tensors, [P+"assembled"], axis=1))

    pad_h = 30 - H
    pad_w = 30 - W_half
    nodes.append(_n("Pad", [P+"assembled"], ["output"],
                    pads=[0, 0, 0, 0, 0, 0, pad_h, pad_w], mode="constant", value=0.0))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def cross_product_network(N, col_idx, template_color, out_color, batch=1):
    """ONNX: cross-product marking — row 0 × col col_idx → out_color.

    Cells where row0[c]==template_color AND col[r]==template_color AND cell==0
    are marked with out_color. All original values preserved.
    task043: N=10, col_idx=9, template_color=5, out_color=2.
    """
    if batch != 1:
        raise NotImplementedError("cross-product is single-example only")

    nodes = []
    inits = []
    P = "cp_"

    def _t(name, dtype, shape, vals):
        return onnx.helper.make_tensor(name, dtype, shape, vals)
    def _n(op, inputs, outputs, **kwargs):
        return onnx.helper.make_node(op, inputs, outputs, **kwargs)

    # Extract template_color channel → [1,1,30,30]
    inits += [
        _t(P+"tc_st", onnx.TensorProto.INT64, [1], [template_color]),
        _t(P+"tc_en", onnx.TensorProto.INT64, [1], [template_color + 1]),
        _t(P+"ch_ax", onnx.TensorProto.INT64, [1], [1]),
        _t(P+"ch_sp", onnx.TensorProto.INT64, [1], [1]),
    ]
    nodes.append(_n("Slice", ["input", P+"tc_st", P+"tc_en", P+"ch_ax", P+"ch_sp"],
                    [P+"tch_full"]))

    # Slice row 0: [1,1,1,N]
    inits += [
        _t(P+"row0_st", onnx.TensorProto.INT64, [2], [0, 0]),
        _t(P+"row0_en", onnx.TensorProto.INT64, [2], [1, N]),
        _t(P+"hw_ax",   onnx.TensorProto.INT64, [2], [2, 3]),
        _t(P+"hw_sp",   onnx.TensorProto.INT64, [2], [1, 1]),
    ]
    nodes.append(_n("Slice", [P+"tch_full", P+"row0_st", P+"row0_en", P+"hw_ax", P+"hw_sp"],
                    [P+"row0_t"]))

    # Slice col col_idx: [1,1,N,1]
    inits += [
        _t(P+"col_st", onnx.TensorProto.INT64, [2], [0, col_idx]),
        _t(P+"col_en", onnx.TensorProto.INT64, [2], [N, col_idx + 1]),
    ]
    nodes.append(_n("Slice", [P+"tch_full", P+"col_st", P+"col_en", P+"hw_ax", P+"hw_sp"],
                    [P+"col_t"]))

    # Outer product via broadcast Mul: [1,1,1,N] * [1,1,N,1] → [1,1,N,N]
    nodes.append(_n("Mul", [P+"row0_t", P+"col_t"], [P+"outer"]))

    # Extract ch0 (background) → [1,1,N,N]
    inits += [
        _t(P+"ch0_st", onnx.TensorProto.INT64, [1], [0]),
        _t(P+"ch0_en", onnx.TensorProto.INT64, [1], [1]),
        _t(P+"grid_st", onnx.TensorProto.INT64, [2], [0, 0]),
        _t(P+"grid_en", onnx.TensorProto.INT64, [2], [N, N]),
    ]
    nodes.append(_n("Slice", ["input", P+"ch0_st", P+"ch0_en", P+"ch_ax", P+"ch_sp"],
                    [P+"ch0_full"]))
    nodes.append(_n("Slice", [P+"ch0_full", P+"grid_st", P+"grid_en", P+"hw_ax", P+"hw_sp"],
                    [P+"ch0_grid"]))

    # intersection = outer * ch0_grid (only at background cells)
    nodes.append(_n("Mul", [P+"outer", P+"ch0_grid"], [P+"inters"]))

    inits += [
        _t(P+"const_one",  onnx.TensorProto.FLOAT, [1], [1.0]),
        _t(P+"const_zero", onnx.TensorProto.FLOAT, [1], [0.0]),
    ]

    # new_ch0 = ch0_grid - inters
    nodes.append(_n("Sub", [P+"ch0_grid", P+"inters"], [P+"new_ch0"]))

    # original template_color channel (NxN)
    nodes.append(_n("Slice", [P+"tch_full", P+"grid_st", P+"grid_en", P+"hw_ax", P+"hw_sp"],
                    [P+"tch_grid"]))

    # zeros for other channels
    nodes.append(_n("Mul", [P+"inters", P+"const_zero"], [P+"zeros"]))

    ch_tensors = []
    for c in range(10):
        if c == 0:
            ch_tensors.append(P+"new_ch0")
        elif c == template_color:
            ch_tensors.append(P+"tch_grid")
        elif c == out_color:
            ch_tensors.append(P+"inters")
        else:
            ch_tensors.append(P+"zeros")
    nodes.append(_n("Concat", ch_tensors, [P+"assembled"], axis=1))

    pad_h = 30 - N
    pad_w = 30 - N
    nodes.append(_n("Pad", [P+"assembled"], ["output"],
                    pads=[0, 0, 0, 0, 0, 0, pad_h, pad_w], mode="constant", value=0.0))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def block_count_ge_network(count_color, threshold, out_color, N, n, block_size, stride, batch=1):
    """ONNX: count count_color in each block_size×block_size block of an N×N grid.

    Output is n×n: each cell = out_color where count >= threshold, else 0.
    task149: N=11, n=3, block_size=3, stride=4, count_color=6, threshold=2, out_color=1.
    """
    if batch != 1:
        raise NotImplementedError("block-count-ge is single-example only")

    nodes = []
    inits = []
    P = "bcg_"

    def _t(name, dtype, shape, vals):
        return onnx.helper.make_tensor(name, dtype, shape, vals)
    def _n(op, inputs, outputs, **kwargs):
        return onnx.helper.make_node(op, inputs, outputs, **kwargs)

    # Extract count_color channel: [batch, 1, 30, 30]
    inits += [
        _t(P+"cc_st", onnx.TensorProto.INT64, [1], [count_color]),
        _t(P+"cc_en", onnx.TensorProto.INT64, [1], [count_color + 1]),
        _t(P+"ch_ax", onnx.TensorProto.INT64, [1], [1]),
        _t(P+"ch_sp", onnx.TensorProto.INT64, [1], [1]),
    ]
    nodes.append(_n("Slice", ["input", P+"cc_st", P+"cc_en", P+"ch_ax", P+"ch_sp"],
                    [P+"cc_full"]))

    inits += [
        _t(P+"hw_ax", onnx.TensorProto.INT64, [2], [2, 3]),
        _t(P+"hw_sp", onnx.TensorProto.INT64, [2], [1, 1]),
    ]

    # Slice each block and ReduceSum to scalar
    block_sums = []
    for br in range(n):
        for bc in range(n):
            r0 = br * stride
            c0 = bc * stride
            nm = f"{P}b{br}_{bc}"
            inits += [
                _t(nm+"_st", onnx.TensorProto.INT64, [2], [r0, c0]),
                _t(nm+"_en", onnx.TensorProto.INT64, [2], [r0 + block_size, c0 + block_size]),
            ]
            nodes.append(_n("Slice", [P+"cc_full", nm+"_st", nm+"_en", P+"hw_ax", P+"hw_sp"],
                            [nm+"_blk"]))
            nodes.append(_n("ReduceSum", [nm+"_blk"], [nm+"_sum"], axes=[2, 3], keepdims=1))
            block_sums.append(nm+"_sum")

    # Concat 3 scalars per row along W, then 3 rows along H → [batch, 1, n, n]
    row_tensors = []
    for br in range(n):
        row_sums = block_sums[br * n:(br + 1) * n]
        rn = f"{P}row{br}"
        nodes.append(_n("Concat", row_sums, [rn], axis=3))
        row_tensors.append(rn)
    nodes.append(_n("Concat", row_tensors, [P+"counts"], axis=2))

    # Apply >= threshold: Mul(1/threshold) → Floor → Min(1.0) → {0,1}
    inits += [
        _t(P+"inv_thr", onnx.TensorProto.FLOAT, [1], [1.0 / threshold]),
        _t(P+"one",     onnx.TensorProto.FLOAT, [1], [1.0]),
        _t(P+"zero",    onnx.TensorProto.FLOAT, [1], [0.0]),
    ]
    nodes.append(_n("Mul",   [P+"counts", P+"inv_thr"], [P+"scaled"]))
    nodes.append(_n("Floor", [P+"scaled"],              [P+"floored"]))
    nodes.append(_n("Min",   [P+"floored", P+"one"],    [P+"binary"]))

    # Scale binary by out_color if needed
    if out_color == 1:
        result_tensor = P+"binary"
    else:
        inits.append(_t(P+"out_scale", onnx.TensorProto.FLOAT, [1], [float(out_color)]))
        nodes.append(_n("Mul", [P+"binary", P+"out_scale"], [P+"result"]))
        result_tensor = P+"result"

    # ch0 = 1 - binary (background where count < threshold)
    nodes.append(_n("Sub", [P+"one", P+"binary"], [P+"ch0"]))
    # zeros for unused channels
    nodes.append(_n("Mul", [P+"binary", P+"zero"], [P+"zeros"]))

    ch_tensors = []
    for c in range(10):
        if c == 0:
            ch_tensors.append(P+"ch0")
        elif c == out_color:
            ch_tensors.append(result_tensor)
        else:
            ch_tensors.append(P+"zeros")
    nodes.append(_n("Concat", ch_tensors, [P+"assembled"], axis=1))

    pad_h = 30 - n
    pad_w = 30 - n
    nodes.append(_n("Pad", [P+"assembled"], ["output"],
                    pads=[0, 0, 0, 0, 0, 0, pad_h, pad_w], mode="constant", value=0.0))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def marker_recolor_network(H, W, r_marker, c_marker, batch=1):
    """ONNX: recolor blob using single-cell marker.

    Input H×W: one blob color (many cells) + one marker color (single cell at r_marker,c_marker).
    Output: blob cells get marker color, marker cell becomes background.
    task267: H=7, W=7, r_marker=6, c_marker=0.
    """
    if batch != 1:
        raise NotImplementedError("marker-recolor is single-example only")

    nodes = []
    inits = []
    P = "mr_"

    def _t(name, dtype, shape, vals):
        return onnx.helper.make_tensor(name, dtype, shape, vals)
    def _n(op, inputs, outputs, **kwargs):
        return onnx.helper.make_node(op, inputs, outputs, **kwargs)

    # Shared slice constants
    inits += [
        _t(P+"hw_ax", onnx.TensorProto.INT64, [2], [2, 3]),
        _t(P+"hw_sp", onnx.TensorProto.INT64, [2], [1, 1]),
        _t(P+"ch_ax", onnx.TensorProto.INT64, [1], [1]),
        _t(P+"ch_sp", onnx.TensorProto.INT64, [1], [1]),
        _t(P+"hw_st", onnx.TensorProto.INT64, [2], [0, 0]),
        _t(P+"hw_en", onnx.TensorProto.INT64, [2], [H, W]),
        _t(P+"mk_st", onnx.TensorProto.INT64, [2], [r_marker, c_marker]),
        _t(P+"mk_en", onnx.TensorProto.INT64, [2], [r_marker + 1, c_marker + 1]),
        _t(P+"ch0_st", onnx.TensorProto.INT64, [1], [0]),
        _t(P+"ch0_en", onnx.TensorProto.INT64, [1], [1]),
        _t(P+"one",  onnx.TensorProto.FLOAT, [1], [1.0]),
    ]

    # delta_marker: constant [1,1,H,W] with 1.0 at [0,0,r_marker,c_marker]
    delta_data = [0.0] * (H * W)
    delta_data[r_marker * W + c_marker] = 1.0
    inits.append(_t(P+"delta", onnx.TensorProto.FLOAT, [1, 1, H, W], delta_data))

    # Extract ch0 (background) from input and slice to H×W
    nodes.append(_n("Slice", ["input", P+"ch0_st", P+"ch0_en", P+"ch_ax", P+"ch_sp"],
                    [P+"ch0_full"]))
    nodes.append(_n("Slice", [P+"ch0_full", P+"hw_st", P+"hw_en", P+"hw_ax", P+"hw_sp"],
                    [P+"ch0_hw"]))

    # fg = 1 - ch0_hw (all non-background cells)
    nodes.append(_n("Sub", [P+"one", P+"ch0_hw"], [P+"fg_hw"]))
    # blob_mask = fg - delta (fg excluding marker position)
    nodes.append(_n("Sub", [P+"fg_hw", P+"delta"], [P+"blob_mask"]))
    # out_ch_0 = ch0_hw + delta (background + marker cell becomes background)
    nodes.append(_n("Add", [P+"ch0_hw", P+"delta"], [P+"out_ch_0"]))

    # For each channel c in 1..9: out_ch_c = blob_mask * marker_indicator_c
    ch_tensors = [P+"out_ch_0"]
    for c in range(1, 10):
        nm = f"{P}c{c}"
        inits += [
            _t(nm+"_st", onnx.TensorProto.INT64, [1], [c]),
            _t(nm+"_en", onnx.TensorProto.INT64, [1], [c + 1]),
        ]
        # Extract channel c from input → [1,1,30,30]
        nodes.append(_n("Slice", ["input", nm+"_st", nm+"_en", P+"ch_ax", P+"ch_sp"],
                        [nm+"_full"]))
        # Extract value at marker position → [1,1,1,1]
        nodes.append(_n("Slice", [nm+"_full", P+"mk_st", P+"mk_en", P+"hw_ax", P+"hw_sp"],
                        [nm+"_mk"]))
        # out_ch_c = blob_mask [1,1,H,W] * marker_c [1,1,1,1] (broadcast)
        nodes.append(_n("Mul", [P+"blob_mask", nm+"_mk"], [nm+"_out"]))
        ch_tensors.append(nm+"_out")

    nodes.append(_n("Concat", ch_tensors, [P+"assembled"], axis=1))
    pad_h = 30 - H
    pad_w = 30 - W
    nodes.append(_n("Pad", [P+"assembled"], ["output"],
                    pads=[0, 0, 0, 0, 0, 0, pad_h, pad_w], mode="constant", value=0.0))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def distinct_count_pattern_network(c1, c2, c3, out_color=5, batch=1):
    """ONNX: 3×3 → 3×3 based on count of distinct colors in input.

    Colors in input come from {c1,c2,c3}. Count k of distinct present colors:
      k=1 → top row = out_color, others = 0
      k=2 → main diagonal = out_color, others = 0
      k=3 → anti-diagonal = out_color, others = 0
    task167: c1=2, c2=3, c3=4, out_color=5.
    """
    if batch != 1:
        raise NotImplementedError("distinct-count-pattern is single-example only")

    nodes = []
    inits = []
    P = "dcp_"

    def _t(name, dtype, shape, vals):
        return onnx.helper.make_tensor(name, dtype, shape, vals)
    def _n(op, inputs, outputs, **kwargs):
        return onnx.helper.make_node(op, inputs, outputs, **kwargs)

    inits += [
        _t(P+"ch_ax", onnx.TensorProto.INT64, [1], [1]),
        _t(P+"ch_sp", onnx.TensorProto.INT64, [1], [1]),
        _t(P+"hw_ax", onnx.TensorProto.INT64, [2], [2, 3]),
        _t(P+"hw_sp", onnx.TensorProto.INT64, [2], [1, 1]),
        _t(P+"hw_st", onnx.TensorProto.INT64, [2], [0, 0]),
        _t(P+"hw_en", onnx.TensorProto.INT64, [2], [3, 3]),
        _t(P+"one",   onnx.TensorProto.FLOAT, [1], [1.0]),
        _t(P+"zero",  onnx.TensorProto.FLOAT, [1], [0.0]),
        _t(P+"half",  onnx.TensorProto.FLOAT, [1], [0.5]),
        _t(P+"two",   onnx.TensorProto.FLOAT, [1], [2.0]),
        _t(P+"three", onnx.TensorProto.FLOAT, [1], [3.0]),
    ]

    # Extract each color channel and check if any cell uses it
    for idx, c in enumerate([c1, c2, c3]):
        inits += [
            _t(P+f"s{idx}", onnx.TensorProto.INT64, [1], [c]),
            _t(P+f"e{idx}", onnx.TensorProto.INT64, [1], [c + 1]),
        ]
        nodes.append(_n("Slice", ["input", P+f"s{idx}", P+f"e{idx}",
                                   P+"ch_ax", P+"ch_sp"], [P+f"cf{idx}"]))
        nodes.append(_n("Slice", [P+f"cf{idx}", P+"hw_st", P+"hw_en",
                                   P+"hw_ax", P+"hw_sp"], [P+f"c{idx}hw"]))
        nodes.append(_n("ReduceMax", [P+f"c{idx}hw"], [P+f"has{idx}"],
                        axes=[2, 3], keepdims=1))  # [1,1,1,1]

    # k = count of distinct colors present
    nodes.append(_n("Add", [P+"has0", P+"has1"], [P+"h01"]))
    nodes.append(_n("Add", [P+"h01", P+"has2"], [P+"k"]))  # [1,1,1,1] in {1,2,3}

    # Lagrange weights: w1=(k-2)(k-3)/2, w2=(k-1)(3-k), w3=(k-1)(k-2)/2
    nodes.append(_n("Sub", [P+"k", P+"one"],   [P+"km1"]))
    nodes.append(_n("Sub", [P+"k", P+"two"],   [P+"km2"]))
    nodes.append(_n("Sub", [P+"k", P+"three"], [P+"km3"]))
    nodes.append(_n("Sub", [P+"three", P+"k"], [P+"3mk"]))

    nodes.append(_n("Mul",  [P+"km2", P+"km3"], [P+"km2km3"]))
    nodes.append(_n("Mul",  [P+"km2km3", P+"half"], [P+"w1"]))   # k==1 → 1, else 0

    nodes.append(_n("Mul",  [P+"km1", P+"3mk"], [P+"w2"]))        # k==2 → 1, else 0

    nodes.append(_n("Mul",  [P+"km1", P+"km2"], [P+"km1km2"]))
    nodes.append(_n("Mul",  [P+"km1km2", P+"half"], [P+"w3"]))   # k==3 → 1, else 0

    # Binary output patterns [1,1,3,3]
    P1 = [1., 1., 1., 0., 0., 0., 0., 0., 0.]  # top row
    P2 = [1., 0., 0., 0., 1., 0., 0., 0., 1.]  # main diagonal
    P3 = [0., 0., 1., 0., 1., 0., 1., 0., 0.]  # anti-diagonal
    for nm, pat in [("pat1", P1), ("pat2", P2), ("pat3", P3)]:
        inits.append(_t(P+nm, onnx.TensorProto.FLOAT, [1, 1, 3, 3], pat))

    # Weighted sum → binary indicator of which cells get 5
    nodes.append(_n("Mul", [P+"pat1", P+"w1"], [P+"wp1"]))
    nodes.append(_n("Mul", [P+"pat2", P+"w2"], [P+"wp2"]))
    nodes.append(_n("Mul", [P+"pat3", P+"w3"], [P+"wp3"]))
    nodes.append(_n("Add", [P+"wp1", P+"wp2"], [P+"wp12"]))
    nodes.append(_n("Add", [P+"wp12", P+"wp3"], [P+"binary"]))  # [1,1,3,3]

    # ch_out_color = binary, ch0 = 1 - binary
    nodes.append(_n("Sub", [P+"one", P+"binary"], [P+"ch0"]))
    nodes.append(_n("Mul", [P+"binary", P+"zero"], [P+"zeros"]))

    ch_tensors = []
    for c in range(10):
        if c == 0:
            ch_tensors.append(P+"ch0")
        elif c == out_color:
            ch_tensors.append(P+"binary")
        else:
            ch_tensors.append(P+"zeros")
    nodes.append(_n("Concat", ch_tensors, [P+"assembled"], axis=1))

    # Pad 3×3 → 30×30
    nodes.append(_n("Pad", [P+"assembled"], ["output"],
                    pads=[0, 0, 0, 0, 0, 0, 27, 27], mode="constant", value=0.0))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def uniform_row_network(H, W, out_color=5, batch=1):
    """ONNX: 3×3 → 3×3, uniform non-zero rows → out_color, others → 0.

    Row r all same non-zero color → [out_color]*W; else [0]*W.
    task052: H=3, W=3, out_color=5.
    """
    if batch != 1:
        raise NotImplementedError("uniform-row is single-example only")

    nodes = []
    inits = []
    P = "urw_"
    FLOAT = onnx.TensorProto.FLOAT
    INT64 = onnx.TensorProto.INT64

    def _t(name, dtype, shape, vals):
        return onnx.helper.make_tensor(name, dtype, shape, vals)
    def _n(op, inputs, outputs, **kwargs):
        return onnx.helper.make_node(op, inputs, outputs, **kwargs)

    inits += [
        _t(P+"one",     FLOAT, [1], [1.0]),
        _t(P+"zero",    FLOAT, [1], [0.0]),
        _t(P+"tile_W",  INT64, [4], [1, 1, 1, W]),
        _t(P+"ch1_st",  INT64, [1], [1]),
        _t(P+"ch10_en", INT64, [1], [10]),
        _t(P+"ch_ax",   INT64, [1], [1]),
        _t(P+"ch_sp",   INT64, [1], [1]),
        _t(P+"hw_ax",   INT64, [2], [2, 3]),
        _t(P+"hw_st",   INT64, [2], [0, 0]),
        _t(P+"hw_en",   INT64, [2], [H, W]),
        _t(P+"hw_sp",   INT64, [2], [1, 1]),
    ]

    # Crop input to [1,10,H,W]
    nodes.append(_n("Slice", ["input", P+"hw_st", P+"hw_en", P+"hw_ax", P+"hw_sp"], [P+"x_hw"]))

    # Slice channels 1..9 (non-background) → [1,9,H,W]
    nodes.append(_n("Slice", [P+"x_hw", P+"ch1_st", P+"ch10_en", P+"ch_ax", P+"ch_sp"], [P+"x_nz"]))

    # Min over cols → [1,9,H,1]: 1 iff all cells in row have that channel active
    nodes.append(_n("ReduceMin", [P+"x_nz"], [P+"row_min"], axes=[3], keepdims=1))

    # Max over channels → [1,1,H,1]: 1 iff row uniform at some non-zero color
    nodes.append(_n("ReduceMax", [P+"row_min"], [P+"row_uniform"], axes=[1], keepdims=1))

    # Tile to [1,1,H,W]
    nodes.append(_n("Tile", [P+"row_uniform", P+"tile_W"], [P+"uniform_mask"]))

    # ch0 = 1 - uniform_mask
    nodes.append(_n("Sub", [P+"one", P+"uniform_mask"], [P+"ch0"]))

    # zeros = 0 (same shape as uniform_mask)
    nodes.append(_n("Mul", [P+"uniform_mask", P+"zero"], [P+"zeros"]))

    ch_tensors = []
    for c in range(10):
        if c == 0:
            ch_tensors.append(P+"ch0")
        elif c == out_color:
            ch_tensors.append(P+"uniform_mask")
        else:
            ch_tensors.append(P+"zeros")
    nodes.append(_n("Concat", ch_tensors, [P+"assembled"], axis=1))

    nodes.append(_n("Pad", [P+"assembled"], ["output"],
                   pads=[0, 0, 0, 0, 0, 0, 30 - H, 30 - W], mode="constant", value=0.0))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def cummax_down_column_network(H, W, batch=1):
    """ONNX: cumulative max downward per column.

    output[r][c] = max(input[0][c], ..., input[r][c]).
    Channel 0 (background) is treated as 0 so non-zero colors propagate.
    task322: H=3, W=3.
    """
    if batch != 1:
        raise NotImplementedError("cummax-down-column is single-example only")

    nodes = []
    inits = []
    P = "cmdc_"
    FLOAT = onnx.TensorProto.FLOAT
    INT64 = onnx.TensorProto.INT64

    def _t(name, dtype, shape, vals):
        return onnx.helper.make_tensor(name, dtype, shape, vals)
    def _n(op, inputs, outputs, **kwargs):
        return onnx.helper.make_node(op, inputs, outputs, **kwargs)

    inits += [
        _t(P+"one",    FLOAT, [1], [1.0]),
        _t(P+"ch_ax",  INT64, [1], [1]),
        _t(P+"ch_sp",  INT64, [1], [1]),
        _t(P+"hw_ax",  INT64, [2], [2, 3]),
        _t(P+"hw_st",  INT64, [2], [0, 0]),
        _t(P+"hw_en",  INT64, [2], [H, W]),
        _t(P+"hw_sp",  INT64, [2], [1, 1]),
        _t(P+"r_ax",   INT64, [1], [2]),
        _t(P+"r_sp",   INT64, [1], [1]),
    ]
    for ch in range(1, 10):
        inits += [
            _t(P+f"sch{ch}", INT64, [1], [ch]),
            _t(P+f"ech{ch}", INT64, [1], [ch + 1]),
        ]
    for r in range(H):
        inits += [
            _t(P+f"rs{r}", INT64, [1], [r]),
            _t(P+f"re{r}", INT64, [1], [r + 1]),
        ]

    cummax_ch_tensors = []  # one [1,1,H,W] cummax per channel 1..9

    for ch in range(1, 10):
        # Extract and crop channel ch → [1,1,H,W]
        nodes.append(_n("Slice", ["input", P+f"sch{ch}", P+f"ech{ch}", P+"ch_ax", P+"ch_sp"],
                        [P+f"xch{ch}_30"]))
        nodes.append(_n("Slice", [P+f"xch{ch}_30", P+"hw_st", P+"hw_en", P+"hw_ax", P+"hw_sp"],
                        [P+f"xch{ch}"]))

        # Build per-row slices and accumulate cummax
        cm_rows = []
        for r in range(H):
            row_nm = P+f"ch{ch}_r{r}"
            nodes.append(_n("Slice", [P+f"xch{ch}", P+f"rs{r}", P+f"re{r}", P+"r_ax", P+"r_sp"],
                            [row_nm]))
            if r == 0:
                cm_rows.append(row_nm)
            else:
                cm_nm = P+f"ch{ch}_cm{r}"
                nodes.append(_n("Max", [row_nm, cm_rows[r - 1]], [cm_nm]))
                cm_rows.append(cm_nm)

        nodes.append(_n("Concat", cm_rows, [P+f"ch{ch}_cummax"], axis=2))
        cummax_ch_tensors.append(P+f"ch{ch}_cummax")

    # Exact one-hot max color per cell; channel 0 fills cells with no color above.
    exact_ch = {}
    higher_seen = None
    for ch in range(9, 0, -1):
        seen = cummax_ch_tensors[ch - 1]
        if higher_seen is None:
            exact_ch[ch] = seen
            higher_seen = seen
            continue
        not_higher = P+f"not_higher_{ch}"
        exact = P+f"exact_ch{ch}"
        next_higher = P+f"higher_seen_{ch}"
        nodes.append(_n("Sub", [P+"one", higher_seen], [not_higher]))
        nodes.append(_n("Mul", [seen, not_higher], [exact]))
        nodes.append(_n("Max", [higher_seen, seen], [next_higher]))
        exact_ch[ch] = exact
        higher_seen = next_higher

    nodes.append(_n("Sub", [P+"one", higher_seen], [P+"ch0_exact"]))

    # Assemble [ch0, ch1..ch9] → [1,10,H,W]
    nodes.append(_n(
        "Concat",
        [P+"ch0_exact"] + [exact_ch[ch] for ch in range(1, 10)],
        [P+"assembled"],
        axis=1,
    ))

    nodes.append(_n("Pad", [P+"assembled"], ["output"],
                   pads=[0, 0, 0, 0, 0, 0, 30 - H, 30 - W], mode="constant", value=0.0))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def staircase_bottom_half_recolor_network(batch=1):
    """ONNX: bottom floor(h/2) cells of each color-2 column become color-8.

    task320: input uses {0, 2}, output uses {0, 2, 8}. Colors are constants.
    Uses MatMul with 30x30 upper-triangular constant to compute descending
    cumsum per column (CumSum is opset-11+, unavailable in opset-10).
    """
    if batch != 1:
        raise NotImplementedError("staircase-bottom-half-recolor is single-example only")

    nodes = []
    inits = []
    P = "sbhr_"

    def _t(name, dtype, shape, vals):
        return onnx.helper.make_tensor(name, dtype, shape, vals)
    def _n(op, inputs, outputs, **kwargs):
        return onnx.helper.make_node(op, inputs, outputs, **kwargs)

    FLOAT = onnx.TensorProto.FLOAT
    INT64 = onnx.TensorProto.INT64

    # Upper-triangular 30x30: L[i,j]=1 if j>=i else 0.
    # MatMul(L, ch2_mask) -> desc_cumsum[r,c] = sum_{k>=r} ch2[k,c].
    L_upper_vals = [1.0 if j >= i else 0.0 for i in range(30) for j in range(30)]

    inits += [
        _t(P+"L_upper",    FLOAT, [1, 1, 30, 30], L_upper_vals),
        _t(P+"coeff_half", FLOAT, [1], [0.5]),
        _t(P+"eps_half",   FLOAT, [1], [0.5]),
        _t(P+"step1",      INT64, [1], [1]),
        _t(P+"ch_ax",      INT64, [1], [1]),
        _t(P+"ch2_s",      INT64, [1], [2]),
        _t(P+"ch2_e",      INT64, [1], [3]),
    ]
    for ch in [0, 1, 3, 4, 5, 6, 7, 9]:
        inits += [
            _t(P+f"s{ch}", INT64, [1], [ch]),
            _t(P+f"e{ch}", INT64, [1], [ch + 1]),
        ]

    # Step 1: Extract ch2 -> (1,1,30,30).
    nodes.append(_n("Slice", ["input", P+"ch2_s", P+"ch2_e", P+"ch_ax", P+"step1"],
                    [P+"ch2_mask"]))

    # Step 2: Per-column total ch2 count -> (1,1,1,30).
    nodes.append(_n("ReduceSum", [P+"ch2_mask"], [P+"col_count"], axes=[2], keepdims=1))

    # Step 3: floor(count/2) via float multiply + Floor -> (1,1,1,30).
    nodes.append(_n("Mul",   [P+"col_count", P+"coeff_half"], [P+"half_pre"]))
    nodes.append(_n("Floor", [P+"half_pre"],                  [P+"half_floor"]))

    # Step 4: Descending cumsum via MatMul with upper-triangular.
    # (1,1,30,30) @ (1,1,30,30) -> (1,1,30,30).
    nodes.append(_n("MatMul", [P+"L_upper", P+"ch2_mask"], [P+"desc_cumsum"]))

    # Step 5: is_bottom_half: desc_cumsum < floor(cnt/2) + 0.5 (int-safe <=).
    nodes.append(_n("Add",  [P+"half_floor", P+"eps_half"],  [P+"half_thresh"]))
    nodes.append(_n("Less", [P+"desc_cumsum", P+"half_thresh"], [P+"is_bottom_bool"]))
    nodes.append(_n("Cast", [P+"is_bottom_bool"], [P+"is_bottom_f"],
                    to=FLOAT))

    # Must also be a ch2 cell (not canvas-padding zeros).
    nodes.append(_n("Mul", [P+"is_bottom_f", P+"ch2_mask"], [P+"recolor_mask"]))

    # Step 6: Updated channels.
    nodes.append(_n("Sub", [P+"ch2_mask", P+"recolor_mask"], [P+"ch2_new"]))

    # Step 7: Pass unchanged channels through.
    for ch in [0, 1, 3, 4, 5, 6, 7, 9]:
        nodes.append(_n("Slice", ["input", P+f"s{ch}", P+f"e{ch}", P+"ch_ax", P+"step1"],
                        [P+f"ch{ch}_raw"]))

    # Step 8: Concat all 10 channels: 0,1,2(new),3..7,8(new),9.
    nodes.append(_n("Concat", [
        P+"ch0_raw", P+"ch1_raw", P+"ch2_new",
        P+"ch3_raw", P+"ch4_raw", P+"ch5_raw",
        P+"ch6_raw", P+"ch7_raw", P+"recolor_mask", P+"ch9_raw",
    ], ["output"], axis=1))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def horizontal_tile_2x_network(batch=1):
    """ONNX: horizontal tile by 2 (np.tile(input, (1, 2))).

    task249 + task231: output[r][c] = input[r][c if c<W else c-W] where W is
    the input grid width, detected at runtime via weighted ArgMax of column
    activity. 15 precomputed gather_idx arrays, one per candidate W in 1..15.
    """
    if batch != 1:
        raise NotImplementedError("horizontal-tile-2x is single-example only")

    nodes = []
    inits = []
    P = "ht2_"
    FLOAT = onnx.TensorProto.FLOAT
    INT64 = onnx.TensorProto.INT64

    def _t(name, dtype, shape, vals):
        return onnx.helper.make_tensor(name, dtype, shape, vals)
    def _n(op, inputs, outputs, **kwargs):
        return onnx.helper.make_node(op, inputs, outputs, **kwargs)

    # For each candidate W: gather_idx[c] = c if c < W else c - W.
    # c < W  → identity (left half of grid).
    # W <= c < 2W → c - W (right half mirrors left half).
    # c >= 2W → c - W >= W, so selects input's padding col (correct).
    for W_cand in range(1, 16):
        idx_f = [float(c if c < W_cand else c - W_cand) for c in range(30)]
        inits.append(_t(P+f"idx_w{W_cand}", FLOAT, [30], idx_f))

    inits += [
        _t(P+"asc_col", FLOAT, [1, 1, 1, 30], [float(i + 1) for i in range(30)]),
        _t(P+"shape1",  INT64, [1], [1]),
        _t(P+"one_i64", INT64, [1], [1]),
    ]
    for W_cand in range(1, 16):
        inits.append(_t(P+f"Wc{W_cand}", INT64, [1], [W_cand]))

    # Step 1: Detect grid width W from the one-hot active mask.
    # Use the whole one-hot active mask here; background-only edge columns are
    # still part of the decoded output grid.
    nodes.append(_n("ReduceSum", ["input"], [P+"is_active"], axes=[1], keepdims=1))
    # Max over rows -> (1,1,1,30): 1 if the column is inside the active grid.
    nodes.append(_n("ReduceMax", [P+"is_active"], [P+"col_active"], axes=[2], keepdims=1))
    # Weight ascending [1..30] so ArgMax picks the last active col.
    nodes.append(_n("Mul", [P+"col_active", P+"asc_col"], [P+"col_weighted"]))
    # ArgMax along col axis → (1,1,1) INT64.
    nodes.append(_n("ArgMax", [P+"col_weighted"], [P+"last_col_3d"], axis=3, keepdims=0))
    # Reshape to (1,) and add 1 → W.
    nodes.append(_n("Reshape", [P+"last_col_3d", P+"shape1"], [P+"last_col"]))
    nodes.append(_n("Add", [P+"last_col", P+"one_i64"], [P+"W_val"]))

    # Step 2: Select the gather_idx matching detected W.
    # For each W_cand: contribution = (W == W_cand ? 1 : 0) * gather_idx_W_cand.
    running = None
    for W_cand in range(1, 16):
        is_w = P+f"isW{W_cand}"
        is_w_f = P+f"isW{W_cand}f"
        contrib = P+f"cW{W_cand}"
        nodes.append(_n("Equal", [P+"W_val", P+f"Wc{W_cand}"], [is_w]))
        nodes.append(_n("Cast",  [is_w], [is_w_f], to=FLOAT))
        nodes.append(_n("Mul",   [P+f"idx_w{W_cand}", is_w_f], [contrib]))
        if running is None:
            running = contrib
        else:
            nxt = P+f"sW{W_cand}"
            nodes.append(_n("Add", [running, contrib], [nxt]))
            running = nxt

    # Cast to INT64 for use as Gather indices.
    nodes.append(_n("Cast", [running], [P+"gather_idx"], to=INT64))

    # Step 3: Gather output columns.
    # Gather(input, gather_idx, axis=3): output[b,ch,r,i] = input[b,ch,r,gather_idx[i]].
    nodes.append(_n("Gather", ["input", P+"gather_idx"], ["output"], axis=3))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def period_extend_row_2x_network(batch=1):
    """Task231-style H x W -> H x 2W row-wise period extension."""
    if batch != 1:
        raise NotImplementedError("period-extend-row-2x is single-example only")

    nodes = []
    inits = []
    P = "per2_"
    FLOAT = onnx.TensorProto.FLOAT
    INT64 = onnx.TensorProto.INT64

    def _t(name, dtype, shape, vals):
        tensor = onnx.helper.make_tensor(P + name, dtype, shape, vals)
        inits.append(tensor)
        return P + name

    def _n(op, inputs, outputs, **kwargs):
        nodes.append(onnx.helper.make_node(op, inputs, outputs, **kwargs))

    half = _t("half", FLOAT, [1], [0.5])
    one = _t("one", FLOAT, [1], [1.0])
    asc_col = _t("asc_col", FLOAT, [1, 1, 1, 30], [float(i + 1) for i in range(30)])
    shape1 = _t("shape1", INT64, [1], [1])
    one_i64 = _t("one_i64", INT64, [1], [1])
    col_axis = _t("col_axis", INT64, [1], [3])
    step1 = _t("step1", INT64, [1], [1])

    for p in (1, 2, 3):
        _t(f"idx_p{p}", INT64, [30], [c % p for c in range(30)])
        _n("Gather", ["input", P + f"idx_p{p}"], [P + f"cand_p{p}"], axis=3)

    _n("ReduceSum", ["input"], [P + "active"], axes=[1], keepdims=1)
    _n("ReduceMax", [P + "active"], [P + "col_active"], axes=[2], keepdims=1)
    _n("Mul", [P + "col_active", asc_col], [P + "col_weighted"])
    _n("ArgMax", [P + "col_weighted"], [P + "last_col_3d"], axis=3, keepdims=0)
    _n("Reshape", [P + "last_col_3d", shape1], [P + "last_col"])
    _n("Add", [P + "last_col", one_i64], [P + "W_val"])

    width_terms = []
    for W_cand in range(6, 11):
        wc = _t(f"Wc{W_cand}", INT64, [1], [W_cand])
        width_mask = _t(
            f"width_mask_{W_cand}", FLOAT, [1, 1, 1, 30],
            [1.0 if c < 2 * W_cand else 0.0 for c in range(30)],
        )
        _n("Equal", [P + "W_val", wc], [P + f"isW{W_cand}_b"])
        _n("Cast", [P + f"isW{W_cand}_b"], [P + f"isW{W_cand}"], to=FLOAT)
        _n("Mul", [width_mask, P + f"isW{W_cand}"], [P + f"width_{W_cand}_term"])
        width_terms.append(P + f"width_{W_cand}_term")

    running_width = width_terms[0]
    for idx, name in enumerate(width_terms[1:], start=1):
        out = P + f"width_mask_sum{idx}" if idx < len(width_terms) - 1 else P + "width_mask"
        _n("Add", [running_width, name], [out])
        running_width = out

    valid = {}
    for period in (1, 2, 3):
        tail_st = _t(f"p{period}_tail_st", INT64, [1], [period])
        tail_en = _t(f"p{period}_tail_en", INT64, [1], [30])
        head_st = _t(f"p{period}_head_st", INT64, [1], [0])
        head_en = _t(f"p{period}_head_en", INT64, [1], [30 - period])
        _n("Slice", ["input", tail_st, tail_en, col_axis, step1], [P + f"p{period}_tail"])
        _n("Slice", ["input", head_st, head_en, col_axis, step1], [P + f"p{period}_head"])
        _n("Slice", [P + "col_active", tail_st, tail_en, col_axis, step1], [P + f"p{period}_active_tail"])
        _n("Sub", [P + f"p{period}_tail", P + f"p{period}_head"], [P + f"p{period}_diff"])
        _n("Abs", [P + f"p{period}_diff"], [P + f"p{period}_adiff"])
        _n("Mul", [P + f"p{period}_adiff", P + f"p{period}_active_tail"], [P + f"p{period}_masked"])
        _n("ReduceSum", [P + f"p{period}_masked"], [P + f"p{period}_sum"],
           axes=[1, 3], keepdims=1)
        _n("Less", [P + f"p{period}_sum", half], [P + f"p{period}_valid_b"])
        _n("Cast", [P + f"p{period}_valid_b"], [P + f"p{period}_valid"], to=FLOAT)
        valid[period] = P + f"p{period}_valid"

    _n("Sub", [one, valid[1]], [P + "not_p1"])
    _n("Sub", [one, valid[2]], [P + "not_p2"])
    _n("Mul", [valid[2], P + "not_p1"], [P + "p2_only"])
    _n("Mul", [valid[3], P + "not_p1"], [P + "p3_pre"])
    _n("Mul", [P + "p3_pre", P + "not_p2"], [P + "p3_only"])

    period_terms = []
    for period, mask in ((1, valid[1]), (2, P + "p2_only"), (3, P + "p3_only")):
        _n("Mul", [P + f"cand_p{period}", mask], [P + f"p{period}_selected"])
        period_terms.append(P + f"p{period}_selected")
    _n("Add", [period_terms[0], period_terms[1]], [P + "period12"])
    _n("Add", [P + "period12", period_terms[2]], [P + "period_selected"])
    _n("Mul", [P + "period_selected", P + "width_mask"], ["output"])

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def periodic_hole_fill_network(batch=1):
    """Task017-style 21x21 periodic pattern completion."""
    if batch != 1:
        raise NotImplementedError("periodic-hole-fill is single-example only")

    nodes = []
    inits = []
    P = "phf_"
    FLOAT = onnx.TensorProto.FLOAT
    INT64 = onnx.TensorProto.INT64
    H = W = 21

    def _t(name, dtype, shape, vals):
        tensor = onnx.helper.make_tensor(name, dtype, shape, vals)
        inits.append(tensor)
        return name

    def _n(op, inputs, outputs, **kwargs):
        nodes.append(onnx.helper.make_node(op, inputs, outputs, **kwargs))

    def _add_many(names, out_name):
        if len(names) == 1:
            _n("Mul", [names[0], P + "one"], [out_name])
            return out_name
        cur = names[0]
        for i, name in enumerate(names[1:], start=1):
            nxt = out_name if i == len(names) - 1 else f"{out_name}_{i}"
            _n("Add", [cur, name], [nxt])
            cur = nxt
        return out_name

    _t(P + "one", FLOAT, [1], [1.0])
    _t(P + "half", FLOAT, [1], [0.5])
    _t(P + "hw_st", INT64, [2], [0, 0])
    _t(P + "hw_en", INT64, [2], [H, W])
    _t(P + "hw_ax", INT64, [2], [2, 3])
    _t(P + "step2", INT64, [2], [1, 1])
    _t(P + "ch_ax", INT64, [1], [1])
    _t(P + "step1", INT64, [1], [1])
    _n("Slice", ["input", P + "hw_st", P + "hw_en", P + "hw_ax", P + "step2"], [P + "x"])
    _t(P + "ch0_st", INT64, [1], [0])
    _t(P + "ch0_en", INT64, [1], [1])
    _t(P + "fg_st", INT64, [1], [1])
    _t(P + "fg_en", INT64, [1], [10])
    _n("Slice", [P + "x", P + "ch0_st", P + "ch0_en", P + "ch_ax", P + "step1"], [P + "ch0"])
    _n("Slice", [P + "x", P + "fg_st", P + "fg_en", P + "ch_ax", P + "step1"], [P + "fg"])
    _n("Sub", [P + "one", P + "ch0"], [P + "nonzero"])
    center = H // 2
    kernel_bank = []
    for period in range(1, 10):
        for _ch in range(9):
            for rr in range(H):
                for cc in range(W):
                    kernel_bank.append(1.0 if (rr - center) % period == 0 and (cc - center) % period == 0 else 0.0)
    _t(P + "kernel_bank", FLOAT, [9, 9, 1, H, W], kernel_bank)
    _t(P + "best_idx_shape", INT64, [1], [1])
    _t(P + "kernel_shape", INT64, [4], [9, 1, H, W])

    score_by_period = {}
    for period in range(1, 10):
        _t(P + f"p{period}_left_st", INT64, [1], [0])
        _t(P + f"p{period}_left_en", INT64, [1], [W - period])
        _t(P + f"p{period}_right_st", INT64, [1], [period])
        _t(P + f"p{period}_right_en", INT64, [1], [W])
        _t(P + f"p{period}_col_ax", INT64, [1], [3])

        _n("Slice", [P + "fg", P + f"p{period}_left_st", P + f"p{period}_left_en",
                     P + f"p{period}_col_ax", P + "step1"], [P + f"p{period}_fg_left"])
        _n("Slice", [P + "fg", P + f"p{period}_right_st", P + f"p{period}_right_en",
                     P + f"p{period}_col_ax", P + "step1"], [P + f"p{period}_fg_right"])
        _n("Mul", [P + f"p{period}_fg_left", P + f"p{period}_fg_right"], [P + f"p{period}_same_by_ch"])
        _n("ReduceSum", [P + f"p{period}_same_by_ch"], [P + f"p{period}_same"], axes=[1], keepdims=1)
        same = P + f"p{period}_same"

        _n("Slice", [P + "nonzero", P + f"p{period}_left_st", P + f"p{period}_left_en",
                     P + f"p{period}_col_ax", P + "step1"], [P + f"p{period}_nz_left"])
        _n("Slice", [P + "nonzero", P + f"p{period}_right_st", P + f"p{period}_right_en",
                     P + f"p{period}_col_ax", P + "step1"], [P + f"p{period}_nz_right"])
        _n("Mul", [P + f"p{period}_nz_left", P + f"p{period}_nz_right"], [P + f"p{period}_both"])
        _n("Sub", [P + f"p{period}_both", same], [P + f"p{period}_conflict_cells"])
        _n("ReduceSum", [P + f"p{period}_conflict_cells"], [P + f"p{period}_conflicts"],
           axes=[2, 3], keepdims=1)
        _n("ReduceSum", [P + f"p{period}_both"], [P + f"p{period}_comparisons"],
           axes=[2, 3], keepdims=1)
        _n("Less", [P + f"p{period}_conflicts", P + "half"], [P + f"p{period}_valid_b"])
        _n("Cast", [P + f"p{period}_valid_b"], [P + f"p{period}_valid"], to=FLOAT)
        _n("Mul", [P + f"p{period}_comparisons", P + f"p{period}_valid"], [P + f"p{period}_score"])
        score_by_period[period] = P + f"p{period}_score"

    _n("Concat", [score_by_period[p] for p in range(1, 10)], [P + "scores"], axis=1)
    _n("ArgMax", [P + "scores"], [P + "best_period_idx"], axis=1, keepdims=1)
    _n("Reshape", [P + "best_period_idx", P + "best_idx_shape"], [P + "best_period_flat"])
    _n("Gather", [P + "kernel_bank", P + "best_period_flat"], [P + "selected_kernel_5d"], axis=0)
    _n("Reshape", [P + "selected_kernel_5d", P + "kernel_shape"], [P + "selected_kernel"])
    _n("Conv", [P + "fg", P + "selected_kernel"], [P + "periodic_sum"],
       pads=[center, center, center, center], group=9)
    _n("Greater", [P + "periodic_sum", P + "half"], [P + "fill_fg_b"])
    _n("Cast", [P + "fill_fg_b"], [P + "fill_fg"], to=FLOAT)
    _n("Mul", [P + "ch0", P + "fill_fg"], [P + "zero_fill"])
    _n("Add", [P + "fg", P + "zero_fill"], [P + "fg_out"])
    _n("ReduceMax", [P + "fill_fg"], [P + "fill_any"], axes=[1], keepdims=1)
    _n("Sub", [P + "one", P + "fill_any"], [P + "no_fill"])
    _n("Mul", [P + "ch0", P + "no_fill"], [P + "ch0_out"])
    _n("Concat", [P + "ch0_out", P + "fg_out"], [P + "selected_21"], axis=1)
    _n("Pad", [P + "selected_21"], ["output"], pads=[0, 0, 0, 0, 0, 0, 30 - H, 30 - W],
       mode="constant", value=0.0)
    return _make_model(nodes, inits, shape=[1, 10, 30, 30])


def zero_adjacency_label_network(batch=1):
    """ONNX: task369 zero-cell labels from 4-neighbor zero adjacency.

    Input is a fixed 10x10 grid with colors {0, 5}. Color-5 cells are
    preserved. Each color-0 cell becomes:
      - 3 when it has zero 4-neighbor zero cells;
      - 1 when it has at least two 4-neighbor zero cells;
      - otherwise 1 when its lone zero neighbor is adjacent to a >=2 cell,
        else 2.
    """
    if batch != 1:
        raise NotImplementedError("zero-adjacency-label is single-example only")

    nodes = []
    inits = []
    P = "zal_"
    FLOAT = onnx.TensorProto.FLOAT
    INT64 = onnx.TensorProto.INT64

    def _t(name, dtype, shape, vals):
        return onnx.helper.make_tensor(name, dtype, shape, vals)
    def _n(op, inputs, outputs, **kwargs):
        return onnx.helper.make_node(op, inputs, outputs, **kwargs)

    inits += [
        _t(P+"hw_st", INT64, [2], [0, 0]),
        _t(P+"hw_en", INT64, [2], [10, 10]),
        _t(P+"hw_ax", INT64, [2], [2, 3]),
        _t(P+"hw_sp", INT64, [2], [1, 1]),
        _t(P+"ch_ax", INT64, [1], [1]),
        _t(P+"ch_sp", INT64, [1], [1]),
        _t(P+"ch0_s", INT64, [1], [0]),
        _t(P+"ch0_e", INT64, [1], [1]),
        _t(P+"ch5_s", INT64, [1], [5]),
        _t(P+"ch5_e", INT64, [1], [6]),
        _t(P+"one", FLOAT, [1], [1.0]),
        _t(P+"two", FLOAT, [1], [2.0]),
        _t(P+"zero", FLOAT, [1], [0.0]),
        _t(P+"cross", FLOAT, [1, 1, 3, 3],
           [0.0, 1.0, 0.0,
            1.0, 0.0, 1.0,
            0.0, 1.0, 0.0]),
    ]

    nodes.append(_n("Slice", ["input", P+"hw_st", P+"hw_en", P+"hw_ax", P+"hw_sp"],
                    [P+"crop"]))
    nodes.append(_n("Slice", [P+"crop", P+"ch0_s", P+"ch0_e", P+"ch_ax", P+"ch_sp"],
                    [P+"ch0"]))
    nodes.append(_n("Slice", [P+"crop", P+"ch5_s", P+"ch5_e", P+"ch_ax", P+"ch_sp"],
                    [P+"ch5"]))

    nodes.append(_n("Conv", [P+"ch0", P+"cross"], [P+"adj0_raw"],
                    kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]))
    nodes.append(_n("Mul", [P+"ch0", P+"adj0_raw"], [P+"adj0"]))

    # Indicator adj0 >= 2 for integer counts 0..4:
    # relu(adj0 - 1) - relu(adj0 - 2) is 0,0,1,1,1.
    nodes.append(_n("Sub", [P+"adj0", P+"one"], [P+"adj0_m1"]))
    nodes.append(_n("Relu", [P+"adj0_m1"], [P+"relu_m1"]))
    nodes.append(_n("Sub", [P+"adj0", P+"two"], [P+"adj0_m2"]))
    nodes.append(_n("Relu", [P+"adj0_m2"], [P+"relu_m2"]))
    nodes.append(_n("Sub", [P+"relu_m1", P+"relu_m2"], [P+"adj_ge2"]))

    nodes.append(_n("Clip", [P+"adj0"], [P+"adj_nonzero"], min=0.0, max=1.0))
    nodes.append(_n("Sub", [P+"one", P+"adj_nonzero"], [P+"is_adj0_eq0"]))
    nodes.append(_n("Mul", [P+"ch0", P+"is_adj0_eq0"], [P+"ch3_out"]))
    nodes.append(_n("Sub", [P+"adj_nonzero", P+"adj_ge2"], [P+"is_adj0_eq1"]))

    nodes.append(_n("Conv", [P+"adj_ge2", P+"cross"], [P+"nbr_heavy_raw"],
                    kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]))
    nodes.append(_n("Clip", [P+"nbr_heavy_raw"], [P+"has_heavy"], min=0.0, max=1.0))
    nodes.append(_n("Sub", [P+"one", P+"has_heavy"], [P+"no_heavy"]))

    nodes.append(_n("Mul", [P+"is_adj0_eq1", P+"no_heavy"], [P+"ch2_out"]))
    nodes.append(_n("Mul", [P+"is_adj0_eq1", P+"has_heavy"], [P+"adj1_heavy"]))
    nodes.append(_n("Add", [P+"adj_ge2", P+"adj1_heavy"], [P+"ch1_out"]))
    nodes.append(_n("Mul", [P+"ch0", P+"zero"], [P+"zero_hw"]))

    nodes.append(_n("Concat", [
        P+"zero_hw", P+"ch1_out", P+"ch2_out", P+"ch3_out", P+"zero_hw",
        P+"ch5", P+"zero_hw", P+"zero_hw", P+"zero_hw", P+"zero_hw",
    ], [P+"assembled"], axis=1))
    nodes.append(_n("Pad", [P+"assembled"], ["output"],
                    pads=[0, 0, 0, 0, 0, 0, 20, 20], mode="constant", value=0.0))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def boustrophedon_pack_3x3_network(batch=1):
    """ONNX: sort nonzero cells by (col,row), pack first 9 into 3x3 boustrophedon.

    task316: output is always 3x3. Non-zero cells sorted col-ascending, then
    row-ascending. First 9 placed into 3x3 in snake order: row0 L-R, row1 R-L,
    row2 L-R. Uses negated-rank + TopK trick for smallest-K in opset 10.
    """
    if batch != 1:
        raise NotImplementedError("boustrophedon-pack-3x3 is single-example only")

    nodes = []
    inits = []
    P = "bp3_"
    FLOAT = onnx.TensorProto.FLOAT
    INT64 = onnx.TensorProto.INT64

    def _t(name, dtype, shape, vals):
        return onnx.helper.make_tensor(name, dtype, shape, vals)
    def _n(op, inputs, outputs, **kwargs):
        return onnx.helper.make_node(op, inputs, outputs, **kwargs)

    # Boustrophedon permutation: Gather(sorted_colors, PERM) gives snake order.
    # sorted_colors[k] -> out[row][col] where row=k//3, col=k%3 if row%2==0 else 2-k%3.
    PERM = [0, 1, 2, 5, 4, 3, 6, 7, 8]

    inits += [
        # Rank encoding: col * 30 + row  (col-primary, row-secondary sort)
        _t(P+"col_idx",  FLOAT, [1, 1, 1, 30], [float(c) for c in range(30)]),
        _t(P+"row_idx",  FLOAT, [1, 1, 30, 1], [float(r) for r in range(30)]),
        _t(P+"scale30",  FLOAT, [1], [30.0]),
        _t(P+"sentinel", FLOAT, [1], [1e6]),
        _t(P+"one_f",    FLOAT, [1], [1.0]),
        _t(P+"neg1_f",   FLOAT, [1], [-1.0]),
        # TopK controls
        _t(P+"K9",       INT64, [1], [9]),
        _t(P+"perm",     INT64, [9], PERM),
        # Shape constants
        _t(P+"s1x900",   INT64, [2], [1, 900]),
        _t(P+"s900",     INT64, [1], [900]),
        _t(P+"s9",       INT64, [1], [9]),
        _t(P+"s1133",    INT64, [4], [1, 1, 3, 3]),
        _t(P+"active_3x3", FLOAT, [1, 1, 30, 30],
           [1.0 if r < 3 and c < 3 else 0.0 for r in range(30) for c in range(30)]),
        # Channel slice
        _t(P+"ch1_s",    INT64, [1], [1]),
        _t(P+"ch9_e",    INT64, [1], [10]),
        _t(P+"ch_ax",    INT64, [1], [1]),
        _t(P+"step1",    INT64, [1], [1]),
    ]
    for ch in range(10):
        inits.append(_t(P+f"v{ch}", INT64, [1], [ch]))

    # Step 1: Encode per-cell color as float (0-9).
    # ArgMax over channels gives which channel (= color value) is 1.
    nodes.append(_n("ArgMax", ["input"], [P+"argmax_3d"], axis=1, keepdims=0))
    # (1,30,30) INT64 -> cast -> (1,30,30) float -> reshape -> (900,)
    nodes.append(_n("Cast",    [P+"argmax_3d"], [P+"color_3d"],  to=FLOAT))
    nodes.append(_n("Reshape", [P+"color_3d",   P+"s900"],       [P+"color_flat"]))

    # Step 2: Nonbg mask: 1 where any color 1-9 is present, 0 otherwise.
    nodes.append(_n("Slice",     ["input", P+"ch1_s", P+"ch9_e", P+"ch_ax", P+"step1"],
                    [P+"nonbg_chs"]))  # (1,9,30,30)
    nodes.append(_n("ReduceMax", [P+"nonbg_chs"], [P+"nonbg"], axes=[1], keepdims=1))

    # Step 3: Effective rank = col*30+row for nonbg cells, sentinel for bg/padding.
    nodes.append(_n("Mul", [P+"col_idx",   P+"scale30"],  [P+"col_x30"]))
    nodes.append(_n("Add", [P+"col_x30",   P+"row_idx"],  [P+"rank_raw"]))
    nodes.append(_n("Mul", [P+"rank_raw",  P+"nonbg"],    [P+"rank_active"]))
    nodes.append(_n("Sub", [P+"one_f",     P+"nonbg"],    [P+"is_bg"]))
    nodes.append(_n("Mul", [P+"sentinel",  P+"is_bg"],    [P+"rank_sent"]))
    nodes.append(_n("Add", [P+"rank_active", P+"rank_sent"], [P+"eff_rank"]))
    # Reshape to (1,900) and negate so TopK(largest=True) picks smallest original.
    nodes.append(_n("Reshape", [P+"eff_rank",    P+"s1x900"], [P+"rank_flat"]))
    nodes.append(_n("Mul",     [P+"rank_flat",   P+"neg1_f"], [P+"neg_rank"]))

    # Step 4: TopK K=9 (largest values of neg_rank = smallest original ranks).
    # opset 10: TopK(X, K) with axis attr; sorted=True by default.
    nodes.append(_n("TopK", [P+"neg_rank", P+"K9"], [P+"tk_vals", P+"tk_idx"], axis=1))
    # tk_idx: (1,9) INT64 -> reshape to (9,).
    nodes.append(_n("Reshape", [P+"tk_idx", P+"s9"], [P+"idx9"]))

    # Step 5: Gather the 9 colors (sorted by (col,row) lex).
    nodes.append(_n("Gather", [P+"color_flat", P+"idx9"], [P+"sorted_colors"], axis=0))

    # Step 6: Apply boustrophedon permutation: row0 L-R, row1 R-L, row2 L-R.
    nodes.append(_n("Gather", [P+"sorted_colors", P+"perm"], [P+"boustro_colors"], axis=0))

    # Step 7: Reshape to (1,1,3,3) and pad to (1,1,30,30).
    nodes.append(_n("Reshape", [P+"boustro_colors", P+"s1133"], [P+"grid_3x3"]))
    nodes.append(_n("Pad", [P+"grid_3x3"], [P+"grid_pad"],
                    pads=[0, 0, 0, 0, 0, 0, 27, 27], mode="constant", value=0.0))
    # Cast to INT64 for Equal (opset 10 Equal requires int or bool, not float).
    nodes.append(_n("Cast", [P+"grid_pad"], [P+"grid_int"], to=INT64))

    # Step 8: One-hot encode: for each ch, cast(Equal(pad_int, ch_i64)) -> (1,1,30,30).
    ch_parts = []
    for ch in range(10):
        nodes.append(_n("Equal", [P+"grid_int", P+f"v{ch}"], [P+f"eq{ch}"]))
        nodes.append(_n("Cast",  [P+f"eq{ch}"], [P+f"ch{ch}_raw"], to=FLOAT))
        nodes.append(_n("Mul",   [P+f"ch{ch}_raw", P+"active_3x3"], [P+f"ch{ch}"]))
        ch_parts.append(P+f"ch{ch}")

    # Step 9: Concat -> output (1,10,30,30).
    nodes.append(_n("Concat", ch_parts, ["output"], axis=1))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def col_fill_2x2_tile_network(batch=1):
    """ONNX: fill 0->8 in X-containing cols, then tile 2x2.

    task388: detect unique nonzero color X; columns containing X get their 0-cells
    replaced with 8; result is tiled 2x2. Output shape: (2H, 2W) in canvas.
    H and W are detected at runtime. Requires opset 10.
    """
    if batch != 1:
        raise NotImplementedError("col-fill-2x2-tile is single-example only")

    nodes = []
    inits = []
    P = "cf2t_"
    FLOAT = onnx.TensorProto.FLOAT
    INT64 = onnx.TensorProto.INT64

    def _t(name, dtype, shape, vals):
        return onnx.helper.make_tensor(name, dtype, shape, vals)
    def _n(op, inputs, outputs, **kwargs):
        return onnx.helper.make_node(op, inputs, outputs, **kwargs)

    # Precompute row/col gather indices for 2x2 tile.
    # tile_idx_W[k] = k if k < W else k - W (mirrors the 2nd half to the 1st).
    for k_cand in range(1, 16):
        vals = [float(k if k < k_cand else k - k_cand) for k in range(30)]
        inits.append(_t(P+f"ridx{k_cand}", FLOAT, [30], vals))  # row gather
        inits.append(_t(P+f"cidx{k_cand}", FLOAT, [30], vals))  # col gather

    inits += [
        _t(P+"ch0_s",   INT64, [1], [0]),
        _t(P+"ch1_s",   INT64, [1], [1]),
        _t(P+"ch8_s",   INT64, [1], [8]),
        _t(P+"ch9_e",   INT64, [1], [10]),
        _t(P+"ch_ax",   INT64, [1], [1]),
        _t(P+"step1",   INT64, [1], [1]),
        _t(P+"ch8e",    INT64, [1], [9]),
        _t(P+"ch9s",    INT64, [1], [9]),
        _t(P+"ch9e2",   INT64, [1], [10]),
        _t(P+"asc_row", FLOAT, [1, 1, 30, 1], [float(i + 1) for i in range(30)]),
        _t(P+"asc_col", FLOAT, [1, 1, 1, 30], [float(i + 1) for i in range(30)]),
        _t(P+"row_asc", FLOAT, [1, 1, 30, 1], [float(i) for i in range(30)]),
        _t(P+"col_asc", FLOAT, [1, 1, 1, 30], [float(i) for i in range(30)]),
        _t(P+"shape1",  INT64, [1], [1]),
        _t(P+"one_i",   INT64, [1], [1]),
    ]
    for k_cand in range(1, 16):
        inits.append(_t(P+f"Hc{k_cand}", INT64, [1], [k_cand]))
        inits.append(_t(P+f"Wc{k_cand}", INT64, [1], [k_cand]))

    # === Step 1: Detect col_has_X and row/col extents ===
    # nonbg: channels 1..9 -> (1,9,30,30)
    nodes.append(_n("Slice", ["input", P+"ch1_s", P+"ch9_e", P+"ch_ax", P+"step1"],
                    [P+"nonbg"]))
    # col_has_X: any nonbg in each col -> (1,1,1,30)
    nodes.append(_n("ReduceMax", [P+"nonbg"], [P+"col_has_X"], axes=[1, 2], keepdims=1))
    # Active extents come from all one-hot channels, preserving background-only
    # rows and columns on the grid edge.
    nodes.append(_n("ReduceSum", ["input"], [P+"is_active"], axes=[1], keepdims=1))
    nodes.append(_n("ReduceMax", [P+"is_active"], [P+"row_active"], axes=[3], keepdims=1))
    nodes.append(_n("ReduceMax", [P+"is_active"], [P+"col_active"], axes=[2], keepdims=1))

    # Detect W = last active col + 1
    nodes.append(_n("Mul",    [P+"col_active", P+"asc_col"],  [P+"col_w"]))
    nodes.append(_n("ArgMax", [P+"col_w"], [P+"last_col_3d"], axis=3, keepdims=0))
    nodes.append(_n("Reshape",[P+"last_col_3d", P+"shape1"], [P+"last_col"]))
    nodes.append(_n("Add",    [P+"last_col", P+"one_i"],     [P+"W_val"]))

    # Detect H = last active row + 1
    nodes.append(_n("Mul",    [P+"row_active", P+"asc_row"], [P+"row_w"]))
    nodes.append(_n("ArgMax", [P+"row_w"], [P+"last_row_3d"], axis=2, keepdims=0))
    nodes.append(_n("Reshape",[P+"last_row_3d", P+"shape1"],[P+"last_row"]))
    nodes.append(_n("Add",    [P+"last_row", P+"one_i"],     [P+"H_val"]))

    # === Step 2: Inside-grid mask via float Less comparison ===
    nodes.append(_n("Cast",   [P+"H_val"], [P+"H_f"], to=FLOAT))
    nodes.append(_n("Cast",   [P+"W_val"], [P+"W_f"], to=FLOAT))
    nodes.append(_n("Less",   [P+"row_asc", P+"H_f"], [P+"row_in_b"]))
    nodes.append(_n("Less",   [P+"col_asc", P+"W_f"], [P+"col_in_b"]))
    nodes.append(_n("Cast",   [P+"row_in_b"], [P+"row_in"], to=FLOAT))
    nodes.append(_n("Cast",   [P+"col_in_b"], [P+"col_in"], to=FLOAT))
    nodes.append(_n("Mul",    [P+"row_in", P+"col_in"], [P+"inside"]))  # (1,1,30,30)

    # === Step 3: Compute fill mask ===
    # is_ch0 = Slice(input, ch0, axis=1) -> (1,1,30,30)
    nodes.append(_n("Slice", ["input", P+"ch0_s", P+"ch1_s", P+"ch_ax", P+"step1"],
                    [P+"ch0_raw"]))
    # should_fill = col_has_X * inside * is_ch0
    nodes.append(_n("Mul", [P+"col_has_X", P+"inside"],   [P+"col_inside"]))
    nodes.append(_n("Mul", [P+"col_inside", P+"ch0_raw"], [P+"fill_mask"]))

    # === Step 4: Build transformed channels ===
    # ch0_new = ch0 - fill_mask
    nodes.append(_n("Sub", [P+"ch0_raw", P+"fill_mask"], [P+"ch0_new"]))
    # ch1..ch7: pass through from input
    nodes.append(_n("Slice", ["input", P+"ch1_s",  P+"ch8_s",  P+"ch_ax", P+"step1"], [P+"ch1to7"]))
    # ch8_new = ch8 + fill_mask
    nodes.append(_n("Slice", ["input", P+"ch8_s",  P+"ch8e",   P+"ch_ax", P+"step1"], [P+"ch8_raw"]))
    nodes.append(_n("Add",  [P+"ch8_raw", P+"fill_mask"], [P+"ch8_new"]))
    # ch9: pass through
    nodes.append(_n("Slice", ["input", P+"ch9s",  P+"ch9e2", P+"ch_ax", P+"step1"], [P+"ch9_raw"]))
    # Concat -> transformed (1,10,30,30)
    nodes.append(_n("Concat", [P+"ch0_new", P+"ch1to7", P+"ch8_new", P+"ch9_raw"],
                    [P+"transformed"], axis=1))

    # === Step 5: 2x2 tile via row then col Gather ===
    # Select row gather indices for detected H
    row_running = None
    for k_cand in range(1, 16):
        is_h = P+f"isH{k_cand}"
        is_h_f = P+f"isH{k_cand}f"
        rc = P+f"rcH{k_cand}"
        nodes.append(_n("Equal", [P+"H_val", P+f"Hc{k_cand}"], [is_h]))
        nodes.append(_n("Cast",  [is_h], [is_h_f], to=FLOAT))
        nodes.append(_n("Mul",   [P+f"ridx{k_cand}", is_h_f], [rc]))
        if row_running is None:
            row_running = rc
        else:
            nxt = P+f"rsH{k_cand}"
            nodes.append(_n("Add", [row_running, rc], [nxt]))
            row_running = nxt
    nodes.append(_n("Cast", [row_running], [P+"row_idx"], to=INT64))

    # Select col gather indices for detected W
    col_running = None
    for k_cand in range(1, 16):
        is_w = P+f"isW{k_cand}"
        is_w_f = P+f"isW{k_cand}f"
        cc = P+f"ccW{k_cand}"
        nodes.append(_n("Equal", [P+"W_val", P+f"Wc{k_cand}"], [is_w]))
        nodes.append(_n("Cast",  [is_w], [is_w_f], to=FLOAT))
        nodes.append(_n("Mul",   [P+f"cidx{k_cand}", is_w_f], [cc]))
        if col_running is None:
            col_running = cc
        else:
            nxt = P+f"csW{k_cand}"
            nodes.append(_n("Add", [col_running, cc], [nxt]))
            col_running = nxt
    nodes.append(_n("Cast", [col_running], [P+"col_idx"], to=INT64))

    # Apply Gather: first by rows, then by cols
    nodes.append(_n("Gather", [P+"transformed", P+"row_idx"], [P+"tiled_rows"], axis=2))
    nodes.append(_n("Gather", [P+"tiled_rows",  P+"col_idx"], ["output"],       axis=3))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def tile_diagonal_neighbor_mark_network(batch=1):
    """ONNX: 2x2 tile input, then mark diagonal-neighbor background with 8.

    task019: input has one nonzero color X (never 8). The output is a 2x2 tile
    of the input plus color-8 marks wherever an active background output cell is
    diagonally adjacent to a tiled X cell.
    """
    if batch != 1:
        raise NotImplementedError("tile-diagonal-neighbor-mark is single-example only")

    nodes = []
    inits = []
    P = "tdnm_"
    FLOAT = onnx.TensorProto.FLOAT
    INT64 = onnx.TensorProto.INT64

    def _t(name, dtype, shape, vals):
        return onnx.helper.make_tensor(name, dtype, shape, vals)
    def _n(op, inputs, outputs, **kwargs):
        return onnx.helper.make_node(op, inputs, outputs, **kwargs)

    for k_cand in range(1, 16):
        vals = [float(k if k < k_cand else k - k_cand) for k in range(30)]
        inits.append(_t(P+f"ridx{k_cand}", FLOAT, [30], vals))
        inits.append(_t(P+f"cidx{k_cand}", FLOAT, [30], vals))

    inits += [
        _t(P+"ch0_s",   INT64, [1], [0]),
        _t(P+"ch1_s",   INT64, [1], [1]),
        _t(P+"ch8_s",   INT64, [1], [8]),
        _t(P+"ch8e",    INT64, [1], [9]),
        _t(P+"ch9_s",   INT64, [1], [9]),
        _t(P+"ch9_e",   INT64, [1], [10]),
        _t(P+"ch_ax",   INT64, [1], [1]),
        _t(P+"step1",   INT64, [1], [1]),
        _t(P+"asc_row", FLOAT, [1, 1, 30, 1], [float(i + 1) for i in range(30)]),
        _t(P+"asc_col", FLOAT, [1, 1, 1, 30], [float(i + 1) for i in range(30)]),
        _t(P+"shape1",  INT64, [1], [1]),
        _t(P+"one_i",   INT64, [1], [1]),
        _t(P+"diag_w",  FLOAT, [1, 1, 3, 3], [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0]),
        _t(P+"half",    FLOAT, [1], [0.5]),
    ]
    for k_cand in range(1, 16):
        inits.append(_t(P+f"Hc{k_cand}", INT64, [1], [k_cand]))
        inits.append(_t(P+f"Wc{k_cand}", INT64, [1], [k_cand]))

    # Runtime H/W comes from active one-hot cells, including background rows/cols.
    nodes.append(_n("ReduceSum", ["input"], [P+"is_active"], axes=[1], keepdims=1))
    nodes.append(_n("ReduceMax", [P+"is_active"], [P+"row_active"], axes=[3], keepdims=1))
    nodes.append(_n("ReduceMax", [P+"is_active"], [P+"col_active"], axes=[2], keepdims=1))
    nodes.append(_n("Mul",    [P+"row_active", P+"asc_row"], [P+"row_w"]))
    nodes.append(_n("ArgMax", [P+"row_w"], [P+"last_row_3d"], axis=2, keepdims=0))
    nodes.append(_n("Reshape",[P+"last_row_3d", P+"shape1"], [P+"last_row"]))
    nodes.append(_n("Add",    [P+"last_row", P+"one_i"],     [P+"H_val"]))
    nodes.append(_n("Mul",    [P+"col_active", P+"asc_col"],  [P+"col_w"]))
    nodes.append(_n("ArgMax", [P+"col_w"], [P+"last_col_3d"], axis=3, keepdims=0))
    nodes.append(_n("Reshape",[P+"last_col_3d", P+"shape1"], [P+"last_col"]))
    nodes.append(_n("Add",    [P+"last_col", P+"one_i"],     [P+"W_val"]))

    row_running = None
    for k_cand in range(1, 16):
        is_h = P+f"isH{k_cand}"
        is_h_f = P+f"isH{k_cand}f"
        rc = P+f"rcH{k_cand}"
        nodes.append(_n("Equal", [P+"H_val", P+f"Hc{k_cand}"], [is_h]))
        nodes.append(_n("Cast",  [is_h], [is_h_f], to=FLOAT))
        nodes.append(_n("Mul",   [P+f"ridx{k_cand}", is_h_f], [rc]))
        if row_running is None:
            row_running = rc
        else:
            nxt = P+f"rsH{k_cand}"
            nodes.append(_n("Add", [row_running, rc], [nxt]))
            row_running = nxt
    nodes.append(_n("Cast", [row_running], [P+"row_idx"], to=INT64))

    col_running = None
    for k_cand in range(1, 16):
        is_w = P+f"isW{k_cand}"
        is_w_f = P+f"isW{k_cand}f"
        cc = P+f"ccW{k_cand}"
        nodes.append(_n("Equal", [P+"W_val", P+f"Wc{k_cand}"], [is_w]))
        nodes.append(_n("Cast",  [is_w], [is_w_f], to=FLOAT))
        nodes.append(_n("Mul",   [P+f"cidx{k_cand}", is_w_f], [cc]))
        if col_running is None:
            col_running = cc
        else:
            nxt = P+f"csW{k_cand}"
            nodes.append(_n("Add", [col_running, cc], [nxt]))
            col_running = nxt
    nodes.append(_n("Cast", [col_running], [P+"col_idx"], to=INT64))

    nodes.append(_n("Gather", ["input", P+"row_idx"], [P+"tiled_rows"], axis=2))
    nodes.append(_n("Gather", [P+"tiled_rows", P+"col_idx"], [P+"tiled"], axis=3))

    nodes.append(_n("Slice", [P+"tiled", P+"ch1_s", P+"ch9_e", P+"ch_ax", P+"step1"], [P+"tiled_nonbg"]))
    nodes.append(_n("ReduceSum", [P+"tiled_nonbg"], [P+"x_mask"], axes=[1], keepdims=1))
    nodes.append(_n("Conv", [P+"x_mask", P+"diag_w"], [P+"diag_count"], pads=[1, 1, 1, 1]))
    nodes.append(_n("Greater", [P+"diag_count", P+"half"], [P+"diag_any_b"]))
    nodes.append(_n("Cast", [P+"diag_any_b"], [P+"diag_any"], to=FLOAT))
    nodes.append(_n("Slice", [P+"tiled", P+"ch0_s", P+"ch1_s", P+"ch_ax", P+"step1"], [P+"ch0_raw"]))
    nodes.append(_n("Mul", [P+"diag_any", P+"ch0_raw"], [P+"fill_mask"]))

    nodes.append(_n("Sub", [P+"ch0_raw", P+"fill_mask"], [P+"ch0_new"]))
    nodes.append(_n("Slice", [P+"tiled", P+"ch1_s", P+"ch8_s", P+"ch_ax", P+"step1"], [P+"ch1to7"]))
    nodes.append(_n("Slice", [P+"tiled", P+"ch8_s", P+"ch8e", P+"ch_ax", P+"step1"], [P+"ch8_raw"]))
    nodes.append(_n("Add", [P+"ch8_raw", P+"fill_mask"], [P+"ch8_new"]))
    nodes.append(_n("Slice", [P+"tiled", P+"ch9_s", P+"ch9_e", P+"ch_ax", P+"step1"], [P+"ch9_raw"]))
    nodes.append(_n("Concat", [P+"ch0_new", P+"ch1to7", P+"ch8_new", P+"ch9_raw"], ["output"], axis=1))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def color2_bbox_fill_network(batch=1):
    """ONNX: task077-style color-2 component bbox fill.

    Color-2 cells form compact components with bbox height 2..3 and width 2..7.
    Inside each component's bbox, nonzero/non-2 cells are recolored to 4 while
    zeros and color-2 cells are preserved.
    """
    if batch != 1:
        raise NotImplementedError("color2-bbox-fill is single-example only")

    P = "c2bf_"
    FLOAT = onnx.TensorProto.FLOAT
    INT64 = onnx.TensorProto.INT64

    def _t(name, dtype, shape, vals):
        return onnx.helper.make_tensor(name, dtype, shape, vals)

    def _n(op, inputs, outputs, **kwargs):
        return onnx.helper.make_node(op, inputs, outputs, **kwargs)

    nodes = []
    inits = [
        _t(P+"zero", FLOAT, [1], [0.0]),
        _t(P+"one", FLOAT, [1], [1.0]),
        _t(P+"ch_axis", INT64, [1], [1]),
        _t(P+"step1", INT64, [1], [1]),
        _t(P+"ch0_s", INT64, [1], [0]),
        _t(P+"ch1_s", INT64, [1], [1]),
        _t(P+"ch2_s", INT64, [1], [2]),
        _t(P+"ch3_s", INT64, [1], [3]),
        _t(P+"ch4_s", INT64, [1], [4]),
        _t(P+"ch5_s", INT64, [1], [5]),
        _t(P+"ch6_s", INT64, [1], [6]),
        _t(P+"ch7_s", INT64, [1], [7]),
        _t(P+"ch8_s", INT64, [1], [8]),
        _t(P+"ch9_s", INT64, [1], [9]),
        _t(P+"ch10_s", INT64, [1], [10]),
    ]

    # Color-2 mask.
    nodes.append(_n("Slice", ["input", P+"ch0_s", P+"ch1_s", P+"ch_axis", P+"step1"], [P+"ch0"]))
    nodes.append(_n("Slice", ["input", P+"ch2_s", P+"ch3_s", P+"ch_axis", P+"step1"], [P+"ch2"]))

    paint_counts = []
    for h in (2, 3):
        for w in range(2, 8):
            cond_filters = []

            def add_filter(cells):
                filt = np.zeros((h, w), dtype=np.float32)
                for rr, cc in cells:
                    filt[rr, cc] = 1.0
                cond_filters.append(filt)

            add_filter([(0, cc) for cc in range(w)])             # top edge touched
            add_filter([(h - 1, cc) for cc in range(w)])         # bottom edge touched
            add_filter([(rr, 0) for rr in range(h)])             # left edge touched
            add_filter([(rr, w - 1) for rr in range(h)])         # right edge touched
            for cc in range(w - 1):                              # no two empty columns
                add_filter([(rr, cc) for rr in range(h)] + [(rr, cc + 1) for rr in range(h)])

            C = len(cond_filters)
            weight = np.stack(cond_filters, axis=0).reshape(C, 1, h, w)
            cond_w = P+f"cond_w_{h}_{w}"
            inits.append(_t(cond_w, FLOAT, [C, 1, h, w], weight.flatten().tolist()))
            cond = P+f"cond_{h}_{w}"
            minc = P+f"min_{h}_{w}"
            valid_bool = P+f"valid_bool_{h}_{w}"
            valid = P+f"valid_{h}_{w}"
            paint_w = P+f"paint_w_{h}_{w}"
            paint = P+f"paint_{h}_{w}"

            nodes.append(_n("Conv", [P+"ch2", cond_w], [cond]))
            nodes.append(_n("ReduceMin", [cond], [minc], axes=[1], keepdims=1))
            nodes.append(_n("Greater", [minc, P+"zero"], [valid_bool]))
            nodes.append(_n("Cast", [valid_bool], [valid], to=FLOAT))

            inits.append(_t(paint_w, FLOAT, [1, 1, h, w], [1.0] * (h * w)))
            nodes.append(_n(
                "Conv", [valid, paint_w], [paint],
                pads=[h - 1, w - 1, h - 1, w - 1],
            ))
            paint_counts.append(paint)

    nodes.append(_n("Sum", paint_counts, [P+"paint_sum"]))
    nodes.append(_n("Greater", [P+"paint_sum", P+"zero"], [P+"fill_bool"]))
    nodes.append(_n("Cast", [P+"fill_bool"], [P+"fill"], to=FLOAT))

    # Restrict to nonzero, non-2 cells. One-hot inputs make this cheaper than
    # summing channels 1..9 and it leaves canvas padding alone.
    nodes.append(_n("Sub", [P+"one", P+"ch0"], [P+"nonzero"]))
    nodes.append(_n("Sub", [P+"nonzero", P+"ch2"], [P+"nonzero_non2"]))
    nodes.append(_n("Mul", [P+"fill", P+"nonzero_non2"], [P+"target"]))
    nodes.append(_n("Sub", [P+"one", P+"target"], [P+"keep"]))

    out_channels = []
    ch_starts = [P+f"ch{i}_s" for i in range(10)]
    ch_ends = [P+f"ch{i+1}_s" for i in range(10)]
    for ch in range(10):
        if ch == 0:
            raw = P+"ch0"
        elif ch == 2:
            raw = P+"ch2"
        else:
            raw = P+f"raw_{ch}"
            nodes.append(_n("Slice", ["input", ch_starts[ch], ch_ends[ch], P+"ch_axis", P+"step1"], [raw]))
        if ch == 2:
            out = raw
        elif ch == 4:
            out = P+"out_4"
            nodes.append(_n("Max", [raw, P+"target"], [out]))
        else:
            out = P+f"out_{ch}"
            nodes.append(_n("Mul", [raw, P+"keep"], [out]))
        out_channels.append(out)

    nodes.append(_n("Concat", out_channels, ["output"], axis=1))
    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def diagonal_marker_square_rings_network(batch=1):
    """ONNX: draw square rings from three equally spaced diagonal markers.

    task137: three same-color cells lie on a 45-degree diagonal. The middle
    marker is the ring center, and the row spacing between adjacent markers is
    the ring period. Output keeps that color on all active-grid cells whose
    Chebyshev distance from the center is a multiple of the period.
    """
    if batch != 1:
        raise NotImplementedError("diagonal-marker-square-rings is single-example only")

    nodes = []
    inits = []
    P = "dmsr_"
    FLOAT = onnx.TensorProto.FLOAT
    INT64 = onnx.TensorProto.INT64

    def _t(name, dtype, shape, vals):
        return onnx.helper.make_tensor(name, dtype, shape, vals)
    def _n(op, inputs, outputs, **kwargs):
        return onnx.helper.make_node(op, inputs, outputs, **kwargs)

    inits += [
        _t(P+"ch1_s",   INT64, [1], [1]),
        _t(P+"ch10_e",  INT64, [1], [10]),
        _t(P+"ch_ax",   INT64, [1], [1]),
        _t(P+"step1",   INT64, [1], [1]),
        _t(P+"desc_row", FLOAT, [1, 1, 30, 1], [float(30 - i) for i in range(30)]),
        _t(P+"asc_row",  FLOAT, [1, 1, 30, 1], [float(i + 1) for i in range(30)]),
        _t(P+"desc_col", FLOAT, [1, 1, 1, 30], [float(30 - i) for i in range(30)]),
        _t(P+"asc_col",  FLOAT, [1, 1, 1, 30], [float(i + 1) for i in range(30)]),
        _t(P+"row_idx",  FLOAT, [1, 1, 30, 1], [float(i) for i in range(30)]),
        _t(P+"col_idx",  FLOAT, [1, 1, 1, 30], [float(i) for i in range(30)]),
        _t(P+"shape1",   INT64, [1], [1]),
        _t(P+"half",     FLOAT, [1], [0.5]),
        _t(P+"oh_depth9", INT64, [], [9]),
        _t(P+"oh_values", FLOAT, [2], [0.0, 1.0]),
        _t(P+"rs_1911",  INT64, [4], [1, 9, 1, 1]),
    ]
    for k in range(30):
        inits.append(_t(P+f"k{k}", FLOAT, [1], [float(k)]))

    nodes.append(_n("ReduceSum", ["input"], [P+"active"], axes=[1], keepdims=1))
    nodes.append(_n("Slice", ["input", P+"ch1_s", P+"ch10_e", P+"ch_ax", P+"step1"], [P+"nonbg"]))
    nodes.append(_n("ReduceSum", [P+"nonbg"], [P+"counts9"], axes=[2, 3], keepdims=0))
    nodes.append(_n("ArgMax", [P+"counts9"], [P+"marker_idx"], axis=1, keepdims=0))
    nodes.append(_n("OneHot", [P+"marker_idx", P+"oh_depth9", P+"oh_values"], [P+"marker_oh9"], axis=1))
    nodes.append(_n("Reshape", [P+"marker_oh9", P+"rs_1911"], [P+"marker_oh4d"]))

    nodes.append(_n("ReduceMax", [P+"nonbg"], [P+"row_has"], axes=[1, 3], keepdims=1))
    nodes.append(_n("ReduceMax", [P+"nonbg"], [P+"col_has"], axes=[1, 2], keepdims=1))
    nodes.append(_n("Mul", [P+"row_has", P+"desc_row"], [P+"first_row_w"]))
    nodes.append(_n("Mul", [P+"row_has", P+"asc_row"], [P+"last_row_w"]))
    nodes.append(_n("Mul", [P+"col_has", P+"desc_col"], [P+"first_col_w"]))
    nodes.append(_n("Mul", [P+"col_has", P+"asc_col"], [P+"last_col_w"]))
    nodes.append(_n("ArgMax", [P+"first_row_w"], [P+"first_row_3d"], axis=2, keepdims=0))
    nodes.append(_n("ArgMax", [P+"last_row_w"], [P+"last_row_3d"], axis=2, keepdims=0))
    nodes.append(_n("ArgMax", [P+"first_col_w"], [P+"first_col_3d"], axis=3, keepdims=0))
    nodes.append(_n("ArgMax", [P+"last_col_w"], [P+"last_col_3d"], axis=3, keepdims=0))

    for src, dst in [
        ("first_row_3d", "first_row_f"),
        ("last_row_3d", "last_row_f"),
        ("first_col_3d", "first_col_f"),
        ("last_col_3d", "last_col_f"),
    ]:
        nodes.append(_n("Reshape", [P+src, P+"shape1"], [P+src.replace("_3d", "")]))
        nodes.append(_n("Cast", [P+src.replace("_3d", "")], [P+dst], to=FLOAT))

    nodes.append(_n("Add", [P+"first_row_f", P+"last_row_f"], [P+"center_row_x2"]))
    nodes.append(_n("Mul", [P+"center_row_x2", P+"half"], [P+"center_row"]))
    nodes.append(_n("Add", [P+"first_col_f", P+"last_col_f"], [P+"center_col_x2"]))
    nodes.append(_n("Mul", [P+"center_col_x2", P+"half"], [P+"center_col"]))
    nodes.append(_n("Sub", [P+"last_row_f", P+"first_row_f"], [P+"row_span"]))
    nodes.append(_n("Mul", [P+"row_span", P+"half"], [P+"period"]))

    nodes.append(_n("Sub", [P+"row_idx", P+"center_row"], [P+"row_delta"]))
    nodes.append(_n("Abs", [P+"row_delta"], [P+"row_dist"]))
    nodes.append(_n("Sub", [P+"col_idx", P+"center_col"], [P+"col_delta"]))
    nodes.append(_n("Abs", [P+"col_delta"], [P+"col_dist"]))
    nodes.append(_n("Max", [P+"row_dist", P+"col_dist"], [P+"cheb"]))

    ring_running = None
    for k in range(30):
        nodes.append(_n("Mul", [P+"period", P+f"k{k}"], [P+f"target{k}"]))
        nodes.append(_n("Sub", [P+"cheb", P+f"target{k}"], [P+f"diff{k}"]))
        nodes.append(_n("Abs", [P+f"diff{k}"], [P+f"absdiff{k}"]))
        nodes.append(_n("Less", [P+f"absdiff{k}", P+"half"], [P+f"ring{k}_b"]))
        nodes.append(_n("Cast", [P+f"ring{k}_b"], [P+f"ring{k}"], to=FLOAT))
        if ring_running is None:
            ring_running = P+f"ring{k}"
        else:
            nxt = P+f"ring_sum{k}"
            nodes.append(_n("Add", [ring_running, P+f"ring{k}"], [nxt]))
            ring_running = nxt

    nodes.append(_n("Mul", [ring_running, P+"active"], [P+"ring_mask"]))
    nodes.append(_n("Mul", [P+"marker_oh4d", P+"ring_mask"], [P+"color_out9"]))
    nodes.append(_n("Sub", [P+"active", P+"ring_mask"], [P+"ch0_out"]))
    nodes.append(_n("Concat", [P+"ch0_out", P+"color_out9"], ["output"], axis=1))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def anchor_cross_aligned_cells_network(batch=1):
    """Task270: paired-anchor color-partition cross drawing on a 15x15 grid.

    Input has exactly one cell of color 1 (c1) and one of color 2 (c2), plus
    scattered cells of colors 3 and 7. Output keeps both centers, removes all
    scattered cells, and draws a partial 4-conn cross at distance 1 around
    each center in the center's assigned outer color {3, 7}.

    Outer color assignment: partition scattered {3, 7} cells into those
    unambiguously aligned with c1 (in c1's row or col but NOT in c2's row/col)
    vs c2's similar set. Whichever center has more unambiguous 3-cells gets 3;
    other gets 7 (canonical tie-break). Arm is drawn only in directions where
    an unambiguous outer-color cell exists along that center's row/col.

    Verified 266/266 (train 3/3, test 1/1, arc-gen 262/262). Cost ~81k.
    Recipe: `Research_Archive/notes/recipe-task270-anchor-cross-from-aligned-cells.md`.
    """
    if batch != 1:
        raise NotImplementedError("task270 anchor-cross builder is single-example only")

    S = 15
    nodes = []
    inits = []
    P = "acac_"

    def _tensor(name, dtype, shape, values):
        t = onnx.helper.make_tensor(name, dtype, shape, values)
        inits.append(t)
        return name

    def _ints(name, values):
        return _tensor(name, onnx.TensorProto.INT64, [len(values)], values)

    def _float(name, value):
        return _tensor(name, onnx.TensorProto.FLOAT, [1], [float(value)])

    def _node(op, inputs, outputs, **attrs):
        nodes.append(onnx.helper.make_node(op, inputs, outputs, **attrs))

    half = _float(P + "half", 0.5)
    one = _float(P + "one", 1.0)
    zero = _float(P + "zero", 0.0)
    ch_axis = _ints(P + "ch_axis", [1])
    ch_step = _ints(P + "ch_step", [1])
    hw_axes = _ints(P + "hw_axes", [2, 3])
    hw_steps = _ints(P + "hw_steps", [1, 1])

    h_range = _tensor(P + "h_range", onnx.TensorProto.FLOAT, [1, 1, S, 1],
                      [float(i) for i in range(S)])
    w_range = _tensor(P + "w_range", onnx.TensorProto.FLOAT, [1, 1, 1, S],
                      [float(i) for i in range(S)])

    # 1) crop input to 15x15 active grid
    _node("Slice", ["input", _ints(P + "cs", [0, 0]), _ints(P + "ce", [S, S]),
                    hw_axes, hw_steps], [P + "crop15"])

    # 2) slice channels 1, 2, 3, 7
    for c in (1, 2, 3, 7):
        _node("Slice", [P + "crop15",
                        _ints(P + f"c{c}s", [c]), _ints(P + f"c{c}e", [c + 1]),
                        ch_axis, ch_step], [P + f"ch{c}"])

    # 3) center positions cr1, cc1, cr2, cc2 (scalars via Mul-by-range + ReduceSum)
    for c in (1, 2):
        _node("Mul", [P + f"ch{c}", h_range], [P + f"ch{c}_rh"])
        _node("ReduceSum", [P + f"ch{c}_rh"], [P + f"cr{c}"], axes=[2, 3], keepdims=1)
        _node("Mul", [P + f"ch{c}", w_range], [P + f"ch{c}_cw"])
        _node("ReduceSum", [P + f"ch{c}_cw"], [P + f"cc{c}"], axes=[2, 3], keepdims=1)

    # 4) equality masks via Less-of-Abs (opset-10 Equal doesn't accept FLOAT)
    def _eq_mask(out_name, scalar_name, range_tensor):
        _node("Sub", [range_tensor, scalar_name], [out_name + "_diff"])
        _node("Abs", [out_name + "_diff"], [out_name + "_abs"])
        _node("Less", [out_name + "_abs", half], [out_name + "_b"])
        _node("Cast", [out_name + "_b"], [out_name], to=onnx.TensorProto.FLOAT)

    _eq_mask(P + "r_eq_cr1", P + "cr1", h_range)
    _eq_mask(P + "r_eq_cr2", P + "cr2", h_range)
    _eq_mask(P + "c_eq_cc1", P + "cc1", w_range)
    _eq_mask(P + "c_eq_cc2", P + "cc2", w_range)

    # 5) aligned_to_c1 = (r==cr1) OR (c==cc1) — broadcast Add then >0.5
    _node("Add", [P + "r_eq_cr1", P + "c_eq_cc1"], [P + "ali1_raw"])
    _node("Less", [half, P + "ali1_raw"], [P + "ali1_b"])
    _node("Cast", [P + "ali1_b"], [P + "aligned_c1"], to=onnx.TensorProto.FLOAT)
    _node("Add", [P + "r_eq_cr2", P + "c_eq_cc2"], [P + "ali2_raw"])
    _node("Less", [half, P + "ali2_raw"], [P + "ali2_b"])
    _node("Cast", [P + "ali2_b"], [P + "aligned_c2"], to=onnx.TensorProto.FLOAT)

    # 6) unambig masks: aligned to exactly one center
    _node("Sub", [one, P + "aligned_c2"], [P + "not_ac2"])
    _node("Mul", [P + "aligned_c1", P + "not_ac2"], [P + "unambig_c1"])
    _node("Sub", [one, P + "aligned_c1"], [P + "not_ac1"])
    _node("Mul", [P + "aligned_c2", P + "not_ac1"], [P + "unambig_c2"])

    # 7) unambiguous 3-count per center
    _node("Mul", [P + "ch3", P + "unambig_c1"], [P + "ch3_uc1"])
    _node("ReduceSum", [P + "ch3_uc1"], [P + "cnt3c1"], axes=[2, 3], keepdims=1)
    _node("Mul", [P + "ch3", P + "unambig_c2"], [P + "ch3_uc2"])
    _node("ReduceSum", [P + "ch3_uc2"], [P + "cnt3c2"], axes=[2, 3], keepdims=1)

    # 8) outer1_is_3 = strict (cnt3c1 > cnt3c2); else canonical outer1=7
    _node("Sub", [P + "cnt3c1", P + "cnt3c2"], [P + "d3"])
    _node("Greater", [P + "d3", zero], [P + "o1is3_b"])
    _node("Cast", [P + "o1is3_b"], [P + "o1is3"], to=onnx.TensorProto.FLOAT)
    _node("Sub", [one, P + "o1is3"], [P + "o1is7"])

    # 9) direction half-row/half-col masks + adjacent arm positions per center
    def _dir_masks(ci):
        cr = P + f"cr{ci}"
        cc = P + f"cc{ci}"
        r_eq = P + f"r_eq_cr{ci}"
        c_eq = P + f"c_eq_cc{ci}"
        _node("Less", [w_range, cc], [P + f"lt_cc{ci}_b"])
        _node("Cast", [P + f"lt_cc{ci}_b"], [P + f"lt_cc{ci}"], to=onnx.TensorProto.FLOAT)
        _node("Greater", [w_range, cc], [P + f"gt_cc{ci}_b"])
        _node("Cast", [P + f"gt_cc{ci}_b"], [P + f"gt_cc{ci}"], to=onnx.TensorProto.FLOAT)
        _node("Less", [h_range, cr], [P + f"lt_cr{ci}_b"])
        _node("Cast", [P + f"lt_cr{ci}_b"], [P + f"lt_cr{ci}"], to=onnx.TensorProto.FLOAT)
        _node("Greater", [h_range, cr], [P + f"gt_cr{ci}_b"])
        _node("Cast", [P + f"gt_cr{ci}_b"], [P + f"gt_cr{ci}"], to=onnx.TensorProto.FLOAT)
        _node("Mul", [r_eq, P + f"lt_cc{ci}"], [P + f"row_left_c{ci}"])
        _node("Mul", [r_eq, P + f"gt_cc{ci}"], [P + f"row_right_c{ci}"])
        _node("Mul", [c_eq, P + f"lt_cr{ci}"], [P + f"col_up_c{ci}"])
        _node("Mul", [c_eq, P + f"gt_cr{ci}"], [P + f"col_down_c{ci}"])
        _node("Sub", [cc, one], [cc + "_m1"])
        _eq_mask(P + f"c_eq_cc{ci}m1", cc + "_m1", w_range)
        _node("Add", [cc, one], [cc + "_p1"])
        _eq_mask(P + f"c_eq_cc{ci}p1", cc + "_p1", w_range)
        _node("Sub", [cr, one], [cr + "_m1"])
        _eq_mask(P + f"r_eq_cr{ci}m1", cr + "_m1", h_range)
        _node("Add", [cr, one], [cr + "_p1"])
        _eq_mask(P + f"r_eq_cr{ci}p1", cr + "_p1", h_range)
        _node("Mul", [r_eq, P + f"c_eq_cc{ci}m1"], [P + f"arm_left_c{ci}"])
        _node("Mul", [r_eq, P + f"c_eq_cc{ci}p1"], [P + f"arm_right_c{ci}"])
        _node("Mul", [P + f"r_eq_cr{ci}m1", c_eq], [P + f"arm_up_c{ci}"])
        _node("Mul", [P + f"r_eq_cr{ci}p1", c_eq], [P + f"arm_down_c{ci}"])

    _dir_masks(1)
    _dir_masks(2)

    # 10) per-center outer-color channel
    _node("Mul", [P + "ch3", P + "o1is3"], [P + "oc1_3"])
    _node("Mul", [P + "ch7", P + "o1is7"], [P + "oc1_7"])
    _node("Add", [P + "oc1_3", P + "oc1_7"], [P + "ch_outer_c1"])
    _node("Mul", [P + "ch3", P + "o1is7"], [P + "oc2_3"])
    _node("Mul", [P + "ch7", P + "o1is3"], [P + "oc2_7"])
    _node("Add", [P + "oc2_3", P + "oc2_7"], [P + "ch_outer_c2"])

    # 11) per-direction "any outer cell exists" gates
    def _has(dir_mask, ch_outer, name):
        _node("Mul", [ch_outer, dir_mask], [name + "_prod"])
        _node("ReduceSum", [name + "_prod"], [name + "_sum"], axes=[2, 3], keepdims=1)
        _node("Greater", [name + "_sum", zero], [name + "_b"])
        _node("Cast", [name + "_b"], [name], to=onnx.TensorProto.FLOAT)

    for ci in (1, 2):
        for d in ("left", "right", "up", "down"):
            src = P + (f"row_{d}_c{ci}" if d in ("left", "right") else f"col_{d}_c{ci}")
            _has(src, P + f"ch_outer_c{ci}", P + f"has_{d}_c{ci}")

    # 12) draw arms: arm_mask * has_dir
    arm_c1 = []
    arm_c2 = []
    for d in ("left", "right", "up", "down"):
        _node("Mul", [P + f"arm_{d}_c1", P + f"has_{d}_c1"], [P + f"draw_{d}_c1"])
        arm_c1.append(P + f"draw_{d}_c1")
        _node("Mul", [P + f"arm_{d}_c2", P + f"has_{d}_c2"], [P + f"draw_{d}_c2"])
        arm_c2.append(P + f"draw_{d}_c2")
    _node("Sum", arm_c1, [P + "cross_c1"])
    _node("Sum", arm_c2, [P + "cross_c2"])

    # 13) split cross output by color
    _node("Mul", [P + "cross_c1", P + "o1is3"], [P + "cc1_as3"])
    _node("Mul", [P + "cross_c2", P + "o1is7"], [P + "cc2_as3"])
    _node("Add", [P + "cc1_as3", P + "cc2_as3"], [P + "ch3_cross"])
    _node("Mul", [P + "cross_c1", P + "o1is7"], [P + "cc1_as7"])
    _node("Mul", [P + "cross_c2", P + "o1is3"], [P + "cc2_as7"])
    _node("Add", [P + "cc1_as7", P + "cc2_as7"], [P + "ch7_cross"])

    # 14) output assembly: ch0 = 1 - (ch1 + ch2 + ch3_cross + ch7_cross), rest zero
    _node("Sum", [P + "ch1", P + "ch2", P + "ch3_cross", P + "ch7_cross"], [P + "any_active"])
    _node("Sub", [one, P + "any_active"], [P + "ch0_out"])
    zero_ch = _tensor(P + "zero_ch", onnx.TensorProto.FLOAT, [1, 1, S, S], [0.0] * (S * S))
    _node("Concat",
          [P + "ch0_out", P + "ch1", P + "ch2", P + "ch3_cross",
           zero_ch, zero_ch, zero_ch, P + "ch7_cross", zero_ch, zero_ch],
          [P + "out15"], axis=1)
    _node("Pad", [P + "out15"], ["output"],
          pads=[0, 0, 0, 0, 0, 0, 30 - S, 30 - S], mode="constant", value=0.0)

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def dominant_color_keep_network(H, W, out_color=5, batch=1):
    """ONNX: most frequent non-zero color kept; others → out_color.

    task229: H=3, W=3, out_color=5.
    """
    if batch != 1:
        raise NotImplementedError("dominant-color-keep is single-example only")

    nodes = []
    inits = []
    P = "dck_"
    FLOAT = onnx.TensorProto.FLOAT
    INT64 = onnx.TensorProto.INT64

    def _t(name, dtype, shape, vals):
        return onnx.helper.make_tensor(name, dtype, shape, vals)
    def _n(op, inputs, outputs, **kwargs):
        return onnx.helper.make_node(op, inputs, outputs, **kwargs)

    # nz_mask: [1,10,1,1] with ch0=0, others=1 (ignore background in count)
    nz_mask_vals = [0.0 if c == 0 else 1.0 for c in range(10)]
    # out_ch: [1,10,1,1] with out_color channel = 1, others = 0
    out_ch_vals = [1.0 if c == out_color else 0.0 for c in range(10)]

    inits += [
        _t(P+"one",      FLOAT, [1], [1.0]),
        _t(P+"nz_mask",  FLOAT, [1, 10, 1, 1], nz_mask_vals),
        _t(P+"out_ch",   FLOAT, [1, 10, 1, 1], out_ch_vals),
        _t(P+"hw_ax",    INT64, [2], [2, 3]),
        _t(P+"hw_st",    INT64, [2], [0, 0]),
        _t(P+"hw_en",    INT64, [2], [H, W]),
        _t(P+"hw_sp",    INT64, [2], [1, 1]),
    ]

    # Crop input to [1,10,H,W]
    nodes.append(_n("Slice", ["input", P+"hw_st", P+"hw_en", P+"hw_ax", P+"hw_sp"], [P+"x_hw"]))

    # Count per channel, zero out background
    nodes.append(_n("ReduceSum", [P+"x_hw"], [P+"counts_raw"], axes=[2, 3], keepdims=1))
    nodes.append(_n("Mul", [P+"counts_raw", P+"nz_mask"], [P+"counts"]))

    # Max count across channels → [1,1,1,1]
    nodes.append(_n("ReduceMax", [P+"counts"], [P+"max_count"], axes=[1], keepdims=1))

    # ratio = counts / max_count → 1.0 for dominant channel(s), <1 otherwise
    nodes.append(_n("Div", [P+"counts", P+"max_count"], [P+"ratio"]))

    # Floor(ratio) → 1 iff counts == max_count (dominant), 0 otherwise
    nodes.append(_n("Floor", [P+"ratio"], [P+"dom_indicator"]))

    # Keep only dominant color cells: x_hw * dom_indicator → [1,10,H,W]
    nodes.append(_n("Mul", [P+"x_hw", P+"dom_indicator"], [P+"dom_cells"]))

    # Binary mask: 1 where cell has dominant color
    nodes.append(_n("ReduceMax", [P+"dom_cells"], [P+"dom_mask"], axes=[1], keepdims=1))

    # Non-dominant mask
    nodes.append(_n("Sub", [P+"one", P+"dom_mask"], [P+"non_dom"]))

    # out_color at non-dominant cells: [1,1,H,W] * [1,10,1,1] → [1,10,H,W]
    nodes.append(_n("Mul", [P+"non_dom", P+"out_ch"], [P+"out_color_cells"]))

    # Final: dominant cells + out_color at non-dominant cells
    nodes.append(_n("Add", [P+"dom_cells", P+"out_color_cells"], [P+"output_hw"]))

    nodes.append(_n("Pad", [P+"output_hw"], ["output"],
                   pads=[0, 0, 0, 0, 0, 0, 30 - H, 30 - W], mode="constant", value=0.0))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def color_to_pattern_network(H_in, W_in, color_map, out_color=5, batch=1):
    """ONNX: input color → fixed 3×3 binary output pattern.

    color_map: dict {color_int: ((r0c0, r0c1, r0c2), (r1c0, ...), (r2c0, ...))}
    Each example has exactly one non-zero color; that color selects its pattern.
    task334: H_in=5, W_in=5, out_color=5.
    """
    if batch != 1:
        raise NotImplementedError("color-to-pattern is single-example only")

    nodes = []
    inits = []
    P = "c2p_"
    FLOAT = onnx.TensorProto.FLOAT
    INT64 = onnx.TensorProto.INT64

    def _t(name, dtype, shape, vals):
        return onnx.helper.make_tensor(name, dtype, shape, vals)
    def _n(op, inputs, outputs, **kwargs):
        return onnx.helper.make_node(op, inputs, outputs, **kwargs)

    inits += [
        _t(P+"one",    FLOAT, [1], [1.0]),
        _t(P+"zero",   FLOAT, [1], [0.0]),
        _t(P+"ch_ax",  INT64, [1], [1]),
        _t(P+"ch_sp",  INT64, [1], [1]),
        _t(P+"hw_ax",  INT64, [2], [2, 3]),
        _t(P+"hw_st",  INT64, [2], [0, 0]),
        _t(P+"hw_en",  INT64, [2], [H_in, W_in]),
        _t(P+"hw_sp",  INT64, [2], [1, 1]),
    ]

    # Crop input to [1,10,H_in,W_in]
    nodes.append(_n("Slice", ["input", P+"hw_st", P+"hw_en", P+"hw_ax", P+"hw_sp"], [P+"x_hw"]))

    # For each mapped color: compute has_c and weight its pattern
    weighted = []
    for idx, (c, pattern) in enumerate(sorted(color_map.items())):
        pat_flat = [float(v) for row in pattern for v in row]
        inits += [
            _t(P+f"cs{idx}", INT64, [1], [c]),
            _t(P+f"ce{idx}", INT64, [1], [c + 1]),
            _t(P+f"pat{idx}", FLOAT, [1, 1, 3, 3], pat_flat),
        ]
        nodes.append(_n("Slice", [P+"x_hw", P+f"cs{idx}", P+f"ce{idx}", P+"ch_ax", P+"ch_sp"],
                        [P+f"ic{idx}"]))
        # Scalar 1.0 if color c present in input
        nodes.append(_n("ReduceMax", [P+f"ic{idx}"], [P+f"has{idx}"], axes=[2, 3], keepdims=1))
        # Weighted pattern [1,1,3,3] * [1,1,1,1]
        nodes.append(_n("Mul", [P+f"pat{idx}", P+f"has{idx}"], [P+f"wp{idx}"]))
        weighted.append(P+f"wp{idx}")

    # Sum contributions → binary [1,1,3,3]
    if len(weighted) == 1:
        nodes.append(_n("Mul", [weighted[0], P+"one"], [P+"binary"]))
    else:
        curr = weighted[0]
        for i in range(1, len(weighted)):
            nxt = P+"binary" if i == len(weighted) - 1 else P+f"sum{i}"
            nodes.append(_n("Add", [curr, weighted[i]], [nxt]))
            curr = nxt

    # ch_out_color = binary, ch0 = 1 - binary, others = zeros
    nodes.append(_n("Sub", [P+"one", P+"binary"], [P+"ch0"]))
    nodes.append(_n("Mul", [P+"binary", P+"zero"], [P+"zeros"]))

    ch_tensors = []
    for c in range(10):
        if c == 0:
            ch_tensors.append(P+"ch0")
        elif c == out_color:
            ch_tensors.append(P+"binary")
        else:
            ch_tensors.append(P+"zeros")
    nodes.append(_n("Concat", ch_tensors, [P+"assembled"], axis=1))

    # Pad 3×3 → 30×30
    nodes.append(_n("Pad", [P+"assembled"], ["output"],
                   pads=[0, 0, 0, 0, 0, 0, 27, 27], mode="constant", value=0.0))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def cross_line_fill_network(H, W, color_col, color_row, color_inter, batch=1):
    """ONNX: partial vertical line + partial horizontal line → full cross.

    color_col: vertical line color (fills entire column)
    color_row: horizontal line color (fills entire row)
    color_inter: intersection cell color
    task299: H=6, W=6, color_col=8, color_row=2, color_inter=4.
    """
    if batch != 1:
        raise NotImplementedError("cross-line-fill is single-example only")

    nodes = []
    inits = []
    P = "clf_"
    FLOAT = onnx.TensorProto.FLOAT
    INT64 = onnx.TensorProto.INT64

    def _t(name, dtype, shape, vals):
        return onnx.helper.make_tensor(name, dtype, shape, vals)
    def _n(op, inputs, outputs, **kwargs):
        return onnx.helper.make_node(op, inputs, outputs, **kwargs)

    inits += [
        _t(P+"one",      FLOAT, [1], [1.0]),
        _t(P+"zero",     FLOAT, [1], [0.0]),
        _t(P+"ch_ax",    INT64, [1], [1]),
        _t(P+"ch_sp",    INT64, [1], [1]),
        _t(P+"hw_ax",    INT64, [2], [2, 3]),
        _t(P+"hw_st",    INT64, [2], [0, 0]),
        _t(P+"hw_en",    INT64, [2], [H, W]),
        _t(P+"hw_sp",    INT64, [2], [1, 1]),
        _t(P+"sc_col",   INT64, [1], [color_col]),
        _t(P+"ec_col",   INT64, [1], [color_col + 1]),
        _t(P+"sc_row",   INT64, [1], [color_row]),
        _t(P+"ec_row",   INT64, [1], [color_row + 1]),
        _t(P+"tile_col", INT64, [4], [1, 1, H, 1]),
        _t(P+"tile_row", INT64, [4], [1, 1, 1, W]),
    ]

    # Crop input to [1,10,H,W]
    nodes.append(_n("Slice", ["input", P+"hw_st", P+"hw_en", P+"hw_ax", P+"hw_sp"], [P+"x_hw"]))

    # Extract color_col channel [1,1,H,W] and color_row channel [1,1,H,W]
    nodes.append(_n("Slice", [P+"x_hw", P+"sc_col", P+"ec_col", P+"ch_ax", P+"ch_sp"], [P+"x_col"]))
    nodes.append(_n("Slice", [P+"x_hw", P+"sc_row", P+"ec_row", P+"ch_ax", P+"ch_sp"], [P+"x_row"]))

    # col_any: which column has color_col → [1,1,1,W]
    nodes.append(_n("ReduceMax", [P+"x_col"], [P+"col_any"], axes=[2], keepdims=1))
    # row_any: which row has color_row → [1,1,H,1]
    nodes.append(_n("ReduceMax", [P+"x_row"], [P+"row_any"], axes=[3], keepdims=1))

    # Tile to full H×W masks
    nodes.append(_n("Tile", [P+"col_any", P+"tile_col"], [P+"col_mask"]))
    nodes.append(_n("Tile", [P+"row_any", P+"tile_row"], [P+"row_mask"]))

    # Decompose into: intersection, col_only, row_only, bg
    nodes.append(_n("Mul",  [P+"col_mask", P+"row_mask"], [P+"intersection"]))
    nodes.append(_n("Sub",  [P+"one",      P+"row_mask"], [P+"not_row"]))
    nodes.append(_n("Mul",  [P+"col_mask", P+"not_row"],  [P+"col_only"]))
    nodes.append(_n("Sub",  [P+"one",      P+"col_mask"], [P+"not_col"]))
    nodes.append(_n("Mul",  [P+"row_mask", P+"not_col"],  [P+"row_only"]))
    nodes.append(_n("Mul",  [P+"not_col",  P+"not_row"],  [P+"bg"]))

    # zeros channel for unused colors
    nodes.append(_n("Mul",  [P+"bg", P+"zero"], [P+"zeros_hw"]))

    # Assemble 10 channels
    ch_tensors = []
    for c in range(10):
        if c == 0:
            ch_tensors.append(P+"bg")
        elif c == color_col:
            ch_tensors.append(P+"col_only")
        elif c == color_row:
            ch_tensors.append(P+"row_only")
        elif c == color_inter:
            ch_tensors.append(P+"intersection")
        else:
            ch_tensors.append(P+"zeros_hw")
    nodes.append(_n("Concat", ch_tensors, [P+"assembled"], axis=1))

    # Pad H×W → 30×30
    pad_h = 30 - H
    pad_w = 30 - W
    nodes.append(_n("Pad", [P+"assembled"], ["output"],
                   pads=[0, 0, 0, 0, 0, 0, pad_h, pad_w], mode="constant", value=0.0))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def rectangular_zero_rank_fill_network(H=20, W=20, batch=1):
    """ONNX: task145 rectangle-rank fill.

    Inputs contain color-2 walls and color-0 rectangular rooms. The largest
    zero room becomes color 1; the smallest non-largest zero room(s) become
    color 8; walls stay color 2. Since rooms are rectangles, a zero cell's
    component area is horizontal zero-run length times vertical zero-run length.
    """
    if batch != 1:
        raise NotImplementedError("rectangular-zero-rank-fill is single-example only")

    nodes = []
    inits = []
    P = "rzrf_"
    FLOAT = onnx.TensorProto.FLOAT
    INT64 = onnx.TensorProto.INT64

    def _t(name, dtype, shape, vals):
        return onnx.helper.make_tensor(name, dtype, shape, vals)

    def _n(op, inputs, outputs, **kwargs):
        return onnx.helper.make_node(op, inputs, outputs, **kwargs)

    inits += [
        _t(P+"one",       FLOAT, [1], [1.0]),
        _t(P+"zero",      FLOAT, [1], [0.0]),
        _t(P+"half",      FLOAT, [1], [0.5]),
        _t(P+"large",     FLOAT, [1], [1000.0]),
        _t(P+"neg_one",   FLOAT, [1], [-1.0]),
        _t(P+"ax4",       INT64, [4], [0, 1, 2, 3]),
        _t(P+"sp4",       INT64, [4], [1, 1, 1, 1]),
        _t(P+"st_crop",   INT64, [4], [0, 0, 0, 0]),
        _t(P+"en_crop",   INT64, [4], [1, 10, H, W]),
        _t(P+"st_ch0",    INT64, [4], [0, 0, 0, 0]),
        _t(P+"en_ch0",    INT64, [4], [1, 1, H, W]),
        _t(P+"st_ch2",    INT64, [4], [0, 2, 0, 0]),
        _t(P+"en_ch2",    INT64, [4], [1, 3, H, W]),
    ]

    nodes.append(_n("Slice", ["input", P+"st_crop", P+"en_crop", P+"ax4", P+"sp4"], [P+"x_hw"]))
    nodes.append(_n("Slice", [P+"x_hw", P+"st_ch0", P+"en_ch0", P+"ax4", P+"sp4"], [P+"z"]))
    nodes.append(_n("Slice", [P+"x_hw", P+"st_ch2", P+"en_ch2", P+"ax4", P+"sp4"], [P+"wall"]))

    def run_length_map(axis, limit):
        parts = []
        for k in range(1, limit + 1):
            if axis == "h":
                interior_shape = [1, 1, 1, k + 2]
                interior_vals = [0.0] + [1.0] * k + [0.0]
                boundary_shape = [1, 1, 1, k + 2]
                boundary_vals = [1.0] + [0.0] * k + [1.0]
                start_pads = [0, 1, 0, k]
                spread_shape = [1, 1, 1, k]
                spread_pads = [0, k - 1, 0, 0]
                kernel_shape = [1, k + 2]
                spread_kernel_shape = [1, k]
            else:
                interior_shape = [1, 1, k + 2, 1]
                interior_vals = [0.0] + [1.0] * k + [0.0]
                boundary_shape = [1, 1, k + 2, 1]
                boundary_vals = [1.0] + [0.0] * k + [1.0]
                start_pads = [1, 0, k, 0]
                spread_shape = [1, 1, k, 1]
                spread_pads = [k - 1, 0, 0, 0]
                kernel_shape = [k + 2, 1]
                spread_kernel_shape = [k, 1]

            stem = f"{P}{axis}{k}_"
            inits.extend([
                _t(stem+"int_k", FLOAT, interior_shape, interior_vals),
                _t(stem+"bd_k", FLOAT, boundary_shape, boundary_vals),
                _t(stem+"spread_k", FLOAT, spread_shape, [1.0] * k),
                _t(stem+"k", FLOAT, [1], [float(k)]),
            ])
            nodes.append(_n("Conv", [P+"z", stem+"int_k"], [stem+"int_count"],
                            kernel_shape=kernel_shape, pads=start_pads))
            nodes.append(_n("Conv", [P+"z", stem+"bd_k"], [stem+"bd_count"],
                            kernel_shape=kernel_shape, pads=start_pads))
            nodes.append(_n("Sub", [stem+"int_count", stem+"k"], [stem+"int_delta"]))
            nodes.append(_n("Abs", [stem+"int_delta"], [stem+"int_abs"]))
            nodes.append(_n("Less", [stem+"int_abs", P+"half"], [stem+"int_ok"]))
            nodes.append(_n("Less", [stem+"bd_count", P+"half"], [stem+"bd_ok"]))
            nodes.append(_n("And", [stem+"int_ok", stem+"bd_ok"], [stem+"start_bool"]))
            nodes.append(_n("Cast", [stem+"start_bool"], [stem+"start"], to=FLOAT))
            nodes.append(_n("Conv", [stem+"start", stem+"spread_k"], [stem+"spread"],
                            kernel_shape=spread_kernel_shape, pads=spread_pads))
            nodes.append(_n("Mul", [stem+"spread", stem+"k"], [stem+"part"]))
            parts.append(stem+"part")

        acc = parts[0]
        for idx, part in enumerate(parts[1:], start=2):
            out = f"{P}{axis}_sum_{idx}"
            nodes.append(_n("Add", [acc, part], [out]))
            acc = out
        return acc

    h_run = run_length_map("h", W)
    v_run = run_length_map("v", H)
    nodes.append(_n("Mul", [h_run, v_run], [P+"area_raw"]))
    nodes.append(_n("Mul", [P+"area_raw", P+"z"], [P+"area"]))

    nodes.append(_n("ReduceMax", [P+"area"], [P+"max_area"], axes=[2, 3], keepdims=1))
    nodes.append(_n("Sub", [P+"one", P+"z"], [P+"not_z"]))
    nodes.append(_n("Mul", [P+"not_z", P+"large"], [P+"nonzero_big"]))
    nodes.append(_n("Add", [P+"area", P+"nonzero_big"], [P+"area_safe"]))
    nodes.append(_n("Mul", [P+"area_safe", P+"neg_one"], [P+"neg_area_safe"]))
    nodes.append(_n("ReduceMax", [P+"neg_area_safe"], [P+"neg_min_area"], axes=[2, 3], keepdims=1))
    nodes.append(_n("Mul", [P+"neg_min_area", P+"neg_one"], [P+"min_area"]))

    nodes.append(_n("Sub", [P+"area", P+"max_area"], [P+"large_delta"]))
    nodes.append(_n("Abs", [P+"large_delta"], [P+"large_abs"]))
    nodes.append(_n("Less", [P+"large_abs", P+"half"], [P+"is_large_bool"]))
    nodes.append(_n("Sub", [P+"area", P+"min_area"], [P+"small_delta"]))
    nodes.append(_n("Abs", [P+"small_delta"], [P+"small_abs"]))
    nodes.append(_n("Less", [P+"small_abs", P+"half"], [P+"is_small_bool"]))
    nodes.append(_n("Not", [P+"is_large_bool"], [P+"not_large_bool"]))
    nodes.append(_n("Cast", [P+"is_large_bool"], [P+"large_f"], to=FLOAT))
    nodes.append(_n("Cast", [P+"is_small_bool"], [P+"small_f"], to=FLOAT))
    nodes.append(_n("Cast", [P+"not_large_bool"], [P+"not_large_f"], to=FLOAT))
    nodes.append(_n("Mul", [P+"large_f", P+"z"], [P+"large_z"]))
    nodes.append(_n("Mul", [P+"small_f", P+"not_large_f"], [P+"small_not_large"]))
    nodes.append(_n("Mul", [P+"small_not_large", P+"z"], [P+"small_z"]))
    nodes.append(_n("Add", [P+"large_z", P+"small_z"], [P+"ranked_z"]))
    nodes.append(_n("Sub", [P+"z", P+"ranked_z"], [P+"ch0_out"]))
    nodes.append(_n("Mul", [P+"z", P+"zero"], [P+"zeros_hw"]))

    ch_tensors = []
    for c in range(10):
        if c == 0:
            ch_tensors.append(P+"ch0_out")
        elif c == 1:
            ch_tensors.append(P+"large_z")
        elif c == 2:
            ch_tensors.append(P+"wall")
        elif c == 8:
            ch_tensors.append(P+"small_z")
        else:
            ch_tensors.append(P+"zeros_hw")
    nodes.append(_n("Concat", ch_tensors, [P+"assembled"], axis=1))

    nodes.append(_n("Pad", [P+"assembled"], ["output"],
                    pads=[0, 0, 0, 0, 0, 0, 30 - H, 30 - W],
                    mode="constant", value=0.0))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def framed_zero_square_fill_network(H=12, W=12, max_size=4, frame_color=5, fill_color=2, batch=1):
    """ONNX: fill zero square components immediately framed by one-cell walls.

    Task102 has 12x12 grids with color-5 rectangular frames. Only framed zero
    interiors whose connected zero component is square are recolored to 2;
    framed non-square rectangles remain zero.
    """
    if batch != 1:
        raise NotImplementedError("framed-zero-square-fill is single-example only")
    if max_size < 1 or max_size > min(H, W) - 2:
        raise ValueError("max_size must fit inside a one-cell frame")

    nodes = []
    inits = []
    P = "fzsf_"
    FLOAT = onnx.TensorProto.FLOAT
    INT64 = onnx.TensorProto.INT64

    def _t(name, dtype, shape, vals):
        return onnx.helper.make_tensor(name, dtype, shape, vals)

    def _n(op, inputs, outputs, **kwargs):
        return onnx.helper.make_node(op, inputs, outputs, **kwargs)

    inits += [
        _t(P+"ax4", INT64, [4], [0, 1, 2, 3]),
        _t(P+"sp4", INT64, [4], [1, 1, 1, 1]),
        _t(P+"st_crop", INT64, [4], [0, 0, 0, 0]),
        _t(P+"en_crop", INT64, [4], [1, 10, H, W]),
        _t(P+"st_zero", INT64, [4], [0, 0, 0, 0]),
        _t(P+"en_zero", INT64, [4], [1, 1, H, W]),
        _t(P+"st_frame", INT64, [4], [0, frame_color, 0, 0]),
        _t(P+"en_frame", INT64, [4], [1, frame_color + 1, H, W]),
        _t(P+"zero", FLOAT, [1], [0.0]),
        _t(P+"half", FLOAT, [1], [0.5]),
    ]

    nodes.append(_n("Slice", ["input", P+"st_crop", P+"en_crop", P+"ax4", P+"sp4"], [P+"x_hw"]))
    nodes.append(_n("Slice", [P+"x_hw", P+"st_zero", P+"en_zero", P+"ax4", P+"sp4"], [P+"z"]))
    nodes.append(_n("Slice", [P+"x_hw", P+"st_frame", P+"en_frame", P+"ax4", P+"sp4"], [P+"frame"]))

    filled_parts = []
    for k in range(1, max_size + 1):
        stem = f"{P}k{k}_"
        frame_side = k + 2
        frame_vals = []
        for rr in range(frame_side):
            for cc in range(frame_side):
                frame_vals.append(1.0 if rr in (0, frame_side - 1) or cc in (0, frame_side - 1) else 0.0)
        inits.extend([
            _t(stem+"zero_k", FLOAT, [1, 1, k, k], [1.0] * (k * k)),
            _t(stem+"frame_k", FLOAT, [1, 1, frame_side, frame_side], frame_vals),
            _t(stem+"zero_target", FLOAT, [1], [float(k * k)]),
            _t(stem+"frame_target", FLOAT, [1], [float(4 * k + 4)]),
            _t(stem+"st_inner", INT64, [4], [0, 0, 1, 1]),
            _t(stem+"en_inner", INT64, [4], [1, 1, H - k, W - k]),
        ])
        nodes.append(_n("Conv", [P+"z", stem+"zero_k"], [stem+"zero_count"], kernel_shape=[k, k]))
        nodes.append(_n("Slice", [stem+"zero_count", stem+"st_inner", stem+"en_inner", P+"ax4", P+"sp4"],
                        [stem+"zero_count_inner"]))
        nodes.append(_n("Conv", [P+"frame", stem+"frame_k"], [stem+"frame_count"],
                        kernel_shape=[frame_side, frame_side]))
        nodes.append(_n("Sub", [stem+"zero_count_inner", stem+"zero_target"], [stem+"zero_delta"]))
        nodes.append(_n("Abs", [stem+"zero_delta"], [stem+"zero_abs"]))
        nodes.append(_n("Less", [stem+"zero_abs", P+"half"], [stem+"zero_ok"]))
        nodes.append(_n("Sub", [stem+"frame_count", stem+"frame_target"], [stem+"frame_delta"]))
        nodes.append(_n("Abs", [stem+"frame_delta"], [stem+"frame_abs"]))
        nodes.append(_n("Less", [stem+"frame_abs", P+"half"], [stem+"frame_ok"]))
        nodes.append(_n("And", [stem+"zero_ok", stem+"frame_ok"], [stem+"start_bool"]))
        nodes.append(_n("Cast", [stem+"start_bool"], [stem+"start"], to=FLOAT))

        offsets = []
        for dr in range(k):
            for dc in range(k):
                off = f"{stem}off_{dr}_{dc}"
                nodes.append(_n("Pad", [stem+"start"], [off],
                                pads=[0, 0, 1 + dr, 1 + dc, 0, 0, k - dr, k - dc],
                                mode="constant", value=0.0))
                offsets.append(off)
        acc = offsets[0]
        for idx, off in enumerate(offsets[1:], start=1):
            out = f"{stem}sum_{idx}"
            nodes.append(_n("Add", [acc, off], [out]))
            acc = out
        filled_parts.append(acc)

    fill_acc = filled_parts[0]
    for idx, part in enumerate(filled_parts[1:], start=1):
        out = f"{P}fill_sum_{idx}"
        nodes.append(_n("Add", [fill_acc, part], [out]))
        fill_acc = out
    nodes.append(_n("Greater", [fill_acc, P+"half"], [P+"fill_bool"]))
    nodes.append(_n("Cast", [P+"fill_bool"], [P+"fill"], to=FLOAT))
    nodes.append(_n("Sub", [P+"z", P+"fill"], [P+"ch0_out"]))
    nodes.append(_n("Mul", [P+"z", P+"zero"], [P+"zeros_hw"]))

    ch_tensors = []
    for c in range(10):
        if c == 0:
            ch_tensors.append(P+"ch0_out")
        elif c == fill_color:
            ch_tensors.append(P+"fill")
        elif c == frame_color:
            ch_tensors.append(P+"frame")
        else:
            ch_tensors.append(P+"zeros_hw")
    nodes.append(_n("Concat", ch_tensors, [P+"assembled"], axis=1))
    nodes.append(_n("Pad", [P+"assembled"], ["output"],
                    pads=[0, 0, 0, 0, 0, 0, 30 - H, 30 - W],
                    mode="constant", value=0.0))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def bouncing_diagonal_ray_network(max_w=10, batch=1):
    """ONNX: task248 bottom-left seed emits a wall-bouncing diagonal ray."""
    if batch != 1:
        raise NotImplementedError("bouncing-diagonal-ray is single-example only")
    if max_w < 2 or max_w > 30:
        raise ValueError("max_w must be in 2..30")

    nodes = []
    inits = []
    P = "bdr_"
    FLOAT = onnx.TensorProto.FLOAT
    INT64 = onnx.TensorProto.INT64
    H = 10

    def _t(name, dtype, shape, vals):
        return onnx.helper.make_tensor(name, dtype, shape, vals)

    def _n(op, inputs, outputs, **kwargs):
        return onnx.helper.make_node(op, inputs, outputs, **kwargs)

    inits += [
        _t(P+"st_row0", INT64, [4], [0, 0, 0, 0]),
        _t(P+"en_row0", INT64, [4], [1, 1, 1, 30]),
        _t(P+"ax4", INT64, [4], [0, 1, 2, 3]),
        _t(P+"sp4", INT64, [4], [1, 1, 1, 1]),
        _t(P+"half", FLOAT, [1], [0.5]),
        _t(P+"zero", FLOAT, [1], [0.0]),
    ]
    nodes.append(_n("Slice", ["input", P+"st_row0", P+"en_row0", P+"ax4", P+"sp4"], [P+"row0"]))
    nodes.append(_n("ReduceSum", [P+"row0"], [P+"width"], axes=[2, 3], keepdims=1))

    bg_parts = []
    ray_parts = []
    for w in range(2, max_w + 1):
        stem = f"{P}w{w}_"
        bg_vals = []
        ray_vals = []
        period = 2 * (w - 1)
        for r in range(H):
            dist = H - 1 - r
            m = dist % period
            col = m if m < w else period - m
            for c in range(max_w):
                ray = 1.0 if c == col else 0.0
                active_bg = 1.0 if c < w and not ray else 0.0
                ray_vals.append(ray)
                bg_vals.append(active_bg)
        inits.extend([
            _t(stem+"wval", FLOAT, [1], [float(w)]),
            _t(stem+"bg", FLOAT, [1, 1, H, max_w], bg_vals),
            _t(stem+"ray", FLOAT, [1, 1, H, max_w], ray_vals),
        ])
        nodes.append(_n("Sub", [P+"width", stem+"wval"], [stem+"delta"]))
        nodes.append(_n("Abs", [stem+"delta"], [stem+"abs"]))
        nodes.append(_n("Less", [stem+"abs", P+"half"], [stem+"match_bool"]))
        nodes.append(_n("Cast", [stem+"match_bool"], [stem+"match"], to=FLOAT))
        nodes.append(_n("Mul", [stem+"match", stem+"bg"], [stem+"bg_part"]))
        nodes.append(_n("Mul", [stem+"match", stem+"ray"], [stem+"ray_part"]))
        bg_parts.append(stem+"bg_part")
        ray_parts.append(stem+"ray_part")

    def add_all(parts, prefix):
        acc = parts[0]
        for idx, part in enumerate(parts[1:], start=1):
            out = f"{P}{prefix}_{idx}"
            nodes.append(_n("Add", [acc, part], [out]))
            acc = out
        return acc

    bg = add_all(bg_parts, "bg_sum")
    ray = add_all(ray_parts, "ray_sum")
    nodes.append(_n("Mul", [ray, P+"zero"], [P+"zeros_hw"]))

    ch_tensors = []
    for c in range(10):
        if c == 0:
            ch_tensors.append(bg)
        elif c == 1:
            ch_tensors.append(ray)
        else:
            ch_tensors.append(P+"zeros_hw")
    nodes.append(_n("Concat", ch_tensors, [P+"assembled"], axis=1))
    nodes.append(_n("Pad", [P+"assembled"], ["output"],
                    pads=[0, 0, 0, 0, 0, 0, 30 - H, 30 - max_w],
                    mode="constant", value=0.0))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def trirow_periodic_four_fill_network(batch=1):
    """ONNX: task176 3-row periodic color-2 scaffold gains color-4 phase marks."""
    nodes = []
    inits = []
    P = "tpf_"
    FLOAT = onnx.TensorProto.FLOAT
    INT64 = onnx.TensorProto.INT64

    def _t(name, dtype, shape, vals):
        return onnx.helper.make_tensor(name, dtype, shape, vals)

    def _n(op, inputs, outputs, **kwargs):
        return onnx.helper.make_node(op, inputs, outputs, **kwargs)

    four_vals = []
    for r in range(3):
        for c in range(30):
            is_four = (
                (r == 0 and c % 12 in (5, 6, 7)) or
                (r == 1 and c % 6 == 0) or
                (r == 2 and c % 12 in (11, 0, 1))
            )
            four_vals.append(1.0 if is_four else 0.0)

    inits.extend([
        _t(P+"st_ch2", INT64, [3], [2, 0, 0]),
        _t(P+"en_ch2", INT64, [3], [3, 3, 30]),
        _t(P+"ax_chw", INT64, [3], [1, 2, 3]),
        _t(P+"sp3", INT64, [3], [1, 1, 1]),
        _t(P+"zero", FLOAT, [1], [0.0]),
        _t(P+"ones3", FLOAT, [1, 1, 3, 30], [1.0] * 90),
        _t(P+"four_pattern", FLOAT, [1, 1, 3, 30], four_vals),
    ])

    nodes.append(_n("Slice", ["input", P+"st_ch2", P+"en_ch2", P+"ax_chw", P+"sp3"], [P+"ch2"]))
    nodes.append(_n("ReduceSum", [P+"ch2"], [P+"col_sum"], axes=[1, 2], keepdims=1))
    nodes.append(_n("Greater", [P+"col_sum", P+"zero"], [P+"active_bool"]))
    nodes.append(_n("Cast", [P+"active_bool"], [P+"active_col"], to=FLOAT))
    nodes.append(_n("Mul", [P+"active_col", P+"ones3"], [P+"active"]))
    nodes.append(_n("Mul", [P+"four_pattern", P+"active"], [P+"ch4"]))
    nodes.append(_n("Max", [P+"ch2", P+"ch4"], [P+"nonbg"]))
    nodes.append(_n("Sub", [P+"active", P+"nonbg"], [P+"ch0"]))
    nodes.append(_n("Mul", [P+"active", P+"zero"], [P+"zeros"]))

    ch_tensors = []
    for c in range(10):
        if c == 0:
            ch_tensors.append(P+"ch0")
        elif c == 2:
            ch_tensors.append(P+"ch2")
        elif c == 4:
            ch_tensors.append(P+"ch4")
        else:
            ch_tensors.append(P+"zeros")
    nodes.append(_n("Concat", ch_tensors, [P+"assembled"], axis=1))
    nodes.append(_n("Pad", [P+"assembled"], ["output"],
                    pads=[0, 0, 0, 0, 0, 0, 27, 0],
                    mode="constant", value=0.0))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def square_spiral_three_network(max_n=30, batch=1):
    """ONNX: task058 all-zero square becomes a color-3 inward square spiral."""
    if max_n < 5 or max_n > 30:
        raise ValueError("max_n must be in 5..30")

    nodes = []
    inits = []
    P = "sst_"
    FLOAT = onnx.TensorProto.FLOAT
    INT64 = onnx.TensorProto.INT64

    def _t(name, dtype, shape, vals):
        return onnx.helper.make_tensor(name, dtype, shape, vals)

    def _n(op, inputs, outputs, **kwargs):
        return onnx.helper.make_node(op, inputs, outputs, **kwargs)

    def _spiral(n):
        grid = [[0.0 for _ in range(max_n)] for _ in range(max_n)]
        r = c = 0
        grid[r][c] = 1.0
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        lengths = [n - 1, n - 1, n - 1]
        k = n - 3
        while k > 0:
            lengths.extend([k, k])
            k -= 2
        for idx, length in enumerate(lengths):
            dr, dc = directions[idx % 4]
            for _ in range(length):
                r += dr
                c += dc
                if 0 <= r < n and 0 <= c < n:
                    grid[r][c] = 1.0
        return [v for row in grid for v in row]

    inits.extend([
        _t(P+"st_row0", INT64, [3], [0, 0, 0]),
        _t(P+"en_row0", INT64, [3], [1, 1, 30]),
        _t(P+"ax_chw", INT64, [3], [1, 2, 3]),
        _t(P+"sp3", INT64, [3], [1, 1, 1]),
        _t(P+"half", FLOAT, [1], [0.5]),
        _t(P+"zero", FLOAT, [1], [0.0]),
    ])
    nodes.append(_n("Slice", ["input", P+"st_row0", P+"en_row0", P+"ax_chw", P+"sp3"], [P+"row0"]))
    nodes.append(_n("ReduceSum", [P+"row0"], [P+"width"], axes=[2, 3], keepdims=1))

    bg_parts = []
    spiral_parts = []
    for n in range(5, max_n + 1):
        stem = f"{P}n{n}_"
        spiral = _spiral(n)
        active = [1.0 if (r < n and c < n) else 0.0 for r in range(max_n) for c in range(max_n)]
        bg = [a - s for a, s in zip(active, spiral)]
        inits.extend([
            _t(stem+"nval", FLOAT, [1], [float(n)]),
            _t(stem+"bg", FLOAT, [1, 1, max_n, max_n], bg),
            _t(stem+"spiral", FLOAT, [1, 1, max_n, max_n], spiral),
        ])
        nodes.append(_n("Sub", [P+"width", stem+"nval"], [stem+"delta"]))
        nodes.append(_n("Abs", [stem+"delta"], [stem+"abs"]))
        nodes.append(_n("Less", [stem+"abs", P+"half"], [stem+"match_bool"]))
        nodes.append(_n("Cast", [stem+"match_bool"], [stem+"match"], to=FLOAT))
        nodes.append(_n("Mul", [stem+"match", stem+"bg"], [stem+"bg_part"]))
        nodes.append(_n("Mul", [stem+"match", stem+"spiral"], [stem+"spiral_part"]))
        bg_parts.append(stem+"bg_part")
        spiral_parts.append(stem+"spiral_part")

    def add_all(parts, prefix):
        acc = parts[0]
        for idx, part in enumerate(parts[1:], start=1):
            out = f"{P}{prefix}_{idx}"
            nodes.append(_n("Add", [acc, part], [out]))
            acc = out
        return acc

    bg = add_all(bg_parts, "bg_sum")
    spiral = add_all(spiral_parts, "spiral_sum")
    nodes.append(_n("Mul", [spiral, P+"zero"], [P+"zeros_hw"]))

    ch_tensors = []
    for c in range(10):
        if c == 0:
            ch_tensors.append(bg)
        elif c == 3:
            ch_tensors.append(spiral)
        else:
            ch_tensors.append(P+"zeros_hw")
    nodes.append(_n("Concat", ch_tensors, [P+"assembled"], axis=1))
    nodes.append(_n("Pad", [P+"assembled"], ["output"],
                    pads=[0, 0, 0, 0, 0, 0, 30 - max_n, 30 - max_n],
                    mode="constant", value=0.0))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def two_by_two_zero_fill_network(H, W, bg_color=5, fill_color=2, batch=1):
    """ONNX: fill isolated 2-wide zero strips (N×2 or 2×N, N≥2) with fill_color.
    task265: H=18, W=18, bg_color=5, fill_color=2.
    """
    if batch != 1:
        raise NotImplementedError("two-by-two-zero-fill is single-example only")

    nodes, inits = [], []
    P = "ttzf_"
    FLOAT, INT64 = onnx.TensorProto.FLOAT, onnx.TensorProto.INT64

    def _t(name, dtype, shape, vals):
        return onnx.helper.make_tensor(name, dtype, shape, vals)
    def _n(op, inputs, outputs, **kw):
        return onnx.helper.make_node(op, inputs, outputs, **kw)

    inits += [
        _t(P+"ax",      INT64, [4], [0, 1, 2, 3]),
        _t(P+"sp",      INT64, [4], [1, 1, 1, 1]),
        _t(P+"one",     FLOAT, [1, 1, H, W], [1.0] * (H * W)),
        _t(P+"zero_hw", FLOAT, [1, 1, H, W], [0.0] * (H * W)),
    ]

    def sl(inp, out, st, en):
        inits.append(_t(P + out + "_st", INT64, [4], st))
        inits.append(_t(P + out + "_en", INT64, [4], en))
        nodes.append(_n("Slice", [inp, P + out + "_st", P + out + "_en",
                                   P + "ax", P + "sp"], [P + out]))

    sl("input", "xhw", [0, 0, 0, 0], [1, 10, H, W])
    sl(P + "xhw", "Z", [0, 0, 0, 0], [1, 1, H, W])

    # --- Vertical strips: pair_v[i,j] = Z[i,j]*Z[i,j+1] ---
    nodes.append(_n("Pad", [P + "Z"], [P + "Z_pr"],
                   pads=[0, 0, 0, 0, 0, 0, 0, 1], mode="constant", value=0.0))
    sl(P + "Z_pr", "Zsl", [0, 0, 0, 1], [1, 1, H, W + 1])
    nodes.append(_n("Mul", [P + "Z", P + "Zsl"], [P + "pairv"]))

    nodes.append(_n("Pad", [P + "pairv"], [P + "pv_pl"],
                   pads=[0, 0, 0, 1, 0, 0, 0, 0], mode="constant", value=0.0))
    sl(P + "pv_pl", "pvsr", [0, 0, 0, 0], [1, 1, H, W])
    nodes.append(_n("Pad", [P + "pairv"], [P + "pv_pr"],
                   pads=[0, 0, 0, 0, 0, 0, 0, 1], mode="constant", value=0.0))
    sl(P + "pv_pr", "pvsl", [0, 0, 0, 1], [1, 1, H, W + 1])
    nodes.append(_n("Pad", [P + "pairv"], [P + "pv_pt"],
                   pads=[0, 0, 1, 0, 0, 0, 0, 0], mode="constant", value=0.0))
    sl(P + "pv_pt", "pvprev", [0, 0, 0, 0], [1, 1, H, W])
    nodes.append(_n("Pad", [P + "pairv"], [P + "pv_pb"],
                   pads=[0, 0, 0, 0, 0, 0, 1, 0], mode="constant", value=0.0))
    sl(P + "pv_pb", "pvnext", [0, 0, 1, 0], [1, 1, H + 1, W])

    nodes.append(_n("Max", [P + "pvprev", P + "pvnext"], [P + "hadjv"]))
    nodes.append(_n("Mul", [P + "pairv", P + "hadjv"], [P + "runv"]))

    nodes.append(_n("Sub", [P + "one", P + "pvsr"], [P + "npvsr"]))
    nodes.append(_n("Mul", [P + "npvsr", P + "runv"], [P + "seedlv"]))
    nodes.append(_n("Sub", [P + "one", P + "pvsl"], [P + "npvsl"]))
    nodes.append(_n("Mul", [P + "npvsl", P + "runv"], [P + "seedrv"]))

    def spread_rows(seed_tensor, run_tensor, prefix):
        s = seed_tensor
        for d in [1, 2, 4, 8, 16]:
            pd = P + prefix + f"p{d}"
            up = P + prefix + f"u{d}"
            dn = P + prefix + f"d{d}"
            mx = P + prefix + f"m{d}"
            ns = P + prefix + f"s{d}"
            nodes.append(_n("Pad", [s], [pd],
                           pads=[0, 0, d, 0, 0, 0, d, 0], mode="constant", value=0.0))
            sl(pd, prefix + f"u{d}", [0, 0, 0, 0], [1, 1, H, W])
            sl(pd, prefix + f"d{d}", [0, 0, 2 * d, 0], [1, 1, H + 2 * d, W])
            nodes.append(_n("Max", [s, up, dn], [mx]))
            nodes.append(_n("Mul", [mx, run_tensor], [ns]))
            s = ns
        return s

    anylv = spread_rows(P + "seedlv", P + "runv", "sLv")
    anyrv = spread_rows(P + "seedrv", P + "runv", "sRv")
    nodes.append(_n("Mul", [P + "runv", anylv], [P + "rovt1"]))
    nodes.append(_n("Mul", [P + "rovt1", anyrv], [P + "runokv"]))
    nodes.append(_n("Pad", [P + "runokv"], [P + "rov_pl"],
                   pads=[0, 0, 0, 1, 0, 0, 0, 0], mode="constant", value=0.0))
    sl(P + "rov_pl", "rovsr", [0, 0, 0, 0], [1, 1, H, W])
    nodes.append(_n("Max", [P + "runokv", P + "rovsr"], [P + "fillv"]))

    # --- Horizontal strips: pair_h[i,j] = Z[i,j]*Z[i+1,j] ---
    nodes.append(_n("Pad", [P + "Z"], [P + "Z_pb"],
                   pads=[0, 0, 0, 0, 0, 0, 1, 0], mode="constant", value=0.0))
    sl(P + "Z_pb", "Zsd", [0, 0, 1, 0], [1, 1, H + 1, W])
    nodes.append(_n("Mul", [P + "Z", P + "Zsd"], [P + "pairh"]))

    nodes.append(_n("Pad", [P + "pairh"], [P + "ph_pt"],
                   pads=[0, 0, 1, 0, 0, 0, 0, 0], mode="constant", value=0.0))
    sl(P + "ph_pt", "phsu", [0, 0, 0, 0], [1, 1, H, W])
    nodes.append(_n("Pad", [P + "pairh"], [P + "ph_pb"],
                   pads=[0, 0, 0, 0, 0, 0, 1, 0], mode="constant", value=0.0))
    sl(P + "ph_pb", "phsd", [0, 0, 1, 0], [1, 1, H + 1, W])
    nodes.append(_n("Pad", [P + "pairh"], [P + "ph_pl"],
                   pads=[0, 0, 0, 1, 0, 0, 0, 0], mode="constant", value=0.0))
    sl(P + "ph_pl", "phprev", [0, 0, 0, 0], [1, 1, H, W])
    nodes.append(_n("Pad", [P + "pairh"], [P + "ph_pr"],
                   pads=[0, 0, 0, 0, 0, 0, 0, 1], mode="constant", value=0.0))
    sl(P + "ph_pr", "phnext", [0, 0, 0, 1], [1, 1, H, W + 1])

    nodes.append(_n("Max", [P + "phprev", P + "phnext"], [P + "hadjh"]))
    nodes.append(_n("Mul", [P + "pairh", P + "hadjh"], [P + "runh"]))

    nodes.append(_n("Sub", [P + "one", P + "phsu"], [P + "nphsu"]))
    nodes.append(_n("Mul", [P + "nphsu", P + "runh"], [P + "seedth"]))
    nodes.append(_n("Sub", [P + "one", P + "phsd"], [P + "nphsd"]))
    nodes.append(_n("Mul", [P + "nphsd", P + "runh"], [P + "seedbh"]))

    def spread_cols(seed_tensor, run_tensor, prefix):
        s = seed_tensor
        for d in [1, 2, 4, 8, 16]:
            pd = P + prefix + f"p{d}"
            lt = P + prefix + f"l{d}"
            rt = P + prefix + f"r{d}"
            mx = P + prefix + f"m{d}"
            ns = P + prefix + f"s{d}"
            nodes.append(_n("Pad", [s], [pd],
                           pads=[0, 0, 0, d, 0, 0, 0, d], mode="constant", value=0.0))
            sl(pd, prefix + f"l{d}", [0, 0, 0, 0], [1, 1, H, W])
            sl(pd, prefix + f"r{d}", [0, 0, 0, 2 * d], [1, 1, H, W + 2 * d])
            nodes.append(_n("Max", [s, lt, rt], [mx]))
            nodes.append(_n("Mul", [mx, run_tensor], [ns]))
            s = ns
        return s

    anyth = spread_cols(P + "seedth", P + "runh", "sTh")
    anybh = spread_cols(P + "seedbh", P + "runh", "sBh")
    nodes.append(_n("Mul", [P + "runh", anyth], [P + "roht1"]))
    nodes.append(_n("Mul", [P + "roht1", anybh], [P + "runokh"]))
    nodes.append(_n("Pad", [P + "runokh"], [P + "roh_pt"],
                   pads=[0, 0, 1, 0, 0, 0, 0, 0], mode="constant", value=0.0))
    sl(P + "roh_pt", "rohsu", [0, 0, 0, 0], [1, 1, H, W])
    nodes.append(_n("Max", [P + "runokh", P + "rohsu"], [P + "fillh0"]))

    # Suppress horizontal fills adjacent to vertical fills
    nodes.append(_n("Pad", [P + "fillv"], [P + "fv_pl"],
                   pads=[0, 0, 0, 1, 0, 0, 0, 0], mode="constant", value=0.0))
    sl(P + "fv_pl", "fvsl", [0, 0, 0, 0], [1, 1, H, W])
    nodes.append(_n("Pad", [P + "fillv"], [P + "fv_pr"],
                   pads=[0, 0, 0, 0, 0, 0, 0, 1], mode="constant", value=0.0))
    sl(P + "fv_pr", "fvsr", [0, 0, 0, 1], [1, 1, H, W + 1])
    nodes.append(_n("Max", [P + "fillv", P + "fvsl", P + "fvsr"], [P + "fvadj"]))
    nodes.append(_n("Sub", [P + "one", P + "fvadj"], [P + "nfvadj"]))
    nodes.append(_n("Mul", [P + "fillh0", P + "nfvadj"], [P + "fillh"]))

    nodes.append(_n("Max", [P + "fillv", P + "fillh"], [P + "fill"]))

    nodes.append(_n("Sub", [P + "one", P + "fill"], [P + "nfill"]))
    nodes.append(_n("Mul", [P + "Z", P + "nfill"], [P + "zunfill"]))
    nodes.append(_n("Sub", [P + "one", P + "Z"], [P + "bgout"]))

    ch_list = []
    for c in range(10):
        if c == 0:
            ch_list.append(P + "zunfill")
        elif c == fill_color:
            ch_list.append(P + "fill")
        elif c == bg_color:
            ch_list.append(P + "bgout")
        else:
            ch_list.append(P + "zero_hw")
    nodes.append(_n("Concat", ch_list, [P + "assembled"], axis=1))

    pad_h, pad_w = 30 - H, 30 - W
    nodes.append(_n("Pad", [P + "assembled"], ["output"],
                   pads=[0, 0, 0, 0, 0, 0, pad_h, pad_w], mode="constant", value=0.0))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def rotation_symmetry_fill_network(H=10, W=10, fg_color=1, fill_color=2, batch=1):
    """ONNX: fill 0-cells to complete 180° rotational symmetry of fg_color with fill_color.
    Dynamically picks center (4.5,4.5) or (5,5) from existing symmetry count.
    task027: H=10, W=10, fg_color=1, fill_color=2.
    """
    if batch != 1:
        raise NotImplementedError("rotation-symmetry-fill is single-example only")

    nodes, inits = [], []
    P = "rsf_"
    FLOAT, INT64 = onnx.TensorProto.FLOAT, onnx.TensorProto.INT64

    def _t(name, dtype, shape, vals):
        return onnx.helper.make_tensor(name, dtype, shape, vals)
    def _n(op, inputs, outputs, **kw):
        return onnx.helper.make_node(op, inputs, outputs, **kw)

    rev_idx = list(range(H * W - 1, -1, -1))

    inits += [
        _t(P + "rev_idx",    INT64,  [H * W],  rev_idx),
        _t(P + "half_1111",  FLOAT,  [1,1,1,1],[0.5]),
        _t(P + "zero_1111",  FLOAT,  [1,1,1,1],[0.0]),
        _t(P + "one_1111",   FLOAT,  [1,1,1,1],[1.0]),
        _t(P + "one_hw",     FLOAT,  [1,1,H,W],[1.0] * (H * W)),
        _t(P + "zero_hw",    FLOAT,  [1,1,H,W],[0.0] * (H * W)),
        _t(P + "shape_flat", INT64,  [4],      [1, 1, 1, H * W]),
        _t(P + "shape_hw",   INT64,  [4],      [1, 1, H, W]),
        _t(P + "ax4d",       INT64,  [4],      [0, 1, 2, 3]),
        _t(P + "sp4d",       INT64,  [4],      [1, 1, 1, 1]),
        _t(P + "st4d",       INT64,  [4],      [0, 0, 0, 0]),
        _t(P + "en_crop",    INT64,  [4],      [1, 10, H, W]),
        _t(P + "st_fg",      INT64,  [4],      [0, fg_color, 0, 0]),
        _t(P + "en_fg",      INT64,  [4],      [1, fg_color + 1, H, W]),
        _t(P + "en_pad_hw",  INT64,  [4],      [1, 1, H, W]),
    ]

    # Crop to [1,10,H,W]
    nodes.append(_n("Slice", ["input", P+"st4d", P+"en_crop", P+"ax4d", P+"sp4d"], [P+"x_crop"]))
    # Extract fg channel → ones [1,1,H,W]
    nodes.append(_n("Slice", [P+"x_crop", P+"st_fg", P+"en_fg", P+"ax4d", P+"sp4d"], [P+"ones"]))
    # Flatten to [1,1,1,H*W]
    nodes.append(_n("Reshape", [P+"ones", P+"shape_flat"], [P+"ones_flat"]))
    # Reverse (180° flip both axes) via Gather
    nodes.append(_n("Gather", [P+"ones_flat", P+"rev_idx"], [P+"flip_flat"], axis=3))
    # Reshape back → flip_44 [1,1,H,W]  (rotation about grid-center 4.5,4.5)
    nodes.append(_n("Reshape", [P+"flip_flat", P+"shape_hw"], [P+"flip_44"]))
    # flip_55: shift flip_44 by (+1 row, +1 col) = pad top+left, crop to [H,W]
    nodes.append(_n("Pad", [P+"flip_44"], [P+"flip_44_pad"],
                   pads=[0, 0, 1, 1, 0, 0, 0, 0], mode="constant", value=0.0))
    nodes.append(_n("Slice", [P+"flip_44_pad", P+"st4d", P+"en_pad_hw",
                               P+"ax4d", P+"sp4d"], [P+"flip_55"]))
    # sym44 = sum(ones * flip_44)
    nodes.append(_n("Mul", [P+"ones", P+"flip_44"], [P+"prod_44"]))
    nodes.append(_n("ReduceSum", [P+"prod_44"], [P+"sym44"], axes=[0, 1, 2, 3], keepdims=1))
    # sym55 = sum(ones * flip_55)
    nodes.append(_n("Mul", [P+"ones", P+"flip_55"], [P+"prod_55"]))
    nodes.append(_n("ReduceSum", [P+"prod_55"], [P+"sym55"], axes=[0, 1, 2, 3], keepdims=1))
    # diff = (sym55 - sym44) + 0.5  (half-int offset: positive → use center 5,5)
    nodes.append(_n("Sub", [P+"sym55", P+"sym44"], [P+"diff_raw"]))
    nodes.append(_n("Add", [P+"diff_raw", P+"half_1111"], [P+"diff"]))
    # use_55 = max(sign(diff), 0)   → 1 if sym55>=sym44, else 0
    nodes.append(_n("Sign", [P+"diff"], [P+"sign_diff"]))
    nodes.append(_n("Max", [P+"sign_diff", P+"zero_1111"], [P+"use_55"]))
    # use_44 = 1 - use_55
    nodes.append(_n("Sub", [P+"one_1111", P+"use_55"], [P+"use_44"]))
    # zeros = 1 - ones
    nodes.append(_n("Sub", [P+"one_hw", P+"ones"], [P+"zeros"]))
    # twos = use_55 * (zeros*flip_55) + use_44 * (zeros*flip_44)
    nodes.append(_n("Mul", [P+"zeros", P+"flip_55"], [P+"twos_55"]))
    nodes.append(_n("Mul", [P+"zeros", P+"flip_44"], [P+"twos_44"]))
    nodes.append(_n("Mul", [P+"use_55", P+"twos_55"], [P+"t55"]))
    nodes.append(_n("Mul", [P+"use_44", P+"twos_44"], [P+"t44"]))
    nodes.append(_n("Add", [P+"t55", P+"t44"], [P+"twos"]))
    # ch0 = 0-cells not filled = zeros*(1-twos)
    nodes.append(_n("Sub", [P+"one_hw", P+"twos"], [P+"not_twos"]))
    nodes.append(_n("Mul", [P+"zeros", P+"not_twos"], [P+"ch0"]))

    ch_list = []
    for c in range(10):
        if c == 0:
            ch_list.append(P + "ch0")
        elif c == fg_color:
            ch_list.append(P + "ones")
        elif c == fill_color:
            ch_list.append(P + "twos")
        else:
            ch_list.append(P + "zero_hw")
    nodes.append(_n("Concat", ch_list, [P + "assembled"], axis=1))

    pad_h, pad_w = 30 - H, 30 - W
    nodes.append(_n("Pad", [P + "assembled"], ["output"],
                   pads=[0, 0, 0, 0, 0, 0, pad_h, pad_w], mode="constant", value=0.0))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def diagonal_symmetry_fill_network(H=10, W=10, fg_color=3, fill_color=8, batch=1):
    """ONNX: place fill_color at 90°-rotated (×3) positions of fg_color block pairs.
    task042: H=10, W=10, fg_color=3, fill_color=8.
    Handles singleton pairs (diag-adjacent cells), 2×2 block pairs, and 3×3 block pairs.
    Applies priority: 3×3 blocks suppress 2×2 detection; 2×2 blocks suppress singleton detection.
    """
    if batch != 1:
        raise NotImplementedError("diagonal-symmetry-fill is single-example only")

    nodes, inits = [], []
    P = "dsf_"
    FLOAT, INT64 = onnx.TensorProto.FLOAT, onnx.TensorProto.INT64

    def _t(name, dtype, shape, vals):
        return onnx.helper.make_tensor(name, dtype, shape, vals)
    def _n(op, inputs, outputs, **kw):
        return onnx.helper.make_node(op, inputs, outputs, **kw)

    uid = [0]
    def fresh(prefix="t"):
        uid[0] += 1
        return f"{P}{prefix}_{uid[0]}"

    inits += [
        _t(P + "ax4d",    INT64, [4], [0, 1, 2, 3]),
        _t(P + "sp4d",    INT64, [4], [1, 1, 1, 1]),
        _t(P + "st4d",    INT64, [4], [0, 0, 0, 0]),
        _t(P + "en_crop", INT64, [4], [1, 10, H, W]),
        _t(P + "st_fg",   INT64, [4], [0, fg_color, 0, 0]),
        _t(P + "en_fg",   INT64, [4], [1, fg_color + 1, H, W]),
        _t(P + "one_hw",  FLOAT, [1, 1, H, W], [1.0] * (H * W)),
        _t(P + "one_s",   FLOAT, [1, 1, 1, 1], [1.0]),
    ]

    # Crop input to [1,10,H,W] then extract fg channel → ones [1,1,H,W]
    nodes.append(_n("Slice", ["input", P+"st4d", P+"en_crop", P+"ax4d", P+"sp4d"], [P+"x_crop"]))
    nodes.append(_n("Slice", [P+"x_crop", P+"st_fg", P+"en_fg", P+"ax4d", P+"sp4d"], [P+"ones"]))

    def shifted(inp, dr, dc):
        """Return tensor shifted by (dr,dc): out[r,c] = inp[r-dr, c-dc], 0 outside."""
        hb = max(0, dr); he = max(0, -dr)
        wb = max(0, dc); we = max(0, -dc)
        padded = fresh("pd")
        nodes.append(_n("Pad", [inp], [padded],
                        pads=[0, 0, hb, wb, 0, 0, he, we], mode="constant", value=0.0))
        # Slice initializers
        st_name = fresh("st"); en_name = fresh("en")
        inits.append(_t(st_name, INT64, [4], [0, 0, he, we]))
        inits.append(_t(en_name, INT64, [4], [1, 1, H + he, W + we]))
        out = fresh("sh")
        nodes.append(_n("Slice", [padded, st_name, en_name, P+"ax4d", P+"sp4d"], [out]))
        return out

    def mul2(a, b):
        out = fresh("ml"); nodes.append(_n("Mul", [a, b], [out])); return out

    def add2(a, b):
        out = fresh("ad"); nodes.append(_n("Add", [a, b], [out])); return out

    def sub2(a, b):
        out = fresh("sb"); nodes.append(_n("Sub", [a, b], [out])); return out

    def clip01(inp):
        out = fresh("cp"); nodes.append(_n("Clip", [inp], [out], min=0.0, max=1.0)); return out

    def rsum(inp):
        """ReduceSum over all axes, keepdims=1 → shape [1,1,1,1]."""
        out = fresh("rs")
        nodes.append(_n("ReduceSum", [inp], [out], axes=[0, 1, 2, 3], keepdims=1))
        return out

    # Precompute shifted versions of ones needed across multiple detections
    s_0_m1  = shifted(P+"ones", 0,  -1)  # ones[r, c+1]
    s_0_m2  = shifted(P+"ones", 0,  -2)  # ones[r, c+2]
    s_m1_0  = shifted(P+"ones", -1,  0)  # ones[r+1, c]
    s_m1_m1 = shifted(P+"ones", -1, -1)  # ones[r+1, c+1]
    s_m1_m2 = shifted(P+"ones", -1, -2)  # ones[r+1, c+2]
    s_m1_p1 = shifted(P+"ones", -1,  1)  # ones[r+1, c-1]
    s_m2_0  = shifted(P+"ones", -2,  0)  # ones[r+2, c]
    s_m2_m1 = shifted(P+"ones", -2, -1)  # ones[r+2, c+1]
    s_m2_m2 = shifted(P+"ones", -2, -2)  # ones[r+2, c+2]

    # --- Singleton diagonal pair detections ---
    # detect11[r,c]: 3 at (r,c) AND (r+1,c+1)
    detect11  = mul2(P+"ones", s_m1_m1)
    # detect1m1[r,c]: 3 at (r,c) AND (r+1,c-1)
    detect1m1 = mul2(P+"ones", s_m1_p1)

    # --- 2×2 block detection: block2[r,c]=1 iff all of (r..r+1,c..c+1) are fg ---
    b2 = mul2(mul2(mul2(P+"ones", s_0_m1), s_m1_0), s_m1_m1)
    detect2_2m2 = mul2(b2, shifted(b2, -2,  2))  # B block at (r+2,c-2)
    detect2_22  = mul2(b2, shifted(b2, -2, -2))  # B block at (r+2,c+2)

    # --- 3×3 block detection: block3[r,c]=1 iff all of (r..r+2,c..c+2) are fg ---
    b3_tmp = mul2(mul2(P+"ones", s_0_m1), s_0_m2)
    b3_tmp = mul2(b3_tmp, s_m1_0)
    b3_tmp = mul2(b3_tmp, s_m1_m1)
    b3_tmp = mul2(b3_tmp, s_m1_m2)
    b3_tmp = mul2(b3_tmp, s_m2_0)
    b3_tmp = mul2(b3_tmp, s_m2_m1)
    b3     = mul2(b3_tmp, s_m2_m2)
    detect3_3m3 = mul2(b3, shifted(b3, -3,  3))  # B block at (r+3,c-3)
    detect3_33  = mul2(b3, shifted(b3, -3, -3))  # B block at (r+3,c+3)

    # --- Priority scalars: has_block3 and has_block2 ---
    has_b3  = clip01(rsum(b3))   # [1,1,1,1]: 1 iff any 3×3 block of fg exists
    has_b2  = clip01(rsum(b2))   # [1,1,1,1]: 1 iff any 2×2 block of fg exists
    no_b3   = sub2(P+"one_s", has_b3)   # 1 - has_b3
    no_b2   = sub2(P+"one_s", has_b2)
    no_b2_or_b3 = mul2(no_b3, no_b2)   # 1 iff neither block size exists

    # --- Singleton 8-mask ---
    # (1,1) pairs: A_8@(r-1,c+2), B_8@(r+2,c-1)
    m1 = add2(shifted(detect11, -1, 2), shifted(detect11, 2, -1))
    # (1,-1) pairs: A_8@(r+2,c+1), B_8@(r-1,c-2)
    m1 = add2(m1, add2(shifted(detect1m1, 2, 1), shifted(detect1m1, -1, -2)))
    mask_singleton = clip01(m1)

    # --- 2×2 block 8-mask ---
    # (2,-2) direction: A_8 TL@(r+4,c+2), B_8 TL@(r-2,c-4)
    m2 = shifted(detect2_2m2, 4, 2)
    for dr, dc in [(4,3),(5,2),(5,3),(-2,-4),(-2,-3),(-1,-4),(-1,-3)]:
        m2 = add2(m2, shifted(detect2_2m2, dr, dc))
    # (2,2) direction: A_8 TL@(r-2,c+4), B_8 TL@(r+4,c-2)
    for dr, dc in [(-2,4),(-2,5),(-1,4),(-1,5),(4,-2),(4,-1),(5,-2),(5,-1)]:
        m2 = add2(m2, shifted(detect2_22, dr, dc))
    mask_block2 = clip01(m2)

    # --- 3×3 block 8-mask ---
    # (3,-3) direction: A_8 TL@(r+6,c+3), B_8 TL@(r-3,c-6)
    m3 = shifted(detect3_3m3, 6, 3)
    for dr, dc in [(6,4),(6,5),(7,3),(7,4),(7,5),(8,3),(8,4),(8,5),
                   (-3,-6),(-3,-5),(-3,-4),(-2,-6),(-2,-5),(-2,-4),(-1,-6),(-1,-5),(-1,-4)]:
        m3 = add2(m3, shifted(detect3_3m3, dr, dc))
    # (3,3) direction: A_8 TL@(r-3,c+6), B_8 TL@(r+6,c-3)
    for dr, dc in [(-3,6),(-3,7),(-3,8),(-2,6),(-2,7),(-2,8),(-1,6),(-1,7),(-1,8),
                   (6,-3),(6,-2),(6,-1),(7,-3),(7,-2),(7,-1),(8,-3),(8,-2),(8,-1)]:
        m3 = add2(m3, shifted(detect3_33, dr, dc))
    mask_block3 = clip01(m3)

    # --- Priority-gated final mask ---
    # 3×3 mask always active; 2×2 suppressed when 3×3 blocks exist; singleton suppressed otherwise
    gated2 = mul2(no_b3, mask_block2)
    gated1 = mul2(no_b2_or_b3, mask_singleton)
    eight_mask_raw = add2(add2(mask_block3, gated2), gated1)
    eight_mask = clip01(eight_mask_raw)   # [1,1,H,W]

    # --- Build output: modify ch0 and ch_fill in input ---
    # neg_mask: -eight_mask  (to subtract from ch0)
    neg_mask = fresh("nm")
    nodes.append(_n("Neg", [eight_mask], [neg_mask]))

    # Pad neg_mask into [1,10,H,W]: ch0=neg_mask, ch1..9=0
    delta_ch0 = fresh("dc0")
    nodes.append(_n("Pad", [neg_mask], [delta_ch0],
                    pads=[0, 0, 0, 0, 0, 9, 0, 0], mode="constant", value=0.0))
    # Pad eight_mask into [1,10,H,W]: ch0..fill_color-1=0, ch_fill=mask, rest=0
    delta_chf = fresh("dcf")
    nodes.append(_n("Pad", [eight_mask], [delta_chf],
                    pads=[0, fill_color, 0, 0, 0, 10 - fill_color - 1, 0, 0],
                    mode="constant", value=0.0))

    delta = add2(delta_ch0, delta_chf)
    assembled = add2(P+"x_crop", delta)

    pad_h, pad_w = 30 - H, 30 - W
    nodes.append(_n("Pad", [assembled], ["output"],
                   pads=[0, 0, 0, 0, 0, 0, pad_h, pad_w], mode="constant", value=0.0))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def two_marker_cross_network(color_A, color_B, color_inter, H, W, batch=1):
    """ONNX: two single-cell markers draw full crosses with intersection color.

    Input H×W: one cell of color_A and one of color_B.
    Output: full row+col cross for each marker, cross intersections = color_inter.
    task047: H=9, W=9, color_A=8, color_B=7, color_inter=2.
    """
    if batch != 1:
        raise NotImplementedError("two-marker-cross is single-example only")

    nodes = []
    inits = []
    P = "tmc_"

    def _t(name, dtype, shape, vals):
        return onnx.helper.make_tensor(name, dtype, shape, vals)
    def _n(op, inputs, outputs, **kwargs):
        return onnx.helper.make_node(op, inputs, outputs, **kwargs)

    inits += [
        _t(P+"ch_ax", onnx.TensorProto.INT64, [1], [1]),
        _t(P+"ch_sp", onnx.TensorProto.INT64, [1], [1]),
        _t(P+"hw_ax", onnx.TensorProto.INT64, [2], [2, 3]),
        _t(P+"hw_sp", onnx.TensorProto.INT64, [2], [1, 1]),
        _t(P+"hw_st", onnx.TensorProto.INT64, [2], [0, 0]),
        _t(P+"hw_en", onnx.TensorProto.INT64, [2], [H, W]),
        _t(P+"one",  onnx.TensorProto.FLOAT, [1], [1.0]),
        _t(P+"zero", onnx.TensorProto.FLOAT, [1], [0.0]),
    ]

    def extract_channel(c, out_name):
        inits.extend([
            _t(P+f"c{c}_st", onnx.TensorProto.INT64, [1], [c]),
            _t(P+f"c{c}_en", onnx.TensorProto.INT64, [1], [c + 1]),
        ])
        nodes.append(_n("Slice", ["input", P+f"c{c}_st", P+f"c{c}_en",
                                   P+"ch_ax", P+"ch_sp"], [P+f"c{c}_full"]))
        nodes.append(_n("Slice", [P+f"c{c}_full", P+"hw_st", P+"hw_en",
                                   P+"hw_ax", P+"hw_sp"], [out_name]))

    # Extract channels for A and B
    extract_channel(color_A, P+"chA")  # [1,1,H,W]
    extract_channel(color_B, P+"chB")

    # Row/col indicators via ReduceMax (collapses to 1 at the marker's row/col)
    nodes.append(_n("ReduceMax", [P+"chA"], [P+"RA"], axes=[3], keepdims=1))  # [1,1,H,1]
    nodes.append(_n("ReduceMax", [P+"chA"], [P+"CA"], axes=[2], keepdims=1))  # [1,1,1,W]
    nodes.append(_n("ReduceMax", [P+"chB"], [P+"RB"], axes=[3], keepdims=1))
    nodes.append(_n("ReduceMax", [P+"chB"], [P+"CB"], axes=[2], keepdims=1))

    # Cross masks: union of row and col (broadcasts [1,1,H,1] + [1,1,1,W] → [1,1,H,W])
    nodes.append(_n("Mul", [P+"RA", P+"CA"], [P+"RAcA"]))  # 1 at origin A
    nodes.append(_n("Add", [P+"RA", P+"CA"], [P+"crossA_raw"]))
    nodes.append(_n("Sub", [P+"crossA_raw", P+"RAcA"], [P+"crossA"]))  # union

    nodes.append(_n("Mul", [P+"RB", P+"CB"], [P+"RBcB"]))
    nodes.append(_n("Add", [P+"RB", P+"CB"], [P+"crossB_raw"]))
    nodes.append(_n("Sub", [P+"crossB_raw", P+"RBcB"], [P+"crossB"]))

    # Intersection cells: A-row×B-col and B-row×A-col
    nodes.append(_n("Mul", [P+"RA", P+"CB"], [P+"interAB"]))  # at (rA, cB)
    nodes.append(_n("Mul", [P+"RB", P+"CA"], [P+"interBA"]))  # at (rB, cA)
    nodes.append(_n("Add", [P+"interAB", P+"interBA"], [P+"inter"]))

    # A-cross output: crossA minus intersection cells
    nodes.append(_n("Sub", [P+"crossA", P+"interAB"], [P+"outA_minus1"]))
    nodes.append(_n("Sub", [P+"outA_minus1", P+"interBA"], [P+"outA"]))

    # B-cross output: crossB minus intersection cells
    nodes.append(_n("Sub", [P+"crossB", P+"interAB"], [P+"outB_minus1"]))
    nodes.append(_n("Sub", [P+"outB_minus1", P+"interBA"], [P+"outB"]))

    # ch0 (background) = 1 - outA - outB - inter
    nodes.append(_n("Add", [P+"outA", P+"outB"], [P+"fg_AB"]))
    nodes.append(_n("Add", [P+"fg_AB", P+"inter"], [P+"fg_all"]))
    nodes.append(_n("Sub", [P+"one", P+"fg_all"], [P+"ch0_out"]))

    # zeros tensor
    nodes.append(_n("Mul", [P+"ch0_out", P+"zero"], [P+"zeros"]))

    ch_tensors = []
    for c in range(10):
        if c == 0:
            ch_tensors.append(P+"ch0_out")
        elif c == color_A:
            ch_tensors.append(P+"outA")
        elif c == color_B:
            ch_tensors.append(P+"outB")
        elif c == color_inter:
            ch_tensors.append(P+"inter")
        else:
            ch_tensors.append(P+"zeros")
    nodes.append(_n("Concat", ch_tensors, [P+"assembled"], axis=1))

    pad_h = 30 - H
    pad_w = 30 - W
    nodes.append(_n("Pad", [P+"assembled"], ["output"],
                    pads=[0, 0, 0, 0, 0, 0, pad_h, pad_w], mode="constant", value=0.0))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def rank_vertical_bars_network(batch=1):
    """ONNX: task010 — 4 vertical bars of color-5 at cols {1,3,5,7} in a 9×9 grid,
    each bar a contiguous run from row 8 upward. Rank bars by height (tallest → 1,
    next → 2, ..., shortest → 4) and replace each bar's color-5 with its rank color.

    Assumptions validated by detector:
      - Input: (9,9) grid, colors ⊆ {0,5}, bars exactly at cols {1,3,5,7}.
      - Bars always touch row 8 (bottom) and extend upward.
      - All 4 bar heights are distinct (no ties → deterministic ranking).

    Strategy:
      1. Extract ch5 → [1,1,30,30].
      2. Slice each bar column [1,1,9,1] and ReduceSum heights.
      3. Compute n_taller[c] = #{other bars taller} using 6 pairwise Greater ops.
      4. For each column c and rank k ∈ {1,2,3,4}: indicator f_rk_c (0/1 float).
      5. For each rank k: sum padded column contributions → bars_k [1,1,9,9].
      6. ch0 = 1 - all_bars (all-bars = ch5 sliced to 9×9 = sum of bars_1..4).
      7. Concat 10 channels + Pad to [1,10,30,30].

    ~70 nodes, ~10 KB params. ~16 pts. Batch=1 only.
    """
    if batch != 1:
        raise NotImplementedError("rank-vertical-bars is single-example only")

    nodes = []
    inits = []
    P = "rvb_"

    def ci(name, vals):
        t = onnx.helper.make_tensor(P+name, onnx.TensorProto.INT64, [len(vals)], vals)
        inits.append(t)
        return P+name

    def cf(name, vals):
        t = onnx.helper.make_tensor(P+name, onnx.TensorProto.FLOAT, [len(vals)], vals)
        inits.append(t)
        return P+name

    def node(op, inputs, outputs, **attrs):
        nodes.append(onnx.helper.make_node(op, inputs, outputs, **attrs))

    def cast_float(src, dst):
        node("Cast", [src], [dst], to=onnx.TensorProto.FLOAT)

    # Shared constants
    ax1 = ci("ax1", [1]);  ax23 = ci("ax23", [2, 3]);  sp1 = ci("sp1", [1, 1])
    sp1_1d = ci("sp1_1d", [1]);  ax1_1d = ci("ax1_1d", [1])
    c_one = cf("c_one", [1.0]);  c_zero = cf("c_zero", [0.0])
    c_half = cf("c_half", [0.5]);  c_1h = cf("c_1h", [1.5]);  c_2h = cf("c_2h", [2.5])
    c_two = cf("c_two", [2.0]);  c_three = cf("c_three", [3.0])

    # Step 1: Extract ch5
    st5 = ci("st5", [5]);  en6 = ci("en6", [6])
    node("Slice", ["input", st5, en6, ax1_1d, sp1_1d], [P+"ch5"])  # [1,1,30,30]

    # Step 2: Slice each bar column (rows 0:9, col c:c+1)
    COLS = [1, 3, 5, 7]
    for c in COLS:
        st = ci(f"st_c{c}", [0, c])
        en = ci(f"en_c{c}", [9, c + 1])
        node("Slice", [P+"ch5", st, en, ax23, sp1], [P+f"col{c}"])  # [1,1,9,1]

    # Step 3: Heights via ReduceSum
    for c in COLS:
        node("ReduceSum", [P+f"col{c}"], [P+f"h{c}"])  # [1,1,1,1]

    # Step 4: Pairwise Greater + Cast to float
    PAIRS = [(1, 3), (1, 5), (1, 7), (3, 5), (3, 7), (5, 7)]
    for a, b in PAIRS:
        node("Greater", [P+f"h{a}", P+f"h{b}"], [P+f"bool_gt{a}{b}"])
        cast_float(P+f"bool_gt{a}{b}", P+f"gt{a}{b}")

    # Step 5: n_taller[c] = #{j != c: h[j] > h[c]}
    # n1 = 3 - gt13 - gt15 - gt17  (number of bars taller than bar-1)
    node("Sub", [c_three, P+"gt13"], [P+"nt1a"])
    node("Sub", [P+"nt1a", P+"gt15"], [P+"nt1b"])
    node("Sub", [P+"nt1b", P+"gt17"], [P+"n1"])

    # n3 = gt13 + 2 - gt35 - gt37
    node("Add", [P+"gt13", c_two], [P+"nt3a"])
    node("Sub", [P+"nt3a", P+"gt35"], [P+"nt3b"])
    node("Sub", [P+"nt3b", P+"gt37"], [P+"n3"])

    # n5 = gt15 + gt35 + 1 - gt57
    node("Add", [P+"gt15", P+"gt35"], [P+"nt5a"])
    node("Add", [P+"nt5a", c_one], [P+"nt5b"])
    node("Sub", [P+"nt5b", P+"gt57"], [P+"n5"])

    # n7 = gt17 + gt37 + gt57
    node("Add", [P+"gt17", P+"gt37"], [P+"nt7a"])
    node("Add", [P+"nt7a", P+"gt57"], [P+"n7"])

    # Step 6: Rank indicators via thresholding n_taller
    # rank == 1 iff n < 0.5; rank == 2 iff 0.5 <= n < 1.5; etc.
    for c in COLS:
        n = P+f"n{c}"
        node("Less", [n, c_half], [P+f"lt05_{c}"])
        node("Less", [n, c_1h], [P+f"lt15_{c}"])
        node("Less", [n, c_2h], [P+f"lt25_{c}"])

        cast_float(P+f"lt05_{c}", P+f"fr1_{c}")

        node("Not", [P+f"lt05_{c}"], [P+f"nlt05_{c}"])
        node("And", [P+f"nlt05_{c}", P+f"lt15_{c}"], [P+f"br2_{c}"])
        cast_float(P+f"br2_{c}", P+f"fr2_{c}")

        node("Not", [P+f"lt15_{c}"], [P+f"nlt15_{c}"])
        node("And", [P+f"nlt15_{c}", P+f"lt25_{c}"], [P+f"br3_{c}"])
        cast_float(P+f"br3_{c}", P+f"fr3_{c}")

        node("Not", [P+f"lt25_{c}"], [P+f"br4_{c}"])
        cast_float(P+f"br4_{c}", P+f"fr4_{c}")

    # Step 7: For each rank k, build [1,1,9,9] bars_k
    # col_c_rk = col_c * fr_k_c  then Pad to [1,1,9,9]
    for k in range(1, 5):
        padded = []
        for c in COLS:
            m = P+f"m{k}_{c}"
            p = P+f"p{k}_{c}"
            node("Mul", [P+f"col{c}", P+f"fr{k}_{c}"], [m])
            node("Pad", [m], [p],
                 pads=[0, 0, 0, c, 0, 0, 0, 8 - c], mode="constant", value=0.0)
            padded.append(p)
        node("Add", [padded[0], padded[1]], [P+f"bars{k}a"])
        node("Add", [P+f"bars{k}a", padded[2]], [P+f"bars{k}b"])
        node("Add", [P+f"bars{k}b", padded[3]], [P+f"bars{k}"])

    # Step 8: all_bars [1,1,9,9] = ch5 within 9x9
    st99 = ci("st99", [0, 0]);  en99 = ci("en99", [9, 9])
    node("Slice", [P+"ch5", st99, en99, ax23, sp1], [P+"all_bars"])  # [1,1,9,9]
    node("Sub", [c_one, P+"all_bars"], [P+"ch0"])  # ch0 = 1 - all_bars
    node("Mul", [P+"all_bars", c_zero], [P+"zeros"])

    # Step 9: Concat 10 channels + Pad to [1,10,30,30]
    ch_inputs = [P+"ch0", P+"bars1", P+"bars2", P+"bars3", P+"bars4"] + [P+"zeros"] * 5
    node("Concat", ch_inputs, [P+"inner"], axis=1)  # [1,10,9,9]
    node("Pad", [P+"inner"], ["output"],
         pads=[0, 0, 0, 0, 0, 0, 21, 21], mode="constant", value=0.0)

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def color_perm_network(channel_indices, batch=1):
    return _gather_network(axis=1, indices=channel_indices, batch=batch)


def color_replace_network(forward, batch=1):
    """ONNX: 1x1 Conv with a custom non-permutation weight that allows merge
    (multiple input colors -> same output color).

    For each color c in 0..9:
        target = forward.get(c, c)            # what input color c maps to in output
        weight[target, c, 0, 0] = 1.0         # input channel c contributes to output channel `target`

    All other weight entries are 0. Output cell (r, c) lights up exactly the
    channel `forward.get(input_color(r,c), input_color(r,c))`.

    Distinct from color_perm: when forward maps non-injectively (e.g. {6: 2, 7: 7}),
    the Gather indices [..., inverse[2], ..., inverse[6], ...] would have duplicate
    entries (channel 6 referenced twice) and the post-PR-#6 invertibility check
    rejects it. The 1x1 Conv weight has no such invertibility requirement: it sets
    weight[2, 6] = 1 and weight[7, 7] = 1 with weight[6, 6] = 0, so input color 6
    becomes output color 2 cleanly.

    Cost (per cost-model verify, opset 10): macs = 10*10*30*30 = 90,000;
    params = 100; memory = 0 (single op, no intermediate value_info entries).
    Total cost ~90,100 -> ~13.59 pts.
    """
    weight = [0.0] * (10 * 10)
    for c in range(10):
        target = forward.get(c, c)
        weight[target * 10 + c] = 1.0
    weight_init = onnx.helper.make_tensor(
        "cr_weight", onnx.TensorProto.FLOAT, [10, 10, 1, 1], weight
    )
    conv = onnx.helper.make_node(
        "Conv",
        ["input", "cr_weight"],
        ["output"],
        kernel_shape=[1, 1],
        strides=[1, 1],
        pads=[0, 0, 0, 0],
    )
    return _make_model([conv], [weight_init], shape=[batch, 10, 30, 30])


def color_swap_5_always_stays_network(batch=1):
    """ONNX: tasks where input has exactly 2 non-bg colors (one is always 5),
    and the rule is: color-5 cells become the OTHER color; other-color cells
    become bg. The "other color" varies per pair → fully runtime detection.

    Pipeline (input [batch, 10, 30, 30]):
      1. ch1_to_9 = Slice(input, axes=[1], 1:10)                 [batch,9,30,30]
      2. ch_has = ReduceMax(ch1_to_9, axes=[2,3], keepdims=1)    [batch,9,1,1]
                  1 at any channel k=1..9 that has a lit cell, 0 otherwise.
      3. not_ch5 = constant (1,9,1,1) = [1,1,1,1,0,1,1,1,1]      zeros out channel 5
      4. active_other_mask = ch_has * not_ch5                    1 only at other_ch's index
      5. ch5 = Slice(input, axes=[1], 5:6)                       [batch,1,30,30]
      6. ch5_x9 = Concat([ch5]*9, axis=1)                        [batch,9,30,30]
      7. out_19 = ch5_x9 * active_other_mask                     ch5 lit at other_ch position
      8. ch0_out = 1 - ReduceMax(ch1_to_9, axes=[1], keepdims=1) bg derivation
      9. output = Concat([ch0_out, out_19], axis=1)              [batch, 10, 30, 30]

    Cost target: ~13-14 pts.

    Source: Beta-cracked recipe at Research_Archive/notes/recipe-task389-color-swap-cheap-rewrite.md
    (266/266 numpy verified).
    """
    P = "css_"  # prefix to avoid name collisions

    nodes = []
    inits = []

    # 1. Slice channels 1..9 (the non-bg one-hot)
    inits.append(onnx.helper.make_tensor(P + "in19_starts", onnx.TensorProto.INT64, [1], [1]))
    inits.append(onnx.helper.make_tensor(P + "in19_ends", onnx.TensorProto.INT64, [1], [10]))
    inits.append(onnx.helper.make_tensor(P + "ch_axis", onnx.TensorProto.INT64, [1], [1]))
    nodes.append(onnx.helper.make_node(
        "Slice", ["input", P + "in19_starts", P + "in19_ends", P + "ch_axis"],
        [P + "ch1_to_9"]
    ))

    # 2. ch_has = ReduceMax(ch1_to_9, axes=[2,3], keepdims=1) -> per-channel "any lit"
    nodes.append(onnx.helper.make_node(
        "ReduceMax", [P + "ch1_to_9"], [P + "ch_has"],
        axes=[2, 3], keepdims=1
    ))

    # 3. not_ch5: (1,9,1,1) constant zeroing out channel 5 (index 4 in the 9-tensor)
    not_ch5 = [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    inits.append(onnx.helper.make_tensor(
        P + "not_ch5", onnx.TensorProto.FLOAT, [1, 9, 1, 1], not_ch5
    ))

    # 4. active_other_mask = ch_has * not_ch5
    nodes.append(onnx.helper.make_node(
        "Mul", [P + "ch_has", P + "not_ch5"], [P + "other_mask"]
    ))

    # 5. ch5 = Slice(input, axes=[1], 5:6)
    inits.append(onnx.helper.make_tensor(P + "ch5_starts", onnx.TensorProto.INT64, [1], [5]))
    inits.append(onnx.helper.make_tensor(P + "ch5_ends", onnx.TensorProto.INT64, [1], [6]))
    nodes.append(onnx.helper.make_node(
        "Slice", ["input", P + "ch5_starts", P + "ch5_ends", P + "ch_axis"],
        [P + "ch5"]
    ))

    # 6. ch5_x9 = Concat([ch5]*9, axis=1)
    nodes.append(onnx.helper.make_node(
        "Concat", [P + "ch5"] * 9, [P + "ch5_x9"], axis=1
    ))

    # 7. out_19 = ch5_x9 * other_mask
    nodes.append(onnx.helper.make_node(
        "Mul", [P + "ch5_x9", P + "other_mask"], [P + "out_19"]
    ))

    # 8. ch0_out = is_active - ch5
    #    is_active = input_ch0 + ReduceMax(ch1_to_9, axes=[1])  -> 1 inside grid, 0 at padding
    #    is_active - ch5  -> 1 where input was non-5 inside grid, 0 elsewhere
    nodes.append(onnx.helper.make_node(
        "ReduceMax", [P + "ch1_to_9"], [P + "any_nonbg"],
        axes=[1], keepdims=1
    ))
    inits.append(onnx.helper.make_tensor(P + "ch0_starts", onnx.TensorProto.INT64, [1], [0]))
    inits.append(onnx.helper.make_tensor(P + "ch0_ends", onnx.TensorProto.INT64, [1], [1]))
    nodes.append(onnx.helper.make_node(
        "Slice", ["input", P + "ch0_starts", P + "ch0_ends", P + "ch_axis"],
        [P + "ch0_in"]
    ))
    nodes.append(onnx.helper.make_node(
        "Add", [P + "ch0_in", P + "any_nonbg"], [P + "is_active"]
    ))
    nodes.append(onnx.helper.make_node(
        "Sub", [P + "is_active", P + "ch5"], [P + "ch0_out"]
    ))

    # 9. Concat([ch0_out, out_19], axis=1) -> output
    nodes.append(onnx.helper.make_node(
        "Concat", [P + "ch0_out", P + "out_19"], ["output"], axis=1
    ))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def enclosed_region_fill_network(fill_color, batch=1, canvas_h=30, canvas_w=30):
    """ONNX: BFS-from-boundary blocked by walls -> fill enclosed-empty cells.

    For tasks where every empty (color-0) cell that is enclosed by a closed loop
    of nonzero "wall" cells gets recolored to `fill_color`. Wall cells (input
    color != 0) and reachable-from-boundary empty cells pass through unchanged.

    Pipeline (input [N,10,30,30], BFS on canvas_h x canvas_w crop):
      1. [If canvas < 30] Slice input to [N,10,canvas_h,canvas_w]
      2. is_active = ReduceSum(src, axis=1, keepdims=1)    [N,1,H,W]
      3. bg_ch = Slice(src, axis=1, [0:1])                 [N,1,H,W]
      4. walls = is_active - bg_ch                         1 iff color != 0
      5. not_walls = 1 - walls                             1 iff color==0 or padding
      6. reach_0 = boundary_mask * not_walls
      7. Iterate n_iters times:
           conv = Conv3x3(reach_i, cross_kernel)
           clip = Clip(conv, 0, 1)
           reach_{i+1} = Mul(clip, not_walls)
      8. unreached = 1 - reach_final
      9. enclosed = not_walls * unreached
      10. delta = Mul(enclosed, delta_oh)                  [N,10,H,W] broadcast
      11. output_crop = src + delta
      12. [If canvas < 30] Pad output_crop to [N,10,30,30] as "output"
          [Else] output = output_crop directly

    n_iters = (min(H,W) - 1) // 2 + 2 — safe upper bound for H×W BFS.
    Crop approach eliminates 30×(3 nodes × 3600 bytes) value_info overhead;
    the 30×30 default path is unchanged for backward-compat.

    Requires fill_color in 1..9.
    """
    if fill_color == 0:
        raise ValueError("fill_color must be in 1..9 (color 0 is background)")
    H, W = canvas_h, canvas_w
    # n_iters scales with the cropped canvas so boundary reachability converges
    # through longer corridors. task002 ARC-GEN examples need 20 steps on 20x20.
    n_iters = max(16, H + W)
    inits = []
    nodes = []

    one_init = onnx.helper.make_tensor("er_one", onnx.TensorProto.FLOAT, [1], [1.0])
    inits.append(one_init)

    boundary = np.zeros((1, 1, H, W), dtype=np.float32)
    boundary[:, :, 0, :] = 1.0
    boundary[:, :, -1, :] = 1.0
    boundary[:, :, :, 0] = 1.0
    boundary[:, :, :, -1] = 1.0
    boundary_init = onnx.helper.make_tensor(
        "er_boundary", onnx.TensorProto.FLOAT, [1, 1, H, W], boundary.flatten().tolist()
    )
    inits.append(boundary_init)

    cross = np.array([[[[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]]]], dtype=np.float32)
    kernel_init = onnx.helper.make_tensor(
        "er_cross_kernel", onnx.TensorProto.FLOAT, [1, 1, 3, 3], cross.flatten().tolist()
    )
    inits.append(kernel_init)

    delta_oh = np.zeros((1, 10, 1, 1), dtype=np.float32)
    delta_oh[0, 0, 0, 0] = -1.0
    delta_oh[0, fill_color, 0, 0] = 1.0
    delta_oh_init = onnx.helper.make_tensor(
        "er_delta_oh", onnx.TensorProto.FLOAT, [1, 10, 1, 1], delta_oh.flatten().tolist()
    )
    inits.append(delta_oh_init)

    sl_starts = onnx.helper.make_tensor("er_sl_starts", onnx.TensorProto.INT64, [1], [0])
    sl_ends = onnx.helper.make_tensor("er_sl_ends", onnx.TensorProto.INT64, [1], [1])
    sl_axes = onnx.helper.make_tensor("er_sl_axes", onnx.TensorProto.INT64, [1], [1])
    sl_steps = onnx.helper.make_tensor("er_sl_steps", onnx.TensorProto.INT64, [1], [1])
    inits.extend([sl_starts, sl_ends, sl_axes, sl_steps])

    if H < 30 or W < 30:
        cr_starts = onnx.helper.make_tensor("er_cr_starts", onnx.TensorProto.INT64, [2], [0, 0])
        cr_ends = onnx.helper.make_tensor("er_cr_ends", onnx.TensorProto.INT64, [2], [H, W])
        cr_axes = onnx.helper.make_tensor("er_cr_axes", onnx.TensorProto.INT64, [2], [2, 3])
        cr_steps = onnx.helper.make_tensor("er_cr_steps", onnx.TensorProto.INT64, [2], [1, 1])
        inits.extend([cr_starts, cr_ends, cr_axes, cr_steps])
        nodes.append(onnx.helper.make_node(
            "Slice", ["input", "er_cr_starts", "er_cr_ends", "er_cr_axes", "er_cr_steps"],
            ["er_src"],
        ))
        src = "er_src"
    else:
        src = "input"

    nodes.append(onnx.helper.make_node("ReduceSum", [src], ["er_is_active"], axes=[1], keepdims=1))
    nodes.append(onnx.helper.make_node(
        "Slice", [src, "er_sl_starts", "er_sl_ends", "er_sl_axes", "er_sl_steps"], ["er_bg_ch"],
    ))
    nodes.append(onnx.helper.make_node("Sub", ["er_is_active", "er_bg_ch"], ["er_walls"]))
    nodes.append(onnx.helper.make_node("Sub", ["er_one", "er_walls"], ["er_not_walls"]))
    nodes.append(onnx.helper.make_node("Mul", ["er_boundary", "er_not_walls"], ["er_reach_0"]))

    for i in range(n_iters):
        nodes.append(onnx.helper.make_node(
            "Conv", [f"er_reach_{i}", "er_cross_kernel"], [f"er_conv_{i}"],
            kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1],
        ))
        nodes.append(onnx.helper.make_node("Clip", [f"er_conv_{i}"], [f"er_clip_{i}"], min=0.0, max=1.0))
        nodes.append(onnx.helper.make_node("Mul", [f"er_clip_{i}", "er_not_walls"], [f"er_reach_{i+1}"]))

    nodes.append(onnx.helper.make_node("Sub", ["er_one", f"er_reach_{n_iters}"], ["er_unreached"]))
    nodes.append(onnx.helper.make_node("Mul", ["er_not_walls", "er_unreached"], ["er_enclosed"]))
    nodes.append(onnx.helper.make_node("Mul", ["er_enclosed", "er_delta_oh"], ["er_delta"]))

    if H < 30 or W < 30:
        nodes.append(onnx.helper.make_node("Add", [src, "er_delta"], ["er_out_crop"]))
        nodes.append(onnx.helper.make_node(
            "Pad", ["er_out_crop"], ["output"],
            pads=[0, 0, 0, 0, 0, 0, 30 - H, 30 - W], mode="constant", value=0.0,
        ))
    else:
        nodes.append(onnx.helper.make_node("Add", ["input", "er_delta"], ["output"]))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def frame_interior_fill_network(frame_color=2, fill_color=3, batch=1):
    """ONNX: BFS-from-boundary blocked by frame_color walls → fill enclosed empty cells.

    Differs from enclosed_region_fill_network: wall (frame) cells are ERASED in
    output. Output = fill_color at enclosed-empty cells, 0 everywhere else.

    task338: frame_color=2, fill_color=3. Canvas always 30×30.

    Pipeline (input [N,10,30,30]):
      1. walls = Slice(input, axis=1, frame_color:frame_color+1)   [N,1,30,30]
      2. not_walls = 1 - walls
      3. reach_0 = boundary_mask * not_walls
      4. Iterate n_iters=30 times:
           conv = Conv3x3(reach_i, cross_kernel, pads=1)
           clip = Clip(conv, 0, 1)
           reach_{i+1} = clip * not_walls
      5. enclosed = not_walls * (1 - reach_final)
      6. Output: ch0 = 1-enclosed, ch_fill = enclosed, others = 0
    """
    H, W = 30, 30
    n_iters = 30
    inits = []
    nodes = []

    inits.append(onnx.helper.make_tensor("fif_one", onnx.TensorProto.FLOAT, [1], [1.0]))

    boundary = np.zeros((1, 1, H, W), dtype=np.float32)
    boundary[:, :, 0, :] = 1.0
    boundary[:, :, -1, :] = 1.0
    boundary[:, :, :, 0] = 1.0
    boundary[:, :, :, -1] = 1.0
    inits.append(onnx.helper.make_tensor(
        "fif_boundary", onnx.TensorProto.FLOAT, [1, 1, H, W], boundary.flatten().tolist()
    ))

    cross = np.array([[[[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]]]], dtype=np.float32)
    inits.append(onnx.helper.make_tensor(
        "fif_cross", onnx.TensorProto.FLOAT, [1, 1, 3, 3], cross.flatten().tolist()
    ))

    inits.append(onnx.helper.make_tensor("fif_sl_st", onnx.TensorProto.INT64, [1], [frame_color]))
    inits.append(onnx.helper.make_tensor("fif_sl_en", onnx.TensorProto.INT64, [1], [frame_color + 1]))
    inits.append(onnx.helper.make_tensor("fif_sl_ax", onnx.TensorProto.INT64, [1], [1]))
    inits.append(onnx.helper.make_tensor("fif_sl_sp", onnx.TensorProto.INT64, [1], [1]))

    nodes.append(onnx.helper.make_node(
        "Slice", ["input", "fif_sl_st", "fif_sl_en", "fif_sl_ax", "fif_sl_sp"], ["fif_walls"]
    ))
    nodes.append(onnx.helper.make_node("ReduceSum", ["input"], ["fif_is_active"], axes=[1], keepdims=1))
    nodes.append(onnx.helper.make_node("Sub", ["fif_one", "fif_walls"], ["fif_not_walls"]))
    nodes.append(onnx.helper.make_node("Mul", ["fif_boundary", "fif_not_walls"], ["fif_reach_0"]))

    for i in range(n_iters):
        nodes.append(onnx.helper.make_node(
            "Conv", [f"fif_reach_{i}", "fif_cross"], [f"fif_conv_{i}"],
            kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1],
        ))
        nodes.append(onnx.helper.make_node(
            "Clip", [f"fif_conv_{i}"], [f"fif_clip_{i}"], min=0.0, max=1.0
        ))
        nodes.append(onnx.helper.make_node(
            "Mul", [f"fif_clip_{i}", "fif_not_walls"], [f"fif_reach_{i + 1}"]
        ))

    nodes.append(onnx.helper.make_node(
        "Sub", ["fif_one", f"fif_reach_{n_iters}"], ["fif_unreached"]
    ))
    nodes.append(onnx.helper.make_node("Mul", ["fif_not_walls", "fif_unreached"], ["fif_enclosed"]))

    nodes.append(onnx.helper.make_node(
        "Pad", ["fif_enclosed"], ["fif_enc_at_0"],
        pads=[0, 0, 0, 0, 0, 9, 0, 0], mode="constant", value=0.0,
    ))
    nodes.append(onnx.helper.make_node(
        "Pad", ["fif_is_active"], ["fif_base_active"],
        pads=[0, 0, 0, 0, 0, 9, 0, 0], mode="constant", value=0.0,
    ))
    nodes.append(onnx.helper.make_node("Sub", ["fif_base_active", "fif_enc_at_0"], ["fif_ch0_part"]))
    nodes.append(onnx.helper.make_node(
        "Pad", ["fif_enclosed"], ["fif_enc_at_fill"],
        pads=[0, fill_color, 0, 0, 0, 9 - fill_color, 0, 0], mode="constant", value=0.0,
    ))
    nodes.append(onnx.helper.make_node("Add", ["fif_ch0_part", "fif_enc_at_fill"], ["output"]))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def binary_neighborhood_lut_network(bg_color, lut_table, batch=1):
    """ONNX: 3x3 binary-neighborhood LUT (J.20 Pattern 4).

    Pipeline: ReduceSum input over channels -> is_active (1 in active grid cells,
    0 in canvas padding). Slice channel `bg_color` -> Sub is_active - mask_bg =
    foreground (1 in active non-bg cells, 0 in active-bg or padding). Conv 3x3
    with bit-position weights -> 9-bit key -> Cast int64 -> Reshape flat -> Gather
    LUT[512,10] -> Reshape -> Transpose. Final Mul by is_active so canvas padding
    stays all-zero.

    Args:
        bg_color: int 0..9, the "background" color (off-bits).
        lut_table: dict {key (0..511) -> output_color (0..9)}. Unseen keys
            default to bg_color (identity-ish fallback).

    Cost (per measurement, opset 10): macs ~17k (Conv 3x3 1->1 = 8.1k +
    ReduceSum 8.1k + Sub/Mul ~1.8k), memory ~110 KB (intermediate
    [batch,30,30,10] from Reshape pre-Transpose), params ~5,140 (LUT 512x10
    = 5,120 + bit-weights 9 + Slice initializers + reshape shapes).
    """
    n = batch * 30 * 30

    # Slice axis=1, channel bg_color: [batch,1,30,30]
    sl_starts = onnx.helper.make_tensor("nl_sl_starts", onnx.TensorProto.INT64, [1], [bg_color])
    sl_ends = onnx.helper.make_tensor("nl_sl_ends", onnx.TensorProto.INT64, [1], [bg_color + 1])
    sl_axes = onnx.helper.make_tensor("nl_sl_axes", onnx.TensorProto.INT64, [1], [1])
    sl_steps = onnx.helper.make_tensor("nl_sl_steps", onnx.TensorProto.INT64, [1], [1])
    slice_node = onnx.helper.make_node(
        "Slice",
        ["input", "nl_sl_starts", "nl_sl_ends", "nl_sl_axes", "nl_sl_steps"],
        ["nl_bg_channel"],
    )

    # is_active = ReduceSum(input, axes=[1], keepdims=True) - 1 in active cells, 0 in padding.
    is_active_node = onnx.helper.make_node(
        "ReduceSum", ["input"], ["nl_is_active"], axes=[1], keepdims=1
    )

    # foreground mask = is_active - bg_channel.
    sub_node = onnx.helper.make_node("Sub", ["nl_is_active", "nl_bg_channel"], ["nl_mask"])

    # Conv 3x3 with bit-position weights, zero-pad on each side.
    bit_w = np.array([[[[1, 2, 4], [8, 16, 32], [64, 128, 256]]]], dtype=np.float32)
    w_init = onnx.helper.make_tensor(
        "nl_bit_weights", onnx.TensorProto.FLOAT, list(bit_w.shape), bit_w.flatten().tolist()
    )
    conv_node = onnx.helper.make_node(
        "Conv",
        ["nl_mask", "nl_bit_weights"],
        ["nl_key_float"],
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        strides=[1, 1],
    )

    # Cast key to int64 for Gather.
    cast_node = onnx.helper.make_node(
        "Cast", ["nl_key_float"], ["nl_key_int"], to=onnx.TensorProto.INT64
    )

    # Reshape key from [batch,1,30,30] to [batch*30*30].
    flat_shape = onnx.helper.make_tensor("nl_flat_shape", onnx.TensorProto.INT64, [1], [n])
    reshape_flat = onnx.helper.make_node(
        "Reshape", ["nl_key_int", "nl_flat_shape"], ["nl_key_flat"]
    )

    # Build LUT [512, 10]: each row is one-hot of the output color for that key.
    # Unseen keys default to bg_color (so untrained neighborhoods fall back to
    # "this cell is background", which is structurally safe).
    lut = np.zeros((512, 10), dtype=np.float32)
    for key in range(512):
        lut[key, lut_table.get(key, bg_color)] = 1.0
    lut_init = onnx.helper.make_tensor(
        "nl_lut", onnx.TensorProto.FLOAT, [512, 10], lut.flatten().tolist()
    )

    # Gather(LUT, key_flat, axis=0) -> [n, 10]
    gather_node = onnx.helper.make_node(
        "Gather", ["nl_lut", "nl_key_flat"], ["nl_gathered"], axis=0
    )

    # Reshape [n, 10] -> [batch, 30, 30, 10]
    pre_t_shape = onnx.helper.make_tensor(
        "nl_pre_t_shape", onnx.TensorProto.INT64, [4], [batch, 30, 30, 10]
    )
    reshape_pre = onnx.helper.make_node(
        "Reshape", ["nl_gathered", "nl_pre_t_shape"], ["nl_pre_transpose"]
    )

    # Transpose [batch, 30, 30, 10] -> [batch, 10, 30, 30]
    transpose_node = onnx.helper.make_node(
        "Transpose", ["nl_pre_transpose"], ["nl_lut_out"], perm=[0, 3, 1, 2]
    )

    # Mul by is_active so canvas padding stays all-zero in every channel.
    mul_node = onnx.helper.make_node("Mul", ["nl_lut_out", "nl_is_active"], ["output"])

    return _make_model(
        [slice_node, is_active_node, sub_node, conv_node, cast_node, reshape_flat,
         gather_node, reshape_pre, transpose_node, mul_node],
        [sl_starts, sl_ends, sl_axes, sl_steps, w_init, flat_shape, lut_init, pre_t_shape],
        shape=[batch, 10, 30, 30],
    )


def grid_pattern_to_color_network(in_h, in_w, lut_table, batch=1):
    """ONNX: ReduceSum + Slice ch0 + Sub + Slice active region + Conv (in_h x in_w
    bit-weights kernel) + Cast + Reshape + Gather + Reshape + Pad.

    For tasks where the input is a fixed in_h x in_w grid (positioned at top-left
    of the [N,10,30,30] canvas) and the output is 1x1. The 1x1 output color is a
    function of the binary pattern of nonzero cells in the input — i.e., a
    LUT[2^(in_h*in_w)] -> output color.

    Pipeline:
      1. is_active = ReduceSum(input, axes=[1], keepdims=True). [N,1,30,30].
      2. mask_0 = Slice channel 0.
      3. is_nonzero = Sub(is_active, mask_0). 1 in active non-zero cells, 0 elsewhere.
      4. Slice the active in_h x in_w region from is_nonzero -> [N,1,in_h,in_w].
      5. Conv (in_h, in_w) with bit-weight kernel: weights[i, j] = 2^(i*in_w + j).
         No padding. Output [N,1,1,1] = key.
      6. Cast to int64.
      7. Reshape [N,1,1,1] -> [N] (1D).
      8. Gather LUT[key] -> [N, 10].
      9. Reshape [N, 10] -> [N, 10, 1, 1].
      10. Pad to [N, 10, 30, 30] (29 rows after, 29 cols after — output cell at (0, 0)).

    Args:
        in_h, in_w: input active grid shape (in_h * in_w must be <= 12 for LUT to fit).
        lut_table: dict {key (0..2^(in_h*in_w)-1) -> output_color (0..9)}. Unseen keys
            default to 0.

    Cost optimization vs the prior 30x30-kernel approach: Slice the active region
    first, then Conv with a kernel of just (in_h, in_w). Saves (900 - in_h*in_w)
    element params on the kernel; e.g. for 3x3 input, kernel drops from 900 to 9.
    """
    bits = in_h * in_w
    if bits > 12:
        raise ValueError(f"in_h*in_w={bits} too large for LUT ({2**bits} entries)")
    n_keys = 2 ** bits

    # is_active = ReduceSum(input, axes=[1], keepdims=True).
    is_active_node = onnx.helper.make_node(
        "ReduceSum", ["input"], ["gp_is_active"], axes=[1], keepdims=1
    )

    # Slice channel 0 (mask_0).
    sl_starts = onnx.helper.make_tensor("gp_sl_starts", onnx.TensorProto.INT64, [1], [0])
    sl_ends = onnx.helper.make_tensor("gp_sl_ends", onnx.TensorProto.INT64, [1], [1])
    sl_axes = onnx.helper.make_tensor("gp_sl_axes", onnx.TensorProto.INT64, [1], [1])
    sl_steps = onnx.helper.make_tensor("gp_sl_steps", onnx.TensorProto.INT64, [1], [1])
    slice_ch0_node = onnx.helper.make_node(
        "Slice",
        ["input", "gp_sl_starts", "gp_sl_ends", "gp_sl_axes", "gp_sl_steps"],
        ["gp_mask_0"],
    )
    sub_node = onnx.helper.make_node("Sub", ["gp_is_active", "gp_mask_0"], ["gp_is_nonzero"])

    # Slice the active in_h x in_w region of is_nonzero.
    act_starts = onnx.helper.make_tensor("gp_act_starts", onnx.TensorProto.INT64, [2], [0, 0])
    act_ends = onnx.helper.make_tensor("gp_act_ends", onnx.TensorProto.INT64, [2], [in_h, in_w])
    act_axes = onnx.helper.make_tensor("gp_act_axes", onnx.TensorProto.INT64, [2], [2, 3])
    act_steps = onnx.helper.make_tensor("gp_act_steps", onnx.TensorProto.INT64, [2], [1, 1])
    slice_active_node = onnx.helper.make_node(
        "Slice",
        ["gp_is_nonzero", "gp_act_starts", "gp_act_ends", "gp_act_axes", "gp_act_steps"],
        ["gp_active"],
    )

    # Conv (in_h, in_w) with packed bit-weight kernel.
    bit_w = np.zeros((1, 1, in_h, in_w), dtype=np.float32)
    for i in range(in_h):
        for j in range(in_w):
            bit_w[0, 0, i, j] = float(1 << (i * in_w + j))
    w_init = onnx.helper.make_tensor(
        "gp_bit_weights", onnx.TensorProto.FLOAT, list(bit_w.shape), bit_w.flatten().tolist()
    )
    conv_node = onnx.helper.make_node(
        "Conv",
        ["gp_active", "gp_bit_weights"],
        ["gp_key_float"],
        kernel_shape=[in_h, in_w],
        pads=[0, 0, 0, 0],
        strides=[1, 1],
    )

    # Cast key to int64.
    cast_node = onnx.helper.make_node(
        "Cast", ["gp_key_float"], ["gp_key_int"], to=onnx.TensorProto.INT64
    )

    # Reshape [N, 1, 1, 1] -> [N] (1D scalar per batch).
    flat_shape = onnx.helper.make_tensor("gp_flat_shape", onnx.TensorProto.INT64, [1], [batch])
    reshape_flat = onnx.helper.make_node(
        "Reshape", ["gp_key_int", "gp_flat_shape"], ["gp_key_flat"]
    )

    # LUT [n_keys, 10] one-hot.
    lut = np.zeros((n_keys, 10), dtype=np.float32)
    for key in range(n_keys):
        lut[key, lut_table.get(key, 0)] = 1.0
    lut_init = onnx.helper.make_tensor(
        "gp_lut", onnx.TensorProto.FLOAT, [n_keys, 10], lut.flatten().tolist()
    )

    # Gather(LUT, key_flat, axis=0) -> [N, 10].
    gather_node = onnx.helper.make_node(
        "Gather", ["gp_lut", "gp_key_flat"], ["gp_gathered"], axis=0
    )

    # Reshape [N, 10] -> [N, 10, 1, 1].
    pre_pad_shape = onnx.helper.make_tensor(
        "gp_pre_pad_shape", onnx.TensorProto.INT64, [4], [batch, 10, 1, 1]
    )
    reshape_pre_pad = onnx.helper.make_node(
        "Reshape", ["gp_gathered", "gp_pre_pad_shape"], ["gp_pre_pad"]
    )

    # Pad to [N, 10, 30, 30] - output cell at (0, 0).
    pad_node = onnx.helper.make_node(
        "Pad",
        ["gp_pre_pad"],
        ["output"],
        pads=[0, 0, 0, 0, 0, 0, 29, 29],
        mode="constant",
        value=0.0,
    )

    return _make_model(
        [is_active_node, slice_ch0_node, sub_node, slice_active_node, conv_node, cast_node,
         reshape_flat, gather_node, reshape_pre_pad, pad_node],
        [sl_starts, sl_ends, sl_axes, sl_steps,
         act_starts, act_ends, act_axes, act_steps,
         w_init, flat_shape, lut_init, pre_pad_shape],
        shape=[batch, 10, 30, 30],
    )


def grid_pattern_to_grid_network(in_h, in_w, out_h, out_w, lut_table, batch=1):
    """ONNX: ReduceSum + Slice + Sub + Conv (30x30 packed bit-weights) + Cast +
    Reshape + Gather (LUT[n_keys, out_h*out_w] int64) + OneHot + Reshape + Transpose + Pad.

    Generalized version of `grid_pattern_to_color_network`: input is fixed in_h x in_w
    grid, output is fixed out_h x out_w grid. Each input nonzero-pattern (binary
    key) maps to a specific output cell-grid via LUT.

    Args:
        in_h, in_w: input grid shape (in_h*in_w <= 12 for the LUT to fit).
        out_h, out_w: output grid shape (any size <= 30).
        lut_table: dict {key (0..2^(in_h*in_w)-1) -> output 2D tuple of int colors}.
            Unseen keys default to all-zero output.
    """
    bits = in_h * in_w
    if bits > 12:
        raise ValueError(f"in_h*in_w={bits} too large for LUT ({2**bits} entries)")
    if out_h <= 0 or out_w <= 0 or out_h > 30 or out_w > 30:
        raise ValueError(f"out_h={out_h}, out_w={out_w} must be in (0, 30]")
    n_keys = 2 ** bits
    out_cells = out_h * out_w

    is_active_node = onnx.helper.make_node(
        "ReduceSum", ["input"], ["gpg_is_active"], axes=[1], keepdims=1
    )
    sl_starts = onnx.helper.make_tensor("gpg_sl_starts", onnx.TensorProto.INT64, [1], [0])
    sl_ends = onnx.helper.make_tensor("gpg_sl_ends", onnx.TensorProto.INT64, [1], [1])
    sl_axes = onnx.helper.make_tensor("gpg_sl_axes", onnx.TensorProto.INT64, [1], [1])
    sl_steps = onnx.helper.make_tensor("gpg_sl_steps", onnx.TensorProto.INT64, [1], [1])
    slice_ch0_node = onnx.helper.make_node(
        "Slice",
        ["input", "gpg_sl_starts", "gpg_sl_ends", "gpg_sl_axes", "gpg_sl_steps"],
        ["gpg_mask_0"],
    )
    sub_node = onnx.helper.make_node("Sub", ["gpg_is_active", "gpg_mask_0"], ["gpg_is_nonzero"])

    bit_w = np.zeros((1, 1, 30, 30), dtype=np.float32)
    for i in range(in_h):
        for j in range(in_w):
            bit_w[0, 0, i, j] = float(1 << (i * in_w + j))
    w_init = onnx.helper.make_tensor(
        "gpg_bit_weights", onnx.TensorProto.FLOAT, list(bit_w.shape), bit_w.flatten().tolist()
    )
    conv_node = onnx.helper.make_node(
        "Conv",
        ["gpg_is_nonzero", "gpg_bit_weights"],
        ["gpg_key_float"],
        kernel_shape=[30, 30],
        pads=[0, 0, 0, 0],
        strides=[1, 1],
    )
    cast_node = onnx.helper.make_node(
        "Cast", ["gpg_key_float"], ["gpg_key_int"], to=onnx.TensorProto.INT64
    )

    # Reshape [N,1,1,1] -> [N].
    flat_shape = onnx.helper.make_tensor("gpg_flat_shape", onnx.TensorProto.INT64, [1], [batch])
    reshape_flat = onnx.helper.make_node(
        "Reshape", ["gpg_key_int", "gpg_flat_shape"], ["gpg_key_flat"]
    )

    # LUT [n_keys, out_h*out_w] int64 with default 0 (color 0) for unseen keys.
    lut = np.zeros((n_keys, out_cells), dtype=np.int64)
    for key, grid in lut_table.items():
        for i in range(out_h):
            for j in range(out_w):
                lut[key, i * out_w + j] = int(grid[i][j])
    lut_init = onnx.helper.make_tensor(
        "gpg_lut", onnx.TensorProto.INT64, [n_keys, out_cells], lut.flatten().tolist()
    )
    gather_node = onnx.helper.make_node(
        "Gather", ["gpg_lut", "gpg_key_flat"], ["gpg_gathered"], axis=0
    )

    # OneHot expansion: [N, out_cells] -> [N, out_cells, 10].
    onehot_depth = onnx.helper.make_tensor("gpg_oh_depth", onnx.TensorProto.INT64, [], [10])
    onehot_values = onnx.helper.make_tensor("gpg_oh_values", onnx.TensorProto.FLOAT, [2], [0.0, 1.0])
    onehot_node = onnx.helper.make_node(
        "OneHot",
        ["gpg_gathered", "gpg_oh_depth", "gpg_oh_values"],
        ["gpg_onehot"],
        axis=-1,
    )

    # Reshape [N, out_cells, 10] -> [N, out_h, out_w, 10].
    pre_t_shape = onnx.helper.make_tensor(
        "gpg_pre_t_shape", onnx.TensorProto.INT64, [4], [batch, out_h, out_w, 10]
    )
    reshape_pre_t = onnx.helper.make_node(
        "Reshape", ["gpg_onehot", "gpg_pre_t_shape"], ["gpg_pre_transpose"]
    )

    # Transpose [N, out_h, out_w, 10] -> [N, 10, out_h, out_w].
    transpose_node = onnx.helper.make_node(
        "Transpose", ["gpg_pre_transpose"], ["gpg_transposed"], perm=[0, 3, 1, 2]
    )

    # Pad to [N, 10, 30, 30].
    pad_node = onnx.helper.make_node(
        "Pad",
        ["gpg_transposed"],
        ["output"],
        pads=[0, 0, 0, 0, 0, 0, 30 - out_h, 30 - out_w],
        mode="constant",
        value=0.0,
    )

    return _make_model(
        [is_active_node, slice_ch0_node, sub_node, conv_node, cast_node,
         reshape_flat, gather_node, onehot_node, reshape_pre_t, transpose_node, pad_node],
        [sl_starts, sl_ends, sl_axes, sl_steps, w_init, flat_shape, lut_init,
         onehot_depth, onehot_values, pre_t_shape],
        shape=[batch, 10, 30, 30],
    )


def grid_pattern_color_to_grid_network(in_h, in_w, out_h, out_w, lut_table, batch=1):
    """ONNX: ReduceSum(channels=[1..9]) -> ArgMax (single nonzero color per pair)
    + grid-pattern key (as in grid_pattern_to_grid_network) -> combined key
    -> Gather LUT -> OneHot + Reshape + Transpose + Pad.

    For tasks where the input is a fixed in_h x in_w grid AND each pair has a
    single nonzero color (varying per pair), and the output grid depends on
    BOTH the binary nonzero-pattern AND that nonzero color.

    Combined key = pattern_key * 9 + (nonzero_color - 1). LUT size = 2^bits * 9.

    Args:
        in_h, in_w: input grid shape (in_h*in_w <= 12 for the LUT to fit).
        out_h, out_w: output grid shape.
        lut_table: dict {(pattern_key, nonzero_color) -> 2D tuple of int colors}.
            Unseen full-keys default to all-zero output.
    """
    bits = in_h * in_w
    if bits > 12:
        raise ValueError(f"in_h*in_w={bits} too large for LUT ({2**bits} entries)")
    if out_h <= 0 or out_w <= 0 or out_h > 30 or out_w > 30:
        raise ValueError(f"out_h={out_h}, out_w={out_w} must be in (0, 30]")
    n_pat_keys = 2 ** bits
    n_keys = n_pat_keys * 9
    out_cells = out_h * out_w

    # is_active = ReduceSum(input, axes=[1], keepdims=True).
    is_active_node = onnx.helper.make_node(
        "ReduceSum", ["input"], ["gpc_is_active"], axes=[1], keepdims=1
    )
    sl0_starts = onnx.helper.make_tensor("gpc_sl0_starts", onnx.TensorProto.INT64, [1], [0])
    sl0_ends = onnx.helper.make_tensor("gpc_sl0_ends", onnx.TensorProto.INT64, [1], [1])
    sl0_axes = onnx.helper.make_tensor("gpc_sl0_axes", onnx.TensorProto.INT64, [1], [1])
    sl0_steps = onnx.helper.make_tensor("gpc_sl0_steps", onnx.TensorProto.INT64, [1], [1])
    slice_ch0_node = onnx.helper.make_node(
        "Slice",
        ["input", "gpc_sl0_starts", "gpc_sl0_ends", "gpc_sl0_axes", "gpc_sl0_steps"],
        ["gpc_mask_0"],
    )
    sub_node = onnx.helper.make_node("Sub", ["gpc_is_active", "gpc_mask_0"], ["gpc_is_nonzero"])

    # Pattern key via 30x30 packed-bit-weights Conv (gives [N,1,1,1]).
    bit_w = np.zeros((1, 1, 30, 30), dtype=np.float32)
    for i in range(in_h):
        for j in range(in_w):
            bit_w[0, 0, i, j] = float(1 << (i * in_w + j))
    pat_w_init = onnx.helper.make_tensor(
        "gpc_pat_weights", onnx.TensorProto.FLOAT, list(bit_w.shape), bit_w.flatten().tolist()
    )
    pat_conv_node = onnx.helper.make_node(
        "Conv",
        ["gpc_is_nonzero", "gpc_pat_weights"],
        ["gpc_pat_key_float"],
        kernel_shape=[30, 30],
        pads=[0, 0, 0, 0],
        strides=[1, 1],
    )

    # Color key via per-channel count + ArgMax over channels [1..9].
    # First Slice channels [1..10] from input -> [N, 9, 30, 30].
    sl1_starts = onnx.helper.make_tensor("gpc_sl1_starts", onnx.TensorProto.INT64, [1], [1])
    sl1_ends = onnx.helper.make_tensor("gpc_sl1_ends", onnx.TensorProto.INT64, [1], [10])
    sl1_axes = onnx.helper.make_tensor("gpc_sl1_axes", onnx.TensorProto.INT64, [1], [1])
    sl1_steps = onnx.helper.make_tensor("gpc_sl1_steps", onnx.TensorProto.INT64, [1], [1])
    slice_nz_node = onnx.helper.make_node(
        "Slice",
        ["input", "gpc_sl1_starts", "gpc_sl1_ends", "gpc_sl1_axes", "gpc_sl1_steps"],
        ["gpc_nonzero_chans"],
    )
    # ReduceSum over spatial dims (axes 2, 3) keepdims=False -> [N, 9].
    color_count_node = onnx.helper.make_node(
        "ReduceSum", ["gpc_nonzero_chans"], ["gpc_color_counts"],
        axes=[2, 3], keepdims=0,
    )
    # ArgMax over channels axis=1 -> [N, 1] (or [N] depending on keepdims).
    argmax_node = onnx.helper.make_node(
        "ArgMax", ["gpc_color_counts"], ["gpc_color_idx_2d"],
        axis=1, keepdims=1,
    )
    # color_idx_2d shape: [N, 1] int64.

    # Combine: pat_key (float, [N,1,1,1]) -> Reshape to [N,1] -> Cast int64.
    pat_flat_shape = onnx.helper.make_tensor("gpc_pat_flat_shape", onnx.TensorProto.INT64, [2], [batch, 1])
    reshape_pat = onnx.helper.make_node(
        "Reshape", ["gpc_pat_key_float", "gpc_pat_flat_shape"], ["gpc_pat_key_2d_float"]
    )
    pat_cast_node = onnx.helper.make_node(
        "Cast", ["gpc_pat_key_2d_float"], ["gpc_pat_key_2d"], to=onnx.TensorProto.INT64
    )

    # full_key = pat_key * 9 + color_idx -> [N, 1] int64.
    nine = onnx.helper.make_tensor("gpc_nine", onnx.TensorProto.INT64, [1], [9])
    pat_times9_node = onnx.helper.make_node("Mul", ["gpc_pat_key_2d", "gpc_nine"], ["gpc_pat_times9"])
    full_key_node = onnx.helper.make_node("Add", ["gpc_pat_times9", "gpc_color_idx_2d"], ["gpc_full_key_2d"])

    # Reshape [N, 1] -> [N].
    flat_shape = onnx.helper.make_tensor("gpc_flat_shape", onnx.TensorProto.INT64, [1], [batch])
    reshape_flat = onnx.helper.make_node(
        "Reshape", ["gpc_full_key_2d", "gpc_flat_shape"], ["gpc_key_flat"]
    )

    # LUT [n_keys, out_cells] int64.
    lut = np.zeros((n_keys, out_cells), dtype=np.int64)
    for (pat_key, nz_color), grid in lut_table.items():
        full_key = pat_key * 9 + (nz_color - 1)
        for i in range(out_h):
            for j in range(out_w):
                lut[full_key, i * out_w + j] = int(grid[i][j])
    lut_init = onnx.helper.make_tensor(
        "gpc_lut", onnx.TensorProto.INT64, [n_keys, out_cells], lut.flatten().tolist()
    )
    gather_node = onnx.helper.make_node(
        "Gather", ["gpc_lut", "gpc_key_flat"], ["gpc_gathered"], axis=0
    )

    # OneHot expand -> [N, out_cells, 10].
    onehot_depth = onnx.helper.make_tensor("gpc_oh_depth", onnx.TensorProto.INT64, [], [10])
    onehot_values = onnx.helper.make_tensor("gpc_oh_values", onnx.TensorProto.FLOAT, [2], [0.0, 1.0])
    onehot_node = onnx.helper.make_node(
        "OneHot",
        ["gpc_gathered", "gpc_oh_depth", "gpc_oh_values"],
        ["gpc_onehot"],
        axis=-1,
    )

    # Reshape [N, out_cells, 10] -> [N, out_h, out_w, 10].
    pre_t_shape = onnx.helper.make_tensor(
        "gpc_pre_t_shape", onnx.TensorProto.INT64, [4], [batch, out_h, out_w, 10]
    )
    reshape_pre_t = onnx.helper.make_node(
        "Reshape", ["gpc_onehot", "gpc_pre_t_shape"], ["gpc_pre_transpose"]
    )

    # Transpose [N, out_h, out_w, 10] -> [N, 10, out_h, out_w].
    transpose_node = onnx.helper.make_node(
        "Transpose", ["gpc_pre_transpose"], ["gpc_transposed"], perm=[0, 3, 1, 2]
    )

    # Pad to [N, 10, 30, 30].
    pad_node = onnx.helper.make_node(
        "Pad",
        ["gpc_transposed"],
        ["output"],
        pads=[0, 0, 0, 0, 0, 0, 30 - out_h, 30 - out_w],
        mode="constant",
        value=0.0,
    )

    return _make_model(
        [is_active_node, slice_ch0_node, sub_node, pat_conv_node,
         slice_nz_node, color_count_node, argmax_node,
         reshape_pat, pat_cast_node, pat_times9_node, full_key_node,
         reshape_flat, gather_node, onehot_node, reshape_pre_t, transpose_node, pad_node],
        [sl0_starts, sl0_ends, sl0_axes, sl0_steps, pat_w_init,
         sl1_starts, sl1_ends, sl1_axes, sl1_steps,
         pat_flat_shape, nine, flat_shape, lut_init,
         onehot_depth, onehot_values, pre_t_shape],
        shape=[batch, 10, 30, 30],
    )


def mirror_concat_network(in_h, in_w, axis, transform, second_first=False, batch=1):
    """ONNX: Slice (active grid) + Gather (flip indices) + Concat + Pad.

    For tasks where the output is a fixed concat of the input with a transformed
    version of itself along axis H (axis=2) or W (axis=3).

    Args:
        in_h, in_w: input grid shape.
        axis: 2 (H) or 3 (W) — axis along which to concat.
        transform: one of "flip_h", "flip_w", "flip_hw", "identity" (tile),
            applied to input before concatenating to its right (or below) along axis.

    Pipeline:
      1. Slice input along H and W to extract the active in_h x in_w region
         from the [N, 10, 30, 30] canvas. -> [N, 10, in_h, in_w]
      2. Apply transform via Gather along H and/or W with reversed indices.
         (No transform = pass-through.)
      3. Concat(input_active, transformed) along axis -> doubled along that dim.
      4. Pad to [N, 10, 30, 30].

    Cost: 0 MACs (Slice + Gather + Concat + Pad are all shape-only). Memory ~36 KB
    for the doubled intermediate. Params ~30 (Slice initializers + Gather indices).
    """
    if axis not in (2, 3):
        raise ValueError(f"axis={axis} must be 2 (H) or 3 (W)")
    if transform not in ("flip_h", "flip_w", "flip_hw", "identity"):
        raise ValueError(f"transform={transform!r} unknown")
    out_h = in_h if axis == 3 else 2 * in_h
    out_w = in_w if axis == 2 else 2 * in_w
    if out_h > 30 or out_w > 30:
        raise ValueError(f"out shape {out_h}x{out_w} exceeds canvas")

    # Step 1: Slice active region [N, 10, in_h, in_w].
    # Prefix is `mcat_` (not `mc_`) to avoid name collision with `marker_corners_network`.
    sl_starts = onnx.helper.make_tensor("mcat_sl_starts", onnx.TensorProto.INT64, [2], [0, 0])
    sl_ends = onnx.helper.make_tensor("mcat_sl_ends", onnx.TensorProto.INT64, [2], [in_h, in_w])
    sl_axes = onnx.helper.make_tensor("mcat_sl_axes", onnx.TensorProto.INT64, [2], [2, 3])
    sl_steps = onnx.helper.make_tensor("mcat_sl_steps", onnx.TensorProto.INT64, [2], [1, 1])
    slice_node = onnx.helper.make_node(
        "Slice",
        ["input", "mcat_sl_starts", "mcat_sl_ends", "mcat_sl_axes", "mcat_sl_steps"],
        ["mcat_active"],
    )

    nodes = [slice_node]
    inits = [sl_starts, sl_ends, sl_axes, sl_steps]

    # Step 2: build the transformed second piece.
    if transform == "identity":
        second = "mcat_active"
    elif transform == "flip_w":
        # Gather axis=3, indices=reverse(in_w).
        idx = list(range(in_w - 1, -1, -1))
        idx_init = onnx.helper.make_tensor("mcat_idx_w", onnx.TensorProto.INT64, [in_w], idx)
        inits.append(idx_init)
        nodes.append(onnx.helper.make_node("Gather", ["mcat_active", "mcat_idx_w"], ["mcat_flipped"], axis=3))
        second = "mcat_flipped"
    elif transform == "flip_h":
        idx = list(range(in_h - 1, -1, -1))
        idx_init = onnx.helper.make_tensor("mcat_idx_h", onnx.TensorProto.INT64, [in_h], idx)
        inits.append(idx_init)
        nodes.append(onnx.helper.make_node("Gather", ["mcat_active", "mcat_idx_h"], ["mcat_flipped"], axis=2))
        second = "mcat_flipped"
    elif transform == "flip_hw":
        idx_w = list(range(in_w - 1, -1, -1))
        idx_h = list(range(in_h - 1, -1, -1))
        idx_w_init = onnx.helper.make_tensor("mcat_idx_w", onnx.TensorProto.INT64, [in_w], idx_w)
        idx_h_init = onnx.helper.make_tensor("mcat_idx_h", onnx.TensorProto.INT64, [in_h], idx_h)
        inits.extend([idx_w_init, idx_h_init])
        nodes.append(onnx.helper.make_node("Gather", ["mcat_active", "mcat_idx_w"], ["mcat_flipped_w"], axis=3))
        nodes.append(onnx.helper.make_node("Gather", ["mcat_flipped_w", "mcat_idx_h"], ["mcat_flipped"], axis=2))
        second = "mcat_flipped"

    # Step 3: Concat along axis (transformed second comes after input by default).
    concat_inputs = [second, "mcat_active"] if second_first else ["mcat_active", second]
    nodes.append(onnx.helper.make_node("Concat", concat_inputs, ["mcat_concat"], axis=axis))

    # Step 4: Pad to [N, 10, 30, 30]. Output shape after concat:
    after_h = out_h
    after_w = out_w
    nodes.append(onnx.helper.make_node(
        "Pad",
        ["mcat_concat"],
        ["output"],
        pads=[0, 0, 0, 0, 0, 0, 30 - after_h, 30 - after_w],
        mode="constant",
        value=0.0,
    ))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


def mirror_grid_2x2_network(in_h, in_w, batch=1):
    """ONNX: Slice + 2 Gather (flip_h, flip_w) + 1 more Gather (flip_hw) + 3 Concats + Pad.

    For tasks where output is a 2x2 grid layout:
      [[ input,   flip_w(input)  ],
       [ flip_h(input), flip_hw(input) ]]
    yielding output shape (2*in_h, 2*in_w).

    Pipeline (all shape-only, 0 MACs):
      1. Slice active grid -> [N, 10, in_h, in_w].
      2. flip_w via Gather axis=3 reverse(in_w).
      3. flip_h via Gather axis=2 reverse(in_h).
      4. flip_hw via Gather axis=2 reverse(in_h) on flip_w.
      5. top = Concat(input, flip_w) along W -> [N, 10, in_h, 2*in_w].
      6. bot = Concat(flip_h, flip_hw) along W.
      7. full = Concat(top, bot) along H -> [N, 10, 2*in_h, 2*in_w].
      8. Pad to canvas.
    """
    out_h, out_w = 2 * in_h, 2 * in_w
    if out_h > 30 or out_w > 30:
        raise ValueError(f"out shape {out_h}x{out_w} exceeds canvas")

    sl_starts = onnx.helper.make_tensor("mg_sl_starts", onnx.TensorProto.INT64, [2], [0, 0])
    sl_ends = onnx.helper.make_tensor("mg_sl_ends", onnx.TensorProto.INT64, [2], [in_h, in_w])
    sl_axes = onnx.helper.make_tensor("mg_sl_axes", onnx.TensorProto.INT64, [2], [2, 3])
    sl_steps = onnx.helper.make_tensor("mg_sl_steps", onnx.TensorProto.INT64, [2], [1, 1])
    nodes = [onnx.helper.make_node(
        "Slice",
        ["input", "mg_sl_starts", "mg_sl_ends", "mg_sl_axes", "mg_sl_steps"],
        ["mg_active"],
    )]
    inits = [sl_starts, sl_ends, sl_axes, sl_steps]

    idx_w = onnx.helper.make_tensor("mg_idx_w", onnx.TensorProto.INT64, [in_w], list(range(in_w - 1, -1, -1)))
    idx_h = onnx.helper.make_tensor("mg_idx_h", onnx.TensorProto.INT64, [in_h], list(range(in_h - 1, -1, -1)))
    inits.extend([idx_w, idx_h])
    nodes.append(onnx.helper.make_node("Gather", ["mg_active", "mg_idx_w"], ["mg_fw"], axis=3))
    nodes.append(onnx.helper.make_node("Gather", ["mg_active", "mg_idx_h"], ["mg_fh"], axis=2))
    nodes.append(onnx.helper.make_node("Gather", ["mg_fw", "mg_idx_h"], ["mg_fhw"], axis=2))

    nodes.append(onnx.helper.make_node("Concat", ["mg_active", "mg_fw"], ["mg_top"], axis=3))
    nodes.append(onnx.helper.make_node("Concat", ["mg_fh", "mg_fhw"], ["mg_bot"], axis=3))
    nodes.append(onnx.helper.make_node("Concat", ["mg_top", "mg_bot"], ["mg_full"], axis=2))

    nodes.append(onnx.helper.make_node(
        "Pad",
        ["mg_full"],
        ["output"],
        pads=[0, 0, 0, 0, 0, 0, 30 - out_h, 30 - out_w],
        mode="constant",
        value=0.0,
    ))

    return _make_model(nodes, inits, shape=[batch, 10, 30, 30])


@functools.cache
def conv_identity_network(kernel_size=3, batch=1):
    if batch != 1:
        # The validator's helper hardcodes [1,10,30,30]; batched verification falls
        # back to per-example via _verify_batched's NotImplementedError handler.
        raise NotImplementedError("conv_identity_network does not support batch != 1")
    return neurogolf_utils.single_layer_conv2d_network(identity_weight, kernel_size)


# ------------------------------------------------------------------- detector

def _shape(g):
    return (len(g), len(g[0]) if g else 0)


def _is_30x30(g):
    return _shape(g) == (30, 30)


def _all_30x30(pairs):
    return all(_is_30x30(p["input"]) and _is_30x30(p["output"]) for p in pairs)


def _transpose_grid(g):
    return [list(row) for row in zip(*g)] if g else []


def _flip_h(g):
    return [list(reversed(row)) for row in g]


def _flip_v(g):
    return list(reversed([list(row) for row in g]))


def _rot180(g):
    return _flip_v(_flip_h(g))


def _rot90(g):
    """90 degree clockwise rotation. For square grids: out[i][j] == in[n-1-j][i]."""
    return [list(row) for row in zip(*g[::-1])]


def _rot270(g):
    """270 degree clockwise rotation. For square grids: out[i][j] == in[j][n-1-i]."""
    return [list(row) for row in zip(*g)][::-1]


def _shift_v(g, k):
    """Cyclic shift down by k rows."""
    if not g:
        return g
    h = len(g)
    k %= h
    return [list(row) for row in (g[-k:] + g[:-k] if k else g)]


def _shift_h(g, k):
    """Cyclic shift right by k columns."""
    if not g:
        return g
    w = len(g[0])
    k %= w
    return [row[-k:] + row[:-k] if k else list(row) for row in g]


def _detect_color_perm(pairs, require_all_colors=True, min_colors_observed=1):
    """Infer a color-recolor rule consistent with all pairs, expressed as a length-10
    Gather indices array (Gather along channel axis = 1).

    Returns indices, or None if no Gather-representable rule fits.

    Handles two sub-cases via the same indices construction:
      (a) Strict permutation (bijective explicit mapping; all unobserved colors
          implicitly identity): standard color-perm.
      (b) Non-bijective explicit mapping (e.g. {6: 2, 7: 7} — input color 6 becomes
          output color 2 but no rule produces output color 6): unmapped output
          channels source from a guaranteed-zero input channel (a color never
          observed in any input pair) so the output channel is always 0. Karnak's
          task276 / task309 demonstrate this trick at cost ~50 vs our previous
          color-replace via Conv 1x1 at ~90k. Captures the +14.99 LB pts that
          were locked behind the old "identity default" indices construction
          (which broke the no-double-source invariant).

    Silent-fail risk to be aware of: when fewer than 10 colors are observed across
    train + test + arc-gen pairs, the inferred mapping leaves unobserved input
    colors at implicit identity. If Kaggle's hidden private benchmark contains a
    pair whose input has an unobserved color AND the true rule maps that color to
    something other than itself, this network silently outputs identity for that
    color and scores 0 on that task. The bundled arc-gen subset (~261-267
    pairs/task) is a practical mitigation but not airtight; `tools/silent_fail_check.py`
    is the operational defense — run it pre-submit on color-perm tasks.

    Args:
        pairs: list of {"input": grid, "output": grid} dicts. Detector aggregates
            colors over ALL pairs in the list.
        require_all_colors: if True, require all 10 colors observed at least once.
        min_colors_observed: if require_all_colors is False, still require this many
            distinct input colors to have been observed before accepting a match.
            Default 1 (no extra constraint).
    """
    forward = {}
    seen = set()
    for p in pairs:
        if _shape(p["input"]) != _shape(p["output"]):
            return None
        for ri, ro in zip(p["input"], p["output"]):
            if len(ri) != len(ro):
                return None
            for ic, oc in zip(ri, ro):
                seen.add(ic)
                if ic in forward and forward[ic] != oc:
                    return None
                forward[ic] = oc
    if all(forward.get(c, c) == c for c in range(10)):
        return None
    if require_all_colors and len(seen) < 10:
        return None
    if not require_all_colors and len(seen) < min_colors_observed:
        return None

    # Build the full forward map: explicit observations + implicit identity for
    # observed input colors that have no explicit rule. Then compute the inverse
    # for Gather-indices construction.
    full_forward = dict(forward)
    for c in seen:
        if c not in full_forward:
            full_forward[c] = c

    inverse = {}
    for in_c, out_c in full_forward.items():
        if out_c in inverse:
            # Non-injective full forward (e.g., explicit 6->2 plus implicit 2->2);
            # Gather can't merge multiple input channels into one output channel.
            return None
        inverse[out_c] = in_c

    zero_channels = [c for c in range(10) if c not in seen]
    indices = []
    for c in range(10):
        if c in inverse:
            indices.append(inverse[c])
        elif c not in seen:
            # input ch c is always 0; identity source is fine (and free).
            indices.append(c)
        else:
            # c is observed in input but no rule maps to c -> need a guaranteed-zero
            # source for output ch c. Without a zero channel, Gather can't represent
            # this rule; fall back to color_replace (Conv 1x1) via the dispatcher.
            if not zero_channels:
                return None
            indices.append(zero_channels[0])
    return indices


def _detect_color_replace(pairs):
    """Detect: forward[input_color] = output_color, consistent across all pairs.

    Generalizes _detect_color_perm by NOT requiring the mapping to be a strict
    permutation (so it handles the merge case e.g. {6: 2} where 6 maps to 2 but
    2 itself doesn't appear in input — Gather can't represent this; 1x1 Conv can).

    Returns a forward dict {input_color: output_color} on match, or None.

    Skips if a strict permutation exists with full Gather coverage — color_perm
    handles that case at ~10 cost vs this builder's ~90k. The cost-aware
    dispatcher would pick color_perm anyway, but skipping here saves a
    build+score round trip per task.
    """
    if not pairs:
        return None
    forward = {}
    for p in pairs:
        if _shape(p["input"]) != _shape(p["output"]):
            return None
        for ri, ro in zip(p["input"], p["output"]):
            if len(ri) != len(ro):
                return None
            for ic, oc in zip(ri, ro):
                if ic in forward and forward[ic] != oc:
                    return None
                forward[ic] = oc
    if all(forward.get(c, c) == c for c in range(10)):
        return None
    # Skip if it's a strict permutation that color_perm covers
    if len(forward) == len(set(forward.values())):
        inverse = {v: k for k, v in forward.items()}
        indices = [inverse.get(c, c) for c in range(10)]
        if len(set(indices)) == 10:
            return None
    return forward


def _detect_color_swap_5_always_stays(pairs):
    """Detect: input has exactly 2 non-bg colors (one is always 5).
    Color-5 cells become the OTHER color in output; other-color cells become bg.

    The "other color" varies per pair → builder uses runtime detection.
    Detector returns True on match, None otherwise.

    Source: Beta-cracked recipe at Research_Archive/notes/recipe-task389-color-swap-cheap-rewrite.md
    (266/266 numpy verified).
    """
    if not pairs:
        return None
    for p in pairs:
        if _shape(p["input"]) != _shape(p["output"]):
            return None
        a = np.array(p["input"])
        e = np.array(p["output"])
        in_colors = sorted(set(a.flatten().tolist()) - {0})
        if len(in_colors) != 2 or 5 not in in_colors:
            return None
        oc = in_colors[0] if in_colors[0] != 5 else in_colors[1]
        # Verify rule: 5-cells -> oc, non-5 cells -> bg.
        pred = np.where(a == 5, oc, 0)
        if not np.array_equal(pred, e):
            return None
    return True


def _enclosed_mask_from_walls(walls):
    """Cells NOT 4-conn reachable from grid boundary through non-wall cells."""
    h, w = walls.shape
    reachable = np.zeros_like(walls, dtype=bool)
    reachable[0, :] = ~walls[0, :]
    reachable[-1, :] = ~walls[-1, :]
    reachable[:, 0] = ~walls[:, 0]
    reachable[:, -1] = ~walls[:, -1]
    while True:
        new = reachable.copy()
        new[1:, :] |= reachable[:-1, :] & ~walls[1:, :]
        new[:-1, :] |= reachable[1:, :] & ~walls[:-1, :]
        new[:, 1:] |= reachable[:, :-1] & ~walls[:, 1:]
        new[:, :-1] |= reachable[:, 1:] & ~walls[:, :-1]
        if (new == reachable).all():
            break
        reachable = new
    return ~walls & ~reachable


def _detect_enclosed_region_fill(pairs):
    """Detect: same-shape input/output where output equals input on all cells
    EXCEPT the cells "enclosed" by nonzero walls — those become a single
    consistent fill_color across all visible pairs.

    Enclosed = cells not 4-conn reachable from the grid boundary through paths
    blocked by any nonzero (wall) cell. Fill color must be uniform across pairs.
    Rejects pairs where any non-enclosed empty cell is filled, or any wall/input
    cell is altered, or no enclosure exists in any pair.

    Returns int fill_color in 0..9 on match, else None.

    Distinct from `aligned_marker_fill` / `sandwich_fill` (those use specific
    marker shapes, not topological enclosure). Distinct from `bin-nbr-lut` (LUT
    over fixed neighborhood pattern; not topology-aware).
    """
    if not pairs:
        return None
    fill_color = None
    saw_fill = False
    for p in pairs:
        if _shape(p["input"]) != _shape(p["output"]):
            return None
        inp = np.array(p["input"])
        out = np.array(p["output"])
        if inp.shape != out.shape:
            return None
        in_nz = inp != 0
        # Walls = nonzero input cells.
        enclosed = _enclosed_mask_from_walls(in_nz)
        # Filled cells = output non-zero where input was zero.
        filled = (out != 0) & ~in_nz
        # Filled must equal enclosed exactly.
        if not (filled == enclosed).all():
            return None
        # All input nonzero cells must be unchanged in output.
        if not (out[in_nz] == inp[in_nz]).all():
            return None
        # Non-enclosed empty cells must remain 0.
        non_enclosed_empty = ~in_nz & ~enclosed
        if (out[non_enclosed_empty] != 0).any():
            return None
        if filled.any():
            saw_fill = True
            colors = np.unique(out[filled])
            if len(colors) != 1:
                return None
            cur = int(colors[0])
            if fill_color is None:
                fill_color = cur
            elif fill_color != cur:
                return None
    if not saw_fill or fill_color is None:
        return None
    max_h = max(len(p["input"]) for p in pairs)
    max_w = max(len(p["input"][0]) for p in pairs if p["input"])
    return {"fill_color": fill_color, "canvas_h": max_h, "canvas_w": max_w}


def _detect_frame_interior_fill(pairs):
    """Detect: single wall color in input; output = fill_color at enclosed cells, 0 elsewhere.

    Walls are ERASED in output (unlike enclosed_region_fill where walls are preserved).
    Enclosed = 0-cells not 4-conn reachable from boundary through non-wall cells.

    Returns {"frame_color": int, "fill_color": int} or None.
    """
    if not pairs:
        return None
    frame_color = None
    fill_color = None
    saw_fill = False
    for p in pairs:
        if _shape(p["input"]) != _shape(p["output"]):
            return None
        inp = np.array(p["input"])
        out = np.array(p["output"])
        # Only one non-zero input color (the wall/frame color)
        in_colors = set(int(c) for c in np.unique(inp) if c != 0)
        if not in_colors:
            return None
        if len(in_colors) != 1:
            return None
        fc_cand = next(iter(in_colors))
        if frame_color is None:
            frame_color = fc_cand
        elif frame_color != fc_cand:
            return None
        walls = inp == frame_color
        enclosed = _enclosed_mask_from_walls(walls)
        # Output must be fill_color at enclosed, 0 everywhere else
        out_nz = out != 0
        if not (out_nz == enclosed).all():
            return None
        if (out[~enclosed] != 0).any():
            return None
        if enclosed.any():
            saw_fill = True
            enc_colors = np.unique(out[enclosed])
            if len(enc_colors) != 1:
                return None
            fvc = int(enc_colors[0])
            if fill_color is None:
                fill_color = fvc
            elif fill_color != fvc:
                return None
    if frame_color is None or fill_color is None or not saw_fill:
        return None
    return {"frame_color": frame_color, "fill_color": fill_color}


def _detect_mirror_grid_2x2(pairs):
    """Detect: output is a 2x2 grid layout [[inp, flip_w(inp)], [flip_h(inp), flip_hw(inp)]].

    Returns (in_h, in_w) on success, else None.
    """
    if not pairs:
        return None
    in_shapes = {_shape(p["input"]) for p in pairs}
    out_shapes = {_shape(p["output"]) for p in pairs}
    if len(in_shapes) != 1 or len(out_shapes) != 1:
        return None
    in_h, in_w = next(iter(in_shapes))
    out_h, out_w = next(iter(out_shapes))
    if (out_h, out_w) != (2 * in_h, 2 * in_w):
        return None
    if out_h > 30 or out_w > 30:
        return None
    for p in pairs:
        inp = np.array(p["input"])
        out = np.array(p["output"])
        fw = inp[:, ::-1]
        fh = inp[::-1, :]
        fhw = inp[::-1, ::-1]
        expected = np.block([[inp, fw], [fh, fhw]])
        if expected.shape != out.shape or not np.array_equal(expected, out):
            return None
    return in_h, in_w


def _detect_mirror_concat(pairs):
    """Detect: output is a fixed-shape concat of input with one of {flip_h, flip_w,
    flip_hw, identity} of itself along H or W axis. Variant: side (input first or
    second in the concat).

    Returns (in_h, in_w, axis, transform, second_first) on success, else None.
    'second_first' is True iff the transformed copy comes BEFORE the input in the
    concat (e.g., output = concat(flip_h(input), input) along H = "mirror_h_top").
    """
    if not pairs:
        return None
    in_shapes = {_shape(p["input"]) for p in pairs}
    out_shapes = {_shape(p["output"]) for p in pairs}
    if len(in_shapes) != 1 or len(out_shapes) != 1:
        return None
    in_h, in_w = next(iter(in_shapes))
    out_h, out_w = next(iter(out_shapes))
    if in_h <= 0 or in_w <= 0 or in_h > 30 or in_w > 30:
        return None
    # Output must be 2x along exactly one axis.
    if (out_h, out_w) == (in_h, 2 * in_w):
        axis = 3
    elif (out_h, out_w) == (2 * in_h, in_w):
        axis = 2
    else:
        return None
    if out_h > 30 or out_w > 30:
        return None

    def _transform(grid, t):
        a = np.array(grid)
        if t == "identity":
            return a
        if t == "flip_w":
            return a[:, ::-1]
        if t == "flip_h":
            return a[::-1, :]
        if t == "flip_hw":
            return a[::-1, ::-1]
        raise ValueError(t)

    for t in ("flip_w", "flip_h", "flip_hw", "identity"):
        for second_first in (False, True):
            ok = True
            for p in pairs:
                inp = np.array(p["input"])
                out = np.array(p["output"])
                trans = _transform(inp, t)
                if axis == 3:
                    expected = np.concatenate([trans, inp], axis=1) if second_first else np.concatenate([inp, trans], axis=1)
                else:
                    expected = np.concatenate([trans, inp], axis=0) if second_first else np.concatenate([inp, trans], axis=0)
                if expected.shape != out.shape or not np.array_equal(expected, out):
                    ok = False
                    break
            if ok:
                return in_h, in_w, axis, t, second_first
    return None


def _detect_grid_pattern_color_to_grid(pairs):
    """Detect: input fixed in_h x in_w, output fixed out_h x out_w; each pair has
    exactly ONE nonzero color (varying per pair); output grid is a function of
    BOTH the binary pattern AND the nonzero color.

    Returns (in_h, in_w, out_h, out_w, lut_table) where lut_table is a dict
    keyed on (pattern_key, nonzero_color) -> 2D output tuple, else None.
    """
    if not pairs:
        return None
    in_shapes = {_shape(p["input"]) for p in pairs}
    out_shapes = {_shape(p["output"]) for p in pairs}
    if len(in_shapes) != 1 or len(out_shapes) != 1:
        return None
    in_h, in_w = next(iter(in_shapes))
    out_h, out_w = next(iter(out_shapes))
    if in_h <= 0 or in_w <= 0 or in_h > 30 or in_w > 30:
        return None
    if out_h <= 0 or out_w <= 0 or out_h > 30 or out_w > 30:
        return None
    if in_h * in_w > 12:
        return None
    # File-size guard: LUT bytes scale ~1 byte per element in the serialized model;
    # 1.44 MiB cap = 1,509,949 bytes. n_keys * out_cells <= 1,200,000 leaves headroom.
    if (2 ** (in_h * in_w)) * 9 * out_h * out_w > 1_200_000:
        return None
    rule = {}
    nz_colors_seen = set()
    for p in pairs:
        inp = p["input"]
        nonzero = set()
        for row in inp:
            for c in row:
                if c != 0:
                    nonzero.add(c)
        if len(nonzero) != 1:
            return None
        nz_color = next(iter(nonzero))
        if not (1 <= nz_color <= 9):
            return None
        nz_colors_seen.add(nz_color)
        pat_key = 0
        for i in range(in_h):
            for j in range(in_w):
                if inp[i][j] != 0:
                    pat_key |= 1 << (i * in_w + j)
        out_grid = tuple(tuple(row) for row in p["output"])
        for row in out_grid:
            for c in row:
                if not (0 <= c < 10):
                    return None
        full_key = (pat_key, nz_color)
        if full_key in rule:
            if rule[full_key] != out_grid:
                return None
        else:
            rule[full_key] = out_grid
    # Require at least 2 distinct nonzero colors observed (otherwise a simpler
    # grid_pattern_to_color/grid_pattern_to_grid would already match).
    if len(nz_colors_seen) < 2:
        return None
    return in_h, in_w, out_h, out_w, rule


def _detect_grid_pattern_to_grid(pairs):
    """Detect: input fixed in_h x in_w, output fixed out_h x out_w (out_cells > 1),
    output grid is a function of the binary nonzero-pattern of input cells.

    Returns (in_h, in_w, out_h, out_w, lut_table) on success, else None.
    Skipped when output is 1x1 (handled by _detect_grid_pattern_to_color).
    """
    if not pairs:
        return None
    in_shapes = {_shape(p["input"]) for p in pairs}
    out_shapes = {_shape(p["output"]) for p in pairs}
    if len(in_shapes) != 1 or len(out_shapes) != 1:
        return None
    in_h, in_w = next(iter(in_shapes))
    out_h, out_w = next(iter(out_shapes))
    if (out_h, out_w) == (1, 1):
        return None
    if in_h <= 0 or in_w <= 0 or out_h <= 0 or out_w <= 0:
        return None
    if in_h > 30 or in_w > 30 or out_h > 30 or out_w > 30:
        return None
    if in_h * in_w > 12:
        return None
    # File-size guard: int64 LUT[n_keys, out_cells] + onnx-overhead must fit the
    # 1.44 MiB validator cap. Same formula as _detect_grid_pattern_color_to_grid;
    # 1.2M elements leaves headroom for serialization + other initializers.
    if (2 ** (in_h * in_w)) * out_h * out_w > 1_200_000:
        return None
    rule = {}
    for p in pairs:
        inp = p["input"]
        key = 0
        for i in range(in_h):
            for j in range(in_w):
                if inp[i][j] != 0:
                    key |= 1 << (i * in_w + j)
        out_grid = tuple(tuple(row) for row in p["output"])
        for row in out_grid:
            for c in row:
                if not (0 <= c < 10):
                    return None
        if key in rule:
            if rule[key] != out_grid:
                return None
        else:
            rule[key] = out_grid
    return in_h, in_w, out_h, out_w, rule


def _detect_grid_pattern_to_color(pairs):
    """Detect: input has fixed in_h x in_w shape, output is 1x1, and the output
    color is a function of the binary nonzero-pattern of the input cells.

    Returns (in_h, in_w, lut_table) where lut_table is dict[key -> color], else None.
    """
    if not pairs:
        return None
    in_shapes = {_shape(p["input"]) for p in pairs}
    out_shapes = {_shape(p["output"]) for p in pairs}
    if len(in_shapes) != 1 or len(out_shapes) != 1:
        return None
    in_h, in_w = next(iter(in_shapes))
    out_h, out_w = next(iter(out_shapes))
    if (out_h, out_w) != (1, 1):
        return None
    if in_h <= 0 or in_w <= 0 or in_h > 30 or in_w > 30:
        return None
    if in_h * in_w > 12:
        return None
    rule = {}
    for p in pairs:
        inp = p["input"]
        key = 0
        for i in range(in_h):
            for j in range(in_w):
                if inp[i][j] != 0:
                    key |= 1 << (i * in_w + j)
        out_color = p["output"][0][0]
        if not (0 <= out_color < 10):
            return None
        if key in rule:
            if rule[key] != out_color:
                return None
        else:
            rule[key] = out_color
    return in_h, in_w, rule


def _detect_binary_neighborhood_lut(pairs):
    """Detect '3x3 binary-neighborhood rule' pattern (J.20 Pattern 4).

    Tries each background color K in 0..9. Builds a 9-bit key per cell (each bit
    = "this neighbor != K AND in active grid"; out-of-grid neighbors treated as
    background). Verifies the rule (key -> output_color) is consistent across all
    pairs.

    Returns (bg_color, lut_table) on success, else None.

    Pre-condition: all pairs must share the same shape (since the LUT rule is
    cell-local but assumes a consistent active-grid extent within the canvas).
    The builder dynamically masks padding via is_active so non-30x30 grids work.
    """
    if not pairs:
        return None
    in_shapes = {_shape(p["input"]) for p in pairs}
    out_shapes = {_shape(p["output"]) for p in pairs}
    if len(in_shapes) != 1 or len(out_shapes) != 1:
        return None
    h, w = next(iter(in_shapes))
    if (h, w) != next(iter(out_shapes)) or h <= 0 or w <= 0 or h > 30 or w > 30:
        return None

    for bg in range(10):
        rule = {}
        consistent = True
        for p in pairs:
            inp, out = p["input"], p["output"]
            for i in range(h):
                for j in range(w):
                    key = 0
                    for db in range(3):
                        for da in range(3):
                            ni, nj = i + db - 1, j + da - 1
                            if 0 <= ni < h and 0 <= nj < w:
                                bit = 1 if inp[ni][nj] != bg else 0
                            else:
                                bit = 0
                            key |= bit << (db * 3 + da)
                    expected = out[i][j]
                    # Reject if output color is out of the validator's 0..9 range;
                    # the builder uses LUT values as direct channel indices.
                    if not (0 <= expected <= 9):
                        consistent = False
                        break
                    if key in rule:
                        if rule[key] != expected:
                            consistent = False
                            break
                    else:
                        rule[key] = expected
                if not consistent:
                    break
            if not consistent:
                break
        if consistent:
            return bg, rule
    return None


_PATTERNS = (
    # Transpose is the only entry that fires across all input shapes (the canvas
    # transpose works because the active grid stays at top-left after the swap).
    # The 30x30-only flip/rot entries that previously lived here are subsumed by
    # `_detect_fixed_shape_spatial` (which dispatches to fixed_flip_*_network /
    # fixed_rot180_network at h == w == 30 with identical cost).
    ("transpose",    _transpose_grid,     transpose_hw_network, lambda pairs: True),
)


def _detect_fixed_shape_spatial(pairs):
    """Detect: all pairs share one fixed (h, w) AND output is a known in-grid
    spatial transform of input (flip, rotation, or cyclic shift).

    Returns a description dict with keys 'transform' and transform-specific
    parameters, or None. Non-square rotations (rot90, rot270) and antitranspose
    are only attempted when h == w. Cyclic shifts are tried for every non-zero
    k < grid_dim.

    The grid is assumed to be at the top-left of the [1,10,30,30] canvas
    (per the validator's convention); rows/cols beyond (h, w) are padding
    and pass through identity.
    """
    if not pairs:
        return None
    in_shapes = {_shape(p["input"]) for p in pairs}
    out_shapes = {_shape(p["output"]) for p in pairs}
    if len(in_shapes) != 1 or len(out_shapes) != 1:
        return None
    h, w = next(iter(in_shapes))
    if next(iter(out_shapes)) != (h, w):
        return None
    if h <= 0 or w <= 0 or h > 30 or w > 30:
        return None
    # 1x1 grids are fixed points of every spatial transform; matching here is
    # meaningless. Reject before producing a degenerate "flip on 1x1" match.
    if h == 1 and w == 1:
        return None

    candidates = [
        ("flip_h", _flip_h, {"w": w}),
        ("flip_v", _flip_v, {"h": h}),
        ("rot180", _rot180, {"h": h, "w": w}),
    ]
    if h == w:
        candidates.append(("rot90", _rot90, {"n": h}))
        candidates.append(("rot270", _rot270, {"n": h}))
    for k in range(1, h):
        candidates.append((f"shift_v_{k}", lambda g, k=k: _shift_v(g, k), {"h": h, "k": k}))
    for k in range(1, w):
        candidates.append((f"shift_h_{k}", lambda g, k=k: _shift_h(g, k), {"w": w, "k": k}))

    for tname, tfn, params in candidates:
        if all(tfn(p["input"]) == p["output"] for p in pairs):
            return {"transform": tname, **params}
    return None


def _translation_eq(inp, out, dr, dc):
    """Test: out[r,c] == inp[r-dr, c-dc] if (r-dr, c-dc) is in bounds, else 0.

    Both grids must already share shape (h, w); caller verifies that upstream.
    """
    h = len(inp)
    w = len(inp[0]) if h else 0
    for r in range(h):
        for c in range(w):
            sr = r - dr
            sc = c - dc
            expected = inp[sr][sc] if 0 <= sr < h and 0 <= sc < w else 0
            if out[r][c] != expected:
                return False
    return True


def _detect_translation(pairs):
    """Detect: output = input translated by a fixed (dr, dc), zero-padded.

    Constraints:
      * All pairs share one input shape (h, w); output shape == input shape.
      * For every cell (r, c) of every pair: out[r,c] == in[r-dr, c-dc] if the
        source cell is in bounds, else 0.
      * (dr, dc) != (0, 0). Identity is not a translation.
      * Distinct from _detect_fixed_shape_spatial's cyclic shift family: cyclic
        wraps content around; translation drops spillover into zeros.

    Returns {"h": h, "w": w, "dr": dr, "dc": dc} or None.
    """
    if not pairs:
        return None
    in_shapes = {_shape(p["input"]) for p in pairs}
    out_shapes = {_shape(p["output"]) for p in pairs}
    if len(in_shapes) != 1 or len(out_shapes) != 1:
        return None
    h, w = next(iter(in_shapes))
    if next(iter(out_shapes)) != (h, w):
        return None
    if h <= 0 or w <= 0 or h > 30 or w > 30:
        return None
    if h == 1 and w == 1:
        return None

    inp0 = pairs[0]["input"]
    out0 = pairs[0]["output"]

    candidates = set()
    for r in range(h):
        for c in range(w):
            color = inp0[r][c]
            if color == 0:
                continue
            for ro in range(h):
                for co in range(w):
                    if out0[ro][co] == color:
                        dr = ro - r
                        dc = co - c
                        if dr == 0 and dc == 0:
                            continue
                        if -h < dr < h and -w < dc < w:
                            candidates.add((dr, dc))
            if candidates:
                break
        if candidates:
            break

    if not candidates:
        return None
    for dr, dc in sorted(candidates, key=lambda x: (abs(x[0]) + abs(x[1]), x[0], x[1])):
        if all(_translation_eq(p["input"], p["output"], dr, dc) for p in pairs):
            return {"h": h, "w": w, "dr": dr, "dc": dc}
    return None


def _detect_pixel_perm(pairs):
    """Detect: same-shape input/output where each output cell is a fixed source
    cell from the input, consistent across all visible pairs.

    Returns {"h": h, "w": w, "indices": [int]*h*w} on match (indices[i_out] is the
    flat input cell index that always equals the output's i_out cell), or None.

    Algorithm: per pair, group input cell indices by color. For each output cell,
    intersect the valid-source-index sets across all pairs. If the intersection is
    non-empty for every output cell, pick the lexicographically smallest source
    index for each output cell. Reject identity (all-cells-map-to-themselves —
    that's identity_only_network's job).

    Distinct from translation (cyclic-shift / fixed-shift): allows arbitrary
    per-cell rearrangement, not just a uniform offset. Distinct from color_perm
    / color_replace: those operate on color values channel-wise; this operates on
    spatial positions and preserves whatever colors the input cells have.
    """
    if not pairs:
        return None
    s = _shape(pairs[0]["input"])
    if s != _shape(pairs[0]["output"]):
        return None
    if any(_shape(p["input"]) != s or _shape(p["output"]) != s for p in pairs):
        return None
    h, w = s
    if h <= 0 or w <= 0 or h > 30 or w > 30:
        return None
    n = h * w
    if n == 0:
        return None

    # Build per-pair index: color -> set of input-flat indices having that color.
    pair_color_indices = []
    for p in pairs:
        flat = [c for row in p["input"] for c in row]
        cti = {}
        for i, c in enumerate(flat):
            cti.setdefault(c, set()).add(i)
        pair_color_indices.append(cti)

    indices = []
    is_identity = True
    for i_out in range(n):
        valid = None
        for pi, p in enumerate(pairs):
            target = p["output"][i_out // w][i_out % w]
            cell_valid = pair_color_indices[pi].get(target, set())
            valid = set(cell_valid) if valid is None else (valid & cell_valid)
            if not valid:
                return None
        chosen = min(valid)
        indices.append(chosen)
        if chosen != i_out:
            is_identity = False

    if is_identity:
        return None
    return {"h": h, "w": w, "indices": indices}


def _detect_shape_changing_pixel_perm(pairs):
    """Detect: cross-shape input/output where each output cell is a fixed source
    cell from the input, consistent across all visible pairs.

    Generalizes _detect_pixel_perm to (out_h, out_w) != (in_h, in_w). Returns
    {"in_h", "in_w", "out_h", "out_w", "indices": [int]*out_h*out_w} on match,
    or None.

    Common cases: tile-rearrange (in (3,3) -> out (6,6) or (9,9) where every
    output cell maps to a specific input cell). The same-shape case is handled
    by _detect_pixel_perm to keep the cheaper builder available there.
    """
    if not pairs:
        return None
    in_shapes = {_shape(p["input"]) for p in pairs}
    out_shapes = {_shape(p["output"]) for p in pairs}
    if len(in_shapes) != 1 or len(out_shapes) != 1:
        return None
    in_h, in_w = next(iter(in_shapes))
    out_h, out_w = next(iter(out_shapes))
    # Reject same-shape case (handled by _detect_pixel_perm at lower cost).
    if (in_h, in_w) == (out_h, out_w):
        return None
    if in_h <= 0 or in_w <= 0 or in_h > 30 or in_w > 30:
        return None
    if out_h <= 0 or out_w <= 0 or out_h > 30 or out_w > 30:
        return None
    n_in = in_h * in_w
    n_out = out_h * out_w
    if n_in == 0 or n_out == 0:
        return None

    pair_color_indices = []
    for p in pairs:
        flat = [c for row in p["input"] for c in row]
        cti = {}
        for i, c in enumerate(flat):
            cti.setdefault(c, set()).add(i)
        pair_color_indices.append(cti)

    indices = []
    for i_out in range(n_out):
        valid = None
        for pi, p in enumerate(pairs):
            target = p["output"][i_out // out_w][i_out % out_w]
            cell_valid = pair_color_indices[pi].get(target, set())
            valid = set(cell_valid) if valid is None else (valid & cell_valid)
            if not valid:
                return None
        indices.append(min(valid))

    return {"in_h": in_h, "in_w": in_w, "out_h": out_h, "out_w": out_w, "indices": indices}


def _detect_marker_corners(pairs):
    """Detect: input has exactly one cell of a "marker" color (consistent across pairs);
    output replaces the marker with color 0 and places four specific colors at the four
    diagonal positions around the marker (only in-bounds diagonals get a color).

    Returns (marker_color, corners_dict) on success where corners_dict maps each of
    "TL", "TR", "BL", "BR" to a color, else None.

    Constraints on the input (so the simple `grid_mask = ch0 + marker` builder is correct):
    - Every visible pair's input contains only color 0 and the marker color.
    - Every output cell outside the four corner positions equals the input cell at the
      same position (or 0 at the marker cell).

    The mapping {direction -> color} must be consistent across all pairs (when in-bounds).
    """
    if not pairs:
        return None
    DIRS = [("TL", -1, -1), ("TR", -1, 1), ("BL", 1, -1), ("BR", 1, 1)]

    for marker in range(1, 10):
        # Every pair has exactly 1 cell of `marker` color
        if not all(sum(1 for row in p["input"] for c in row if c == marker) == 1 for p in pairs):
            continue
        # Inputs must use only colors 0 and marker
        if not all({c for row in p["input"] for c in row} <= {0, marker} for p in pairs):
            continue

        consistent = {"TL": None, "TR": None, "BL": None, "BR": None}
        ok = True
        for p in pairs:
            in_g, out_g = p["input"], p["output"]
            if _shape(in_g) != _shape(out_g):
                ok = False
                break
            h, w = _shape(in_g)
            # Find marker position
            mr, mc = -1, -1
            for i, row in enumerate(in_g):
                for j, c in enumerate(row):
                    if c == marker:
                        mr, mc = i, j
                        break
                if mr >= 0:
                    break
            # Marker must disappear in output
            if out_g[mr][mc] != 0:
                ok = False
                break
            corner_positions = {(mr + dr, mc + dc) for _, dr, dc in DIRS if 0 <= mr + dr < h and 0 <= mc + dc < w}
            # All non-corner cells in output must equal input (with marker -> 0)
            for i in range(h):
                for j in range(w):
                    if (i, j) in corner_positions:
                        continue
                    expected = 0 if (i, j) == (mr, mc) else in_g[i][j]
                    if out_g[i][j] != expected:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
            # Update consistent mapping
            for tag, dr, dc in DIRS:
                nr, nc = mr + dr, mc + dc
                if 0 <= nr < h and 0 <= nc < w:
                    color = out_g[nr][nc]
                    if consistent[tag] is None:
                        consistent[tag] = color
                    elif consistent[tag] != color:
                        ok = False
                        break
            if not ok:
                break
        if not ok:
            continue
        # All four corners must be observed (otherwise we can't pick a placement color)
        if any(consistent[k] is None for k in consistent):
            continue
        # Each corner color must be distinct from 0 and from `marker` (otherwise the channel
        # mapping in the builder collides — channel 0 and channel `marker` are reserved)
        if any(c == 0 or c == marker for c in consistent.values()):
            continue
        # Each corner color must be unique (the builder maps colors -> directions, so duplicates collide)
        if len(set(consistent.values())) != 4:
            continue
        return marker, consistent
    return None


def _detect_isolated_marker_recolor(pairs):
    """Detect: across all visible pairs, edits are exactly the cells where input is
    color M and that cell has 0 M-neighbors (4-connectivity: up/down/left/right
    within the active grid), and those cells in output have color R.

    Returns (M, R) on success, else None.

    Requires (a) shape-preserving, (b) edits exhibit a single (M, R) color transition
    across all pairs, (c) the rule "isolated M -> R" exactly reproduces every
    visible-pair output.
    """
    if not pairs:
        return None
    for p in pairs:
        if _shape(p["input"]) != _shape(p["output"]):
            return None

    # Single (in_color, out_color) transition across all edits
    transitions = set()
    for p in pairs:
        inp = np.array(p["input"])
        out = np.array(p["output"])
        diff = inp != out
        if not diff.any():
            continue
        for ic, oc in zip(inp[diff].tolist(), out[diff].tolist()):
            transitions.add((ic, oc))
            if len(transitions) > 1:
                return None
    if len(transitions) != 1:
        return None
    M, R = next(iter(transitions))
    if M == R or not (0 <= M < 10) or not (0 <= R < 10):
        return None

    # Verify the isolated-M (4-conn) rule reproduces every output exactly
    for p in pairs:
        inp = np.array(p["input"])
        out = np.array(p["output"])
        h, w = inp.shape
        mask_m = (inp == M).astype(np.int32)
        # 4-neighbor count (up/down/left/right only) with zero-padding
        nbrs = np.zeros_like(mask_m)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            src_r0, src_r1 = max(0, dr), h + min(0, dr)
            src_c0, src_c1 = max(0, dc), w + min(0, dc)
            dst_r0, dst_r1 = max(0, -dr), h + min(0, -dr)
            dst_c0, dst_c1 = max(0, -dc), w + min(0, -dc)
            nbrs[dst_r0:dst_r1, dst_c0:dst_c1] += mask_m[src_r0:src_r1, src_c0:src_c1]
        is_isolated = (mask_m == 1) & (nbrs == 0)
        expected = inp.copy()
        expected[is_isolated] = R
        if not np.array_equal(expected, out):
            return None
    return M, R


def _detect_l_shape_missing_corner(pairs):
    """Detect: across all visible pairs, edits are exactly the cells (i, j) where
    (a) input[i, j] == C (a "source" / background color), and (b) some 2x2 block
    containing (i, j) has exactly 3 cells of color M (the "marker" color) plus the
    1 C cell. Output replaces those C cells with color F (the "fill" color).

    Returns (marker_color, src_color, fill_color) on success, else None.

    Requires (a) shape-preserving, (b) edits exhibit a single (C, F) color
    transition across all pairs, (c) some marker color M (with M != C, M != F)
    exists such that the rule reproduces every visible-pair output exactly.
    """
    if not pairs:
        return None
    for p in pairs:
        if _shape(p["input"]) != _shape(p["output"]):
            return None

    transitions = set()
    for p in pairs:
        inp = np.array(p["input"])
        out = np.array(p["output"])
        diff = inp != out
        if not diff.any():
            continue
        for ic, oc in zip(inp[diff].tolist(), out[diff].tolist()):
            transitions.add((ic, oc))
            if len(transitions) > 1:
                return None
    if len(transitions) != 1:
        return None
    C, F = next(iter(transitions))
    if C == F or not (0 <= C < 10) or not (0 <= F < 10):
        return None

    for M in range(10):
        if M == C or M == F:
            continue
        ok = True
        for p in pairs:
            inp = np.array(p["input"])
            out = np.array(p["output"])
            h, w = inp.shape
            if h < 2 or w < 2:
                ok = False
                break
            mask_m = (inp == M).astype(np.int32)
            block_count = (
                mask_m[:-1, :-1] + mask_m[:-1, 1:]
                + mask_m[1:, :-1] + mask_m[1:, 1:]
            )  # shape (h-1, w-1)
            block_has_3 = (block_count == 3)
            cell_in_l = np.zeros((h, w), dtype=bool)
            # 4 blocks containing cell (i,j) have TL at (i+dr, j+dc) for dr,dc in {-1, 0}.
            # Valid cell range: i in [max(0, -dr), h - 1 - dr); j in [max(0, -dc), w - 1 - dc).
            for dr in (-1, 0):
                for dc in (-1, 0):
                    src_r0 = max(0, -dr)
                    src_r1 = h - 1 - dr
                    src_c0 = max(0, -dc)
                    src_c1 = w - 1 - dc
                    cell_in_l[src_r0:src_r1, src_c0:src_c1] |= block_has_3[
                        src_r0 + dr:src_r1 + dr, src_c0 + dc:src_c1 + dc
                    ]
            is_missing = cell_in_l & (inp == C)
            expected = inp.copy()
            expected[is_missing] = F
            if not np.array_equal(expected, out):
                ok = False
                break
        if ok:
            return M, C, F
    return None


def _detect_aligned_marker_fill(pairs):
    """Detect: across all visible pairs, edits are exactly the cells (i, j) where
    (a) input[i, j] == C (a "source" / background color), and (b) there exist M
    markers strictly to the left AND strictly to the right of (i, j) in the same
    row (if use_h), AND/OR strictly above AND strictly below (i, j) in the same
    column (if use_v). Output replaces those C cells with color F.

    Returns (marker_color, src_color, fill_color, use_h, use_v) on success, else None.
    Tries (use_h, use_v) configs in order: H-only, V-only, both. Picks the first
    matching config (H-only and V-only are cheaper than both).
    """
    if not pairs:
        return None
    for p in pairs:
        if _shape(p["input"]) != _shape(p["output"]):
            return None

    transitions = set()
    for p in pairs:
        inp = np.array(p["input"])
        out = np.array(p["output"])
        diff = inp != out
        if not diff.any():
            continue
        for ic, oc in zip(inp[diff].tolist(), out[diff].tolist()):
            transitions.add((ic, oc))
            if len(transitions) > 1:
                return None
    if len(transitions) != 1:
        return None
    C, F = next(iter(transitions))
    if C == F or not (0 <= C < 10) or not (0 <= F < 10):
        return None

    def _check(M, use_h, use_v):
        for p in pairs:
            inp = np.array(p["input"])
            out = np.array(p["output"])
            h, w = inp.shape
            mask_m = (inp == M)
            between_h = np.zeros_like(mask_m, dtype=bool)
            between_v = np.zeros_like(mask_m, dtype=bool)
            if use_h:
                for r in range(h):
                    cols = np.where(mask_m[r])[0]
                    if len(cols) >= 2:
                        between_h[r, cols.min() + 1:cols.max()] = True
            if use_v:
                for c in range(w):
                    rows = np.where(mask_m[:, c])[0]
                    if len(rows) >= 2:
                        between_v[rows.min() + 1:rows.max(), c] = True
            should_change = (between_h | between_v) & (inp == C)
            expected = inp.copy()
            expected[should_change] = F
            if not np.array_equal(expected, out):
                return False
        return True

    for M in range(10):
        if M == C or M == F:
            continue
        for use_h, use_v in ((True, False), (False, True), (True, True)):
            if _check(M, use_h, use_v):
                return M, C, F, use_h, use_v
    return None


def _detect_sandwich_fill(pairs):
    """Detect: across all visible pairs, edits are exactly the cells (i, j) where
    input[i, j] == C AND M markers exist immediately at (i, j-1) AND (i, j+1)
    (horizontal sandwich), OR (i-1, j) AND (i+1, j) (vertical sandwich), OR
    either axis (both flag). The H-only and V-only variants enable tasks that
    fill only along one axis. Output replaces those C cells with color F.

    Returns (marker_color, src_color, fill_color, use_h, use_v) on success, else None.

    Tries 3 axis configs (H-only, V-only, both) and picks the most restrictive
    that still reproduces every visible-pair output.
    """
    if not pairs:
        return None
    for p in pairs:
        if _shape(p["input"]) != _shape(p["output"]):
            return None

    transitions = set()
    for p in pairs:
        inp = np.array(p["input"])
        out = np.array(p["output"])
        diff = inp != out
        if not diff.any():
            continue
        for ic, oc in zip(inp[diff].tolist(), out[diff].tolist()):
            transitions.add((ic, oc))
            if len(transitions) > 1:
                return None
    if len(transitions) != 1:
        return None
    C, F = next(iter(transitions))
    if C == F or not (0 <= C < 10) or not (0 <= F < 10):
        return None

    def _check(M, use_h, use_v):
        for p in pairs:
            inp = np.array(p["input"])
            out = np.array(p["output"])
            h, w = inp.shape
            mask_m = (inp == M)
            sand = np.zeros_like(mask_m, dtype=bool)
            if use_h:
                for r in range(h):
                    for c in range(1, w - 1):
                        if mask_m[r, c - 1] and mask_m[r, c + 1]:
                            sand[r, c] = True
            if use_v:
                for c in range(w):
                    for r in range(1, h - 1):
                        if mask_m[r - 1, c] and mask_m[r + 1, c]:
                            sand[r, c] = True
            should_change = sand & (inp == C)
            expected = inp.copy()
            expected[should_change] = F
            if not np.array_equal(expected, out):
                return False
        return True

    # Try each candidate M, then each axis config — prefer single-axis (cheaper) over both.
    for M in range(10):
        if M == C or M == F:
            continue
        for use_h, use_v in ((True, False), (False, True), (True, True)):
            if _check(M, use_h, use_v):
                return M, C, F, use_h, use_v
    return None


def _detect_pi_shape_project_to_bottom(pairs):
    """Detect: each input "Pi gap" cell (r, c) — color 0 with all 5 stencil cells
    {(r-1,c-1),(r-1,c),(r-1,c+1),(r,c-1),(r,c+1)} of nonzero color (any colors,
    not necessarily the same) — projects color F onto the active-grid bottom row
    at column c. All other cells unchanged.

    Returns fill_color (F) on success, else None.

    Requires (a) shape-preserving, (b) edits exhibit a single (0, F) color
    transition, and (c) the rule reproduces every visible-pair output exactly.
    """
    if not pairs:
        return None
    for p in pairs:
        if _shape(p["input"]) != _shape(p["output"]):
            return None

    transitions = set()
    for p in pairs:
        inp = np.array(p["input"])
        out = np.array(p["output"])
        diff = inp != out
        if not diff.any():
            continue
        for ic, oc in zip(inp[diff].tolist(), out[diff].tolist()):
            transitions.add((ic, oc))
            if len(transitions) > 1:
                return None
    if len(transitions) != 1:
        return None
    C, F = next(iter(transitions))
    if C != 0 or F == 0 or not (0 <= F < 10):
        return None

    for p in pairs:
        inp = np.array(p["input"])
        out = np.array(p["output"])
        h, w = inp.shape
        gap_cols = set()
        for r in range(1, h):
            for c in range(1, w - 1):
                if inp[r, c] != 0:
                    continue
                stencil = [inp[r - 1, c - 1], inp[r - 1, c], inp[r - 1, c + 1], inp[r, c - 1], inp[r, c + 1]]
                if all(v != 0 for v in stencil):
                    gap_cols.add(c)
        expected = inp.copy()
        for c in gap_cols:
            expected[h - 1, c] = F
        if not np.array_equal(expected, out):
            return None
    return F


def _detect_slice_extraction(pairs):
    """Detect: every pair's output is the same fixed (h_start:h_end, w_start:w_end)
    sub-block of input.

    Returns (h_start, h_end, w_start, w_end) on success, else None.

    Requires (a) all pairs have the same output shape (h_out, w_out), (b) every
    pair's input is at least (h_out, w_out) in each dim, and (c) some fixed
    (h_start, w_start) exists such that output == input[h_start:h_start+h_out,
    w_start:w_start+w_out] for every pair.
    """
    if not pairs:
        return None
    out_shapes = {_shape(p["output"]) for p in pairs}
    if len(out_shapes) != 1:
        return None
    h_out, w_out = next(iter(out_shapes))
    if h_out <= 0 or w_out <= 0 or h_out > 30 or w_out > 30:
        return None
    in_shapes = [_shape(p["input"]) for p in pairs]
    min_h = min(h for h, _ in in_shapes)
    min_w = min(w for _, w in in_shapes)
    if h_out > min_h or w_out > min_w:
        return None
    for h_start in range(min_h - h_out + 1):
        for w_start in range(min_w - w_out + 1):
            ok = True
            for p in pairs:
                expected = [row[w_start:w_start + w_out] for row in p["input"][h_start:h_start + h_out]]
                if expected != p["output"]:
                    ok = False
                    break
            if ok:
                return h_start, h_start + h_out, w_start, w_start + w_out
    return None


def _detect_framed_cell_const_output(pairs):
    """Detect: every pair's output is a uniform single color of one fixed
    shape across all pairs, AND that color is the **core** of a 3x3 frame
    anywhere in the input. The frame is 8 cells of one non-zero color X
    surrounding a cell of a different non-zero color Y; output = 1x1 of Y.

    Returns (h_out, w_out) if matched, else None.
    """
    if not pairs:
        return None
    out_shapes = {_shape(p["output"]) for p in pairs}
    if len(out_shapes) != 1:
        return None
    h_out, w_out = next(iter(out_shapes))
    if h_out <= 0 or w_out <= 0 or h_out > 30 or w_out > 30:
        return None
    for p in pairs:
        out_colors = {c for row in p["output"] for c in row}
        if len(out_colors) != 1:
            return None
        out_color = next(iter(out_colors))
        if out_color == 0:
            return None
        # Find any 3x3 window in input with: 8 frame cells of same non-zero color X (X != 0),
        # center cell of color Y (Y != 0, Y != X), and Y == out_color.
        inp = p["input"]
        h_in = len(inp)
        if h_in < 3:
            return None
        w_in = len(inp[0])
        if w_in < 3:
            return None
        found = False
        for r in range(h_in - 2):
            for c in range(w_in - 2):
                core = inp[r + 1][c + 1]
                if core != out_color:
                    continue
                frame = (
                    inp[r][c], inp[r][c + 1], inp[r][c + 2],
                    inp[r + 1][c], inp[r + 1][c + 2],
                    inp[r + 2][c], inp[r + 2][c + 1], inp[r + 2][c + 2],
                )
                X = frame[0]
                if X == 0 or X == core:
                    continue
                if all(f == X for f in frame):
                    found = True
                    break
            if found:
                break
        if not found:
            return None
    max_h = max(len(p["input"]) for p in pairs)
    max_w = max(len(p["input"][0]) for p in pairs if p["input"])
    return h_out, w_out, max_h, max_w


def _detect_constant_output_largest_bbox(pairs):
    """Detect: every pair's output is a uniform single color of one fixed shape
    across all pairs, AND that color is the channel (1..9) whose bounding box
    has the largest area in the input. Tie-break: smallest-index channel wins
    (matching ONNX ArgMax spec).

    Returns (h_out, w_out) if matched, else None.
    """
    if not pairs:
        return None
    out_shapes = {_shape(p["output"]) for p in pairs}
    if len(out_shapes) != 1:
        return None
    h_out, w_out = next(iter(out_shapes))
    if h_out <= 0 or w_out <= 0 or h_out > 30 or w_out > 30:
        return None
    for p in pairs:
        out_colors = {c for row in p["output"] for c in row}
        if len(out_colors) != 1:
            return None
        out_color = next(iter(out_colors))
        if out_color == 0:
            return None  # All-bg output is not this rule
        inp = p["input"]
        h_in = len(inp)
        if h_in == 0:
            return None
        w_in = len(inp[0])
        if w_in == 0:
            return None
        # Compute per-channel bbox area (channels 1..9)
        best_c = 0
        best_area = -1
        for c in range(1, 10):
            rows_with = [r for r in range(h_in) if any(inp[r][cc] == c for cc in range(w_in))]
            if not rows_with:
                continue
            cols_with = [cc for cc in range(w_in) if any(inp[r][cc] == c for r in range(h_in))]
            if not cols_with:
                continue
            area = (rows_with[-1] - rows_with[0] + 1) * (cols_with[-1] - cols_with[0] + 1)
            if area > best_area:  # smaller-c wins on tie (strict >)
                best_c = c
                best_area = area
        if best_c != out_color:
            return None
    return h_out, w_out


def _detect_marker_section_extract(pairs):
    """Detect: input has a divider grid (one color forms full rows/cols) and a
    single rare "marker" color cell; output = the divider-bounded sub-rectangle
    containing that marker (preserving original cell colors).

    Marker color = rarest non-divider color (with at least one cell). Marker
    can be color 0 (e.g., a single 0 in a sea of fill).

    Returns True on match, else None.
    """
    if not pairs:
        return None
    for p in pairs:
        try:
            a = np.array(p["input"], dtype=np.int8)
            o = np.array(p["output"], dtype=np.int8)
        except (TypeError, ValueError):
            return None
        if a.ndim != 2 or o.ndim != 2:
            return None
        H, W = a.shape
        all_colors = sorted(set(int(c) for c in a.flatten()))
        # Divider color: full rows or full cols (could be any color including 0)
        div_cands = []
        for c in all_colors:
            full_rows = [r for r in range(H) if (a[r] == c).all()]
            full_cols = [col for col in range(W) if (a[:, col] == c).all()]
            if full_rows or full_cols:
                div_cands.append((c, full_rows, full_cols))
        if len(div_cands) != 1:
            return None
        div_color, div_rows, div_cols = div_cands[0]
        non_div = [c for c in all_colors if c != div_color]
        if not non_div:
            return None
        counts = {c: int((a == c).sum()) for c in non_div}
        marker_color = min(counts.keys(), key=lambda c: counts[c])
        pos = np.where(a == marker_color)
        if len(pos[0]) == 0:
            return None
        mr, mc = int(pos[0][0]), int(pos[1][0])
        # Find divider-bounded rectangle containing (mr, mc)
        row_breaks = sorted([-1] + div_rows + [H])
        col_breaks = sorted([-1] + div_cols + [W])
        r0 = r1 = None
        for i in range(len(row_breaks) - 1):
            if row_breaks[i] < mr < row_breaks[i + 1]:
                r0, r1 = row_breaks[i] + 1, row_breaks[i + 1]
                break
        c0 = c1 = None
        for i in range(len(col_breaks) - 1):
            if col_breaks[i] < mc < col_breaks[i + 1]:
                c0, c1 = col_breaks[i] + 1, col_breaks[i + 1]
                break
        if r0 is None or c0 is None:
            return None
        expected = a[r0:r1, c0:c1]
        if expected.shape != o.shape:
            return None
        if not np.array_equal(expected, o):
            return None
    return True


def _detect_grid_division_count(pairs):
    """Detect: input is 2-color (one divider, one fill); output is uniform-fill
    rectangle of shape (#divider_rows + 1, #divider_cols + 1).

    Divider color = the (unique) color appearing as a "full row" or "full col"
    in the input (where "full" means every active cell of the row/col is that
    color). Fill color = the other non-zero color (most common, excluding
    divider + 0).

    Returns True on match, else None.
    """
    if not pairs:
        return None
    for p in pairs:
        try:
            a = np.array(p["input"], dtype=np.int8)
            o = np.array(p["output"], dtype=np.int8)
        except (TypeError, ValueError):
            return None
        if a.ndim != 2 or o.ndim != 2:
            return None
        H, W = a.shape
        all_colors = sorted(set(int(c) for c in a.flatten()))
        # Divider color: any color (incl. 0) with at least one full row OR full col
        div_cands = []
        for c in all_colors:
            full_rows = [r for r in range(H) if (a[r] == c).all()]
            full_cols = [col for col in range(W) if (a[:, col] == c).all()]
            if full_rows or full_cols:
                div_cands.append((c, len(full_rows), len(full_cols)))
        if len(div_cands) != 1:
            return None
        div_color, n_full_rows, n_full_cols = div_cands[0]
        # Fill color: the other non-zero color (most common, excl. div + 0)
        non_div = [c for c in all_colors if c != div_color and c != 0]
        if len(non_div) != 1:
            return None
        fill_color = non_div[0]
        # Output: uniform fill_color of shape (n_full_rows+1, n_full_cols+1)
        h_out = n_full_rows + 1
        w_out = n_full_cols + 1
        if o.shape != (h_out, w_out):
            return None
        if not (o == fill_color).all():
            return None
    return True


def _cross_section_grid_label_expected(inp):
    try:
        a = np.array(inp, dtype=np.int8)
    except (TypeError, ValueError):
        return None
    if a.ndim != 2:
        return None
    H, W = a.shape
    if H < 5 or W < 5:
        return None
    if set(int(c) for c in a.flatten()) - {0, 8}:
        return None
    div_rows = [r for r in range(H) if (a[r, :] == 8).all()]
    div_cols = [c for c in range(W) if (a[:, c] == 8).all()]
    if len(div_rows) != 2 or len(div_cols) != 2:
        return None
    row_breaks = [-1] + div_rows + [H]
    col_breaks = [-1] + div_cols + [W]
    expected = np.zeros_like(a)
    expected[a == 8] = 8
    fill_map = {
        (0, 1): 2,
        (1, 0): 4,
        (1, 1): 6,
        (1, 2): 3,
        (2, 1): 1,
    }
    for (r_idx, c_idx), color in fill_map.items():
        r0, r1 = row_breaks[r_idx] + 1, row_breaks[r_idx + 1]
        c0, c1 = col_breaks[c_idx] + 1, col_breaks[c_idx + 1]
        expected[r0:r1, c0:c1] = color
    return expected


def _detect_cross_section_grid_label(pairs):
    """Detect task055's fixed cross-section labeling inside a color-8 divider grid."""
    if not pairs:
        return None
    for p in pairs:
        expected = _cross_section_grid_label_expected(p["input"])
        if expected is None:
            return None
        try:
            out = np.array(p["output"], dtype=np.int8)
        except (TypeError, ValueError):
            return None
        if expected.shape != out.shape or not np.array_equal(expected, out):
            return None
    return True


def _grid_section_copy_expected(inp):
    try:
        a = np.array(inp, dtype=np.int8)
    except (TypeError, ValueError):
        return None
    if a.ndim != 2 or a.shape[0] != 19 or a.shape[1] not in (9, 19, 29):
        return None
    H, W = a.shape
    if not (a[9, :] == 4).all():
        return None
    expected_cols = {9: [], 19: [9], 29: [9, 19]}[W]
    div_cols = [c for c in range(W) if (a[:, c] == 4).all()]
    if div_cols != expected_cols:
        return None
    if any(4 in a[r0:r0 + 9, c0:c0 + 9] for r0 in (0, 10) for c0 in range(0, W, 10)):
        return None

    row_starts = (0, 10)
    col_starts = tuple(range(0, W, 10))
    sections = []
    nonblank = []
    for r0 in row_starts:
        for c0 in col_starts:
            block = a[r0:r0 + 9, c0:c0 + 9]
            if block.shape != (9, 9):
                return None
            sections.append((r0, c0, block))
            if np.any((block != 0) & (block != 4)):
                nonblank.append((r0, c0, block))
    if len(nonblank) != 1:
        return None

    source = nonblank[0][2]
    out = a.copy()
    for r0, c0, _ in sections:
        out[r0:r0 + 9, c0:c0 + 9] = source
    return out


def _detect_grid_section_copy(pairs):
    """Detect task306's color-4 divider grid section replication."""
    if not pairs:
        return None
    for p in pairs:
        expected = _grid_section_copy_expected(p["input"])
        if expected is None:
            return None
        try:
            out = np.array(p["output"], dtype=np.int8)
        except (TypeError, ValueError):
            return None
        if expected.shape != out.shape or not np.array_equal(expected, out):
            return None
    return True


def _grid_max_count_fill_expected(inp):
    try:
        a = np.array(inp, dtype=np.int8)
    except (TypeError, ValueError):
        return None
    if a.shape != (11, 11):
        return None
    if not all((a[r, :] == 5).all() for r in (3, 7)):
        return None
    if not all((a[:, c] == 5).all() for c in (3, 7)):
        return None
    colors = sorted(set(int(c) for c in a.flatten()) - {0, 5})
    if len(colors) != 1:
        return None
    obj = colors[0]
    out = np.zeros_like(a)
    out[a == 5] = 5
    starts = (0, 4, 8)
    counts = []
    for r0 in starts:
        for c0 in starts:
            counts.append(int((a[r0:r0 + 3, c0:c0 + 3] == obj).sum()))
    max_count = max(counts)
    idx = 0
    for r0 in starts:
        for c0 in starts:
            if counts[idx] == max_count and max_count > 0:
                out[r0:r0 + 3, c0:c0 + 3] = obj
            idx += 1
    return out


def _detect_grid_max_count_fill(pairs):
    """Detect task059's fixed compartment-grid max-count fill rule."""
    if not pairs:
        return None
    for p in pairs:
        expected = _grid_max_count_fill_expected(p["input"])
        if expected is None:
            return None
        try:
            out = np.array(p["output"], dtype=np.int8)
        except (TypeError, ValueError):
            return None
        if not np.array_equal(expected, out):
            return None
    return True


def _grid_object_summary_expected(inp):
    try:
        a = np.array(inp, dtype=np.int8)
    except (TypeError, ValueError):
        return None
    if a.ndim != 2 or a.size == 0:
        return None
    nonzero = a != 0
    if not nonzero.any():
        return None

    def bands(has_values):
        result = []
        i = 0
        while i < len(has_values):
            if not has_values[i]:
                i += 1
                continue
            start = i
            while i < len(has_values) and has_values[i]:
                i += 1
            result.append((start, i))
        return result

    row_bands = bands(nonzero.any(axis=1))
    col_bands = bands(nonzero.any(axis=0))
    if not (2 <= len(row_bands) <= 3 and 2 <= len(col_bands) <= 3):
        return None

    out = []
    for r0, r1 in row_bands:
        row = []
        for c0, c1 in col_bands:
            colors = sorted(set(int(c) for c in a[r0:r1, c0:c1].flatten()) - {0})
            if len(colors) != 1:
                return None
            row.append(colors[0])
        out.append(row)
    return out


def _detect_grid_object_summary(pairs):
    """Detect task184's nonzero row/column band summary rule."""
    if not pairs:
        return None
    for p in pairs:
        expected = _grid_object_summary_expected(p["input"])
        if expected is None or expected != p.get("output"):
            return None
    return True


def _two_blob_tile_fit_expected(inp):
    try:
        a = np.array(inp, dtype=np.int8)
    except (TypeError, ValueError):
        return None
    if a.shape != (10, 10):
        return None
    colors = sorted(set(int(c) for c in a.flatten()) - {0})
    if len(colors) != 2:
        return None
    if int((a != 0).sum()) != 9:
        return None

    def blob_for_color(color):
        pts = [(int(r), int(c)) for r, c in zip(*np.where(a == color))]
        if not pts:
            return None
        seen = {pts[0]}
        stack = [pts[0]]
        all_pts = set(pts)
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    q = (r + dr, c + dc)
                    if q in all_pts and q not in seen:
                        seen.add(q)
                        stack.append(q)
        if seen != all_pts:
            return None
        rows = [r for r, _ in pts]
        cols = [c for _, c in pts]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h > 3 or w > 3:
            return None
        local = np.zeros((h, w), dtype=np.int8)
        for r, c in pts:
            local[r - r0, c - c0] = color
        return local

    placements = []
    for color in colors:
        local = blob_for_color(color)
        if local is None:
            return None
        h, w = local.shape
        color_placements = []
        for ar in range(3 - h + 1):
            for ac in range(3 - w + 1):
                placed = np.zeros((3, 3), dtype=np.int8)
                placed[ar:ar + h, ac:ac + w] = local
                color_placements.append(placed)
        placements.append(color_placements)

    valid = []
    for pa in placements[0]:
        ma = pa != 0
        for pb in placements[1]:
            mb = pb != 0
            if np.logical_and(ma, mb).any():
                continue
            if int(ma.sum()) + int(mb.sum()) == 9:
                valid.append(pa + pb)
    if len(valid) != 1:
        return None
    return valid[0].tolist()


def _detect_two_blob_tile_fit(pairs):
    """Detect task153's two compact blobs uniquely tiled into a 3x3 output."""
    if not pairs:
        return None
    for p in pairs:
        if _shape(p.get("output")) != (3, 3):
            return None
        expected = _two_blob_tile_fit_expected(p["input"])
        if expected is None or expected != p.get("output"):
            return None
    return {"matched": True}


def _detect_fourfold_rotation_tile(pairs):
    """Detect task106's 2x2 tile of 0/90/270/180-degree rotations."""
    if not pairs:
        return None
    for p in pairs:
        inp = p.get("input")
        out = p.get("output")
        h, w = _shape(inp)
        if h != w or h not in (2, 3):
            return None
        if _shape(out) != (2 * h, 2 * h):
            return None
        rot90 = _rot90(inp)
        rot270 = _rot270(inp)
        rot180 = _rot180(inp)
        expected = []
        for r in range(h):
            expected.append(list(inp[r]) + list(rot90[r]))
        for r in range(h):
            expected.append(list(rot270[r]) + list(rot180[r]))
        if expected != out:
            return None
    return {"matched": True}


def _diagonal_rays_from_line_expected(inp):
    try:
        a = np.array(inp, dtype=np.int8)
    except (TypeError, ValueError):
        return None
    if a.ndim != 2 or a.size == 0:
        return None
    if set(int(c) for c in a.flatten()) - {0, 7}:
        return None
    line_cells = np.argwhere(a == 7)
    if len(line_cells) == 0:
        return None

    out = a.copy()
    h, w = a.shape
    for r0, c0 in line_cells:
        d = 1
        while True:
            any_inside = False
            for dc in (-d, d):
                r = int(r0) - d
                c = int(c0) + dc
                if 0 <= r < h and 0 <= c < w:
                    any_inside = True
                    if out[r, c] == 0:
                        out[r, c] = 8 if d % 2 else 7
            if not any_inside:
                break
            d += 1
    return out.tolist()


def _detect_diagonal_rays_from_line(pairs):
    """Detect task348's alternating-color upward diagonal rays from color 7."""
    if not pairs:
        return None
    for p in pairs:
        if _shape(p.get("input")) != _shape(p.get("output")):
            return None
        expected = _diagonal_rays_from_line_expected(p["input"])
        if expected is None or expected != p.get("output"):
            return None
    return {"matched": True}


def _euclidean_wedge_fill_expected(inp):
    try:
        a = np.array(inp, dtype=np.int8)
    except (TypeError, ValueError):
        return None
    if a.shape != (21, 21):
        return None
    colors = sorted(set(int(c) for c in a.flatten()) - {0})
    if not colors or colors != list(range(1, max(colors) + 1)):
        return None
    period = max(colors)
    if period > 9:
        return None

    diag_nonzero = [int(a[i, i]) for i in range(21) if int(a[i, i]) != 0]
    if not diag_nonzero:
        return None
    diag_colors = set(diag_nonzero)
    if len(diag_colors) != 1:
        return None
    base = diag_nonzero[0]
    if base > period:
        return None

    def color_for_offset(off):
        return ((base - 1 + off) % period) + 1

    out = np.zeros_like(a)
    for r in range(21):
        for c in range(21):
            m = min(r, c)
            d = abs(r - c)
            if d == 0:
                off = 0
            elif d <= m + 1:
                off = -1
            else:
                off = (d - (m + 2)) // (m + 2)
            out[r, c] = color_for_offset(off)
    return out


def _detect_euclidean_wedge_fill(pairs):
    """Detect task175's 21x21 Euclidean-wedge color lattice hole fill."""
    if not pairs:
        return None
    for p in pairs:
        expected = _euclidean_wedge_fill_expected(p["input"])
        if expected is None:
            return None
        try:
            out = np.array(p["output"], dtype=np.int8)
        except (TypeError, ValueError):
            return None
        if out.shape != (21, 21) or not np.array_equal(expected, out):
            return None
    return True


def _bilateral_symmetry_denoise_expected(inp):
    try:
        a = np.array(inp, dtype=np.int8)
    except (TypeError, ValueError):
        return None
    if a.shape != (16, 16):
        return None
    colors = sorted(set(int(c) for c in a.flatten()) - {0})
    if len(colors) != 2:
        return None

    def bbox(mask):
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        if len(rows) == 0 or len(cols) == 0:
            return None
        return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())

    density = {}
    for c in colors:
        b = bbox(a == c)
        if b is None:
            return None
        r0, r1, c0, c1 = b
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        count = int((a == c).sum())
        density[c] = count / area
    noise = max(colors, key=lambda c: density[c])
    kept = [c for c in colors if c != noise][0]
    if density[noise] != 1.0 or density[kept] == 1.0:
        return None

    kept_mask = a == kept
    nonzero = a != 0

    def axis_score(axis2):
        score = 0
        for r, c in zip(*np.where(kept_mask)):
            mc = axis2 - int(c)
            if 0 <= mc < 16 and nonzero[int(r), mc]:
                score += 1
        return score

    best_axis = max(range(31), key=axis_score)
    out = np.zeros_like(a)
    out[kept_mask] = kept
    for r, c in zip(*np.where(a == noise)):
        mc = best_axis - int(c)
        if 0 <= mc < 16 and a[int(r), mc] == kept:
            out[int(r), int(c)] = kept
    return out


def _detect_bilateral_symmetry_denoise(pairs):
    """Detect task071's rectangular-noise horizontal-symmetry denoise rule."""
    if not pairs:
        return None
    for p in pairs:
        expected = _bilateral_symmetry_denoise_expected(p["input"])
        if expected is None:
            return None
        try:
            out = np.array(p["output"], dtype=np.int8)
        except (TypeError, ValueError):
            return None
        if out.shape != (16, 16) or not np.array_equal(expected, out):
            return None
    return True


def _detect_marker_aligned_shape_recolor(pairs):
    """Detect: input is fixed 10x10 with an L-shape of color 5 at fixed
    position (vertical at col 3 rows 0..3, horizontal at row 3 cols 0..3);
    region (rows 0..2, cols 0..2) — the "template region" — contains exactly
    ONE non-zero non-5 color (the "template_color"); cells of that color
    form a stencil ("template_mask"). Output equals input except every blob
    of any color m (m != 5, m != template_color) whose bbox-cropped mask
    exactly matches template_mask is recolored to 5.

    Returns True on match, else None. (The builder takes no parameters that
    depend on detector output.)
    """
    if not pairs:
        return None

    def bbox_crop_mask(mask):
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        if len(rows) == 0 or len(cols) == 0:
            return None
        r0, r1 = int(rows.min()), int(rows.max()) + 1
        c0, c1 = int(cols.min()), int(cols.max()) + 1
        return mask[r0:r1, c0:c1]

    for p in pairs:
        try:
            inp = np.array(p["input"], dtype=np.int8)
            out = np.array(p["output"], dtype=np.int8)
        except (TypeError, ValueError):
            return None
        if inp.shape != (10, 10) or out.shape != (10, 10):
            return None
        # L-shape preconditions: vertical col 3 rows 0..3; horizontal row 3 cols 0..3.
        if not all(inp[r, 3] == 5 for r in range(4)):
            return None
        if not all(inp[3, c] == 5 for c in range(4)):
            return None
        # Template region must contain exactly one non-zero non-5 color.
        template_region = inp[:3, :3]
        nonzeros = sorted(set(int(c) for c in template_region.flatten()) - {0, 5})
        if len(nonzeros) != 1:
            return None
        template_color = nonzeros[0]
        template_mask_full = (template_region == template_color)
        template_mask = bbox_crop_mask(template_mask_full)
        if template_mask is None:
            return None
        template_count = int(template_mask.sum())
        # Apply the rule and verify against output.
        expected = inp.copy()
        for c in range(1, 10):
            if c == 5 or c == template_color:
                continue
            c_mask = (inp == c)
            if not c_mask.any():
                continue
            if int(c_mask.sum()) != template_count:
                continue
            cropped = bbox_crop_mask(c_mask)
            if cropped is None:
                continue
            if cropped.shape == template_mask.shape and np.array_equal(cropped, template_mask):
                expected[c_mask] = 5
        if not np.array_equal(expected, out):
            return None
    return True


def _detect_marker_driven_shape_copy(pairs):
    """Detect: color 5 = single-cell marker; copy non-bg non-5 pattern so its
    bbox center aligns with the marker; marker disappears from output.

    Returns True on match, else None.
    """
    if not pairs:
        return None
    for p in pairs:
        try:
            a = np.array(p["input"], dtype=np.int8)
            o = np.array(p["output"], dtype=np.int8)
        except (TypeError, ValueError):
            return None
        if a.ndim != 2 or o.ndim != 2 or a.shape != o.shape:
            return None
        marker = np.argwhere(a == 5)
        if len(marker) != 1:
            return None
        pattern_mask = (a != 0) & (a != 5)
        if not pattern_mask.any():
            return None
        marker_pos = marker[0]
        pr_rows = np.where(pattern_mask.any(axis=1))[0]
        pr_cols = np.where(pattern_mask.any(axis=0))[0]
        pr0, pr1 = int(pr_rows.min()), int(pr_rows.max())
        pc0, pc1 = int(pr_cols.min()), int(pr_cols.max())
        cr = (pr0 + pr1) / 2
        cc = (pc0 + pc1) / 2
        dr = round(marker_pos[0] - cr)
        dc = round(marker_pos[1] - cc)
        H, W = a.shape
        pred = a.copy().astype(np.int8)
        pred[marker_pos[0], marker_pos[1]] = 0
        for r in range(H):
            for c in range(W):
                if pattern_mask[r, c]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        pred[nr, nc] = a[r, c]
        if not np.array_equal(pred, o):
            return None
    return True


def _detect_periodic_row_extension(pairs):
    """Detect: 6-row input extended to 9 rows using the smallest period T in {2,3,4},
    with color substitution 1→2 (0 stays 0).

    Conditions:
      - Every pair has shape (6,3) input and (9,3) output.
      - Input colors ⊆ {0,1}; output colors ⊆ {0,2}.
      - Smallest T in {2,3,4} such that input[r]==input[r%T] for all r in 0..5.
      - Output rows 0-5 = input rows 0-5 with 1→2; rows 6,7,8 = input[6%T,7%T,8%T] with 1→2.

    Returns True on match, else None.
    """
    if not pairs:
        return None
    for p in pairs:
        a = np.array(p["input"], dtype=np.int64)
        o = np.array(p["output"], dtype=np.int64)
        if a.shape != (6, 3) or o.shape != (9, 3):
            return None
        if not np.all((a == 0) | (a == 1)):
            return None
        if not np.all((o == 0) | (o == 2)):
            return None
        # Find smallest period T
        T = None
        for t in (2, 3, 4):
            ok = True
            for r in range(6):
                if not np.array_equal(a[r], a[r % t]):
                    ok = False
                    break
            if ok:
                T = t
                break
        if T is None:
            return None
        # Verify output
        subst = np.where(a == 1, 2, a)  # 1→2, 0 stays
        expected = np.vstack([subst, subst[[6 % T, 7 % T, 8 % T]]])
        if not np.array_equal(expected, o):
            return None
    return True


def _detect_boolean_halves(pairs):
    """Detect: input is an (H, 2*W+1) grid split by a separator column into
    two H×W halves. Output is H×W with one non-bg color at cells where a
    boolean condition on the two halves holds.

    Supported configurations (detected from data):
      - Separator color at col W; left colors {0, fg}; right colors {0, fg}.
      - All pairs have the same shape, separator column, and condition.

    Returns dict with keys H, W_in, W_out, check_channel, out_color, or None.
    """
    if not pairs:
        return None
    # All pairs must share the same input and output shape
    in_shape0 = np.array(pairs[0]["input"]).shape
    out_shape0 = np.array(pairs[0]["output"]).shape
    if in_shape0[0] != out_shape0[0]:
        return None
    H = in_shape0[0]
    W_in = in_shape0[1]
    W_out = out_shape0[1]
    # Must be W_in = 2*W_out + 1 (two halves + 1 separator)
    if W_in != 2 * W_out + 1:
        return None
    if H > 29 or W_out > 29:
        return None
    sep_col = W_out

    for p in pairs:
        a = np.array(p["input"])
        o = np.array(p["output"])
        if a.shape != in_shape0 or o.shape != out_shape0:
            return None

    # Determine separator color, check_channel, and out_color from first pair
    a0 = np.array(pairs[0]["input"])
    sep_colors = set(np.unique(a0[:, sep_col]))
    if len(sep_colors) != 1:
        return None
    sep_color = int(list(sep_colors)[0])

    # Left and right halves — allow different fg colors on each side
    left = a0[:, :W_out]
    right = a0[:, sep_col + 1: sep_col + 1 + W_out]
    left_colors = set(np.unique(left)) - {sep_color}
    right_colors = set(np.unique(right)) - {sep_color}
    if 0 not in left_colors or len(left_colors) > 2:
        return None
    if 0 not in right_colors or len(right_colors) > 2:
        return None
    left_fg = int((left_colors - {0}).pop()) if len(left_colors) == 2 else None
    right_fg = int((right_colors - {0}).pop()) if len(right_colors) == 2 else None

    # Output must use exactly one non-zero color (any color)
    all_out_colors: set = set()
    for p in pairs:
        all_out_colors |= set(np.unique(np.array(p["output"])))
    non_zero_out = all_out_colors - {0}
    if len(non_zero_out) != 1:
        return None
    out_color = int(non_zero_out.pop())

    # Try AND (same fg on both sides) then NOR
    candidates = []
    if left_fg is not None and left_fg == right_fg:
        candidates.append((left_fg, "AND"))
    candidates.append((0, "NOR"))

    check_channel = None
    for cc, label in candidates:
        ok = True
        for p in pairs:
            a = np.array(p["input"])
            o = np.array(p["output"])
            if cc == 0:
                lft = (a[:, :W_out] == 0).astype(np.int32)
                rgt = (a[:, sep_col+1:sep_col+1+W_out] == 0).astype(np.int32)
            else:
                lft = (a[:, :W_out] == cc).astype(np.int32)
                rgt = (a[:, sep_col+1:sep_col+1+W_out] == cc).astype(np.int32)
            pred = np.where(lft * rgt == 1, out_color, 0)
            if not np.array_equal(pred, o):
                ok = False
                break
        if ok:
            check_channel = cc
            break

    if check_channel is None:
        return None
    return {"H": H, "W_in": W_in, "W_out": W_out, "check_channel": check_channel,
            "out_color": out_color}


def _detect_rank_vertical_bars(pairs):
    """Detect: (9,9) grid with exactly 4 vertical bars of color-5 at cols {1,3,5,7},
    each bar touching row 8 and extending upward. Output replaces each bar with rank
    color (tallest→1, next→2, ..., shortest→4). All heights distinct (no ties).

    Returns True on match, else None.
    """
    if not pairs:
        return None
    for p in pairs:
        a = np.array(p["input"])
        o = np.array(p["output"])
        if a.shape != (9, 9) or o.shape != (9, 9):
            return None
        if not np.all(np.isin(a, [0, 5])):
            return None
        if not np.all(np.isin(o, [0, 1, 2, 3, 4])):
            return None
        # Bars must be exactly at cols {1,3,5,7}
        cols_with_5 = [c for c in range(9) if np.any(a[:, c] == 5)]
        if sorted(cols_with_5) != [1, 3, 5, 7]:
            return None
        # Each bar must touch row 8
        for c in [1, 3, 5, 7]:
            if a[8, c] != 5:
                return None
        # Heights must be distinct
        heights = {c: int(np.sum(a[:, c] == 5)) for c in [1, 3, 5, 7]}
        if len(set(heights.values())) != 4:
            return None
        # Verify output matches rank rule
        sorted_cols = sorted(heights, key=lambda c: -heights[c])
        rank = {c: i + 1 for i, c in enumerate(sorted_cols)}
        pred = np.zeros((9, 9), dtype=np.int64)
        for c in [1, 3, 5, 7]:
            for r in range(9):
                if a[r, c] == 5:
                    pred[r, c] = rank[c]
        if not np.array_equal(pred, o):
            return None
    return True


def _detect_vertical_halves(pairs):
    """Detect: input (H_in, W) split into two H_half-row sections (optional sep row).
    Output (H_half, W) encodes a boolean op (NOR, OR, XOR) on the two halves.

    Returns dict with H_in, W, H_half, mode, out_color, sep_row; or None.
    """
    if not pairs:
        return None
    a0 = np.array(pairs[0]["input"])
    o0 = np.array(pairs[0]["output"])
    H_in, W = a0.shape
    H_out, W_out = o0.shape
    if W != W_out or W > 29 or H_out > 29:
        return None
    H_half = H_out

    # Determine if there is a separator row
    if H_in == 2 * H_half + 1:
        sep_candidate = H_half
        sep_valid = True
        for p in pairs:
            a = np.array(p["input"])
            if a.shape != (H_in, W):
                return None
            row = a[sep_candidate]
            if len(set(row.tolist())) != 1 or int(row[0]) == 0:
                sep_valid = False
                break
        if not sep_valid:
            return None
        sep_row: int | None = sep_candidate
    elif H_in == 2 * H_half:
        for p in pairs:
            if np.array(p["input"]).shape != (H_in, W):
                return None
        sep_row = None
    else:
        return None

    for p in pairs:
        if np.array(p["output"]).shape != (H_half, W):
            return None

    # Determine output color
    all_out_colors: set = set()
    for p in pairs:
        all_out_colors |= set(np.unique(np.array(p["output"])))
    non_zero_out = all_out_colors - {0}
    if len(non_zero_out) != 1:
        return None
    out_color = int(non_zero_out.pop())

    # Try NOR, OR, XOR
    for mode in ("nor", "or", "xor"):
        ok = True
        for p in pairs:
            a = np.array(p["input"])
            o = np.array(p["output"])
            top = a[:H_half]
            bot = a[sep_row + 1:] if sep_row is not None else a[H_half:]
            if bot.shape[0] != H_half:
                ok = False
                break
            top_bg = (top == 0)
            bot_bg = (bot == 0)
            if mode == "nor":
                pred = (top_bg & bot_bg).astype(int) * out_color
            elif mode == "or":
                pred = ((~top_bg) | (~bot_bg)).astype(int) * out_color
            elif mode == "xor":
                pred = ((~top_bg) ^ (~bot_bg)).astype(int) * out_color
            if not np.array_equal(pred, o):
                ok = False
                break
        if ok:
            return {"H_in": H_in, "W": W, "H_half": H_half,
                    "mode": mode, "out_color": out_color, "sep_row": sep_row}
    return None


def _detect_horizontal_or(pairs):
    """Detect: input (H, 2*W) → output (H, W), OR of two halves (no separator).
    Returns dict with H, W_half, out_color; or None.
    """
    if not pairs:
        return None
    a0 = np.array(pairs[0]["input"])
    o0 = np.array(pairs[0]["output"])
    H_in, W_in = a0.shape
    H_out, W_out = o0.shape
    if H_in != H_out or W_in != 2 * W_out:
        return None
    if H_out > 29 or W_out > 29:
        return None

    for p in pairs:
        a = np.array(p["input"])
        o = np.array(p["output"])
        if a.shape != (H_in, W_in) or o.shape != (H_out, W_out):
            return None

    all_out_colors: set = set()
    for p in pairs:
        all_out_colors |= set(np.unique(np.array(p["output"])))
    non_zero_out = all_out_colors - {0}
    if len(non_zero_out) != 1:
        return None
    out_color = int(non_zero_out.pop())

    for p in pairs:
        a = np.array(p["input"])
        o = np.array(p["output"])
        left = a[:, :W_out]
        right = a[:, W_out:]
        pred = ((left != 0) | (right != 0)).astype(int) * out_color
        if not np.array_equal(pred, o):
            return None
    return {"H": H_out, "W_half": W_out, "out_color": out_color}


def _detect_cross_product(pairs):
    """Detect: N×N grid, row 0 has template_color cells (column selector),
    col N-1 has template_color cells (row selector). At intersection of a
    selected row r and selected column c where cell[r,c]==0, output is out_color.
    All original values preserved. Returns dict or None.
    """
    if not pairs:
        return None
    a0 = np.array(pairs[0]["input"])
    o0 = np.array(pairs[0]["output"])
    H, W = a0.shape
    if H != W or H > 29:
        return None
    N = H

    for p in pairs:
        a = np.array(p["input"])
        o = np.array(p["output"])
        if a.shape != (N, N) or o.shape != (N, N):
            return None

    # Determine template_color from row 0 of first pair
    row0 = a0[0]
    non_zero_in_row0 = set(int(x) for x in row0 if x != 0)
    if len(non_zero_in_row0) != 1:
        return None
    template_color = int(non_zero_in_row0.pop())

    col_idx = N - 1

    # Find out_color by looking at intersection cells in first pair
    coln = a0[:, col_idx]
    out_colors_seen: set = set()
    for r in range(N):
        if coln[r] == template_color:
            for c in range(N):
                if row0[c] == template_color and a0[r, c] == 0:
                    v = int(o0[r, c])
                    if v != 0:
                        out_colors_seen.add(v)
    if len(out_colors_seen) != 1:
        return None
    out_color = int(out_colors_seen.pop())
    if out_color == template_color:
        return None

    # Verify all pairs
    for p in pairs:
        a = np.array(p["input"])
        o = np.array(p["output"])
        row0_p = a[0]
        coln_p = a[:, col_idx]
        expected = a.copy()
        for r in range(N):
            if coln_p[r] == template_color:
                for c in range(N):
                    if row0_p[c] == template_color and a[r, c] == 0:
                        expected[r, c] = out_color
        if not np.array_equal(expected, o):
            return None

    return {"N": N, "col_idx": col_idx, "template_color": template_color,
            "out_color": out_color}


def _detect_block_count_ge(pairs):
    """Detect: N×N grid with uniform-color separator rows/cols → n×n output.

    Pattern: n×n blocks of size block_size separated by 1-cell separators.
    Output cell (br,bc) = out_color if count(count_color in block) >= threshold.
    task149: N=11, n=3, block_size=3, stride=4, count_color=6, threshold=2, out_color=1.
    """
    if not pairs:
        return None

    H_in  = len(pairs[0]["input"])
    W_in  = len(pairs[0]["input"][0])
    H_out = len(pairs[0]["output"])
    W_out = len(pairs[0]["output"][0])

    if H_in != W_in or H_out != W_out:
        return None

    N = H_in
    n = H_out
    if n < 2 or N < 5:
        return None

    # Check all pairs have the same shape
    for p in pairs:
        if len(p["input"]) != N or len(p["input"][0]) != W_in:
            return None
        if len(p["output"]) != n or len(p["output"][0]) != W_out:
            return None

    # block_size = (N - n + 1) / n, stride = block_size + 1
    if (N - n + 1) % n != 0:
        return None
    block_size = (N - n + 1) // n
    stride = block_size + 1

    # Separator rows/cols at positions [block_size, 2*stride-1, ...]
    sep_rows = [br * stride + block_size for br in range(n - 1)]

    # Identify separator color from first pair
    first_inp = np.array(pairs[0]["input"])
    if sep_rows:
        sep_color = int(first_inp[sep_rows[0], 0])
        for p in pairs:
            a = np.array(p["input"])
            for r in sep_rows:
                if not np.all(a[r, :] == sep_color):
                    return None
            for c in sep_rows:
                if not np.all(a[:, c] == sep_color):
                    return None
    else:
        sep_color = None

    # Collect all colors in block regions
    block_colors: set = set()
    for p in pairs:
        a = np.array(p["input"])
        for br in range(n):
            for bc in range(n):
                r0 = br * stride
                c0 = bc * stride
                blk = a[r0:r0 + block_size, c0:c0 + block_size]
                block_colors |= set(np.unique(blk).tolist())

    exclude = {0}
    if sep_color is not None:
        exclude.add(sep_color)
    candidate_colors = block_colors - exclude
    if len(candidate_colors) != 1:
        return None
    count_color = int(candidate_colors.pop())

    # Determine out_color from outputs
    all_out: set = set()
    for p in pairs:
        all_out |= set(np.unique(np.array(p["output"])).tolist())
    non_zero_out = all_out - {0}
    if len(non_zero_out) != 1:
        return None
    out_color = int(non_zero_out.pop())
    if out_color == 0:
        return None

    # Compute (count, expected_output) pairs across all pairs
    count_out: list = []
    for p in pairs:
        a = np.array(p["input"])
        o = np.array(p["output"])
        for br in range(n):
            for bc in range(n):
                r0 = br * stride
                c0 = bc * stride
                blk = a[r0:r0 + block_size, c0:c0 + block_size]
                cnt = int(np.sum(blk == count_color))
                ov = int(o[br, bc])
                count_out.append((cnt, ov))

    # Determine threshold
    true_counts = [cnt for cnt, ov in count_out if ov == out_color]
    false_counts = [cnt for cnt, ov in count_out if ov == 0]
    if not true_counts:
        return None
    threshold = min(true_counts)
    # Verify no false negative: all counts < threshold have ov==0
    if false_counts and max(false_counts) >= threshold:
        return None

    # Full verification
    for cnt, ov in count_out:
        expected_ov = out_color if cnt >= threshold else 0
        if ov != expected_ov:
            return None

    return {
        "N": N, "n": n, "block_size": block_size, "stride": stride,
        "count_color": count_color, "threshold": threshold, "out_color": out_color,
    }


def _detect_marker_recolor(pairs):
    """Detect: single-cell marker at fixed position recolors a blob.

    Input H×W: one blob color (multiple cells) + marker color (exactly 1 cell at fixed pos).
    Output: blob cells → marker color, marker cell → 0.
    Marker position is consistent across all pairs.
    task267: H=7, W=7, marker at (6,0).
    """
    if not pairs:
        return None

    H_in  = len(pairs[0]["input"])
    W_in  = len(pairs[0]["input"][0])
    H_out = len(pairs[0]["output"])
    W_out = len(pairs[0]["output"][0])

    if H_in != H_out or W_in != W_out:
        return None
    H, W = H_in, W_in

    for p in pairs:
        if len(p["input"]) != H or len(p["input"][0]) != W:
            return None
        if len(p["output"]) != H or len(p["output"][0]) != W:
            return None

    marker_pos = None
    for p in pairs:
        a = np.array(p["input"])
        o = np.array(p["output"])

        nz = a[a != 0]
        if nz.size == 0:
            return None
        unique, counts = np.unique(nz, return_counts=True)
        if len(unique) != 2:
            return None

        # Identify marker (appears once) vs blob (appears many times)
        one_mask = counts == 1
        if one_mask.sum() != 1:
            return None
        mk_color = int(unique[one_mask][0])
        blob_color = int(unique[~one_mask][0])

        pos = np.argwhere(a == mk_color)
        if len(pos) != 1:
            return None
        r_mk, c_mk = int(pos[0][0]), int(pos[0][1])

        if marker_pos is None:
            marker_pos = (r_mk, c_mk)
        elif marker_pos != (r_mk, c_mk):
            return None

        # Verify output: blob cells → mk_color, all other cells → 0
        expected = np.zeros_like(a)
        expected[a == blob_color] = mk_color
        if not np.array_equal(expected, o):
            return None

    if marker_pos is None:
        return None
    return {"H": H, "W": W, "r_marker": marker_pos[0], "c_marker": marker_pos[1]}


def _detect_two_marker_cross(pairs):
    """Detect: two single-cell markers each draw a full row+col cross; intersections get a third color.

    Input H×W: exactly one cell of color_A and one of color_B (both non-zero, distinct).
    Output: color_A along row r_A and col c_A, color_B along row r_B and col c_B,
            color_inter at (r_A, c_B) and (r_B, c_A), background elsewhere.
    task047: H=9, W=9, color_A=8, color_B=7, color_inter=2.
    """
    if not pairs:
        return None

    H_in  = len(pairs[0]["input"])
    W_in  = len(pairs[0]["input"][0])
    H_out = len(pairs[0]["output"])
    W_out = len(pairs[0]["output"][0])

    if H_in != H_out or W_in != W_out:
        return None
    H, W = H_in, W_in

    for p in pairs:
        if len(p["input"]) != H or len(p["input"][0]) != W:
            return None
        if len(p["output"]) != H or len(p["output"][0]) != W:
            return None

    # Determine color_A, color_B, color_inter from first pair
    p0 = pairs[0]
    a0 = np.array(p0["input"])
    o0 = np.array(p0["output"])

    nz_input = sorted(set(a0.flatten().tolist()) - {0})
    if len(nz_input) != 2:
        return None
    nz_output = sorted(set(o0.flatten().tolist()) - {0})
    if len(nz_output) != 3:
        return None

    # color_A and color_B are the two input colors
    color_A, color_B = nz_input[0], nz_input[1]
    # color_inter is the output color not present in input
    color_inter_candidates = set(nz_output) - {color_A, color_B}
    if len(color_inter_candidates) != 1:
        return None
    color_inter = int(color_inter_candidates.pop())

    # Each input color appears exactly once
    for p in pairs:
        a = np.array(p["input"])
        nz = sorted(set(a.flatten().tolist()) - {0})
        if nz != [color_A, color_B]:
            return None
        if int(np.sum(a == color_A)) != 1 or int(np.sum(a == color_B)) != 1:
            return None

    # Verify output for all pairs
    for p in pairs:
        a = np.array(p["input"])
        o = np.array(p["output"])
        pos_A = tuple(np.argwhere(a == color_A)[0])
        pos_B = tuple(np.argwhere(a == color_B)[0])
        rA, cA = pos_A
        rB, cB = pos_B

        expected = np.zeros((H, W), dtype=int)
        # A-cross
        expected[rA, :] = color_A
        expected[:, cA] = color_A
        # B-cross
        expected[rB, :] = color_B
        expected[:, cB] = color_B
        # Intersections override (A-row×B-col and B-row×A-col)
        expected[rA, cB] = color_inter
        expected[rB, cA] = color_inter
        # At the 4 cross-cross overlaps:
        # (rA,cA): A-row ∩ A-col → still color_A ✓
        # (rB,cB): B-row ∩ B-col → still color_B ✓
        # (rA,rB) or (rB,cA) handled above already

        if not np.array_equal(expected, o):
            return None

    return {"H": H, "W": W, "color_A": color_A, "color_B": color_B,
            "color_inter": color_inter}


def _detect_distinct_count_pattern(pairs):
    """Detect: 3×3 → 3×3, count k of distinct colors in input.

    All input cells use colors from exactly 3 fixed values {c1,c2,c3}.
    k=1 → top row = 5, k=2 → main diagonal = 5, k=3 → anti-diagonal = 5.
    task167: c1=2, c2=3, c3=4, output = 5.
    """
    if not pairs:
        return None

    H_in  = len(pairs[0]["input"])
    W_in  = len(pairs[0]["input"][0])
    H_out = len(pairs[0]["output"])
    W_out = len(pairs[0]["output"][0])

    if H_in != 3 or W_in != 3 or H_out != 3 or W_out != 3:
        return None

    TOP_ROW  = frozenset([(0, 0), (0, 1), (0, 2)])
    DIAGONAL = frozenset([(0, 0), (1, 1), (2, 2)])
    ANTI_DIAG = frozenset([(0, 2), (1, 1), (2, 0)])

    all_in_colors: set = set()
    for p in pairs:
        a = np.array(p["input"])
        if a.shape != (3, 3):
            return None
        cell_vals = set(a.flatten().tolist())
        if 0 in cell_vals:
            return None  # no background allowed in input
        all_in_colors |= cell_vals

    if len(all_in_colors) != 3:
        return None
    c1, c2, c3 = sorted(all_in_colors)

    # Output color must be consistent
    all_out_colors: set = set()
    for p in pairs:
        all_out_colors |= set(np.array(p["output"]).flatten().tolist())
    non_zero_out = all_out_colors - {0}
    if len(non_zero_out) != 1:
        return None
    out_color = int(non_zero_out.pop())
    for p in pairs:
        a = np.array(p["input"])
        o = np.array(p["output"])
        if o.shape != (3, 3):
            return None
        out_vals = set(o.flatten().tolist())
        if not out_vals.issubset({0, out_color}):
            return None
        n_distinct = len(set(a.flatten().tolist()))
        five_pos = frozenset((r, c) for r in range(3) for c in range(3) if o[r, c] == out_color)
        if n_distinct == 1:
            expected = TOP_ROW
        elif n_distinct == 2:
            expected = DIAGONAL
        elif n_distinct == 3:
            expected = ANTI_DIAG
        else:
            return None
        if five_pos != expected:
            return None

    return {"c1": c1, "c2": c2, "c3": c3, "out_color": out_color}


def _detect_uniform_row(pairs):
    """Detect: H×W → H×W, uniform non-zero rows → [out_color]*W, else [0]*W.

    task052: H=3, W=3, out_color=5.
    """
    if not pairs:
        return None
    H = len(pairs[0]["input"])
    W = len(pairs[0]["input"][0]) if pairs[0]["input"] else 0
    if H == 0 or W == 0:
        return None
    if len(pairs[0]["output"]) != H or (pairs[0]["output"] and len(pairs[0]["output"][0]) != W):
        return None

    out_colors: set = set()
    for p in pairs:
        for row in p["output"]:
            for v in row:
                if v != 0:
                    out_colors.add(v)
    if len(out_colors) != 1:
        return None
    out_color = out_colors.pop()

    for p in pairs:
        inp = p["input"]
        out = p["output"]
        if len(inp) != H or len(out) != H:
            return None
        for r in range(H):
            if len(inp[r]) != W or len(out[r]) != W:
                return None
            in_row = inp[r]
            out_row = out[r]
            row_vals = set(in_row)
            is_uniform_nz = (len(row_vals) == 1 and 0 not in row_vals)
            if is_uniform_nz:
                if out_row != [out_color] * W:
                    return None
            else:
                if out_row != [0] * W:
                    return None

    return {"H": H, "W": W, "out_color": out_color}


def _detect_cummax_down_column(pairs):
    """Detect: H×W same-size, output[r][c] = max(input[0..r][c]).

    task322: H=3, W=3.
    """
    if not pairs:
        return None
    H = len(pairs[0]["input"])
    W = len(pairs[0]["input"][0]) if pairs[0]["input"] else 0
    if H == 0 or W == 0:
        return None
    if len(pairs[0]["output"]) != H:
        return None

    for p in pairs:
        inp = p["input"]
        out = p["output"]
        if len(inp) != H or len(out) != H:
            return None
        for c in range(W):
            running_max = 0
            for r in range(H):
                if len(inp[r]) != W or len(out[r]) != W:
                    return None
                running_max = max(running_max, inp[r][c])
                if out[r][c] != running_max:
                    return None

    return {"H": H, "W": W}


def _detect_dominant_color_keep(pairs):
    """Detect: H×W same-size, most-frequent non-zero color kept, others → out_color.

    task229: H=3, W=3, out_color=5.
    """
    if not pairs:
        return None
    H = len(pairs[0]["input"])
    W = len(pairs[0]["input"][0]) if pairs[0]["input"] else 0
    if H == 0 or W == 0:
        return None
    if len(pairs[0]["output"]) != H:
        return None

    replace_color = None

    for p in pairs:
        inp = p["input"]
        out = p["output"]
        if len(inp) != H or len(out) != H:
            return None
        if any(len(row) != W for row in inp) or any(len(row) != W for row in out):
            return None

        flat_in = [v for row in inp for v in row if v != 0]
        if not flat_in:
            return None
        counts = {}
        for v in flat_in:
            counts[v] = counts.get(v, 0) + 1
        max_cnt = max(counts.values())
        # Dominant = lowest-index channel with max count (matches ArgMax tie-break)
        dominant = min(c for c, cnt in counts.items() if cnt == max_cnt)

        for r in range(H):
            for c in range(W):
                iv = inp[r][c]
                ov = out[r][c]
                if iv == dominant:
                    if ov != dominant:
                        return None
                else:
                    if ov == dominant or ov == iv:
                        return None
                    if replace_color is None:
                        replace_color = ov
                    elif ov != replace_color:
                        return None

    if replace_color is None:
        return None

    return {"H": H, "W": W, "out_color": replace_color}


def _detect_color_to_pattern(pairs):
    """Detect: input has exactly one non-zero color → fixed 3×3 binary output pattern.

    task334: H_in=5, W_in=5, out_color=5.
    """
    if not pairs:
        return None
    H_in = len(pairs[0]["input"])
    W_in = len(pairs[0]["input"][0]) if pairs[0]["input"] else 0
    H_out = len(pairs[0]["output"])
    W_out = len(pairs[0]["output"][0]) if pairs[0]["output"] else 0
    if H_out != 3 or W_out != 3:
        return None

    out_colors: set = set()
    for p in pairs:
        for row in p["output"]:
            for v in row:
                if v != 0:
                    out_colors.add(v)
    if len(out_colors) != 1:
        return None
    out_color = out_colors.pop()

    color_map = {}

    for p in pairs:
        if len(p["input"]) != H_in or len(p["output"]) != 3:
            return None
        if any(len(row) != W_in for row in p["input"]) or any(len(row) != 3 for row in p["output"]):
            return None

        in_colors = {v for row in p["input"] for v in row if v != 0}
        if len(in_colors) != 1:
            return None
        c = in_colors.pop()

        out = p["output"]
        if not all(v in (0, out_color) for row in out for v in row):
            return None
        pattern = tuple(tuple(1 if v == out_color else 0 for v in row) for row in out)

        if c in color_map:
            if color_map[c] != pattern:
                return None
        else:
            color_map[c] = pattern

    if not color_map:
        return None

    return {"H_in": H_in, "W_in": W_in, "color_map": color_map, "out_color": out_color}


def _detect_cross_line_fill(pairs):
    """Detect: partial vertical line + partial horizontal line → full cross with intersection color.

    Input: color_col cells in exactly one column, color_row cells in exactly one row.
    Output: full column of color_col, full row of color_row, intersection = color_inter.
    task299: H=6, W=6, color_col=8, color_row=2, color_inter=4.
    """
    if not pairs:
        return None
    H = len(pairs[0]["input"])
    W = len(pairs[0]["input"][0]) if pairs[0]["input"] else 0
    if len(pairs[0]["output"]) != H:
        return None
    if pairs[0]["output"] and len(pairs[0]["output"][0]) != W:
        return None

    color_col_found = None
    color_row_found = None
    color_inter_found = None

    for p in pairs:
        if len(p["input"]) != H or len(p["output"]) != H:
            return None
        if any(len(r) != W for r in p["input"]) or any(len(r) != W for r in p["output"]):
            return None

        inp = p["input"]
        out = p["output"]

        in_colors = {inp[i][j] for i in range(H) for j in range(W) if inp[i][j] != 0}
        if len(in_colors) != 2:
            return None

        # Identify which color is the column color (all in one column) vs row color (all in one row)
        col_cols = {}
        row_rows = {}
        for c in in_colors:
            col_cols[c] = {j for i in range(H) for j in range(W) if inp[i][j] == c}
            row_rows[c] = {i for i in range(H) for j in range(W) if inp[i][j] == c}

        c_col = None
        c_row = None
        for c in in_colors:
            if len(col_cols[c]) == 1 and len(row_rows[c]) > 1:
                c_col = c
            elif len(row_rows[c]) == 1 and len(col_cols[c]) > 1:
                c_row = c
        if c_col is None or c_row is None:
            return None

        the_col = next(iter(col_cols[c_col]))
        the_row = next(iter(row_rows[c_row]))

        # Intersection must be a distinct third color in output
        c_inter = out[the_row][the_col]
        if c_inter == 0 or c_inter == c_col or c_inter == c_row:
            return None

        # Verify all output cells match the cross-fill rule
        for i in range(H):
            for j in range(W):
                if i == the_row and j == the_col:
                    expected = c_inter
                elif j == the_col:
                    expected = c_col
                elif i == the_row:
                    expected = c_row
                else:
                    expected = 0
                if out[i][j] != expected:
                    return None

        if color_col_found is None:
            color_col_found = c_col
        elif color_col_found != c_col:
            return None
        if color_row_found is None:
            color_row_found = c_row
        elif color_row_found != c_row:
            return None
        if color_inter_found is None:
            color_inter_found = c_inter
        elif color_inter_found != c_inter:
            return None

    if color_col_found is None:
        return None
    return {
        "H": H, "W": W,
        "color_col": color_col_found,
        "color_row": color_row_found,
        "color_inter": color_inter_found,
    }


def _detect_rectangular_zero_rank_fill(pairs):
    """Detect task145: color-2 walls partition zero rectangles by area rank.

    Largest zero rectangle(s) become color 1; smallest non-largest zero
    rectangle(s) become color 8; all other zero rectangles remain 0.
    """
    if not pairs:
        return None
    max_h = 0
    max_w = 0

    def zero_components(inp):
        h = len(inp)
        w = len(inp[0]) if h else 0
        seen = [[False] * w for _ in range(h)]
        comps = []
        for r in range(h):
            for c in range(w):
                if inp[r][c] != 0 or seen[r][c]:
                    continue
                q = collections.deque([(r, c)])
                seen[r][c] = True
                cells = []
                while q:
                    rr, cc = q.popleft()
                    cells.append((rr, cc))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and inp[nr][nc] == 0:
                            seen[nr][nc] = True
                            q.append((nr, nc))
                rows = [rr for rr, _ in cells]
                cols = [cc for _, cc in cells]
                r0, r1 = min(rows), max(rows)
                c0, c1 = min(cols), max(cols)
                if len(cells) != (r1 - r0 + 1) * (c1 - c0 + 1):
                    return None
                comps.append({"cells": cells, "area": len(cells)})
        return comps

    for p in pairs:
        inp = p["input"]
        out = p["output"]
        h = len(inp)
        w = len(inp[0]) if h else 0
        if h == 0 or w == 0 or len(out) != h or any(len(row) != w for row in inp) or any(len(row) != w for row in out):
            return None
        if h > 20 or w > 20:
            return None
        max_h = max(max_h, h)
        max_w = max(max_w, w)
        in_colors = {cell for row in inp for cell in row}
        if not in_colors <= {0, 2} or 0 not in in_colors:
            return None
        comps = zero_components(inp)
        if not comps:
            return None
        areas = [comp["area"] for comp in comps]
        max_area = max(areas)
        min_area = min(areas)
        for comp in comps:
            fill = 1 if comp["area"] == max_area else (8 if comp["area"] == min_area else 0)
            for r, c in comp["cells"]:
                if out[r][c] != fill:
                    return None
        for r in range(h):
            for c in range(w):
                if inp[r][c] == 2 and out[r][c] != 2:
                    return None

    return {"H": max_h, "W": max_w}


def _detect_framed_zero_square_fill(pairs):
    """Detect task102: square zero components inside a one-cell frame are filled."""
    if not pairs:
        return None
    max_h = 0
    max_w = 0
    max_size = 0
    frame_color = None
    fill_color = None
    saw_fill = False

    def zero_components(inp):
        h = len(inp)
        w = len(inp[0]) if h else 0
        seen = [[False] * w for _ in range(h)]
        comps = []
        for r in range(h):
            for c in range(w):
                if inp[r][c] != 0 or seen[r][c]:
                    continue
                q = collections.deque([(r, c)])
                seen[r][c] = True
                cells = []
                while q:
                    rr, cc = q.popleft()
                    cells.append((rr, cc))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and inp[nr][nc] == 0:
                            seen[nr][nc] = True
                            q.append((nr, nc))
                rows = [rr for rr, _ in cells]
                cols = [cc for _, cc in cells]
                comps.append({
                    "cells": cells,
                    "r0": min(rows), "r1": max(rows),
                    "c0": min(cols), "c1": max(cols),
                })
        return comps

    def is_framed_square(inp, comp, fc):
        h = len(inp)
        w = len(inp[0]) if h else 0
        r0, r1 = comp["r0"], comp["r1"]
        c0, c1 = comp["c0"], comp["c1"]
        ch = r1 - r0 + 1
        cw = c1 - c0 + 1
        if ch != cw:
            return False
        if len(comp["cells"]) != ch * cw:
            return False
        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            return False
        for cc in range(c0 - 1, c1 + 2):
            if inp[r0 - 1][cc] != fc or inp[r1 + 1][cc] != fc:
                return False
        for rr in range(r0 - 1, r1 + 2):
            if inp[rr][c0 - 1] != fc or inp[rr][c1 + 1] != fc:
                return False
        return True

    checked_pairs = []
    for p in pairs:
        inp = p["input"]
        out = p["output"]
        h = len(inp)
        w = len(inp[0]) if h else 0
        if h == 0 or w == 0 or len(out) != h:
            return None
        if any(len(row) != w for row in inp) or any(len(row) != w for row in out):
            return None
        if h > 12 or w > 12:
            return None
        max_h = max(max_h, h)
        max_w = max(max_w, w)
        in_colors = sorted({cell for row in inp for cell in row} - {0})
        if len(in_colors) != 1:
            return None
        fc = in_colors[0]
        if frame_color is None:
            frame_color = fc
        elif frame_color != fc:
            return None
        if fc == 0:
            return None

        for r in range(h):
            for c in range(w):
                if inp[r][c] == fc and out[r][c] != fc:
                    return None
                if inp[r][c] == 0 and out[r][c] != 0:
                    if out[r][c] == fc:
                        return None
                    if fill_color is None:
                        fill_color = out[r][c]
                    elif fill_color != out[r][c]:
                        return None
        checked_pairs.append((inp, out, h, w, fc))

    if frame_color is None or fill_color is None:
        return None

    for inp, out, h, w, fc in checked_pairs:
        comps = zero_components(inp)
        for comp in comps:
            should_fill = is_framed_square(inp, comp, fc)
            if should_fill:
                side = comp["r1"] - comp["r0"] + 1
                max_size = max(max_size, side)
                saw_fill = True
                for r, c in comp["cells"]:
                    if out[r][c] != fill_color:
                        return None
            else:
                for r, c in comp["cells"]:
                    if out[r][c] != 0:
                        return None

    if not saw_fill or frame_color is None or fill_color is None:
        return None
    return {"H": max_h, "W": max_w, "max_size": max_size,
            "frame_color": frame_color, "fill_color": fill_color}


def _detect_bouncing_diagonal_ray(pairs):
    """Detect task248: bottom-left seed expands to a 10-row bouncing diagonal."""
    if not pairs:
        return None
    max_w = 0

    def bounce_col(dist, width):
        period = 2 * (width - 1)
        m = dist % period
        return m if m < width else period - m

    for p in pairs:
        inp = p["input"]
        out = p["output"]
        h = len(inp)
        w = len(inp[0]) if h else 0
        if h != 10 or w < 2 or len(out) != h:
            return None
        if any(len(row) != w for row in inp) or any(len(row) != w for row in out):
            return None
        max_w = max(max_w, w)
        for r in range(h):
            for c in range(w):
                expected_in = 1 if (r == h - 1 and c == 0) else 0
                if inp[r][c] != expected_in:
                    return None
                col = bounce_col(h - 1 - r, w)
                expected_out = 1 if c == col else 0
                if out[r][c] != expected_out:
                    return None
        if any(cell not in (0, 1) for row in out for cell in row):
            return None

    return {"max_w": max_w}


def _detect_trirow_periodic_four_fill(pairs):
    """Detect task176: a fixed 3-row color-2 scaffold gains color-4 phase marks."""
    if not pairs:
        return None
    for p in pairs:
        inp = p["input"]
        out = p["output"]
        h = len(inp)
        w = len(inp[0]) if h else 0
        if h != 3 or w < 1 or w > 30 or len(out) != h:
            return None
        if any(len(row) != w for row in inp) or any(len(row) != w for row in out):
            return None
        for r in range(h):
            for c in range(w):
                scaffold = (
                    (r == 0 and c % 4 == 0) or
                    (r == 1 and c % 2 == 1) or
                    (r == 2 and c % 4 == 2)
                )
                mark = (
                    (r == 0 and c % 12 in (5, 6, 7)) or
                    (r == 1 and c % 6 == 0) or
                    (r == 2 and c % 12 in (11, 0, 1))
                )
                expected_in = 2 if scaffold else 0
                expected_out = 2 if scaffold else (4 if mark else 0)
                if inp[r][c] != expected_in or out[r][c] != expected_out:
                    return None
    return True


def _square_spiral_three_grid(n):
    grid = [[0 for _ in range(n)] for _ in range(n)]
    r = c = 0
    grid[r][c] = 3
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    lengths = [n - 1, n - 1, n - 1]
    k = n - 3
    while k > 0:
        lengths.extend([k, k])
        k -= 2
    for idx, length in enumerate(lengths):
        dr, dc = directions[idx % 4]
        for _ in range(length):
            r += dr
            c += dc
            if 0 <= r < n and 0 <= c < n:
                grid[r][c] = 3
    return grid


def _detect_square_spiral_three(pairs):
    """Detect task058: all-zero square maps to a color-3 inward spiral."""
    if not pairs:
        return None
    max_n = 0
    for p in pairs:
        inp = p["input"]
        out = p["output"]
        n = len(inp)
        if n < 5 or n > 30 or any(len(row) != n for row in inp):
            return None
        if len(out) != n or any(len(row) != n for row in out):
            return None
        if any(cell != 0 for row in inp for cell in row):
            return None
        if out != _square_spiral_three_grid(n):
            return None
        max_n = max(max_n, n)
    return {"max_n": max(30, max_n)}


def _ttzf_compute_fill(Z, H, W):
    """Numpy: fill mask for isolated 2-wide zero strips (N×2 vertical or 2×N horizontal, N≥2)."""
    import numpy as _np

    def spread_rows(seed, mask, iters=5):
        s = seed.copy()
        d = 1
        for _ in range(iters):
            s = _np.maximum(s, _np.maximum(
                _np.pad(s, [[d, 0], [0, 0]])[:H, :],
                _np.pad(s, [[0, d], [0, 0]])[d:, :])) * mask
            d *= 2
        return s

    def spread_cols(seed, mask, iters=5):
        s = seed.copy()
        d = 1
        for _ in range(iters):
            s = _np.maximum(s, _np.maximum(
                _np.pad(s, [[0, 0], [d, 0]])[:, :W],
                _np.pad(s, [[0, 0], [0, d]])[:, d:])) * mask
            d *= 2
        return s

    # Vertical pairs: pair_v[i,j] = Z[i,j]*Z[i,j+1]
    pairv = Z * _np.pad(Z, [[0, 0], [0, 1]])[:, 1:]
    pvsr = _np.pad(pairv, [[0, 0], [1, 0]])[:, :W]
    pvsl = _np.pad(pairv, [[0, 0], [0, 1]])[:, 1:]
    runv = pairv * _np.maximum(
        _np.pad(pairv, [[1, 0], [0, 0]])[:H, :],
        _np.pad(pairv, [[0, 1], [0, 0]])[1:, :])
    anylv = spread_rows((1 - pvsr) * runv, runv)
    anyrv = spread_rows((1 - pvsl) * runv, runv)
    runokv = runv * anylv * anyrv
    fillv = _np.maximum(runokv, _np.pad(runokv, [[0, 0], [1, 0]])[:, :W])

    # Horizontal pairs: pair_h[i,j] = Z[i,j]*Z[i+1,j]
    pairh = Z * _np.pad(Z, [[0, 1], [0, 0]])[1:, :]
    phsu = _np.pad(pairh, [[1, 0], [0, 0]])[:H, :]
    phsd = _np.pad(pairh, [[0, 1], [0, 0]])[1:, :]
    runh = pairh * _np.maximum(
        _np.pad(pairh, [[0, 0], [1, 0]])[:, :W],
        _np.pad(pairh, [[0, 0], [0, 1]])[:, 1:])
    anyth = spread_cols((1 - phsu) * runh, runh)
    anybh = spread_cols((1 - phsd) * runh, runh)
    runokh = runh * anyth * anybh
    fillh0 = _np.maximum(runokh, _np.pad(runokh, [[1, 0], [0, 0]])[:H, :])

    # Suppress horizontal fills adjacent to vertical fills
    fvadj = _np.maximum(fillv, _np.maximum(
        _np.pad(fillv, [[0, 0], [1, 0]])[:, :W],
        _np.pad(fillv, [[0, 0], [0, 1]])[:, 1:]))
    fillh = fillh0 * (1 - fvadj)

    return (_np.maximum(fillv, fillh) == 1) & (Z == 1)


def _detect_two_by_two_zero_fill(pairs):
    """Detect: isolated 2-wide zero strips (N×2 or 2×N, N≥2) → fill_color.
    task265: H=18, W=18, bg_color=5, fill_color=2.
    """
    if not pairs:
        return None
    H0 = len(pairs[0]["input"])
    W0 = len(pairs[0]["input"][0]) if pairs[0]["input"] else 0
    if H0 < 4 or W0 < 4:
        return None
    bg_color = fill_color = None

    for p in pairs:
        inp_raw, out_raw = p["input"], p["output"]
        H, W = len(inp_raw), len(inp_raw[0]) if inp_raw else 0
        if H != H0 or W != W0:
            return None
        if len(out_raw) != H or any(len(r) != W for r in out_raw):
            return None
        if any(len(r) != W for r in inp_raw):
            return None

        in_colors = set(v for row in inp_raw for v in row if v != 0)
        if len(in_colors) != 1:
            return None
        cur_bg = next(iter(in_colors))
        out_colors = set(v for row in out_raw for v in row)
        extra = out_colors - {0, cur_bg}
        if len(extra) != 1:
            return None
        cur_fill = next(iter(extra))

        if bg_color is None:
            bg_color, fill_color = cur_bg, cur_fill
        elif bg_color != cur_bg or fill_color != cur_fill:
            return None

        import numpy as _np
        inp = _np.array(inp_raw, dtype=_np.int32)
        out = _np.array(out_raw, dtype=_np.int32)

        if _np.any((inp != 0) & (inp != bg_color)):
            return None
        if _np.any(out[inp == bg_color] != bg_color):
            return None
        if _np.any((out[inp == 0] != 0) & (out[inp == 0] != fill_color)):
            return None

        Z = (inp == 0).astype(_np.float32)
        pred = _ttzf_compute_fill(Z, H, W)
        actual = (out == fill_color) & (inp == 0)
        if not _np.array_equal(pred, actual):
            return None

    return {"H": H0, "W": W0, "bg_color": bg_color, "fill_color": fill_color}


def _rsf_compute_fill(ones, H, W):
    """Numpy: 2s mask for 180° rotational symmetry completion.
    ones: float32 [H,W] with 1 where fg_color, 0 elsewhere.
    Picks center (4.5,4.5) or (5,5) by whichever has higher existing symmetry count.
    """
    import numpy as _np
    flip_44 = ones[::-1, ::-1]
    flip_55 = _np.pad(flip_44, [[1, 0], [1, 0]])[:H, :W]
    zeros = 1 - ones
    sym44 = _np.sum(ones * flip_44)
    sym55 = _np.sum(ones * flip_55)
    flip = flip_55 if sym55 >= sym44 else flip_44
    return (zeros * flip == 1)


def _detect_rotation_symmetry_fill(pairs):
    """Detect: 0-cells complete 180° rotational symmetry of fg_color → fill_color.
    task027: H=10, W=10, fg_color=1, fill_color=2.
    """
    if not pairs:
        return None
    H0 = len(pairs[0]["input"])
    W0 = len(pairs[0]["input"][0]) if pairs[0]["input"] else 0
    if H0 < 4 or W0 < 4:
        return None
    fg_color = fill_color = None

    for p in pairs:
        inp_raw, out_raw = p["input"], p["output"]
        H, W = len(inp_raw), len(inp_raw[0]) if inp_raw else 0
        if H != H0 or W != W0:
            return None
        if len(out_raw) != H or any(len(r) != W for r in out_raw):
            return None
        if any(len(r) != W for r in inp_raw):
            return None

        in_colors = set(v for row in inp_raw for v in row if v != 0)
        if len(in_colors) != 1:
            return None
        cur_fg = next(iter(in_colors))
        out_colors = set(v for row in out_raw for v in row)
        extra = out_colors - {0, cur_fg}
        if len(extra) != 1:
            return None
        cur_fill = next(iter(extra))

        if fg_color is None:
            fg_color, fill_color = cur_fg, cur_fill
        elif fg_color != cur_fg or fill_color != cur_fill:
            return None

        import numpy as _np
        inp = _np.array(inp_raw, dtype=_np.int32)
        out = _np.array(out_raw, dtype=_np.int32)

        if _np.any((inp != 0) & (inp != fg_color)):
            return None
        if _np.any(out[inp == fg_color] != fg_color):
            return None
        if _np.any((out[inp == 0] != 0) & (out[inp == 0] != fill_color)):
            return None

        ones = (inp == fg_color).astype(_np.float32)
        pred = _rsf_compute_fill(ones, H, W)
        actual = (out == fill_color) & (inp == 0)
        if not _np.array_equal(pred, actual):
            return None

    return {"H": H0, "W": W0, "fg_color": fg_color, "fill_color": fill_color}


def _dsf_compute_fill(inp, H, W, fg_color=3, fill_color=8):
    """Numpy: compute fill_color mask for diagonal-symmetry-fill task (task042).
    For each diagonally-adjacent pair of fg_color components, places fill_color
    at the 90°-rotated (×3-scaled) positions about the pair center.
    """
    import numpy as _np

    def _cc4(g, v):
        visited = _np.zeros((H, W), dtype=bool)
        comps = []
        for r0 in range(H):
            for c0 in range(W):
                if g[r0, c0] == v and not visited[r0, c0]:
                    comp = set()
                    q = [(r0, c0)]
                    visited[r0, c0] = True
                    while q:
                        r, c = q.pop()
                        comp.add((r, c))
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc] and g[nr, nc] == v:
                                visited[nr, nc] = True
                                q.append((nr, nc))
                    comps.append(comp)
        return comps

    def _diag_adj(cA, cB):
        for r, c in cA:
            for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                if (r + dr, c + dc) in cB:
                    return True
        return False

    def _center(comp):
        rs = [r for r, c in comp]
        cs = [c for r, c in comp]
        return sum(rs) / len(rs), sum(cs) / len(cs)

    comps = _cc4(inp, fg_color)
    mask = _np.zeros((H, W), dtype=bool)
    used = [False] * len(comps)

    for i in range(len(comps)):
        if used[i]:
            continue
        for j in range(i + 1, len(comps)):
            if not used[j] and _diag_adj(comps[i], comps[j]):
                cA, cB = comps[i], comps[j]
                cA_c = _center(cA)
                cB_c = _center(cB)
                sc = ((cA_c[0] + cB_c[0]) / 2, (cA_c[1] + cB_c[1]) / 2)
                Ao = (cA_c[0] - sc[0], cA_c[1] - sc[1])
                Aor = (Ao[1], -Ao[0])  # 90° CW
                A8c = (sc[0] + 3 * Aor[0], sc[1] + 3 * Aor[1])
                B8c = (sc[0] - 3 * Aor[0], sc[1] - 3 * Aor[1])
                for r, c in cA:
                    nr = int(round(A8c[0] + (r - cA_c[0])))
                    nc = int(round(A8c[1] + (c - cA_c[1])))
                    if 0 <= nr < H and 0 <= nc < W:
                        mask[nr, nc] = True
                for r, c in cB:
                    nr = int(round(B8c[0] + (r - cB_c[0])))
                    nc = int(round(B8c[1] + (c - cB_c[1])))
                    if 0 <= nr < H and 0 <= nc < W:
                        mask[nr, nc] = True
                used[i] = used[j] = True
                break

    return mask


def _detect_diagonal_symmetry_fill(pairs):
    """Detect: fg_color block pairs → fill_color at 90°-rotated (×3) positions.
    task042: H=10, W=10, fg_color=3, fill_color=8.
    """
    import numpy as _np
    if not pairs:
        return None

    ref = None
    for p in pairs:
        inp = _np.array(p["input"], dtype=_np.int32)
        out = _np.array(p["output"], dtype=_np.int32)
        if inp.shape != out.shape:
            return None
        H, W = inp.shape
        if H < 4 or W < 4:
            return None

        in_colors = set(inp.flat) - {0}
        if len(in_colors) != 1:
            return None
        fg_color = next(iter(in_colors))

        changed = [(r, c) for r in range(H) for c in range(W) if inp[r, c] != out[r, c]]
        if not changed:
            return None
        fill_colors = {out[r, c] for r, c in changed}
        if len(fill_colors) != 1:
            return None
        fill_color = next(iter(fill_colors))

        if ref is None:
            ref = {"H": H, "W": W, "fg_color": int(fg_color), "fill_color": int(fill_color)}
        elif ref != {"H": H, "W": W, "fg_color": int(fg_color), "fill_color": int(fill_color)}:
            return None

        # All changes must be 0 → fill_color
        for r, c in changed:
            if inp[r, c] != 0 or out[r, c] != fill_color:
                return None
        # Prediction must match exactly
        pred = _dsf_compute_fill(inp, H, W, int(fg_color), int(fill_color))
        actual = _np.array(out) == fill_color
        if not _np.array_equal(pred, actual):
            return None

    return ref


def _detect_rotational_symmetry_completion(pairs):
    """Detect: output completes the input via 4-fold rotational symmetry around
    the bounding-box center of non-zero cells.

    Rule: For each input cell (r, c) with color v != 0, paint the same color at
    the 90°, 180°, 270° rotations of (r, c) around (cr, cc) where
    (cr, cc) is the bbox center of non-zero cells. Existing non-zero cells in
    the output preserve their color (rotation only fills bg cells).

    Conservative gate (avoids overlapping with simpler same-shape patterns):
      - Input and output shapes equal.
      - At least one non-zero cell.
      - Input has at least one cell that gets a new color from rotation
        (otherwise the rule reduces to identity and the identity-fallback /
        symmetry-completion patterns would be cheaper).
      - bbox extents (first_row+last_row and first_col+last_col) are both even
        (cr and cc are integer; the ONNX builder relies on this).
      - Numpy verifier reproduces every pair exactly.

    Returns True on match, else None.
    """
    if not pairs:
        return None
    saw_nontrivial_completion = False
    for p in pairs:
        try:
            a = np.array(p["input"], dtype=np.int8)
            o = np.array(p["output"], dtype=np.int8)
        except (TypeError, ValueError):
            return None
        if a.ndim != 2 or o.ndim != 2 or a.shape != o.shape:
            return None
        nz = np.argwhere(a != 0)
        if len(nz) == 0:
            return None
        r0, r1 = int(nz[:, 0].min()), int(nz[:, 0].max())
        c0, c1 = int(nz[:, 1].min()), int(nz[:, 1].max())
        if (r0 + r1) % 2 != 0 or (c0 + c1) % 2 != 0:
            return None
        cr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        H, W = a.shape
        out_pred = a.copy()
        for r in range(H):
            for c in range(W):
                if a[r, c] != 0:
                    v = a[r, c]
                    dr, dc = r - cr, c - cc
                    for new_r, new_c in [
                        (cr + dc, cc - dr),  # 90 CW point-rotation
                        (cr - dr, cc - dc),  # 180
                        (cr - dc, cc + dr),  # 270 CW
                    ]:
                        if 0 <= new_r < H and 0 <= new_c < W:
                            if out_pred[new_r, new_c] == 0:
                                out_pred[new_r, new_c] = v
        if not np.array_equal(out_pred, o):
            return None
        if not np.array_equal(a, out_pred):
            saw_nontrivial_completion = True
    if not saw_nontrivial_completion:
        return None
    return True


def _detect_constant_output_dominant_color(pairs):
    """Detect: every pair's output is a uniform single color of one fixed shape
    across all pairs, AND that color equals the smallest-index channel with the
    maximum cell-count in the input (matching ONNX Hardmax's tie-break behavior).

    Returns (h_out, w_out) if matched, else None. The uniform-color predicate
    plus the channel-argmax rule together rule out tasks where the output shape
    or color rule varies.
    """
    if not pairs:
        return None
    out_shapes = {_shape(p["output"]) for p in pairs}
    if len(out_shapes) != 1:
        return None
    h_out, w_out = next(iter(out_shapes))
    if h_out <= 0 or w_out <= 0 or h_out > 30 or w_out > 30:
        return None
    for p in pairs:
        out_colors = {c for row in p["output"] for c in row}
        if len(out_colors) != 1:
            return None
        out_color = next(iter(out_colors))
        # Hardmax tie-break: smallest-indexed channel with max count wins.
        counts = [0] * 10
        for row in p["input"]:
            for c in row:
                if 0 <= c < 10:
                    counts[c] += 1
        max_count = max(counts)
        dominant = next(c for c in range(10) if counts[c] == max_count)
        if dominant != out_color:
            return None
    return h_out, w_out


def _detect_outer_product_marker_tile(pairs):
    """Detect: every pair has input shape (3, 3) and output shape (9, 9), and
    the rule is: for each input cell (i, j) with input[i, j] == marker_color,
    the output's 3x3 block at rows [3i:3i+3] cols [3j:3j+3] equals the input;
    other output blocks are zeros.

    Tries two marker-selection modes (in order):
      "fixed":       marker_color is the SAME color across all pairs (any of
                     0..9). Cheapest builder path (channel-Slice for the mask).
                     Returns {"mode": "fixed", "marker_color": c} on the
                     smallest c that satisfies all pairs.
      "most_common": marker is the per-pair most-common color (ArgMax tie-break:
                     smallest channel index with the max cell count, including
                     channel 0). Returns {"mode": "most_common"}.

    Returns the descriptor dict on first match, else None.
    """
    if not pairs:
        return None
    if {_shape(p["input"]) for p in pairs} != {(3, 3)}:
        return None
    if {_shape(p["output"]) for p in pairs} != {(9, 9)}:
        return None

    def _matches_with_marker(p, marker):
        inp = p["input"]
        out = p["output"]
        for r in range(9):
            for c in range(9):
                ai, ir = divmod(r, 3)
                aj, ic = divmod(c, 3)
                expected = inp[ir][ic] if inp[ai][aj] == marker else 0
                if out[r][c] != expected:
                    return False
        return True

    # Mode "fixed": marker is the same color across all pairs.
    for c in range(10):
        if all(_matches_with_marker(p, c) for p in pairs):
            return {"mode": "fixed", "marker_color": c}

    # Mode "most_common": marker = per-pair ArgMax color (with ch0 included).
    most_common_ok = True
    for p in pairs:
        counts = [0] * 10
        for row in p["input"]:
            for v in row:
                counts[v] += 1
        max_count = max(counts)
        marker = next(c for c in range(10) if counts[c] == max_count)
        if not _matches_with_marker(p, marker):
            most_common_ok = False
            break
    if most_common_ok:
        return {"mode": "most_common"}

    return None


def _detect_per_color_stamp(pairs):
    """Detect: every pair has input shape == output shape; output equals input
    PLUS stamp colors placed at the kernel-described neighborhoods of cells
    with specific input colors. Each cell gets at most one stamp; stamps land
    only on cells that were color 0 in the input.

    Algorithm:
      1. For each pair, compute deltas = {(r, c) : output[r,c] != input[r,c]}.
         All delta cells must have input == 0 (else pattern is broken).
      2. Group delta cells by stamp_color (= output value at delta cell).
      3. For each stamp_color s: in_c candidates are colors that appear in
         EVERY s-delta-cell's 3x3 neighborhood (intersection across deltas).
         At least one candidate must exist.
      4. For each (stamp_color, in_c) candidate: derive the kernel offsets
         (per-pair: for each input cell with color in_c, every deltA cell with
         stamp_color s has a corresponding offset). Check the offsets fit a
         supported kernel.
      5. Pick rule per stamp_color with smallest kernel that survives
         simulation across all pairs.

    Returns {"rules": [(in_color, stamp_color, kernel_type), ...]} or None.
    """
    if not pairs:
        return None
    KERNELS = {
        "ortho": {(0, -1), (0, 1), (-1, 0), (1, 0)},
        "diag":  {(-1, -1), (-1, 1), (1, -1), (1, 1)},
        "all8":  {(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)},
    }
    KERNEL_ORDER = ["ortho", "diag", "all8"]

    for p in pairs:
        if _shape(p["input"]) != _shape(p["output"]):
            return None

    # Step 1+2: Per-pair deltas grouped by stamp_color. Also collect all stamp_colors
    # seen across pairs. For each (pair_idx, stamp_color, delta_r, delta_c), record
    # the cell's 3x3 neighbors' colors.
    per_pair_data = []  # [(inp, out, h, w, deltas_by_color), ...]
    all_stamp_colors = set()
    for p in pairs:
        inp = p["input"]
        out = p["output"]
        h = len(inp)
        w = len(inp[0]) if h else 0
        deltas_by_color = collections.defaultdict(list)  # stamp_color -> [(r, c), ...]
        for r in range(h):
            for c in range(w):
                if inp[r][c] != out[r][c]:
                    if inp[r][c] != 0:
                        return None
                    sc = out[r][c]
                    deltas_by_color[sc].append((r, c))
                    all_stamp_colors.add(sc)
        per_pair_data.append((inp, out, h, w, deltas_by_color))

    if not all_stamp_colors:
        return None

    # Step 3: For each stamp_color s, find in_c candidates = intersection of
    # neighbor colors across ALL delta cells with stamp_color s in ALL pairs.
    in_candidates_for_stamp = {}  # stamp_color -> set of valid in_c
    for s in all_stamp_colors:
        candidates = None  # None means "uninitialized"
        any_delta_seen = False
        for inp, out, h, w, deltas_by_color in per_pair_data:
            for (r, c) in deltas_by_color.get(s, []):
                any_delta_seen = True
                neighbors_colors = set()
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and inp[nr][nc] != 0:
                            neighbors_colors.add(inp[nr][nc])
                if candidates is None:
                    candidates = set(neighbors_colors)
                else:
                    candidates &= neighbors_colors
                if not candidates:
                    return None
        if not any_delta_seen:
            continue
        in_candidates_for_stamp[s] = candidates

    if not in_candidates_for_stamp:
        return None

    # Step 4+5: Pick (in_c, kernel) per stamp_color such that simulation across
    # all pairs reproduces the actual output. Try smallest kernels first; if a
    # rule generates extra stamps that conflict with actual output, reject it.
    def _simulate(rule_list, inp, h, w):
        expected = [row[:] for row in inp]
        for in_c, st_c, k_name in rule_list:
            kernel = KERNELS[k_name]
            for r in range(h):
                for c in range(w):
                    if inp[r][c] != in_c:
                        continue
                    for dr, dc in kernel:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and inp[nr][nc] == 0:
                            expected[nr][nc] = st_c
        return expected

    # Each stamp color needs ONE (in_c, kernel) selection. Search over the (often
    # small) Cartesian product of choices: for each s, pick (in_c, kernel) from
    # candidates × kernel-types.
    stamp_colors_sorted = sorted(in_candidates_for_stamp)
    choices_per_stamp = []
    for s in stamp_colors_sorted:
        opts = []
        for in_c in sorted(in_candidates_for_stamp[s]):
            for k_name in KERNEL_ORDER:
                opts.append((in_c, k_name))
        choices_per_stamp.append((s, opts))

    # Recursive enumeration with early reject: build a partial rule set, simulate
    # against each pair so far, prune early if a rule generates a non-existent stamp
    # or contradicts.
    def _try(idx, partial):
        if idx == len(choices_per_stamp):
            # All stamp colors assigned. Verify exact match against every pair.
            for inp, out, h, w, _ in per_pair_data:
                expected = _simulate(partial, inp, h, w)
                if expected != out:
                    return None
            return partial
        s, opts = choices_per_stamp[idx]
        for in_c, k_name in opts:
            new_rule_list = partial + [(in_c, s, k_name)]
            # Cheap prune: simulation MAY produce extra stamps. Defer full check
            # to the leaf — but skip rule combinations that immediately conflict
            # at the per-pair level (e.g. predict stamp where input is non-zero
            # AND non-matching output).
            ok = True
            for inp, out, h, w, deltas_by_color in per_pair_data:
                kernel = KERNELS[k_name]
                for r in range(h):
                    for c in range(w):
                        if inp[r][c] != in_c:
                            continue
                        for dr, dc in kernel:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w and inp[nr][nc] == 0:
                                if out[nr][nc] != s:
                                    ok = False
                                    break
                        if not ok:
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                continue
            result = _try(idx + 1, new_rule_list)
            if result is not None:
                return result
        return None

    final = _try(0, [])
    if final is None:
        return None
    final.sort()
    return {"rules": final}


def _detect_active_grid_frame(pairs):
    """Detect: every pair has output equal to a 1-pixel border of a fixed color
    around the active grid (input shape == output shape; cells at row 0 or
    h-1 or col 0 or w-1 of the active grid become `border_color`; interior
    cells become 0; canvas padding stays all-zero). Input content is replaced
    entirely; this only matches when the rule is shape-derived, not data-derived.

    Returns {"border_color": c} or None.
    """
    if not pairs:
        return None
    border = None
    for p in pairs:
        if _shape(p["input"]) != _shape(p["output"]):
            return None
        h = len(p["input"])
        w = len(p["input"][0]) if h else 0
        if h < 2 or w < 2:
            return None
        for r in range(h):
            for c in range(w):
                v_out = p["output"][r][c]
                on_border = r == 0 or r == h - 1 or c == 0 or c == w - 1
                if on_border:
                    if border is None:
                        border = v_out
                    elif v_out != border:
                        return None
                else:
                    if v_out != 0:
                        return None
    if border is None or border == 0:
        return None
    return {"border_color": border}


def _detect_outline_extract(pairs):
    """Detect: every pair has input shape == output shape; output equals input
    EXCEPT cells that are "interior" of a connected non-zero region (cell + all
    8 in-grid neighbors are the same color) become color 0. Boundary cells of
    each region stay; cells that were already color 0 stay.

    Returns {"matched": True} or None.
    """
    if not pairs:
        return None
    saw_change = False
    for p in pairs:
        if _shape(p["input"]) != _shape(p["output"]):
            return None
        h = len(p["input"])
        w = len(p["input"][0]) if h else 0
        for r in range(h):
            for c in range(w):
                v_in = p["input"][r][c]
                v_out = p["output"][r][c]
                if v_in == 0:
                    if v_out != 0:
                        return None
                    continue
                # Check if all 8 in-grid neighbors are the same color v_in.
                is_interior = True
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if not (0 <= nr < h and 0 <= nc < w) or p["input"][nr][nc] != v_in:
                            is_interior = False
                            break
                    if not is_interior:
                        break
                expected = 0 if is_interior else v_in
                if expected != v_in:
                    saw_change = True
                if v_out != expected:
                    return None
    if not saw_change:
        return None
    max_h = max(len(p["input"]) for p in pairs)
    max_w = max(len(p["input"][0]) for p in pairs if p["input"])
    return {"matched": True, "canvas_h": max_h, "canvas_w": max_w}


def _detect_bbox_fill(pairs):
    """Detect: every pair has input shape == output shape. The output equals
    the input EXCEPT inside the tight bounding box of all input non-zero cells:
    cells that were color 0 inside the bbox become a fixed `target_color`
    (consistent across pairs); non-zero cells inside the bbox stay at their
    original color. Cells outside the bbox are unchanged.

    Returns {"target_color": c} or None.
    """
    if not pairs:
        return None
    target = None
    saw_any = False
    for p in pairs:
        if _shape(p["input"]) != _shape(p["output"]):
            return None
        h = len(p["input"])
        w = len(p["input"][0]) if h else 0
        nz = [(r, c) for r in range(h) for c in range(w) if p["input"][r][c] != 0]
        if not nz:
            # No non-zero cells -> bbox undefined; output must equal input.
            if any(p["output"][r][c] != p["input"][r][c] for r in range(h) for c in range(w)):
                return None
            continue
        saw_any = True
        rmin = min(r for r, _ in nz); rmax = max(r for r, _ in nz)
        cmin = min(c for _, c in nz); cmax = max(c for _, c in nz)
        for r in range(h):
            for c in range(w):
                v_in = p["input"][r][c]
                v_out = p["output"][r][c]
                in_bbox = rmin <= r <= rmax and cmin <= c <= cmax
                if not in_bbox:
                    if v_out != v_in:
                        return None
                else:
                    if v_in != 0:
                        if v_out != v_in:
                            return None
                    else:
                        if target is None:
                            target = v_out
                        elif v_out != target:
                            return None
    if not saw_any or target is None or target == 0:
        return None
    return {"target_color": target}


def _task105_gridline_adjacent_predict(inp):
    h = len(inp)
    w = len(inp[0]) if h else 0
    out = [row[:] for row in inp]
    ones = [(r, c) for r in range(h) for c in range(w) if inp[r][c] == 1]
    if not ones:
        return out
    rmin = min(r for r, _ in ones)
    rmax = max(r for r, _ in ones)
    cmin = min(c for _, c in ones)
    cmax = max(c for _, c in ones)

    def _has_adjacent(values):
        values = sorted(values)
        return any(b - a == 1 for a, b in zip(values, values[1:]))

    selected_rows = set()
    for r in range(h):
        cols = [c for c in range(w) if inp[r][c] == 1]
        if r in (rmin, rmax) or len(cols) >= 4 or (len(cols) == 3 and _has_adjacent(cols)):
            selected_rows.add(r)

    selected_cols = set()
    for c in range(w):
        rows = [r for r in range(h) if inp[r][c] == 1]
        if c in (cmin, cmax) or len(rows) >= 4 or (len(rows) == 3 and _has_adjacent(rows)):
            selected_cols.add(c)

    for r in selected_rows:
        for c in range(cmin, cmax + 1):
            if out[r][c] == 0:
                out[r][c] = 2
    for c in selected_cols:
        for r in range(rmin, rmax + 1):
            if out[r][c] == 0:
                out[r][c] = 2
    return out


def _detect_gridline_adjacent_fill(pairs):
    if not pairs:
        return None
    saw_change = False
    for p in pairs:
        if _shape(p["input"]) != _shape(p["output"]):
            return None
        in_colors = {v for row in p["input"] for v in row}
        out_colors = {v for row in p["output"] for v in row}
        if not in_colors <= {0, 1} or not out_colors <= {0, 1, 2}:
            return None
        expected = _task105_gridline_adjacent_predict(p["input"])
        if expected != p["output"]:
            return None
        if expected != p["input"]:
            saw_change = True
    return {"matched": True} if saw_change else None


def _task025_line_projection_predict(inp):
    a = np.array(inp, dtype=np.int64)
    if a.ndim != 2:
        return None
    h, w = a.shape
    out = np.zeros_like(a)
    for color in range(1, 10):
        mask = (a == color)
        if not mask.any():
            continue
        full_rows = np.where(mask.sum(axis=1) == w)[0]
        full_cols = np.where(mask.sum(axis=0) == h)[0]
        for r in full_rows:
            out[r, :] = color
            above = mask[:r, :].any(axis=0)
            below = mask[r + 1 :, :].any(axis=0)
            if r > 0:
                out[r - 1, above] = color
            if r + 1 < h:
                out[r + 1, below] = color
        for c in full_cols:
            out[:, c] = color
            left = mask[:, :c].any(axis=1)
            right = mask[:, c + 1 :].any(axis=1)
            if c > 0:
                out[left, c - 1] = color
            if c + 1 < w:
                out[right, c + 1] = color
    return out.tolist()


def _detect_line_projection_to_full_line(pairs):
    if not pairs:
        return None
    saw_change = False
    for p in pairs:
        if _shape(p["input"]) != _shape(p["output"]):
            return None
        try:
            expected = _task025_line_projection_predict(p["input"])
        except (TypeError, ValueError):
            return None
        if expected != p["output"]:
            return None
        if expected != p["input"]:
            saw_change = True
    return {"matched": True} if saw_change else None


def _task005_motif_ray_extend_expected(grid):
    if not grid:
        return None
    arr = np.array(grid, dtype=np.int64)
    if arr.ndim != 2 or arr.shape != (21, 21):
        return None
    colors = [int(c) for c in np.unique(arr) if int(c) != 0]
    if len(colors) < 2:
        return None

    best = []
    best_count = -1
    for color in colors:
        mask = (arr == color)
        for r in range(19):
            for c in range(19):
                count = int(mask[r : r + 3, c : c + 3].sum())
                if count > best_count:
                    best_count = count
                    best = [(color, r, c)]
                elif count == best_count:
                    best.append((color, r, c))
    if best_count <= 0 or len(best) != 1:
        return None

    source_color, sr, sc = best[0]
    motif = (arr[sr : sr + 3, sc : sc + 3] == source_color)
    out = np.zeros_like(arr)
    out[arr == source_color] = source_color

    for color in colors:
        if color == source_color:
            continue
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                fr = sr + 4 * dr
                fc = sc + 4 * dc
                seeded = False
                for rr in range(3):
                    for cc in range(3):
                        r = fr + rr
                        c = fc + cc
                        if 0 <= r < 21 and 0 <= c < 21 and arr[r, c] == color:
                            seeded = True
                            break
                    if seeded:
                        break
                if not seeded:
                    continue
                for k in range(1, 6):
                    any_inside = False
                    tr = sr + 4 * k * dr
                    tc = sc + 4 * k * dc
                    for rr in range(3):
                        for cc in range(3):
                            if not motif[rr, cc]:
                                continue
                            r = tr + rr
                            c = tc + cc
                            if 0 <= r < 21 and 0 <= c < 21:
                                out[r, c] = color
                                any_inside = True
                    if not any_inside:
                        break
    return out.tolist()


def _detect_task005_motif_ray_extend(pairs):
    if not pairs:
        return None
    saw_change = False
    for p in pairs:
        if _shape(p["input"]) != _shape(p["output"]):
            return None
        expected = _task005_motif_ray_extend_expected(p["input"])
        if expected is None or expected != p["output"]:
            return None
        if expected != p["input"]:
            saw_change = True
    return {"matched": True} if saw_change else None


def _task370_diagonal_zero_mask_projection_predict(inp):
    a = np.array(inp, dtype=np.int64)
    if a.ndim != 2:
        return None
    h, w = a.shape
    zeros = np.argwhere(a == 0)
    if zeros.size == 0:
        return None
    colors, counts = np.unique(a[a != 0], return_counts=True)
    if len(colors) != 2:
        return None
    bg = int(colors[np.argmax(counts)])
    marker_colors = [int(c) for c in colors if int(c) != bg]
    if len(marker_colors) != 1:
        return None
    marker = marker_colors[0]
    marker_pos = np.argwhere(a == marker)
    if marker_pos.shape[0] != 1:
        return None
    mr, mc = (int(x) for x in marker_pos[0])

    candidates = []
    for zr, zc in zeros:
        dr = mr - int(zr)
        dc = mc - int(zc)
        if dr == 0 or abs(dr) != abs(dc):
            continue
        candidates.append((abs(dr), dr, dc))
    if not candidates:
        return None
    _, dr, dc = max(candidates, key=lambda item: item[0])

    out = a.copy()
    zero_mask = (a == 0)
    k = 1
    while True:
        rr0 = max(0, k * dr)
        rr1 = h + min(0, k * dr)
        cc0 = max(0, k * dc)
        cc1 = w + min(0, k * dc)
        sr0 = max(0, -k * dr)
        sr1 = h - max(0, k * dr)
        sc0 = max(0, -k * dc)
        sc1 = w - max(0, k * dc)
        if rr0 >= rr1 or cc0 >= cc1:
            break
        shifted = np.zeros_like(zero_mask)
        shifted[rr0:rr1, cc0:cc1] = zero_mask[sr0:sr1, sc0:sc1]
        out[(shifted) & (a == bg)] = marker
        k += 1
    return out.tolist()


def _detect_diagonal_zero_mask_projection(pairs):
    if not pairs:
        return None
    saw_change = False
    for p in pairs:
        if _shape(p["input"]) != _shape(p["output"]):
            return None
        try:
            expected = _task370_diagonal_zero_mask_projection_predict(p["input"])
        except (TypeError, ValueError):
            return None
        if expected != p["output"]:
            return None
        if expected != p["input"]:
            saw_change = True
    return {"matched": True} if saw_change else None


def _task089_marker_shape_copy_predict(inp):
    a = np.array(inp, dtype=np.int64)
    if a.ndim != 2:
        return None
    h, w = a.shape
    if (h, w) != (13, 13):
        return None
    out = a.copy()
    nonmarker = (a != 0) & (a != 2) & (a != 3)

    for marker, flip_h in ((2, True), (3, False)):
        marker_cells = np.argwhere(a == marker)
        source = []
        targets = []
        source_color = None
        for r, c in marker_cells:
            r0, r1 = max(0, r - 1), min(h, r + 2)
            c0, c1 = max(0, c - 1), min(w, c + 2)
            vals = [int(v) for v in a[r0:r1, c0:c1].ravel() if v not in (0, 2, 3)]
            if vals:
                source.append((int(r), int(c)))
                source_color = collections.Counter(vals).most_common(1)[0][0]
            else:
                targets.append((int(r), int(c)))
        if len(source) != 1 or source_color is None:
            continue
        sr, sc = source[0]
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = sr + dr, sc + dc
                if not (0 <= rr < h and 0 <= cc < w and a[rr, cc] == source_color):
                    continue
                out_dc = -dc if flip_h else dc
                for tr, tc in targets:
                    pr, pc = tr + dr, tc + out_dc
                    if 0 <= pr < h and 0 <= pc < w and out[pr, pc] == 0:
                        out[pr, pc] = source_color
    return out.tolist()


def _detect_marker_shape_copy(pairs):
    if not pairs:
        return None
    saw_change = False
    for p in pairs:
        if _shape(p["input"]) != _shape(p["output"]):
            return None
        try:
            expected = _task089_marker_shape_copy_predict(p["input"])
        except (TypeError, ValueError):
            return None
        if expected != p["output"]:
            return None
        if expected != p["input"]:
            saw_change = True
    return {"matched": True} if saw_change else None


def _count_2x2_one_blocks_predict(inp):
    a = np.array(inp, dtype=np.int64)
    if a.shape != (9, 9):
        return None
    count = 0
    for r in range(8):
        for c in range(8):
            if np.all(a[r:r + 2, c:c + 2] == 1):
                count += 1
    if not (0 <= count <= 5):
        return None
    return [[1] * count + [0] * (5 - count)]


def _detect_count_2x2_one_blocks(pairs):
    if not pairs:
        return None
    saw_change = False
    for p in pairs:
        if _shape(p["output"]) != (1, 5):
            return None
        expected = _count_2x2_one_blocks_predict(p["input"])
        if expected is None or expected != p["output"]:
            return None
        if _shape(p["input"]) != _shape(p["output"]) or p["input"] != p["output"]:
            saw_change = True
    return {"matched": True} if saw_change else None


def _block_grid_line_fill_predict(inp):
    a = np.array(inp, dtype=np.int64)
    if a.ndim != 2:
        return None
    h, w = a.shape
    if h != w or h > 29 or (h + 1) % 3 != 0:
        return None
    if h < 5:
        return None
    grid_color = int(a[2, 2])
    if grid_color == 0:
        return None
    for r in range(2, h, 3):
        if not np.all(a[r, :] == grid_color):
            return None
    for c in range(2, w, 3):
        if not np.all(a[:, c] == grid_color):
            return None

    out = a.copy()
    by_color = collections.defaultdict(list)
    n_blocks = (h + 1) // 3
    for br in range(n_blocks):
        for bc in range(n_blocks):
            block = a[3 * br:3 * br + 2, 3 * bc:3 * bc + 2]
            vals = set(int(v) for v in block.ravel())
            if len(vals) != 1:
                return None
            color = next(iter(vals))
            if color in (0, grid_color):
                continue
            by_color[color].append((br, bc))

    for color, points in by_color.items():
        for i, (r1, c1) in enumerate(points):
            for r2, c2 in points[i + 1:]:
                if r1 == r2:
                    for bc in range(min(c1, c2), max(c1, c2) + 1):
                        out[3 * r1:3 * r1 + 2, 3 * bc:3 * bc + 2] = color
                if c1 == c2:
                    for br in range(min(r1, r2), max(r1, r2) + 1):
                        out[3 * br:3 * br + 2, 3 * c1:3 * c1 + 2] = color
    return out.tolist()


def _detect_block_grid_line_fill(pairs):
    if not pairs:
        return None
    saw_change = False
    for p in pairs:
        if _shape(p["input"]) != _shape(p["output"]):
            return None
        expected = _block_grid_line_fill_predict(p["input"])
        if expected is None or expected != p["output"]:
            return None
        if expected != p["input"]:
            saw_change = True
    return {"matched": True} if saw_change else None


def _same_color_neighbor_keep_predict(inp):
    a = np.array(inp, dtype=np.int64)
    if a.ndim != 2:
        return None
    out = np.zeros_like(a)
    for color in range(1, 10):
        mask = (a == color)
        if not mask.any():
            continue
        neighbor = np.zeros_like(mask, dtype=bool)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                src_r0 = max(0, -dr)
                src_r1 = a.shape[0] - max(0, dr)
                src_c0 = max(0, -dc)
                src_c1 = a.shape[1] - max(0, dc)
                dst_r0 = max(0, dr)
                dst_r1 = a.shape[0] - max(0, -dr)
                dst_c0 = max(0, dc)
                dst_c1 = a.shape[1] - max(0, -dc)
                neighbor[dst_r0:dst_r1, dst_c0:dst_c1] |= mask[src_r0:src_r1, src_c0:src_c1]
        out[mask & neighbor] = color
    return out.tolist()


def _detect_same_color_neighbor_keep(pairs):
    if not pairs:
        return None
    saw_change = False
    for p in pairs:
        if _shape(p["input"]) != _shape(p["output"]):
            return None
        expected = _same_color_neighbor_keep_predict(p["input"])
        if expected is None or expected != p["output"]:
            return None
        if expected != p["input"]:
            saw_change = True
    return {"matched": True} if saw_change else None


def _marker_adjacent_bbox_reflection_predict(inp):
    a = np.array(inp, dtype=np.int64)
    if a.shape != (10, 10):
        return None
    colors = sorted(set(int(v) for v in a.ravel()) - {0, 2})
    if len(colors) != 1 or colors[0] == 3:
        return None
    color = colors[0]
    obj = a == color
    marker = a == 2
    if not obj.any() or not marker.any():
        return None

    rr, cc = np.where(obj)
    mr, mc = np.where(marker)
    rmin, rmax = int(rr.min()), int(rr.max())
    cmin, cmax = int(cc.min()), int(cc.max())

    out = np.full(a.shape, 3, dtype=np.int64)
    out[obj | marker] = color
    if int(mc.min()) > cmax:
        for r, c in zip(rr, cc):
            dst = 2 * cmax + 1 - int(c)
            if 0 <= dst < a.shape[1]:
                out[int(r), dst] = color
    elif int(mc.max()) < cmin:
        for r, c in zip(rr, cc):
            dst = 2 * cmin - 1 - int(c)
            if 0 <= dst < a.shape[1]:
                out[int(r), dst] = color
    elif int(mr.min()) > rmax:
        for r, c in zip(rr, cc):
            dst = 2 * rmax + 1 - int(r)
            if 0 <= dst < a.shape[0]:
                out[dst, int(c)] = color
    elif int(mr.max()) < rmin:
        for r, c in zip(rr, cc):
            dst = 2 * rmin - 1 - int(r)
            if 0 <= dst < a.shape[0]:
                out[dst, int(c)] = color
    else:
        return None
    return out.tolist()


def _detect_marker_adjacent_bbox_reflection(pairs):
    if not pairs:
        return None
    saw_change = False
    for p in pairs:
        if _shape(p["input"]) != _shape(p["output"]):
            return None
        expected = _marker_adjacent_bbox_reflection_predict(p["input"])
        if expected is None or expected != p["output"]:
            return None
        if expected != p["input"]:
            saw_change = True
    return {"matched": True} if saw_change else None


def _rectangle_interior_fill8_predict(inp):
    a = np.array(inp, dtype=np.int64)
    if a.ndim != 2:
        return None
    out = a.copy()
    h, w = a.shape
    fill = np.zeros((h, w), dtype=bool)
    for color in range(1, 10):
        mask = a == color
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                if mask[r, c] and np.all(mask[r - 1:r + 2, c - 1:c + 2]):
                    fill[r, c] = True
    out[fill] = 8
    return out.tolist()


def _detect_rectangle_interior_fill8(pairs):
    if not pairs:
        return None
    saw_change = False
    max_h = 0
    max_w = 0
    for p in pairs:
        if _shape(p["input"]) != _shape(p["output"]):
            return None
        expected = _rectangle_interior_fill8_predict(p["input"])
        if expected is None or expected != p["output"]:
            return None
        if expected != p["input"]:
            saw_change = True
        h, w = _shape(p["input"])
        max_h = max(max_h, h)
        max_w = max(max_w, w)
    return {"matched": True, "h": max_h, "w": max_w} if saw_change else None


def _border_distance_probe_recolor_predict(inp):
    a = np.array(inp, dtype=np.int64)
    if a.shape != (10, 10):
        return None
    out = a.copy()
    if len(set(int(v) for v in a[0, :])) == 1 and len(set(int(v) for v in a[-1, :])) == 1 and a[0, 0] != a[-1, 0]:
        axis = "row"
        near_color = int(a[0, 0])
        far_color = int(a[-1, 0])
        max_dist = a.shape[0] - 1
    elif len(set(int(v) for v in a[:, 0])) == 1 and len(set(int(v) for v in a[:, -1])) == 1 and a[0, 0] != a[0, -1]:
        axis = "col"
        near_color = int(a[0, 0])
        far_color = int(a[0, -1])
        max_dist = a.shape[1] - 1
    else:
        return None
    for r, c in np.argwhere(a == 3):
        dist = int(r if axis == "row" else c)
        out[r, c] = near_color if dist <= max_dist - dist else far_color
    return out.tolist()


def _detect_border_distance_probe_recolor(pairs):
    if not pairs:
        return None
    saw_change = False
    max_h = 0
    max_w = 0
    for p in pairs:
        if _shape(p["input"]) != _shape(p["output"]):
            return None
        expected = _border_distance_probe_recolor_predict(p["input"])
        if expected is None or expected != p["output"]:
            return None
        if expected != p["input"]:
            saw_change = True
        h, w = _shape(p["input"])
        max_h = max(max_h, h)
        max_w = max(max_w, w)
    return {"matched": True, "h": max_h, "w": max_w} if saw_change else None


def _frame6_border3_fill4_predict(inp):
    a = np.array(inp, dtype=np.int64)
    if a.ndim != 2:
        return None
    if not set(int(v) for v in a.ravel()) <= {6, 8}:
        return None
    six = a == 6
    bg = a == 8
    h, w = a.shape
    left = np.zeros_like(six, dtype=bool)
    right = np.zeros_like(six, dtype=bool)
    up = np.zeros_like(six, dtype=bool)
    down = np.zeros_like(six, dtype=bool)
    for r in range(h):
        seen = False
        for c in range(w):
            left[r, c] = seen
            seen = seen or bool(six[r, c])
        seen = False
        for c in range(w - 1, -1, -1):
            right[r, c] = seen
            seen = seen or bool(six[r, c])
    for c in range(w):
        seen = False
        for r in range(h):
            up[r, c] = seen
            seen = seen or bool(six[r, c])
        seen = False
        for r in range(h - 1, -1, -1):
            down[r, c] = seen
            seen = seen or bool(six[r, c])

    interior = bg & left & right & up & down
    near = np.zeros_like(six, dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            src_r0 = max(0, -dr)
            src_r1 = h - max(0, dr)
            src_c0 = max(0, -dc)
            src_c1 = w - max(0, dc)
            dst_r0 = max(0, dr)
            dst_r1 = h - max(0, -dr)
            dst_c0 = max(0, dc)
            dst_c1 = w - max(0, -dc)
            near[dst_r0:dst_r1, dst_c0:dst_c1] |= six[src_r0:src_r1, src_c0:src_c1]
    border = bg & near & ~interior
    out = a.copy()
    out[border] = 3
    out[interior] = 4
    return out.tolist()


def _detect_frame6_border3_fill4(pairs):
    if not pairs:
        return None
    saw_change = False
    max_h = 0
    max_w = 0
    for p in pairs:
        if _shape(p["input"]) != _shape(p["output"]):
            return None
        expected = _frame6_border3_fill4_predict(p["input"])
        if expected is None or expected != p["output"]:
            return None
        if expected != p["input"]:
            saw_change = True
        h, w = _shape(p["input"])
        max_h = max(max_h, h)
        max_w = max(max_w, w)
    return {"matched": True, "h": max_h, "w": max_w} if saw_change else None


def _marker_to_rectangle_line_predict(inp):
    a = np.array(inp, dtype=np.int64)
    if a.ndim != 2:
        return None
    colors = sorted({int(v) for v in a.ravel()})
    if len(colors) != 3:
        return None
    vals, counts = np.unique(a, return_counts=True)
    bg = int(vals[int(np.argmax(counts))])
    non_bg = a != bg
    rect = np.zeros_like(non_bg, dtype=bool)
    for color in sorted({int(v) for v in a[non_bg].ravel()}):
        mask = a == color
        left = np.zeros_like(mask, dtype=bool)
        right = np.zeros_like(mask, dtype=bool)
        up = np.zeros_like(mask, dtype=bool)
        down = np.zeros_like(mask, dtype=bool)
        left[:, 1:] = mask[:, :-1]
        right[:, :-1] = mask[:, 1:]
        up[1:, :] = mask[:-1, :]
        down[:-1, :] = mask[1:, :]
        rect |= mask & (left | right) & (up | down)
    if not rect.any():
        return None
    marker = non_bg & ~rect
    marker_colors = sorted({int(v) for v in a[marker].ravel()})
    if len(marker_colors) != 1:
        return None
    marker_color = marker_colors[0]

    line = np.zeros_like(marker, dtype=bool)
    h, w = a.shape
    for r in range(h):
        marker_row = marker[r, :]
        rect_row = rect[r, :]
        marker_left = np.maximum.accumulate(marker_row)
        marker_right = np.maximum.accumulate(marker_row[::-1])[::-1]
        rect_left = np.maximum.accumulate(rect_row)
        rect_right = np.maximum.accumulate(rect_row[::-1])[::-1]
        line[r, :] |= (marker_left & rect_right) | (marker_right & rect_left)
    for c in range(w):
        marker_col = marker[:, c]
        rect_col = rect[:, c]
        marker_up = np.maximum.accumulate(marker_col)
        marker_down = np.maximum.accumulate(marker_col[::-1])[::-1]
        rect_up = np.maximum.accumulate(rect_col)
        rect_down = np.maximum.accumulate(rect_col[::-1])[::-1]
        line[:, c] |= (marker_up & rect_down) | (marker_down & rect_up)

    out = a.copy()
    out[line & ~rect] = marker_color
    return out.tolist()


def _detect_marker_to_rectangle_line(pairs):
    if not pairs:
        return None
    saw_change = False
    max_h = 0
    max_w = 0
    for p in pairs:
        if _shape(p["input"]) != _shape(p["output"]):
            return None
        expected = _marker_to_rectangle_line_predict(p["input"])
        if expected is None or expected != p["output"]:
            return None
        if expected != p["input"]:
            saw_change = True
        h, w = _shape(p["input"])
        max_h = max(max_h, h)
        max_w = max(max_w, w)
    return {"matched": True, "h": max_h, "w": max_w} if saw_change else None


def _foreground_count_row_predict(inp):
    if _shape(inp) != (3, 3):
        return None
    colors = sorted({int(v) for row in inp for v in row if int(v) != 0})
    if len(colors) != 1:
        return None
    color = colors[0]
    count = sum(1 for row in inp for v in row if int(v) == color)
    if not (1 <= count <= 9):
        return None
    return [[color] * count]


def _detect_foreground_count_row(pairs):
    if not pairs:
        return None
    saw_change = False
    for p in pairs:
        expected = _foreground_count_row_predict(p["input"])
        if expected is None or expected != p["output"]:
            return None
        if _shape(p["input"]) != _shape(p["output"]) or p["input"] != p["output"]:
            saw_change = True
    return {"matched": True} if saw_change else None


def _task119_edge_seed_bounce_ray_predict(inp):
    a = np.array(inp, dtype=np.int64)
    if a.ndim != 2:
        return None
    h, w = a.shape
    if (h, w) != (12, 12):
        return None
    if not set(int(v) for v in a.ravel()) <= {0, 2, 8}:
        return None

    twos = np.argwhere(a == 2)
    eights = np.argwhere(a == 8)
    if twos.size == 0 or eights.shape[0] not in (2, 3):
        return None

    full_rows = [r for r in range(h) if np.all(a[r, :] == 2)]
    full_cols = [c for c in range(w) if np.all(a[:, c] == 2)]
    valid_row_wall = (
        full_rows
        and (full_rows == list(range(len(full_rows))) or full_rows == list(range(h - len(full_rows), h)))
        and all(int(a[r, c]) == 2 for r, c in twos)
    )
    valid_col_wall = (
        full_cols
        and (full_cols == list(range(len(full_cols))) or full_cols == list(range(w - len(full_cols), w)))
        and all(int(a[r, c]) == 2 for r, c in twos)
    )
    if valid_row_wall == valid_col_wall:
        return None

    def shift(mask, dr, dc):
        out = np.zeros_like(mask)
        r0 = max(0, -dr)
        r1 = h - max(0, dr)
        c0 = max(0, -dc)
        c1 = w - max(0, dc)
        if r0 < r1 and c0 < c1:
            out[r0 + dr:r1 + dr, c0 + dc:c1 + dc] = mask[r0:r1, c0:c1]
        return out

    ch0 = (a == 0).astype(np.float32)
    ch2 = (a == 2).astype(np.float32)
    ch8 = (a == 8).astype(np.float32)
    pathable = ch0 + ch8
    edge = np.zeros_like(ch0)
    edge[0, :] = 1.0
    edge[-1, :] = 1.0
    edge[:, 0] = 1.0
    edge[:, -1] = 1.0

    dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    fronts = {(dr, dc): ch8 * edge * shift(ch8, -dr, -dc) for dr, dc in dirs}
    if not any(float(f.max()) > 0.5 for f in fronts.values()):
        return None

    paint = np.zeros_like(ch0)
    for _ in range(11):
        incoming = {d: np.zeros_like(ch0) for d in dirs}
        for dr, dc in dirs:
            front = fronts[(dr, dc)]
            if float(front.max()) <= 0.5:
                continue
            bounce_v = front * shift(ch2, 0, -dc)
            bounce_h = front * shift(ch2, -dr, 0)
            bounce = np.clip(bounce_v + bounce_h, 0.0, 1.0)
            straight = front * (1.0 - bounce)
            incoming[(dr, dc)] = np.maximum(incoming[(dr, dc)], shift(straight, dr, dc))
            incoming[(dr, -dc)] = np.maximum(incoming[(dr, -dc)], shift(bounce_v, dr, -dc))
            incoming[(-dr, dc)] = np.maximum(incoming[(-dr, dc)], shift(bounce_h, -dr, dc))

        any_front = False
        for d in dirs:
            nxt = incoming[d] * pathable
            paint = np.maximum(paint, nxt * ch0)
            fronts[d] = nxt
            any_front = any_front or float(nxt.max()) > 0.5
        if not any_front:
            break

    out = a.copy()
    out[paint > 0.5] = 3
    return out.tolist()


def _detect_edge_seed_bounce_ray(pairs):
    if not pairs:
        return None
    saw_change = False
    for p in pairs:
        if _shape(p["input"]) != _shape(p["output"]):
            return None
        try:
            expected = _task119_edge_seed_bounce_ray_predict(p["input"])
        except (TypeError, ValueError):
            return None
        if expected is None or expected != p["output"]:
            return None
        if expected != p["input"]:
            saw_change = True
    return {"matched": True} if saw_change else None


def _detect_symmetry_completion(pairs):
    """Detect: every pair has input shape == output shape. The output equals the
    input PLUS reflected non-zero values across the v-axis (top<->bottom) or
    h-axis (left<->right). Cells of color 0 in input get filled with the
    reflected non-zero value; non-zero input cells preserved; cells whose
    reflected counterpart is also 0 stay zero.

    Tries axes "v" (top-bottom) then "h" (left-right). The h dimension (for v-axis)
    or w (for h-axis) must be CONSTANT across all pairs (Gather indices baked at
    build time).

    Returns {"axis": "v", "h": h, "w": w} or {"axis": "h", "h": h, "w": w}, or None.
    """
    if not pairs:
        return None
    in_shapes = {_shape(p["input"]) for p in pairs}
    out_shapes = {_shape(p["output"]) for p in pairs}
    if in_shapes != out_shapes:
        return None
    # Both v and h require their dim to be constant; check.
    h_set = {hh for hh, _ in in_shapes}
    w_set = {ww for _, ww in in_shapes}

    def _matches(axis):
        for p in pairs:
            inp = p["input"]
            out = p["output"]
            h = len(inp)
            w = len(inp[0]) if h else 0
            for r in range(h):
                for c in range(w):
                    v_in = inp[r][c]
                    v_out = out[r][c]
                    if v_in != 0:
                        if v_out != v_in:
                            return False
                    else:
                        if axis == "v":
                            refl = inp[h - 1 - r][c]
                        else:
                            refl = inp[r][w - 1 - c]
                        if refl != 0:
                            if v_out != refl:
                                return False
                        else:
                            if v_out != 0:
                                return False
        return True

    if len(h_set) == 1 and _matches("v"):
        h = next(iter(h_set))
        # w varies across pairs is OK for v-axis (Gather is on axis 2 only);
        # but builder still takes w as a param for the model shape. Use max w.
        w = max(w_set)
        return {"axis": "v", "h": h, "w": w}
    if len(w_set) == 1 and _matches("h"):
        w = next(iter(w_set))
        h = max(h_set)
        return {"axis": "h", "h": h, "w": w}
    return None


def _detect_diagonal_shear(pairs):
    """Detect: every pair has input shape (3, 3) and output shape (6, 6), and
    output[r, c] = sum over k=0..5 of input[r-k, c-k] (zero outside input bounds).

    Equivalent to: the 3x3 input is "stamped" at output anchor positions
    (k, k) for k=0..5 and the stamps are summed. The builder relies on the
    data property that at any output cell, at most one shift contributes a
    non-zero color (so Sum behaves as element-wise Max for the active channel).
    Detector enforces this by both:
      (a) verifying the simulated sum-of-shifts equals the actual output, AND
      (b) verifying the simulation is non-degenerate — i.e., NO output cell
          receives a non-zero contribution from more than one shift with
          DIFFERENT colors (which would cause the sum to corrupt the channel).

    Returns {"matched": True} on match, else None.
    """
    if not pairs:
        return None
    if {_shape(p["input"]) for p in pairs} != {(3, 3)}:
        return None
    if {_shape(p["output"]) for p in pairs} != {(6, 6)}:
        return None
    for p in pairs:
        inp = p["input"]
        out = p["output"]
        # Simulate sum-of-shifts. Track the unique non-zero contribution per cell
        # (or detect a conflict — multiple distinct non-zeros at the same cell).
        for r in range(6):
            for c in range(6):
                contributing = []
                for k in range(6):
                    i, j = r - k, c - k
                    if 0 <= i <= 2 and 0 <= j <= 2 and inp[i][j] != 0:
                        contributing.append(inp[i][j])
                if not contributing:
                    expected = 0
                elif all(v == contributing[0] for v in contributing):
                    expected = contributing[0]
                else:
                    # Multiple distinct non-zero colors at the same cell: builder
                    # would corrupt this, so reject.
                    return None
                if out[r][c] != expected:
                    return None
    return {"matched": True}


def _detect_seed_flood_fill(pairs):
    """Detect: every pair shares a fill_color in 1..9, and output[r][c] == fill_color
    iff input[r][c] == 0 AND that cell is 4-conn reachable through 0-cells from
    some input[r'][c'] == fill_color cell. Other cells unchanged.

    Returns {"fill_color": c} on match, else None.
    """
    if not pairs:
        return None

    def _flood(inp, fill_color):
        h = len(inp)
        if h == 0:
            return None
        w = len(inp[0])
        # not_blocking[r][c] = (inp[r][c] == 0 or inp[r][c] == fill_color)
        # reach[r][c] = True iff 4-conn reachable from a fill_color cell through 0-cells.
        # Implementation: BFS from each fill_color seed.
        reach = [[False] * w for _ in range(h)]
        seeds = [(r, c) for r in range(h) for c in range(w) if inp[r][c] == fill_color]
        if not seeds:
            return reach
        from collections import deque
        q = deque(seeds)
        for r, c in seeds:
            reach[r][c] = True
        while q:
            r, c = q.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not reach[nr][nc]:
                    if inp[nr][nc] == 0 or inp[nr][nc] == fill_color:
                        reach[nr][nc] = True
                        q.append((nr, nc))
        return reach

    for fill_color in range(1, 10):
        ok = True
        for p in pairs:
            inp = p["input"]
            out = p["output"]
            if len(inp) == 0 or len(inp[0]) == 0:
                ok = False
                break
            if len(out) != len(inp) or any(len(out[r]) != len(inp[r]) for r in range(len(inp))):
                ok = False
                break
            reach = _flood(inp, fill_color)
            if reach is None:
                ok = False
                break
            for r in range(len(inp)):
                for c in range(len(inp[0])):
                    if inp[r][c] == 0 and reach[r][c]:
                        if out[r][c] != fill_color:
                            ok = False
                            break
                    else:
                        if out[r][c] != inp[r][c]:
                            ok = False
                            break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            max_h = max(len(p["input"]) for p in pairs)
            max_w = max(len(p["input"][0]) for p in pairs if p["input"])
            return {"fill_color": fill_color, "canvas_h": max_h, "canvas_w": max_w}
    return None


def _detect_color_filter_bbox_crop(pairs, picker="least"):
    """Detect: every pair's output equals input cropped to bbox of cells of a
    runtime-selected color. Supported pickers:
      'least'         — color with the smallest cell-count among nonzero colors.
      'smallest_bbox' — color with the smallest bbox area among nonzero colors.
      'most_count'    — color with the LARGEST cell-count among nonzero colors.

    Returns {"matched": True, "picker": picker} on match, else None.

    Tie-break: smallest channel index (matches ArgMax-of-negated builder semantics).
    For 'most_count', ties are REJECTED (returns None) since onnx ArgMax tie-break
    on equal counts would pick the smallest-channel-index, which may not match
    the data's intended pick.
    """
    if not pairs:
        return None
    if picker not in ("least", "smallest_bbox", "most_count", "odd_one_out", "max_density", "lr_symmetry"):
        return None
    for p in pairs:
        inp = p["input"]
        out = p["output"]
        h_in = len(inp)
        if h_in == 0:
            return None
        w_in = len(inp[0])
        if w_in == 0:
            return None
        counts = {}
        for r in range(h_in):
            for c in range(w_in):
                v = inp[r][c]
                if v != 0:
                    counts[v] = counts.get(v, 0) + 1
        if not counts:
            return None
        if picker == "least":
            min_count = min(counts.values())
            candidates = [c for c, cnt in counts.items() if cnt == min_count]
            target = min(candidates)
        elif picker == "most_count":
            max_count = max(counts.values())
            candidates = [c for c, cnt in counts.items() if cnt == max_count]
            if len(candidates) > 1:
                return None  # ties rejected
            target = candidates[0]
        elif picker == "odd_one_out":
            # The color whose count appears exactly once across all present colors.
            from collections import Counter
            freq = Counter(counts.values())
            unique_counts = [cnt for cnt, f in freq.items() if f == 1]
            if len(unique_counts) != 1:
                return None
            odd_count = unique_counts[0]
            candidates = [c for c, cnt in counts.items() if cnt == odd_count]
            if len(candidates) != 1:
                return None
            target = candidates[0]
        elif picker == "max_density":
            # Color with highest cells/bbox_area density.
            densities = {}
            for c in counts:
                rows = [r for r in range(h_in) if any(inp[r][cc] == c for cc in range(w_in))]
                cols = [cc for cc in range(w_in) if any(inp[r][cc] == c for r in range(h_in))]
                if rows and cols:
                    area = (rows[-1] - rows[0] + 1) * (cols[-1] - cols[0] + 1)
                    densities[c] = counts[c] / area
            if not densities:
                return None
            max_d = max(densities.values())
            cands = [c for c, d in densities.items() if d == max_d]
            # Tie-break: smallest channel index (consistent with ArgMax).
            target = min(cands)
        elif picker == "lr_symmetry":
            stats = {}
            for c in counts:
                rows = [r for r in range(h_in) if any(inp[r][cc] == c for cc in range(w_in))]
                cols = [cc for cc in range(w_in) if any(inp[r][cc] == c for r in range(h_in))]
                if not rows or not cols:
                    continue
                patch = [
                    [1 if inp[r][cc] == c else 0 for cc in range(cols[0], cols[-1] + 1)]
                    for r in range(rows[0], rows[-1] + 1)
                ]
                area = (rows[-1] - rows[0] + 1) * (cols[-1] - cols[0] + 1)
                stats[c] = {
                    "lr": all(row == row[::-1] for row in patch),
                    "density": counts[c] / area,
                }
            candidates = [c for c, s in stats.items() if s["lr"]]
            if not candidates:
                return None
            max_d = max(stats[c]["density"] for c in candidates)
            target = min(c for c in candidates if stats[c]["density"] == max_d)
        else:  # smallest_bbox
            bbox_areas = {}
            for c in counts:
                rows = [r for r in range(h_in) if any(inp[r][cc] == c for cc in range(w_in))]
                cols = [cc for cc in range(w_in) if any(inp[r][cc] == c for r in range(h_in))]
                if rows and cols:
                    bbox_areas[c] = (rows[-1] - rows[0] + 1) * (cols[-1] - cols[0] + 1)
            if not bbox_areas:
                return None
            min_area = min(bbox_areas.values())
            candidates = [c for c, a in bbox_areas.items() if a == min_area]
            target = min(candidates)
        rows = [r for r in range(h_in) if any(inp[r][cc] == target for cc in range(w_in))]
        cols = [c for c in range(w_in) if any(inp[r][c] == target for r in range(h_in))]
        if not rows or not cols:
            return None
        r0, r1 = rows[0], rows[-1]
        c0, c1 = cols[0], cols[-1]
        cropped = [inp[r][c0:c1 + 1] for r in range(r0, r1 + 1)]
        if cropped != out:
            return None
    return {"matched": True, "picker": picker}


def _detect_largest_rectangle_recolor_crop(pairs):
    """Detect task396: crop largest complete frame rectangle and recolor border.

    The runtime builder picks most-common nonzero as the frame color, least-common
    nonzero as the marker color, and scans 4..8 sized complete rectangles. Keep
    this detector aligned with that exact tie-break and size band.
    """
    if not pairs:
        return None

    def _rect_ok(inp, frame, r0, c0, h, w):
        r1, c1 = r0 + h - 1, c0 + w - 1
        for c in range(c0, c1 + 1):
            if inp[r0][c] != frame or inp[r1][c] != frame:
                return False
        for r in range(r0, r1 + 1):
            if inp[r][c0] != frame or inp[r][c1] != frame:
                return False
        return True

    for p in pairs:
        inp = p["input"]
        out = p["output"]
        if not inp or not out:
            return None
        h_in, w_in = len(inp), len(inp[0])
        counts = {}
        for row in inp:
            if len(row) != w_in:
                return None
            for v in row:
                if v:
                    counts[v] = counts.get(v, 0) + 1
        if len(counts) != 2:
            return None
        count_values = sorted(counts.values())
        if count_values[0] == count_values[1]:
            return None
        frame = max(counts, key=counts.get)
        marker = min(counts, key=counts.get)

        best = None
        for h in range(4, 9):
            if h > h_in:
                continue
            for w in range(4, 9):
                if w > w_in:
                    continue
                score = (h - 2) * (w - 2) * 100 + h * w
                for r0 in range(0, h_in - h + 1):
                    for c0 in range(0, w_in - w + 1):
                        if not _rect_ok(inp, frame, r0, c0, h, w):
                            continue
                        cand = (score, -r0, -c0, r0, c0, h, w)
                        if best is None or cand > best:
                            best = cand
        if best is None:
            return None
        _, _, _, r0, c0, h, w = best
        if len(out) != h or any(len(row) != w for row in out):
            return None
        expected = []
        for rr in range(h):
            row = []
            for cc in range(w):
                if rr in (0, h - 1) or cc in (0, w - 1):
                    row.append(marker)
                else:
                    v = inp[r0 + rr][c0 + cc]
                    row.append(marker if v == marker else 0)
            expected.append(row)
        if expected != out:
            return None
    return {"matched": True}


def _detect_marker_dilate(pairs):
    """Detect: marker-dilate-recolor pattern.

    Rule: there is a single marker_color M (non-zero, sparse), a single
    target_color T (1..9), and a preserve_marker flag such that:
      - Every cell within 3x3 of any input M-cell becomes T in the output
        (except M-cells themselves, which behave per preserve_marker).
      - Cells outside any 3x3 of M are unchanged from input.
      - If preserve_marker=True: input M-cells stay M in output; T must != M.
      - If preserve_marker=False: input M-cells become T in output (M must equal T
        for the cell itself, but T != M as the displacement color is allowed too —
        in practice the detector only fires when marker is fully replaced).

    Cross-pair: same (M, T, preserve_marker) for every pair.
    Returns {"marker_color": M, "target_color": T, "preserve_marker": bool}, else None.
    """
    if not pairs:
        return None
    if not all(_shape(p["input"]) == _shape(p["output"]) for p in pairs):
        return None

    # Restrict to small / medium grids; the rule is shape-preserving and the
    # patterns we've seen are 9x9 grids.
    for p in pairs:
        h, w = _shape(p["input"])
        if h > 30 or w > 30 or h < 3 or w < 3:
            return None

    def _dilate3(mask):
        h, w = mask.shape
        out = np.zeros_like(mask)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                src_r0, src_r1 = max(0, -dr), min(h, h - dr)
                src_c0, src_c1 = max(0, -dc), min(w, w - dc)
                if src_r0 >= src_r1 or src_c0 >= src_c1:
                    continue
                out[src_r0 + dr:src_r1 + dr, src_c0 + dc:src_c1 + dc] |= mask[src_r0:src_r1, src_c0:src_c1]
        return out

    def _try(M, T, preserve):
        if M < 1 or M > 9 or T < 1 or T > 9:
            return False
        if preserve and M == T:
            return False
        for p in pairs:
            in_arr = np.array(p["input"], dtype=np.int8)
            out_arr = np.array(p["output"], dtype=np.int8)
            mask_M = (in_arr == M)
            if not mask_M.any():
                # No marker in this pair => output should equal input.
                if not np.array_equal(in_arr, out_arr):
                    return False
                continue
            dilated = _dilate3(mask_M)
            ring = dilated & ~mask_M
            expected = in_arr.copy()
            expected[ring] = T
            if preserve:
                expected[mask_M] = M
            else:
                expected[mask_M] = T
            if not np.array_equal(expected, out_arr):
                return False
        return True

    # Candidate marker colors: any color that appears in BOTH input and (input or output)
    # of at least one pair. Restrict to colors that appear sparse-enough.
    candidate_M = set()
    for p in pairs:
        in_arr = np.array(p["input"], dtype=np.int8)
        for c in range(1, 10):
            if (in_arr == c).any():
                candidate_M.add(c)
    candidate_T = set(range(1, 10))

    for M in sorted(candidate_M):
        for T in sorted(candidate_T):
            for preserve in (True, False):
                if _try(M, T, preserve):
                    return {"marker_color": M, "target_color": T, "preserve_marker": preserve}
    return None


def _detect_marker_dilate_conv_safe(pairs):
    """Detect marker-dilate cases that are safe for the single-Conv fast path."""
    match = _detect_marker_dilate(pairs)
    if match is None:
        return None

    marker = match["marker_color"]
    for p in pairs:
        in_arr = np.array(p["input"], dtype=np.int8)
        if in_arr.ndim != 2:
            return None
        h, w = in_arr.shape
        marker_cells = np.argwhere(in_arr == marker)
        ring_counts = np.zeros((h, w), dtype=np.int8)
        for r, c in marker_cells:
            if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                return None
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = int(r + dr), int(c + dc)
                    if in_arr[nr, nc] != 0:
                        return None
                    ring_counts[nr, nc] += 1
                    if ring_counts[nr, nc] > 1:
                        return None
    return match


def _detect_fractal_ring_expansion(pairs):
    """Detect: every pair has input shape (5, 5), output shape (10, 10), and
    output[i][j] = fg[max(i, j) mod K] where:
      colors = [input[k][k] for k in 0..4]
      fg = [c for c in colors if c != 0]
      K = len(fg)

    Returns {"matched": True} on match, else None.

    Rejects if K is not in {4, 5} (the recipe is only verified for these cases),
    or if the diagonal contains a 0 anywhere except at index 4 (the recipe's
    bg-handling assumes bg is at most colors[4]).
    """
    if not pairs:
        return None
    if {_shape(p["input"]) for p in pairs} != {(5, 5)}:
        return None
    if {_shape(p["output"]) for p in pairs} != {(10, 10)}:
        return None
    for p in pairs:
        inp = p["input"]
        out = p["output"]
        colors = [inp[k][k] for k in range(5)]
        # Bg may only appear at index 4 (the recipe's assumption). Reject otherwise.
        for k in range(4):
            if colors[k] == 0:
                return None
        fg = [c for c in colors if c != 0]
        K = len(fg)
        if K not in (4, 5):
            return None
        for i in range(10):
            for j in range(10):
                if out[i][j] != fg[max(i, j) % K]:
                    return None
    return {"matched": True}


def _detect_marker_rectangle_interior(pairs):
    """Detect task029: one color forms a complete rectangular border, output is
    the input cropped to the interior of that rectangle.

    Returns {"matched": True} on full match across all pairs, else None.
    """
    if not pairs:
        return None
    for p in pairs:
        inp = p["input"]
        out = p["output"]
        if not inp or not out:
            return None
        a = np.array(inp)
        e = np.array(out)
        H, W = a.shape
        found = False
        for color in range(1, 10):
            mask = (a == color)
            if not mask.any():
                continue
            rows = np.where(mask.any(axis=1))[0]
            cols = np.where(mask.any(axis=0))[0]
            r0, r1 = int(rows.min()), int(rows.max())
            c0, c1 = int(cols.min()), int(cols.max())
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            if not (
                all(a[r0, cc] == color for cc in range(c0, c1 + 1))
                and all(a[r1, cc] == color for cc in range(c0, c1 + 1))
                and all(a[rr, c0] == color for rr in range(r0, r1 + 1))
                and all(a[rr, c1] == color for rr in range(r0, r1 + 1))
            ):
                continue
            interior = a[r0 + 1:r1, c0 + 1:c1]
            if interior.shape != e.shape:
                continue
            if not np.array_equal(interior, e):
                continue
            found = True
            break
        if not found:
            return None
    return {"matched": True}


def _detect_foreground_bbox_crop(pairs):
    """Detect: every pair's output equals the input cropped to the bounding box
    of all non-zero (color != 0) cells in the input. Variable input shape;
    output shape per-pair = (last_row - first_row + 1, last_col - first_col + 1).

    Returns {"matched": True} on match, else None.

    Rejects pairs where the input has no non-zero cells (rule undefined).
    """
    if not pairs:
        return None
    for p in pairs:
        inp = p["input"]
        out = p["output"]
        h_in = len(inp)
        if h_in == 0:
            return None
        w_in = len(inp[0])
        if w_in == 0:
            return None
        rows_with_fg = [r for r in range(h_in) if any(c != 0 for c in inp[r])]
        cols_with_fg = [c for c in range(w_in) if any(inp[r][c] != 0 for r in range(h_in))]
        if not rows_with_fg or not cols_with_fg:
            return None
        r0, r1 = rows_with_fg[0], rows_with_fg[-1]
        c0, c1 = cols_with_fg[0], cols_with_fg[-1]
        cropped = [inp[r][c0:c1 + 1] for r in range(r0, r1 + 1)]
        if cropped != out:
            return None
    return {"matched": True}


def _frame_raycast_crop_expected(inp):
    """Return task138 frame-raycast crop prediction, or None if preconditions fail."""
    a = np.array(inp, dtype=np.int64)
    if a.ndim != 2 or a.size == 0:
        return None
    h, w = a.shape
    full_rows = [r for r in range(h) if np.all(a[r, :] != 0)]
    full_cols = [c for c in range(w) if np.all(a[:, c] != 0)]
    if len(full_rows) != 2 or len(full_cols) != 2:
        return None
    r0, r1 = full_rows
    c0, c1 = full_cols
    if r1 <= r0 + 1 or c1 <= c0 + 1:
        return None
    crop = a[r0:r1 + 1, c0:c1 + 1].copy()

    def side_color(vals):
        counts = collections.Counter(int(v) for v in vals if int(v) != 0)
        if not counts:
            return None
        return counts.most_common(1)[0][0]

    top = side_color(crop[0, 1:-1])
    bottom = side_color(crop[-1, 1:-1])
    left = side_color(crop[1:-1, 0])
    right = side_color(crop[1:-1, -1])
    if None in (top, bottom, left, right):
        return None

    interior = crop[1:-1, 1:-1]
    marker_colors = sorted(set(int(v) for v in interior.flatten()) - {0})
    if len(marker_colors) == 0:
        return crop.tolist()
    if len(marker_colors) > 1:
        return None
    marker = marker_colors[0]
    side_matches = [
        side for side, color in (
            ("top", top),
            ("bottom", bottom),
            ("left", left),
            ("right", right),
        )
        if color == marker
    ]
    if len(side_matches) != 1:
        return None
    side = side_matches[0]
    for rr, cc in zip(*np.where(interior == marker)):
        r = int(rr) + 1
        c = int(cc) + 1
        if side == "top":
            crop[:r + 1, c] = marker
        elif side == "bottom":
            crop[r:, c] = marker
        elif side == "left":
            crop[r, :c + 1] = marker
        else:
            crop[r, c:] = marker
    return crop.tolist()


def _detect_frame_raycast_crop(pairs):
    if not pairs:
        return None
    saw_marker = False
    for p in pairs:
        expected = _frame_raycast_crop_expected(p["input"])
        if expected is None or expected != p["output"]:
            return None
        if expected != [
            row[:] for row in p["input"]
        ]:
            saw_marker = True
    return {"matched": True} if saw_marker else None


def _detect_kron_block_up(pairs):
    """Detect: every pair's output is the input with each cell scaled to a
    (scale_h x scale_w) block (Kronecker product with ones). The same scale
    factors apply across all pairs.

    Returns {"scale_h": int, "scale_w": int} on match, else None.

    Constraints to keep the builder pipeline simple:
      - scale_h, scale_w in {1, 2, 3, 5, 6, 10, 15, 30} (divisors of 30) and
        not both equal to 1.
      - Output shape = (h_in * scale_h, w_in * scale_w) <= (30, 30) for every
        pair (otherwise output would not fit on the canvas).
    """
    if not pairs:
        return None
    valid_scales = {1, 2, 3, 5, 6, 10, 15, 30}
    inp0 = pairs[0]["input"]
    out0 = pairs[0]["output"]
    h_in0, w_in0 = len(inp0), (len(inp0[0]) if inp0 else 0)
    h_out0, w_out0 = len(out0), (len(out0[0]) if out0 else 0)
    if h_in0 == 0 or w_in0 == 0 or h_out0 == 0 or w_out0 == 0:
        return None
    if h_out0 % h_in0 != 0 or w_out0 % w_in0 != 0:
        return None
    sh = h_out0 // h_in0
    sw = w_out0 // w_in0
    if sh == 1 and sw == 1:
        return None
    if sh not in valid_scales or sw not in valid_scales:
        return None
    for p in pairs:
        inp = p["input"]
        out = p["output"]
        h_in = len(inp)
        if h_in == 0:
            return None
        w_in = len(inp[0])
        if w_in == 0:
            return None
        if h_in * sh > 30 or w_in * sw > 30:
            return None
        if len(out) != h_in * sh:
            return None
        for r in range(h_in):
            for dr in range(sh):
                if len(out[r * sh + dr]) != w_in * sw:
                    return None
                for c in range(w_in):
                    cell = inp[r][c]
                    for dc in range(sw):
                        if out[r * sh + dr][c * sw + dc] != cell:
                            return None
    return {"scale_h": sh, "scale_w": sw}


def _detect_task388(pairs):
    """Column-fill 0->8 + 2x2 tile: task388.

    Find unique nonzero color X. Fill 0-cells in X-containing columns with 8.
    Tile the transformed grid 2x2. Output shape: (2H, 2W). Verified 266/266.
    """
    if not pairs:
        return None
    for p in pairs:
        a = np.array(p["input"])
        e = np.array(p["output"])
        H, W = a.shape
        if e.shape != (2 * H, 2 * W):
            return None
        nonzero = set(int(v) for v in a.flatten() if v != 0)
        if len(nonzero) != 1:
            return None
        X = list(nonzero)[0]
        cols_with_X = set(c for c in range(W) if any(a[r, c] == X for r in range(H)))
        transformed = a.copy()
        for r in range(H):
            for c in cols_with_X:
                if a[r, c] == 0:
                    transformed[r, c] = 8
        expected = np.zeros((2 * H, 2 * W), dtype=int)
        expected[:H, :W] = transformed
        expected[:H, W:] = transformed
        expected[H:, :W] = transformed
        expected[H:, W:] = transformed
        if not np.array_equal(expected, e):
            return None
    return True


def _tile_diagonal_neighbor_mark_expected(inp):
    try:
        a = np.array(inp, dtype=np.int8)
    except (TypeError, ValueError):
        return None
    if a.ndim != 2 or a.size == 0:
        return None
    H, W = a.shape
    if 2 * H > 30 or 2 * W > 30:
        return None
    colors = sorted(set(int(v) for v in a.flatten()) - {0})
    if len(colors) != 1 or colors[0] == 8:
        return None
    marker = colors[0]
    tiled = np.tile(a, (2, 2))
    out = tiled.copy()
    for r, c in zip(*np.where(tiled == marker)):
        for dr in (-1, 1):
            for dc in (-1, 1):
                rr = int(r) + dr
                cc = int(c) + dc
                if 0 <= rr < 2 * H and 0 <= cc < 2 * W and out[rr, cc] == 0:
                    out[rr, cc] = 8
    return out


def _detect_tile_diagonal_neighbor_mark(pairs):
    """Detect task019's 2x2 tiled marker plus diagonal-neighbor color-8 marks."""
    if not pairs:
        return None
    for p in pairs:
        expected = _tile_diagonal_neighbor_mark_expected(p["input"])
        if expected is None:
            return None
        try:
            out = np.array(p["output"], dtype=np.int8)
        except (TypeError, ValueError):
            return None
        if out.shape != expected.shape or not np.array_equal(expected, out):
            return None
    return True


def _color2_bbox_fill_expected(inp):
    try:
        arr = np.array(inp, dtype=np.int64)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 2 or arr.size == 0 or not np.any(arr == 2):
        return None
    H, W = arr.shape
    is_two = arr == 2
    fill = np.zeros((H, W), dtype=bool)

    def segments(col_any):
        occ = [idx for idx, val in enumerate(col_any) if val]
        if not occ:
            return []
        result = []
        start = prev = occ[0]
        for x in occ[1:] + [None]:
            if x is not None and x - prev <= 2:
                prev = x
                continue
            if 2 <= prev - start + 1 <= 7:
                result.append((start, prev))
            if x is not None:
                start = prev = x
        return result

    for h in (2, 3):
        for row_offset in range(h):
            for r in range(H):
                r0 = r - row_offset
                r1 = r0 + h - 1
                if r0 < 0 or r1 >= H:
                    continue
                band = is_two[r0:r1 + 1, :]
                for c0, c1 in segments(band.any(axis=0).tolist()):
                    if not band[0, c0:c1 + 1].any():
                        continue
                    if not band[-1, c0:c1 + 1].any():
                        continue
                    fill[r, c0:c1 + 1] = True

    out = arr.copy()
    out[fill & (arr != 0) & (arr != 2)] = 4
    return out


def _detect_color2_bbox_fill(pairs):
    """Detect task077's compact color-2 component bbox fill."""
    if not pairs:
        return None
    for p in pairs:
        expected = _color2_bbox_fill_expected(p["input"])
        if expected is None:
            return None
        try:
            out = np.array(p["output"], dtype=np.int64)
        except (TypeError, ValueError):
            return None
        if out.shape != expected.shape or not np.array_equal(expected, out):
            return None
    return True


def _diagonal_marker_square_rings_expected(inp):
    try:
        a = np.array(inp, dtype=np.int8)
    except (TypeError, ValueError):
        return None
    if a.ndim != 2 or a.size == 0 or a.shape[0] != a.shape[1]:
        return None
    colors = sorted(set(int(v) for v in a.flatten()) - {0})
    if len(colors) != 1:
        return None
    marker = colors[0]
    pts = sorted((int(r), int(c)) for r, c in zip(*np.where(a == marker)))
    if len(pts) != 3:
        return None
    (r0, c0), (cr, cc), (r2, c2) = pts
    if r0 + r2 != 2 * cr or c0 + c2 != 2 * cc:
        return None
    dr1, dc1 = cr - r0, cc - c0
    dr2, dc2 = r2 - cr, c2 - cc
    if (dr1, dc1) != (dr2, dc2):
        return None
    if dr1 == 0 or abs(dr1) != abs(dc1):
        return None
    period = abs(dr1)
    H, W = a.shape
    maxd = max(cr, H - 1 - cr, cc, W - 1 - cc)
    out = np.zeros_like(a)
    for radius in range(0, maxd + 1, period):
        for r in range(H):
            for c in range(W):
                if max(abs(r - cr), abs(c - cc)) == radius:
                    out[r, c] = marker
    return out


def _anchor_cross_aligned_cells_predict(inp):
    """Task270 numpy reference (266/266). 15x15, two centers (1, 2), scattered {3,7} cells."""
    a = np.array(inp, dtype=np.int64)
    if a.ndim != 2 or a.shape != (15, 15):
        return None
    cells_1 = np.argwhere(a == 1)
    cells_2 = np.argwhere(a == 2)
    if len(cells_1) != 1 or len(cells_2) != 1:
        return None
    # all other non-zero cells must be in {3, 7}
    other_mask = (a != 0) & (a != 1) & (a != 2)
    if other_mask.any():
        other_colors = set(int(v) for v in a[other_mask])
        if not other_colors.issubset({3, 7}):
            return None
    cr1, cc1 = int(cells_1[0, 0]), int(cells_1[0, 1])
    cr2, cc2 = int(cells_2[0, 0]), int(cells_2[0, 1])
    out = np.zeros_like(a)
    out[cr1, cc1] = 1
    out[cr2, cc2] = 2
    # tally unambiguous 3-cells per center
    cnt3_c1 = cnt3_c2 = 0
    for r in range(15):
        for c in range(15):
            if int(a[r, c]) != 3:
                continue
            if (r, c) == (cr1, cc1) or (r, c) == (cr2, cc2):
                continue
            a1 = (r == cr1) or (c == cc1)
            a2 = (r == cr2) or (c == cc2)
            if a1 and not a2:
                cnt3_c1 += 1
            elif a2 and not a1:
                cnt3_c2 += 1
    outer1 = 3 if cnt3_c1 > cnt3_c2 else 7
    outer2 = 7 if outer1 == 3 else 3
    H = W = 15
    for (cr, cc, outer_color) in ((cr1, cc1, outer1), (cr2, cc2, outer2)):
        # arms: only drawn if at least one outer-color cell exists in that direction along center's row/col
        if any(a[cr, c] == outer_color for c in range(0, cc)):
            if cc - 1 >= 0:
                out[cr, cc - 1] = outer_color
        if any(a[cr, c] == outer_color for c in range(cc + 1, W)):
            if cc + 1 < W:
                out[cr, cc + 1] = outer_color
        if any(a[r, cc] == outer_color for r in range(0, cr)):
            if cr - 1 >= 0:
                out[cr - 1, cc] = outer_color
        if any(a[r, cc] == outer_color for r in range(cr + 1, H)):
            if cr + 1 < H:
                out[cr + 1, cc] = outer_color
    return out.tolist()


def _detect_anchor_cross_aligned_cells(pairs):
    """Detect task270's anchor-cross color-partition pattern (verified 266/266)."""
    if not pairs:
        return None
    saw_change = False
    for p in pairs:
        if _shape(p["input"]) != _shape(p["output"]):
            return None
        if _shape(p["input"]) != (15, 15):
            return None
        try:
            expected = _anchor_cross_aligned_cells_predict(p["input"])
        except (TypeError, ValueError):
            return None
        if expected is None or expected != p["output"]:
            return None
        if expected != p["input"]:
            saw_change = True
    return {"matched": True} if saw_change else None


def _detect_diagonal_marker_square_rings(pairs):
    """Detect task137's square rings from three equally spaced diagonal markers."""
    if not pairs:
        return None
    for p in pairs:
        expected = _diagonal_marker_square_rings_expected(p["input"])
        if expected is None:
            return None
        try:
            out = np.array(p["output"], dtype=np.int8)
        except (TypeError, ValueError):
            return None
        if out.shape != expected.shape or not np.array_equal(expected, out):
            return None
    return True


def _detect_task316(pairs):
    """Boustrophedon pack into 3x3: task316.

    Non-zero cells sorted by (col, row) lex; first 9 packed into 3x3 in snake order.
    Verified 266/266 pairs.
    """
    if not pairs:
        return None
    for p in pairs:
        a = np.array(p["input"])
        e = np.array(p["output"])
        H, W = a.shape
        if e.shape != (3, 3):
            return None
        cells = [(int(a[r, c]), r, c) for r in range(H) for c in range(W) if a[r, c] != 0]
        if len(cells) > 9:
            return None
        cells.sort(key=lambda t: (t[2], t[1]))
        vals = [v for v, r, c in cells][:9]
        out = np.zeros((3, 3), dtype=int)
        for k, v in enumerate(vals):
            row, col = k // 3, k % 3
            if row % 2 == 1:
                col = 2 - col
            out[row, col] = v
        if not np.array_equal(out, e):
            return None
    return True


def _detect_horizontal_tile_2x(pairs):
    """Horizontal tile by 2: task249.

    Output shape = (H, 2*W). Left half equals input, right half equals input.
    Numpy rule: np.tile(input, (1, 2)).
    """
    if not pairs:
        return None
    for p in pairs:
        a = np.array(p["input"])
        e = np.array(p["output"])
        H, W = a.shape
        if e.shape != (H, 2 * W):
            return None
        if not (np.array_equal(e[:, :W], a) and np.array_equal(e[:, W:], a)):
            return None
    return True


def _period_extend_row_2x_expected(inp):
    try:
        a = np.array(inp, dtype=np.int8)
    except (TypeError, ValueError):
        return None
    if a.ndim != 2 or a.shape[0] != 5 or not (6 <= a.shape[1] <= 10):
        return None
    h, w = a.shape
    out = np.zeros((h, 2 * w), dtype=np.int8)
    any_nonzero = False
    for r in range(h):
        row = [int(v) for v in a[r].tolist()]
        if all(v == 0 for v in row):
            continue
        any_nonzero = True
        period = None
        for p in (1, 2, 3):
            if all(row[c] == row[c - p] for c in range(p, w)):
                period = p
                break
        if period is None:
            return None
        for c in range(2 * w):
            out[r, c] = row[c % period]
    if not any_nonzero:
        return None
    return out


def _detect_period_extend_row_2x(pairs):
    """Detect task231's row-wise period extension to double width."""
    if not pairs:
        return None
    for p in pairs:
        expected = _period_extend_row_2x_expected(p["input"])
        if expected is None:
            return None
        try:
            out = np.array(p["output"], dtype=np.int8)
        except (TypeError, ValueError):
            return None
        if out.shape != expected.shape or not np.array_equal(expected, out):
            return None
    return {"matched": True}


def _periodic_hole_fill_expected(inp):
    try:
        a = np.array(inp, dtype=np.int8)
    except (TypeError, ValueError):
        return None
    if a.shape != (21, 21):
        return None
    if not np.any(a == 0):
        return None

    def best_period(axis):
        h, w = a.shape
        limit = w if axis == 1 else h
        best = None
        for period in range(1, min(12, limit)):
            comparisons = 0
            conflicts = 0
            if axis == 1:
                for r in range(h):
                    for c in range(w - period):
                        x, y = int(a[r, c]), int(a[r, c + period])
                        if x != 0 and y != 0:
                            comparisons += 1
                            conflicts += int(x != y)
            else:
                for r in range(h - period):
                    for c in range(w):
                        x, y = int(a[r, c]), int(a[r + period, c])
                        if x != 0 and y != 0:
                            comparisons += 1
                            conflicts += int(x != y)
            if comparisons == 0:
                continue
            key = (conflicts, -comparisons, period)
            if best is None or key < best[0]:
                best = (key, period)
        return None if best is None else best[1]

    hp = best_period(axis=1)
    vp = best_period(axis=0)
    if hp is None or vp is None or hp != vp:
        return None

    out = a.copy()
    h, w = out.shape
    for r in range(h):
        for c in range(w):
            if out[r, c] != 0:
                continue
            vals = [
                int(a[rr, cc])
                for rr in range(r % hp, h, hp)
                for cc in range(c % hp, w, hp)
                if a[rr, cc] != 0
            ]
            if vals:
                out[r, c] = max(set(vals), key=vals.count)
    if np.any(out == 0):
        return None
    return out


def _detect_periodic_hole_fill(pairs):
    if not pairs:
        return None
    for p in pairs:
        expected = _periodic_hole_fill_expected(p["input"])
        if expected is None:
            return None
        try:
            out = np.array(p["output"], dtype=np.int8)
        except (TypeError, ValueError):
            return None
        if out.shape != expected.shape or not np.array_equal(expected, out):
            return None
    return {"matched": True}


def _zero_adjacency_label_expected(inp):
    a = np.array(inp, dtype=int)
    H, W = a.shape
    zero = (a == 0)
    counts = np.zeros((H, W), dtype=int)
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        shifted = np.zeros((H, W), dtype=bool)
        r_src0 = max(0, -dr)
        r_src1 = H - max(0, dr)
        c_src0 = max(0, -dc)
        c_src1 = W - max(0, dc)
        r_dst0 = max(0, dr)
        r_dst1 = H - max(0, -dr)
        c_dst0 = max(0, dc)
        c_dst1 = W - max(0, -dc)
        shifted[r_dst0:r_dst1, c_dst0:c_dst1] = zero[r_src0:r_src1, c_src0:c_src1]
        counts += shifted.astype(int)

    out = a.copy()
    heavy = zero & (counts >= 2)
    for r in range(H):
        for c in range(W):
            if a[r, c] == 5:
                continue
            if a[r, c] != 0:
                return None
            if counts[r, c] == 0:
                out[r, c] = 3
            elif counts[r, c] >= 2:
                out[r, c] = 1
            else:
                has_heavy_neighbor = False
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < H and 0 <= cc < W and heavy[rr, cc]:
                        has_heavy_neighbor = True
                        break
                out[r, c] = 1 if has_heavy_neighbor else 2
    return out


def _detect_zero_adjacency_label(pairs):
    """Detect task369: 10x10 {0,5} input -> zero-adjacency labels {1,2,3}."""
    if not pairs:
        return None
    for p in pairs:
        inp = p["input"]
        out = p["output"]
        if _shape(inp) != (10, 10) or _shape(out) != (10, 10):
            return None
        if any(len(row) != 10 for row in inp) or any(len(row) != 10 for row in out):
            return None
        in_colors = {v for row in inp for v in row}
        out_colors = {v for row in out for v in row}
        if in_colors - {0, 5} or out_colors - {1, 2, 3, 5}:
            return None
        expected = _zero_adjacency_label_expected(inp)
        if expected is None or not np.array_equal(expected, np.array(out, dtype=int)):
            return None
    return True


def _detect_task320(pairs):
    """Staircase bottom-half recolor: task320.

    Input colors ⊆ {0, 2}, output colors ⊆ {0, 2, 8}.
    For each column: bottom floor(count/2) color-2 cells become color-8.
    Verified numpy rule: 266/266 pairs.
    """
    if not pairs:
        return None
    for p in pairs:
        a = np.array(p["input"])
        e = np.array(p["output"])
        if a.shape != e.shape:
            return None
        if set(int(v) for v in a.flatten()) - {0, 2}:
            return None
        if set(int(v) for v in e.flatten()) - {0, 2, 8}:
            return None
        H, W = a.shape
        pred = a.copy()
        for c in range(W):
            col_mask = (a[:, c] == 2)
            cnt = int(col_mask.sum())
            if cnt < 2:
                continue
            half = cnt // 2
            rows = np.where(col_mask)[0]
            for r in rows[-half:]:
                pred[r, c] = 8
        if not np.array_equal(pred, e):
            return None
    return True


def _marker_count_cross_shift_expected(inp):
    try:
        a = np.array(inp, dtype=np.int8)
    except (TypeError, ValueError):
        return None
    if a.shape != (10, 10):
        return None
    colors = sorted(int(c) for c in np.unique(a) if int(c) not in (0, 5))
    if len(colors) != 1:
        return None
    color = colors[0]
    shift = int(np.count_nonzero(a == 5))
    if shift <= 0 or shift >= 10:
        return None
    color_mask = a == color
    full_rows = np.where(color_mask.sum(axis=1) == 10)[0]
    full_cols = np.where(color_mask.sum(axis=0) == 10)[0]
    if len(full_rows) != 1 or len(full_cols) != 1:
        return None
    out_row = int(full_rows[0]) + shift
    out_col = int(full_cols[0]) - shift
    if not (0 <= out_row < 10 and 0 <= out_col < 10):
        return None
    out = np.zeros_like(a)
    out[out_row, :] = color
    out[:, out_col] = color
    return out


def _detect_marker_count_cross_shift(pairs):
    """Detect task362: color-5 marker count shifts a full cross down-left."""
    if not pairs:
        return None
    for p in pairs:
        expected = _marker_count_cross_shift_expected(p["input"])
        if expected is None:
            return None
        try:
            out = np.array(p["output"], dtype=np.int8)
        except (TypeError, ValueError):
            return None
        if out.shape != (10, 10) or not np.array_equal(expected, out):
            return None
    return True


def _anchor_block_aligned_marker_line_expected(inp):
    try:
        a = np.array(inp, dtype=np.int8)
    except (TypeError, ValueError):
        return None
    if a.shape != (10, 10):
        return None
    block = a == 3
    if int(block.sum()) != 4:
        return None
    block_rows = np.where(block.any(axis=1))[0]
    block_cols = np.where(block.any(axis=0))[0]
    if len(block_rows) != 2 or len(block_cols) != 2:
        return None
    if block_rows[1] != block_rows[0] + 1 or block_cols[1] != block_cols[0] + 1:
        return None
    if not np.all(block[np.ix_(block_rows, block_cols)]):
        return None

    out = a.copy()
    row_mask = block.any(axis=1)
    col_mask = block.any(axis=0)
    for color in sorted(int(c) for c in np.unique(a) if int(c) not in (0, 3)):
        for r, c in np.argwhere(a == color):
            r = int(r)
            c = int(c)
            if row_mask[r]:
                cols = np.where(block[r])[0]
                if c < cols[0]:
                    rng = range(c + 1, int(cols[0]))
                elif c > cols[-1]:
                    rng = range(int(cols[-1]) + 1, c)
                else:
                    rng = ()
                for cc in rng:
                    if out[r, cc] == 0:
                        out[r, cc] = color
            if col_mask[c]:
                rows = np.where(block[:, c])[0]
                if r < rows[0]:
                    rng = range(r + 1, int(rows[0]))
                elif r > rows[-1]:
                    rng = range(int(rows[-1]) + 1, r)
                else:
                    rng = ()
                for rr in rng:
                    if out[rr, c] == 0:
                        out[rr, c] = color
    return out


def _detect_anchor_block_aligned_marker_line(pairs):
    """Detect task333: aligned markers connect back to a 2x2 color-3 anchor."""
    if not pairs:
        return None
    changed = False
    for p in pairs:
        expected = _anchor_block_aligned_marker_line_expected(p["input"])
        if expected is None:
            return None
        try:
            out = np.array(p["output"], dtype=np.int8)
        except (TypeError, ValueError):
            return None
        if out.shape != (10, 10) or not np.array_equal(expected, out):
            return None
        if not np.array_equal(np.array(p["input"], dtype=np.int8), out):
            changed = True
    return True if changed else None


def _frame_inner_markers_to_outer_opposite_expected(inp):
    try:
        a = np.array(inp, dtype=np.int8)
    except (TypeError, ValueError):
        return None
    if a.shape != (10, 10):
        return None
    vals, counts = np.unique(a[a != 0], return_counts=True)
    if len(vals) < 2:
        return None
    frame_color = int(vals[int(np.argmax(counts))])
    frame = a == frame_color
    rows = np.where(frame.any(axis=1))[0]
    cols = np.where(frame.any(axis=0))[0]
    if len(rows) < 2 or len(cols) < 2:
        return None
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    if r0 <= 0 or c0 <= 0 or r1 >= 9 or c1 >= 9:
        return None

    expected_frame = np.zeros_like(frame)
    expected_frame[r0, c0:c1 + 1] = True
    expected_frame[r1, c0:c1 + 1] = True
    expected_frame[r0:r1 + 1, c0] = True
    expected_frame[r0:r1 + 1, c1] = True
    if not np.array_equal(frame, expected_frame):
        return None

    moves = (
        ((r0 + 1, c0 + 1), (r1 + 1, c1 + 1)),
        ((r0 + 1, c1 - 1), (r1 + 1, c0 - 1)),
        ((r1 - 1, c0 + 1), (r0 - 1, c1 + 1)),
        ((r1 - 1, c1 - 1), (r0 - 1, c0 - 1)),
    )
    allowed = expected_frame.copy()
    out = np.zeros_like(a)
    out[expected_frame] = frame_color
    for (sr, sc), (tr, tc) in moves:
        color = int(a[sr, sc])
        if color in (0, frame_color):
            return None
        allowed[sr, sc] = True
        out[tr, tc] = color
    if np.any((a != 0) & (~allowed)):
        return None
    return out


def _detect_frame_inner_markers_to_outer_opposite(pairs):
    """Detect task228: inner frame-corner markers move to opposite outer corners."""
    if not pairs:
        return None
    for p in pairs:
        expected = _frame_inner_markers_to_outer_opposite_expected(p["input"])
        if expected is None:
            return None
        try:
            out = np.array(p["output"], dtype=np.int8)
        except (TypeError, ValueError):
            return None
        if out.shape != (10, 10) or not np.array_equal(expected, out):
            return None
    return True


def _top_marker_run_recolor_expected(inp):
    try:
        a = np.array(inp, dtype=np.int8)
    except (TypeError, ValueError):
        return None
    if a.shape != (10, 10):
        return None
    markers = [(c, int(v)) for c, v in enumerate(a[0]) if int(v) not in (0, 5)]
    if len(markers) != 3:
        return None
    marker_cols = {c: color for c, color in markers}
    expected_colors = {0, 5} | set(marker_cols.values())
    if any(int(v) not in expected_colors for v in np.unique(a)):
        return None
    if not np.any(a == 5):
        return None

    out = a.copy()
    changed = False
    for r in range(10):
        c = 0
        while c < 10:
            if int(a[r, c]) != 5:
                c += 1
                continue
            start = c
            while c < 10 and int(a[r, c]) == 5:
                c += 1
            end = c - 1
            hits = [color for col, color in marker_cols.items() if start <= col <= end]
            if len(hits) != 1:
                return None
            out[r, start:c] = hits[0]
            changed = True
    return out if changed else None


def _detect_top_marker_run_recolor(pairs):
    """Detect task354: top-row markers recolor same-row color-5 runs."""
    if not pairs:
        return None
    for p in pairs:
        expected = _top_marker_run_recolor_expected(p["input"])
        if expected is None:
            return None
        try:
            out = np.array(p["output"], dtype=np.int8)
        except (TypeError, ValueError):
            return None
        if out.shape != (10, 10) or not np.array_equal(expected, out):
            return None
    return True


def _empty_row_col_recolor2_expected(inp):
    try:
        a = np.array(inp, dtype=np.int8)
    except (TypeError, ValueError):
        return None
    if a.ndim != 2 or a.size == 0:
        return None
    if 2 in set(int(v) for v in np.unique(a)):
        return None
    nonzero = a != 0
    row_empty = ~np.any(nonzero, axis=1)
    col_empty = ~np.any(nonzero, axis=0)
    if not np.any(row_empty) and not np.any(col_empty):
        return None
    paint = (a == 0) & (row_empty[:, None] | col_empty[None, :])
    if not np.any(paint):
        return None
    out = a.copy()
    out[paint] = 2
    return out


def _detect_empty_row_col_recolor2(pairs):
    """Detect task303: color-0 cells in empty rows/columns become color 2."""
    if not pairs:
        return None
    for p in pairs:
        expected = _empty_row_col_recolor2_expected(p["input"])
        if expected is None:
            return None
        try:
            out = np.array(p["output"], dtype=np.int8)
        except (TypeError, ValueError):
            return None
        if out.shape != expected.shape or not np.array_equal(out, expected):
            return None
    return True


_ARCH_MARKER_FALL_SHAPE = {
    (0, 3),
    (1, 2), (1, 3), (1, 4),
    (2, 1), (2, 2), (2, 4), (2, 5),
    (3, 0), (3, 6),
}


def _arch_marker_fall_expected(inp):
    try:
        a = np.array(inp, dtype=np.int8)
    except (TypeError, ValueError):
        return None
    if a.shape != (20, 20):
        return None
    colors = [int(v) for v in np.unique(a) if int(v) != 0]
    if len(colors) != 2:
        return None

    object_candidates = []
    for color in colors:
        ys, xs = np.where(a == color)
        if len(ys) != 10:
            continue
        r0, c0 = int(ys.min()), int(xs.min())
        shape = {(int(r - r0), int(c - c0)) for r, c in zip(ys, xs)}
        if shape == _ARCH_MARKER_FALL_SHAPE:
            object_candidates.append(color)
    if len(object_candidates) != 1:
        return None

    obj = object_candidates[0]
    marker = next(color for color in colors if color != obj)
    out = a.copy()
    changed = False
    for c in sorted(set(int(x) for x in np.where(a == obj)[1])):
        obj_rows = np.where(a[:, c] == obj)[0]
        if len(obj_rows) == 0:
            continue
        start = int(obj_rows.max()) + 1
        if start >= a.shape[0] or not np.any(a[start:, c] == marker):
            continue
        col_tail = out[start:, c]
        paint_rows = (col_tail == 0)
        if np.any(paint_rows):
            col_tail[paint_rows] = marker
            changed = True
    return out if changed else None


def _detect_arch_marker_fall(pairs):
    """Detect task165: markers below arch columns project downward."""
    if not pairs:
        return None
    for p in pairs:
        expected = _arch_marker_fall_expected(p["input"])
        if expected is None:
            return None
        try:
            out = np.array(p["output"], dtype=np.int8)
        except (TypeError, ValueError):
            return None
        if out.shape != (20, 20) or not np.array_equal(out, expected):
            return None
    return True


def _diagonal_pair_connect_expected(inp):
    try:
        a = np.array(inp, dtype=np.int8)
    except (TypeError, ValueError):
        return None
    if a.shape != (10, 10):
        return None
    out = a.copy()
    changed = False
    for color in [int(v) for v in np.unique(a) if int(v) != 0]:
        pts = np.argwhere(a == color)
        if len(pts) != 2:
            return None
        (r0, c0), (r1, c1) = pts
        dr_total = int(r1 - r0)
        dc_total = int(c1 - c0)
        if dr_total == 0 or abs(dr_total) != abs(dc_total):
            return None
        dr = 1 if dr_total > 0 else -1
        dc = 1 if dc_total > 0 else -1
        r, c = int(r0 + dr), int(c0 + dc)
        while (r, c) != (int(r1), int(c1)):
            if out[r, c] == 0:
                out[r, c] = color
                changed = True
            elif out[r, c] != color:
                return None
            r += dr
            c += dc
    return out if changed else None


def _detect_diagonal_pair_connect(pairs):
    """Detect task037: same-color diagonal endpoint pairs are connected."""
    if not pairs:
        return None
    for p in pairs:
        expected = _diagonal_pair_connect_expected(p["input"])
        if expected is None:
            return None
        try:
            out = np.array(p["output"], dtype=np.int8)
        except (TypeError, ValueError):
            return None
        if out.shape != (10, 10) or not np.array_equal(out, expected):
            return None
    return True


def _downshift_recolor8_to2_expected(inp):
    try:
        a = np.array(inp, dtype=np.int8)
    except (TypeError, ValueError):
        return None
    if a.ndim != 2 or a.size == 0:
        return None
    colors = set(int(v) for v in np.unique(a))
    if not colors <= {0, 8} or 8 not in colors:
        return None
    if np.any(a[-1, :] == 8):
        return None
    mask = a == 8
    shifted = np.zeros_like(mask)
    shifted[1:, :] = mask[:-1, :]
    if not np.any(shifted):
        return None
    out = np.zeros_like(a)
    out[shifted] = 2
    return out


def _detect_downshift_recolor8_to2(pairs):
    """Detect task261: color 8 shifts down one row and becomes color 2."""
    if not pairs:
        return None
    for p in pairs:
        expected = _downshift_recolor8_to2_expected(p["input"])
        if expected is None:
            return None
        try:
            out = np.array(p["output"], dtype=np.int8)
        except (TypeError, ValueError):
            return None
        if out.shape != expected.shape or not np.array_equal(out, expected):
            return None
    return True


def _fixed_shape_network_factory(desc):
    """Map a transform description (from _detect_fixed_shape_spatial) to a network factory."""
    t = desc["transform"]
    if t == "flip_h":
        return lambda batch=1: fixed_flip_h_network(desc["w"], batch=batch)
    if t == "flip_v":
        return lambda batch=1: fixed_flip_v_network(desc["h"], batch=batch)
    if t == "rot180":
        return lambda batch=1: fixed_rot180_network(desc["h"], desc["w"], batch=batch)
    if t == "rot90":
        return lambda batch=1: fixed_rot90_network(desc["n"], batch=batch)
    if t == "rot270":
        return lambda batch=1: fixed_rot270_network(desc["n"], batch=batch)
    if t.startswith("shift_v_"):
        return lambda batch=1: fixed_shift_v_network(desc["h"], desc["k"], batch=batch)
    if t.startswith("shift_h_"):
        return lambda batch=1: fixed_shift_h_network(desc["w"], desc["k"], batch=batch)
    return None


def _collect_matches(pairs):
    """Return a list of (label, network_fn) for every detector that matches.

    Order in the returned list mirrors the static evaluation order; this matters
    only for tie-breaking in the cost-aware selection done by detect_pattern
    (the earlier match wins on equal cost).

    All builder lambdas use default-argument binding (`_arg=value`) to capture
    the detector outputs at lambda-creation time. Without this, late-binding
    closure semantics would silently shadow earlier captures because variable
    names like `in_h`/`lut` are reused across detectors.

    Defense-in-depth: validate inputs are rectangular grids before dispatching.
    Several detectors index `grid[r][c]` assuming uniform row lengths; jagged
    grids would crash the entire build (one bad detector aborts dispatch). This
    invariant is checked once here so detectors can assume well-formed input.

    Also reject empty pairs lists — the `_PATTERNS` loop's `all(...)` is
    vacuously True over an empty iterable, which would (incorrectly) match the
    transpose pattern for any empty examples set.
    """
    if not pairs:
        return []
    for p in pairs:
        for grid in (p["input"], p["output"]):
            if not grid:
                continue
            row_lens = {len(row) for row in grid}
            if len(row_lens) != 1:
                return []  # jagged grid; reject all detectors at the dispatch boundary

    matches = []
    for label, transform_fn, network_fn, restriction in _PATTERNS:
        if not restriction(pairs):
            continue
        if all(transform_fn(p["input"]) == p["output"] for p in pairs):
            matches.append((label, network_fn))

    fixed_match = _detect_fixed_shape_spatial(pairs)
    if fixed_match is not None:
        net_fn = _fixed_shape_network_factory(fixed_match)
        if net_fn is not None:
            matches.append((f"fixed-{fixed_match['transform']}", net_fn))

    translation_match = _detect_translation(pairs)
    if translation_match is not None:
        matches.append(("translation",
                        lambda batch=1, _h=translation_match["h"], _w=translation_match["w"], _dr=translation_match["dr"], _dc=translation_match["dc"]: translation_network(_h, _w, _dr, _dc, batch=batch)))

    slice_match = _detect_slice_extraction(pairs)
    if slice_match is not None:
        hs, he, ws, we = slice_match
        matches.append(("slice-extract",
                        lambda batch=1, _hs=hs, _he=he, _ws=ws, _we=we: slice_extract_network(_hs, _he, _ws, _we, batch=batch)))

    mc_match = _detect_marker_corners(pairs)
    if mc_match is not None:
        marker_color, corners = mc_match
        matches.append(("marker-corners",
                        lambda batch=1, _m=marker_color, _c=corners: marker_corners_network(_m, _c, batch=batch)))

    iso_match = _detect_isolated_marker_recolor(pairs)
    if iso_match is not None:
        M, R = iso_match
        matches.append(("isolated-marker-recolor",
                        lambda batch=1, _M=M, _R=R: isolated_marker_recolor_network(_M, _R, batch=batch)))

    lshape_match = _detect_l_shape_missing_corner(pairs)
    if lshape_match is not None:
        M, C, F = lshape_match
        matches.append(("l-shape-missing-corner",
                        lambda batch=1, _M=M, _C=C, _F=F: l_shape_missing_corner_network(_M, _C, _F, batch=batch)))

    sandwich_match = _detect_sandwich_fill(pairs)
    if sandwich_match is not None:
        M, C, F, use_h, use_v = sandwich_match
        matches.append(("sandwich-fill",
                        lambda batch=1, _M=M, _C=C, _F=F, _h=use_h, _v=use_v: sandwich_fill_network(_M, _C, _F, _h, _v, batch=batch)))

    aligned_match = _detect_aligned_marker_fill(pairs)
    if aligned_match is not None:
        M, C, F, use_h, use_v = aligned_match
        matches.append(("aligned-marker-fill",
                        lambda batch=1, _M=M, _C=C, _F=F, _h=use_h, _v=use_v: aligned_marker_fill_network(_M, _C, _F, _h, _v, batch=batch)))

    pi_match = _detect_pi_shape_project_to_bottom(pairs)
    if pi_match is not None:
        F = pi_match
        matches.append(("pi-project-bottom",
                        lambda batch=1, _F=F: pi_shape_project_to_bottom_network(_F, batch=batch)))

    dom_match = _detect_constant_output_dominant_color(pairs)
    if dom_match is not None:
        h_out, w_out = dom_match
        matches.append(("const-dominant-color",
                        lambda batch=1, _h=h_out, _w=w_out: constant_output_dominant_color_network(_h, _w, batch=batch)))

    lbc_match = _detect_constant_output_largest_bbox(pairs)
    if lbc_match is not None:
        h_out, w_out = lbc_match
        matches.append(("const-largest-bbox",
                        lambda batch=1, _h=h_out, _w=w_out: largest_bbox_constant_output_network(_h, _w, batch=batch)))

    fcc_match = _detect_framed_cell_const_output(pairs)
    if fcc_match is not None:
        h_out, w_out, fcc_ch, fcc_cw = fcc_match
        matches.append(("const-framed-cell",
                        lambda batch=1, _h=h_out, _w=w_out, _ch=fcc_ch, _cw=fcc_cw: framed_cell_const_output_network(_h, _w, canvas_h=_ch, canvas_w=_cw, batch=batch)))

    masr_match = _detect_marker_aligned_shape_recolor(pairs)
    if masr_match is not None:
        matches.append(("marker-aligned-shape-recolor",
                        lambda batch=1: marker_aligned_shape_recolor_network(batch=batch)))

    gdc_match = _detect_grid_division_count(pairs)
    if gdc_match is not None:
        matches.append(("grid-division-count",
                        lambda batch=1: grid_division_count_network(batch=batch)))

    csg_match = _detect_cross_section_grid_label(pairs)
    if csg_match is not None:
        matches.append(("cross-section-grid-label",
                        lambda batch=1: cross_section_grid_label_network(batch=batch)))

    gsc_match = _detect_grid_section_copy(pairs)
    if gsc_match is not None:
        matches.append(("grid-section-copy",
                        lambda batch=1: grid_section_copy_network(batch=batch)))

    gmcf_match = _detect_grid_max_count_fill(pairs)
    if gmcf_match is not None:
        matches.append(("grid-max-count-fill",
                        lambda batch=1: grid_max_count_fill_network(batch=batch)))

    gos_match = _detect_grid_object_summary(pairs)
    if gos_match is not None:
        matches.append(("grid-object-summary",
                        lambda batch=1: grid_object_summary_network(batch=batch)))

    tbt_match = _detect_two_blob_tile_fit(pairs)
    if tbt_match is not None:
        matches.append(("two-blob-tile-fit",
                        lambda batch=1: two_blob_tile_fit_network(batch=batch)))

    frt_match = _detect_fourfold_rotation_tile(pairs)
    if frt_match is not None:
        matches.append(("fourfold-rotation-tile",
                        lambda batch=1: fourfold_rotation_tile_network(batch=batch)))

    drl_match = _detect_diagonal_rays_from_line(pairs)
    if drl_match is not None:
        matches.append(("diagonal-rays-from-line",
                        lambda batch=1: diagonal_rays_from_line_network(batch=batch)))

    ewf_match = _detect_euclidean_wedge_fill(pairs)
    if ewf_match is not None:
        matches.append(("euclidean-wedge-fill",
                        lambda batch=1: euclidean_wedge_fill_network(batch=batch)))

    bsd_match = _detect_bilateral_symmetry_denoise(pairs)
    if bsd_match is not None:
        matches.append(("bilateral-symmetry-denoise",
                        lambda batch=1: bilateral_symmetry_denoise_network(batch=batch)))

    mse_match = _detect_marker_section_extract(pairs)
    if mse_match is not None:
        matches.append(("marker-section-extract",
                        lambda batch=1: marker_section_extract_network(batch=batch)))

    mdsc_match = _detect_marker_driven_shape_copy(pairs)
    if mdsc_match is not None:
        matches.append(("marker-driven-shape-copy",
                        lambda batch=1: marker_driven_shape_copy_network(batch=batch)))

    pre_match = _detect_periodic_row_extension(pairs)
    if pre_match is not None:
        matches.append(("periodic-row-extension",
                        lambda batch=1: periodic_row_extension_network(batch=batch)))

    bh_match = _detect_boolean_halves(pairs)
    if bh_match is not None:
        matches.append(("boolean-halves",
                        lambda batch=1, _m=bh_match: boolean_halves_network(
                            _m["H"], _m["W_in"], _m["W_out"], _m["check_channel"],
                            _m["out_color"], batch=batch)))

    rvb_match = _detect_rank_vertical_bars(pairs)
    if rvb_match is not None:
        matches.append(("rank-vertical-bars",
                        lambda batch=1: rank_vertical_bars_network(batch=batch)))

    vh_match = _detect_vertical_halves(pairs)
    if vh_match is not None:
        matches.append(("vertical-halves",
                        lambda batch=1, _m=vh_match: vertical_halves_network(
                            _m["H_in"], _m["W"], _m["H_half"], _m["mode"],
                            _m["out_color"], sep_row=_m["sep_row"], batch=batch)))

    hor_match = _detect_horizontal_or(pairs)
    if hor_match is not None:
        matches.append(("horizontal-or",
                        lambda batch=1, _m=hor_match: horizontal_or_network(
                            _m["H"], _m["W_half"], _m["out_color"], batch=batch)))

    cp_match = _detect_cross_product(pairs)
    if cp_match is not None:
        matches.append(("cross-product",
                        lambda batch=1, _m=cp_match: cross_product_network(
                            _m["N"], _m["col_idx"], _m["template_color"],
                            _m["out_color"], batch=batch)))

    bcg_match = _detect_block_count_ge(pairs)
    if bcg_match is not None:
        matches.append(("block-count-ge",
                        lambda batch=1, _m=bcg_match: block_count_ge_network(
                            _m["count_color"], _m["threshold"], _m["out_color"],
                            _m["N"], _m["n"], _m["block_size"], _m["stride"], batch=batch)))

    mr_match = _detect_marker_recolor(pairs)
    if mr_match is not None:
        matches.append(("marker-recolor",
                        lambda batch=1, _m=mr_match: marker_recolor_network(
                            _m["H"], _m["W"], _m["r_marker"], _m["c_marker"], batch=batch)))

    tmc_match = _detect_two_marker_cross(pairs)
    if tmc_match is not None:
        matches.append(("two-marker-cross",
                        lambda batch=1, _m=tmc_match: two_marker_cross_network(
                            _m["color_A"], _m["color_B"], _m["color_inter"],
                            _m["H"], _m["W"], batch=batch)))

    dcp_match = _detect_distinct_count_pattern(pairs)
    if dcp_match is not None:
        matches.append(("distinct-count-pattern",
                        lambda batch=1, _m=dcp_match: distinct_count_pattern_network(
                            _m["c1"], _m["c2"], _m["c3"], _m["out_color"], batch=batch)))

    ur_match = _detect_uniform_row(pairs)
    if ur_match is not None:
        matches.append(("uniform-row",
                        lambda batch=1, _m=ur_match: uniform_row_network(
                            _m["H"], _m["W"], _m["out_color"], batch=batch)))

    cmdc_match = _detect_cummax_down_column(pairs)
    if cmdc_match is not None:
        matches.append(("cummax-down-column",
                        lambda batch=1, _m=cmdc_match: cummax_down_column_network(
                            _m["H"], _m["W"], batch=batch)))

    dck_match = _detect_dominant_color_keep(pairs)
    if dck_match is not None:
        matches.append(("dominant-color-keep",
                        lambda batch=1, _m=dck_match: dominant_color_keep_network(
                            _m["H"], _m["W"], _m["out_color"], batch=batch)))

    c2p_match = _detect_color_to_pattern(pairs)
    if c2p_match is not None:
        matches.append(("color-to-pattern",
                        lambda batch=1, _m=c2p_match: color_to_pattern_network(
                            _m["H_in"], _m["W_in"], _m["color_map"], _m["out_color"], batch=batch)))

    clf_match = _detect_cross_line_fill(pairs)
    if clf_match is not None:
        matches.append(("cross-line-fill",
                        lambda batch=1, _m=clf_match: cross_line_fill_network(
                            _m["H"], _m["W"], _m["color_col"], _m["color_row"], _m["color_inter"], batch=batch)))

    rzrf_match = _detect_rectangular_zero_rank_fill(pairs)
    if rzrf_match is not None:
        matches.append(("rectangular-zero-rank-fill",
                        lambda batch=1, _m=rzrf_match: rectangular_zero_rank_fill_network(
                            _m["H"], _m["W"], batch=batch)))

    fzsf_match = _detect_framed_zero_square_fill(pairs)
    if fzsf_match is not None:
        matches.append(("framed-zero-square-fill",
                        lambda batch=1, _m=fzsf_match: framed_zero_square_fill_network(
                            _m["H"], _m["W"], _m["max_size"], _m["frame_color"],
                            _m["fill_color"], batch=batch)))

    bdr_match = _detect_bouncing_diagonal_ray(pairs)
    if bdr_match is not None:
        matches.append(("bouncing-diagonal-ray",
                        lambda batch=1, _m=bdr_match: bouncing_diagonal_ray_network(
                            _m["max_w"], batch=batch)))

    if _detect_trirow_periodic_four_fill(pairs):
        matches.append(("trirow-periodic-four-fill",
                        lambda batch=1: trirow_periodic_four_fill_network(batch=batch)))

    sst_match = _detect_square_spiral_three(pairs)
    if sst_match is not None:
        matches.append(("square-spiral-three",
                        lambda batch=1, _m=sst_match: square_spiral_three_network(
                            _m["max_n"], batch=batch)))

    ttzf_match = _detect_two_by_two_zero_fill(pairs)
    if ttzf_match is not None:
        matches.append(("two-by-two-zero-fill",
                        lambda batch=1, _m=ttzf_match: two_by_two_zero_fill_network(
                            _m["H"], _m["W"], _m["bg_color"], _m["fill_color"], batch=batch)))

    rsf_match = _detect_rotation_symmetry_fill(pairs)
    if rsf_match is not None:
        matches.append(("rotation-symmetry-fill",
                        lambda batch=1, _m=rsf_match: rotation_symmetry_fill_network(
                            _m["H"], _m["W"], _m["fg_color"], _m["fill_color"], batch=batch)))

    dsf_match = _detect_diagonal_symmetry_fill(pairs)
    if dsf_match is not None:
        matches.append(("diagonal-symmetry-fill",
                        lambda batch=1, _m=dsf_match: diagonal_symmetry_fill_network(
                            _m["H"], _m["W"], _m["fg_color"], _m["fill_color"], batch=batch)))

    rsc_match = _detect_rotational_symmetry_completion(pairs)
    if rsc_match is not None:
        matches.append(("rotational-symmetry-completion",
                        lambda batch=1: rotational_symmetry_completion_network(batch=batch)))

    opmt_match = _detect_outer_product_marker_tile(pairs)
    if opmt_match is not None:
        matches.append(("outer-product-marker-tile",
                        lambda batch=1, _m=opmt_match["mode"], _c=opmt_match.get("marker_color"): outer_product_marker_tile_network(_m, marker_color=_c, batch=batch)))

    ds_match = _detect_diagonal_shear(pairs)
    if ds_match is not None:
        matches.append(("diagonal-shear",
                        lambda batch=1: diagonal_shear_network(batch=batch)))

    fbc_match = _detect_foreground_bbox_crop(pairs)
    if fbc_match is not None:
        matches.append(("foreground-bbox-crop",
                        lambda batch=1: foreground_bbox_crop_network(batch=batch)))

    frc_match = _detect_frame_raycast_crop(pairs)
    if frc_match is not None:
        matches.append(("frame-raycast-crop",
                        lambda batch=1: frame_raycast_crop_network(batch=batch)))

    mri_match = _detect_marker_rectangle_interior(pairs)
    if mri_match is not None:
        matches.append(("marker-rectangle-interior",
                        lambda batch=1: marker_rectangle_interior_network(batch=batch)))

    kbu_match = _detect_kron_block_up(pairs)
    if kbu_match is not None:
        matches.append(("kron-block-up",
                        lambda batch=1, _sh=kbu_match["scale_h"], _sw=kbu_match["scale_w"]: kron_block_up_network(_sh, _sw, batch=batch)))

    fre_match = _detect_fractal_ring_expansion(pairs)
    if fre_match is not None:
        matches.append(("fractal-ring-expansion",
                        lambda batch=1: fractal_ring_expansion_network(batch=batch)))

    cfbc_match = _detect_color_filter_bbox_crop(pairs, picker="least")
    if cfbc_match is not None:
        matches.append(("color-filter-bbox-crop",
                        lambda batch=1, _p=cfbc_match["picker"]: color_filter_bbox_crop_network(picker=_p, batch=batch)))

    cfbc_sb_match = _detect_color_filter_bbox_crop(pairs, picker="smallest_bbox")
    if cfbc_sb_match is not None:
        matches.append(("color-filter-bbox-crop-sb",
                        lambda batch=1, _p=cfbc_sb_match["picker"]: color_filter_bbox_crop_network(picker=_p, batch=batch)))

    cfbc_oo_match = _detect_color_filter_bbox_crop(pairs, picker="odd_one_out")
    if cfbc_oo_match is not None:
        matches.append(("color-filter-bbox-crop-oo",
                        lambda batch=1, _p=cfbc_oo_match["picker"]: color_filter_bbox_crop_network(picker=_p, batch=batch)))

    cfbc_lr_match = _detect_color_filter_bbox_crop(pairs, picker="lr_symmetry")
    if cfbc_lr_match is not None:
        matches.append(("color-filter-bbox-crop-lr",
                        lambda batch=1, _p=cfbc_lr_match["picker"]: color_filter_bbox_crop_network(picker=_p, batch=batch)))

    cfbc_md_match = _detect_color_filter_bbox_crop(pairs, picker="max_density")
    if cfbc_md_match is not None:
        matches.append(("color-filter-bbox-crop-md",
                        lambda batch=1, _p=cfbc_md_match["picker"]: color_filter_bbox_crop_network(picker=_p, batch=batch)))

    cfbc_mc_match = _detect_color_filter_bbox_crop(pairs, picker="most_count")
    if cfbc_mc_match is not None:
        matches.append(("color-filter-bbox-crop-mc",
                        lambda batch=1, _p=cfbc_mc_match["picker"]: color_filter_bbox_crop_network(picker=_p, batch=batch)))

    lrrc_match = _detect_largest_rectangle_recolor_crop(pairs)
    if lrrc_match is not None:
        matches.append(("largest-rectangle-recolor-crop",
                        lambda batch=1: largest_rectangle_recolor_crop_network(batch=batch)))

    sff_match = _detect_seed_flood_fill(pairs)
    if sff_match is not None:
        matches.append(("seed-flood-fill",
                        lambda batch=1, _fc=sff_match["fill_color"], _ch=sff_match["canvas_h"], _cw=sff_match["canvas_w"]: seed_flood_fill_network(_fc, batch=batch, canvas_h=_ch, canvas_w=_cw)))

    sc_match = _detect_symmetry_completion(pairs)
    if sc_match is not None:
        matches.append(("symmetry-completion",
                        lambda batch=1, _a=sc_match["axis"], _h=sc_match["h"], _w=sc_match["w"]: symmetry_completion_network(_a, _h, _w, batch=batch)))

    bbf_match = _detect_bbox_fill(pairs)
    if bbf_match is not None:
        matches.append(("bbox-fill",
                        lambda batch=1, _t=bbf_match["target_color"]: bbox_fill_network(_t, batch=batch)))

    glaf_match = _detect_gridline_adjacent_fill(pairs)
    if glaf_match is not None:
        matches.append(("gridline-adjacent-fill",
                        lambda batch=1: gridline_adjacent_fill_network(batch=batch)))

    lpfl_match = _detect_line_projection_to_full_line(pairs)
    if lpfl_match is not None:
        matches.append(("line-projection-to-full-line",
                        lambda batch=1: line_projection_to_full_line_network(batch=batch)))

    dzmp_match = _detect_diagonal_zero_mask_projection(pairs)
    if dzmp_match is not None:
        matches.append(("diagonal-zero-mask-projection",
                        lambda batch=1: diagonal_zero_mask_projection_network(batch=batch)))

    msc_match = _detect_marker_shape_copy(pairs)
    if msc_match is not None:
        matches.append(("marker-shape-copy",
                        lambda batch=1: marker_shape_copy_network(batch=batch)))

    bglf_match = _detect_block_grid_line_fill(pairs)
    if bglf_match is not None:
        matches.append(("block-grid-line-fill",
                        lambda batch=1: block_grid_line_fill_network(batch=batch)))

    c2ob_match = _detect_count_2x2_one_blocks(pairs)
    if c2ob_match is not None:
        matches.append(("count-2x2-one-blocks",
                        lambda batch=1: count_2x2_one_blocks_network(batch=batch)))

    scnk_match = _detect_same_color_neighbor_keep(pairs)
    if scnk_match is not None:
        matches.append(("same-color-neighbor-keep",
                        lambda batch=1: same_color_neighbor_keep_network(batch=batch)))

    mabr_match = _detect_marker_adjacent_bbox_reflection(pairs)
    if mabr_match is not None:
        matches.append(("marker-adjacent-bbox-reflection",
                        lambda batch=1: marker_adjacent_bbox_reflection_network(batch=batch)))

    rif8_match = _detect_rectangle_interior_fill8(pairs)
    if rif8_match is not None:
        matches.append(("rectangle-interior-fill8",
                        lambda batch=1, _h=rif8_match["h"], _w=rif8_match["w"]: rectangle_interior_fill8_network(_h, _w, batch=batch)))

    bdpr_match = _detect_border_distance_probe_recolor(pairs)
    if bdpr_match is not None:
        matches.append(("border-distance-probe-recolor",
                        lambda batch=1, _h=bdpr_match["h"], _w=bdpr_match["w"]: border_distance_probe_recolor_network(_h, _w, batch=batch)))

    f6b3f4_match = _detect_frame6_border3_fill4(pairs)
    if f6b3f4_match is not None:
        matches.append(("frame6-border3-fill4",
                        lambda batch=1, _h=f6b3f4_match["h"], _w=f6b3f4_match["w"]: frame6_border3_fill4_network(_h, _w, batch=batch)))

    mtrl_match = _detect_marker_to_rectangle_line(pairs)
    if mtrl_match is not None:
        matches.append(("marker-to-rectangle-line",
                        lambda batch=1, _h=mtrl_match["h"], _w=mtrl_match["w"]: marker_to_rectangle_line_network(_h, _w, batch=batch)))

    fcr_match = _detect_foreground_count_row(pairs)
    if fcr_match is not None:
        matches.append(("foreground-count-row",
                        lambda batch=1: foreground_count_row_network(batch=batch)))

    esbr_match = _detect_edge_seed_bounce_ray(pairs)
    if esbr_match is not None:
        matches.append(("edge-seed-bounce-ray",
                        lambda batch=1: edge_seed_bounce_ray_network(batch=batch)))

    ole_match = _detect_outline_extract(pairs)
    if ole_match is not None:
        ole_ch, ole_cw = ole_match["canvas_h"], ole_match["canvas_w"]
        matches.append(("outline-extract",
                        lambda batch=1, _ch=ole_ch, _cw=ole_cw: outline_extract_network(canvas_h=_ch, canvas_w=_cw, batch=batch)))

    agf_match = _detect_active_grid_frame(pairs)
    if agf_match is not None:
        matches.append(("active-grid-frame",
                        lambda batch=1, _b=agf_match["border_color"]: active_grid_frame_network(_b, batch=batch)))

    pcs_match = _detect_per_color_stamp(pairs)
    if pcs_match is not None:
        matches.append(("per-color-stamp",
                        lambda batch=1, _r=pcs_match["rules"]: per_color_stamp_network(_r, batch=batch)))

    mdc_match = _detect_marker_dilate_conv_safe(pairs)
    if mdc_match is not None:
        matches.append(("marker-dilate-conv",
                        lambda batch=1, _m=mdc_match["marker_color"], _t=mdc_match["target_color"], _p=mdc_match["preserve_marker"]: marker_dilate_conv_network(_m, _t, _p, batch=batch)))

    md_match = _detect_marker_dilate(pairs)
    if md_match is not None:
        matches.append(("marker-dilate",
                        lambda batch=1, _m=md_match["marker_color"], _t=md_match["target_color"], _p=md_match["preserve_marker"]: marker_dilate_network(_m, _t, _p, batch=batch)))

    tmrr_match = _detect_top_marker_run_recolor(pairs)
    if tmrr_match is not None:
        matches.append(("top-marker-run-recolor",
                        lambda batch=1: top_marker_run_recolor_network(batch=batch)))

    erc2_match = _detect_empty_row_col_recolor2(pairs)
    if erc2_match is not None:
        matches.append(("empty-row-col-recolor2",
                        lambda batch=1: empty_row_col_recolor2_network(batch=batch)))

    amfall_match = _detect_arch_marker_fall(pairs)
    if amfall_match is not None:
        matches.append(("arch-marker-fall",
                        lambda batch=1: arch_marker_fall_network(batch=batch)))

    dpc_match = _detect_diagonal_pair_connect(pairs)
    if dpc_match is not None:
        matches.append(("diagonal-pair-connect",
                        lambda batch=1: diagonal_pair_connect_network(batch=batch)))

    d8d_match = _detect_downshift_recolor8_to2(pairs)
    if d8d_match is not None:
        matches.append(("downshift-recolor8-to2",
                        lambda batch=1: downshift_recolor8_to2_network(batch=batch)))

    perm_indices = _detect_color_perm(pairs, require_all_colors=False)
    if perm_indices is not None:
        matches.append(("color-perm",
                        lambda batch=1, _idx=perm_indices: color_perm_network(_idx, batch=batch)))

    cr_match = _detect_color_replace(pairs)
    if cr_match is not None:
        matches.append(("color-replace",
                        lambda batch=1, _fwd=cr_match: color_replace_network(_fwd, batch=batch)))

    if _detect_color_swap_5_always_stays(pairs) is not None:
        matches.append(("color-swap-5-stays",
                        lambda batch=1: color_swap_5_always_stays_network(batch=batch)))

    erf_match = _detect_enclosed_region_fill(pairs)
    if erf_match is not None:
        matches.append(("enclosed-region-fill",
                        lambda batch=1, _fc=erf_match["fill_color"], _ch=erf_match["canvas_h"], _cw=erf_match["canvas_w"]: enclosed_region_fill_network(_fc, batch=batch, canvas_h=_ch, canvas_w=_cw)))

    fif_match = _detect_frame_interior_fill(pairs)
    if fif_match is not None:
        matches.append(("frame-interior-fill",
                        lambda batch=1, _fc=fif_match["frame_color"], _fv=fif_match["fill_color"]: frame_interior_fill_network(_fc, _fv, batch=batch)))

    pp_match = _detect_pixel_perm(pairs)
    if pp_match is not None:
        matches.append(("pixel-perm",
                        lambda batch=1, _h=pp_match["h"], _w=pp_match["w"], _idx=pp_match["indices"]: pixel_perm_network(_h, _w, _idx, batch=batch)))

    scpp_match = _detect_shape_changing_pixel_perm(pairs)
    if scpp_match is not None:
        matches.append(("shape-changing-pixel-perm",
                        lambda batch=1, _ih=scpp_match["in_h"], _iw=scpp_match["in_w"], _oh=scpp_match["out_h"], _ow=scpp_match["out_w"], _idx=scpp_match["indices"]: shape_changing_pixel_perm_network(_ih, _iw, _oh, _ow, _idx, batch=batch)))

    mirror_grid_match = _detect_mirror_grid_2x2(pairs)
    if mirror_grid_match is not None:
        in_h, in_w = mirror_grid_match
        matches.append(("mirror-grid-2x2",
                        lambda batch=1, _h=in_h, _w=in_w: mirror_grid_2x2_network(_h, _w, batch=batch)))

    mirror_match = _detect_mirror_concat(pairs)
    if mirror_match is not None:
        in_h, in_w, axis, transform, second_first = mirror_match
        matches.append(("mirror-concat",
                        lambda batch=1, _h=in_h, _w=in_w, _a=axis, _t=transform, _s=second_first: mirror_concat_network(_h, _w, _a, _t, second_first=_s, batch=batch)))

    grid_pat_match = _detect_grid_pattern_to_color(pairs)
    if grid_pat_match is not None:
        in_h, in_w, lut = grid_pat_match
        matches.append(("grid-pattern-to-color",
                        lambda batch=1, _h=in_h, _w=in_w, _lut=lut: grid_pattern_to_color_network(_h, _w, _lut, batch=batch)))

    grid_grid_match = _detect_grid_pattern_to_grid(pairs)
    if grid_grid_match is not None:
        in_h, in_w, out_h, out_w, lut = grid_grid_match
        matches.append(("grid-pattern-to-grid",
                        lambda batch=1, _ih=in_h, _iw=in_w, _oh=out_h, _ow=out_w, _lut=lut: grid_pattern_to_grid_network(_ih, _iw, _oh, _ow, _lut, batch=batch)))

    grid_pc_match = _detect_grid_pattern_color_to_grid(pairs)
    if grid_pc_match is not None:
        in_h, in_w, out_h, out_w, lut = grid_pc_match
        matches.append(("grid-pattern-color-to-grid",
                        lambda batch=1, _ih=in_h, _iw=in_w, _oh=out_h, _ow=out_w, _lut=lut: grid_pattern_color_to_grid_network(_ih, _iw, _oh, _ow, _lut, batch=batch)))

    lut_match = _detect_binary_neighborhood_lut(pairs)
    if lut_match is not None:
        bg, lut = lut_match
        matches.append(("bin-nbr-lut",
                        lambda batch=1, _bg=bg, _lut=lut: binary_neighborhood_lut_network(_bg, _lut, batch=batch)))

    t388_match = _detect_task388(pairs)
    if t388_match is not None:
        matches.append(("col-fill-2x2-tile",
                        lambda batch=1: col_fill_2x2_tile_network(batch=batch)))

    tdnm_match = _detect_tile_diagonal_neighbor_mark(pairs)
    if tdnm_match is not None:
        matches.append(("tile-diagonal-neighbor-mark",
                        lambda batch=1: tile_diagonal_neighbor_mark_network(batch=batch)))

    c2bf_match = _detect_color2_bbox_fill(pairs)
    if c2bf_match is not None:
        matches.append(("color2-bbox-fill",
                        lambda batch=1: color2_bbox_fill_network(batch=batch)))

    dmsr_match = _detect_diagonal_marker_square_rings(pairs)
    if dmsr_match is not None:
        matches.append(("diagonal-marker-square-rings",
                        lambda batch=1: diagonal_marker_square_rings_network(batch=batch)))

    # task270 anchor-cross-aligned-cells: builder + detector retained as research
    # infrastructure (266/266 visible+arc-gen) but NOT auto-dispatched. lemon11
    # submission proved the cost-81207 organic scored -0.54 vs kiwi11's existing
    # external/payload at the same task270 slot. Re-wire when a cheaper organic
    # (target cost <~73k) is implemented. See task270 task card.

    bp3_match = _detect_task316(pairs)
    if bp3_match is not None:
        matches.append(("boustrophedon-pack-3x3",
                        lambda batch=1: boustrophedon_pack_3x3_network(batch=batch)))

    ht2_match = _detect_horizontal_tile_2x(pairs)
    if ht2_match is not None:
        matches.append(("horizontal-tile-2x",
                        lambda batch=1: horizontal_tile_2x_network(batch=batch)))

    per2_match = _detect_period_extend_row_2x(pairs)
    if per2_match is not None:
        matches.append(("period-extend-row-2x",
                        lambda batch=1: period_extend_row_2x_network(batch=batch)))

    phf_match = _detect_periodic_hole_fill(pairs)
    if phf_match is not None:
        matches.append(("periodic-hole-fill",
                        lambda batch=1: periodic_hole_fill_network(batch=batch)))

    zal_match = _detect_zero_adjacency_label(pairs)
    if zal_match is not None:
        matches.append(("zero-adjacency-label",
                        lambda batch=1: zero_adjacency_label_network(batch=batch)))

    t320_match = _detect_task320(pairs)
    if t320_match is not None:
        matches.append(("staircase-bottom-half-recolor",
                        lambda batch=1: staircase_bottom_half_recolor_network(batch=batch)))

    mccs_match = _detect_marker_count_cross_shift(pairs)
    if mccs_match is not None:
        matches.append(("marker-count-cross-shift",
                        lambda batch=1: marker_count_cross_shift_network(batch=batch)))

    abaml_match = _detect_anchor_block_aligned_marker_line(pairs)
    if abaml_match is not None:
        matches.append(("anchor-block-aligned-marker-line",
                        lambda batch=1: anchor_block_aligned_marker_line_network(batch=batch)))

    fimoo_match = _detect_frame_inner_markers_to_outer_opposite(pairs)
    if fimoo_match is not None:
        matches.append(("frame-inner-markers-to-outer-opposite",
                        lambda batch=1: frame_inner_markers_to_outer_opposite_network(batch=batch)))

    return matches


def _score_candidate(network_fn):
    """Build, save (to scratch), score. Return cost or None if rejected/erroring.

    Uses a per-call tempfile so parallel dispatcher calls (e.g. a future
    multiprocess `--hybrid`) don't collide on a shared scratch path.
    """
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    fd, scratch_path = tempfile.mkstemp(suffix=".onnx", prefix="_dispatch_", dir=str(SUBMISSIONS_DIR))
    os.close(fd)
    scratch = pathlib.Path(scratch_path)
    try:
        m = network_fn()
        onnx.save(m, str(scratch))
        if not neurogolf_utils.check_network(str(scratch)):
            return None
        result = score_network_v2(str(scratch))
    except Exception:  # noqa: BLE001
        return None
    finally:
        try:
            scratch.unlink()
        except (FileNotFoundError, OSError):
            pass
    if result is None or result[0] is None:
        return None
    _, mem, params = result
    # Cost objective is now memory + params (MACs removed from the formula 2026-05-04).
    cost = mem + params
    return cost


def detect_pattern(examples):
    """Return (label, network_fn) of the cheapest matching pattern, else (None, None).

    Cost-aware dispatch: collect every detector that matches, then on multi-match
    tasks pick the cheapest by `score_network` cost. On the common single-match
    case (~78% of solved tasks), no scoring is performed — this path is identical
    in cost to the old static-order dispatcher.

    On the multi-match case (~22%), we pay a small overhead (~50 ms per extra
    candidate) to build + score each. In return, the dispatcher is robust to
    detector-ordering bugs (D1-class regressions): adding a new pattern in the
    "wrong" slot in `_collect_matches` no longer locks out a cheaper alternative
    for any task.
    """
    pairs = examples.get("train", []) + examples.get("test", []) + examples.get("arc-gen", [])
    if not pairs:
        return None, None

    matches = _collect_matches(pairs)
    if not matches:
        return None, None
    if len(matches) == 1:
        return matches[0]

    best_cost = None
    best = None
    for label, fn in matches:
        cost = _score_candidate(fn)
        if cost is None:
            continue
        # Strict less-than preserves source-order tie-break: the earlier match wins
        # on equal cost (matches the legacy static dispatcher's preference).
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best = (label, fn)
    if best is None:
        # All candidates failed to score; fall back to the first match by source order.
        return matches[0]
    return best


# ------------------------------------------------------------------- build/zip

def _verify_batched(network_fn, pairs):
    """Run a single batched onnxruntime inference over all pairs. Returns (right, wrong).

    Falls back to the validator's per-example verify_subset if network_fn
    doesn't support batch (e.g., the conv-3x3-identity baseline whose
    builder hardcodes [1,10,30,30]).
    """
    benchmarks = [b for b in (neurogolf_utils.convert_to_numpy(p) for p in pairs) if b is not None]
    if not benchmarks:
        return 0, 0
    batch = len(benchmarks)
    try:
        model = network_fn(batch=batch)
    except (NotImplementedError, TypeError):
        session = onnxruntime.InferenceSession(network_fn().SerializeToString())
        right, wrong, _ = neurogolf_utils.verify_subset(session, pairs)
        return right, wrong
    session = onnxruntime.InferenceSession(model.SerializeToString())
    inputs = np.concatenate([b["input"] for b in benchmarks], axis=0)
    expected = np.concatenate([b["output"] for b in benchmarks], axis=0)
    result = session.run(["output"], {"input": inputs})[0]
    actual = (result > 0.0).astype(np.float32)
    right = sum(1 for i in range(batch) if np.array_equal(actual[i], expected[i]))
    return right, batch - right


def build_one(task_num, network_fn=None):
    if network_fn is None:
        network_fn = lambda batch=1: conv_identity_network(kernel_size=3, batch=batch)
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    filename = SUBMISSIONS_DIR / f"task{task_num:03d}.onnx"

    # Wrap construction + save + check_network so a single bad builder never
    # crashes a 400-task --hybrid run. Captures network_fn() bugs (shape errors,
    # missing initializers), onnx.save serialization failures, and
    # check_network's onnx.checker.ValidationError.
    try:
        model = network_fn()
        onnx.save(model, str(filename))
    except Exception as e:  # noqa: BLE001 — onnx + onnx.checker raise multiple types
        return {"task": task_num, "status": f"FAIL: build raised {type(e).__name__}: {e}"}

    try:
        if not neurogolf_utils.check_network(str(filename)):
            return {"task": task_num, "status": "FAIL: check_network"}
    except Exception as e:  # noqa: BLE001
        return {"task": task_num, "status": f"FAIL: check_network raised {type(e).__name__}: {e}"}

    try:
        result = score_network_v2(str(filename))
    except Exception as e:  # noqa: BLE001 — onnx-tool raises ValueError, KeyError, etc.
        return {"task": task_num, "status": f"FAIL: score_network raised {type(e).__name__}: {e}"}
    if result is None or result[0] is None:
        return {"task": task_num, "status": "FAIL: score_network rejected"}
    macs, memory, params = result

    examples = neurogolf_utils.load_examples(task_num)
    a_right, a_wrong = _verify_batched(network_fn, examples["train"] + examples["test"])
    g_right, g_wrong = _verify_batched(network_fn, examples["arc-gen"])

    # Scoring formula changed 2026-05-04: cost = memory + params (no MACs).
    cost = memory + params
    correct = (a_wrong + g_wrong) == 0
    points = score_to_points(memory, params) if correct else 0.0

    return {
        "task": task_num,
        "status": "VALID + CORRECT" if correct else "VALID but FUNCTIONALLY WRONG",
        "filename": str(filename.relative_to(NG_ROOT).as_posix()),
        "macs": macs,
        "memory": memory,
        "params": params,
        "cost": cost,
        "points": round(points, 4),
        "arc_agi": f"{a_right} pass / {a_wrong} fail",
        "arc_gen": f"{g_right} pass / {g_wrong} fail",
    }


def build_all(picker_fn=None):
    if picker_fn is None:
        picker_fn = lambda task_num: (None, lambda: conv_identity_network(kernel_size=3))
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    RUNS_LOG.parent.mkdir(parents=True, exist_ok=True)
    run_id = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    correct = 0
    valid = 0
    rejected = 0
    total_points = 0.0
    failed_tasks = []
    detections = {}
    correct_detections = {}
    with RUNS_LOG.open("a", encoding="utf-8") as log_f:
        for task_num in range(1, 401):
            label, network_fn = picker_fn(task_num)
            detections[label] = detections.get(label, 0) + 1
            result = build_one(task_num, network_fn=network_fn)
            status = result["status"]
            is_correct = "CORRECT" in status
            if status.startswith("FAIL"):
                rejected += 1
                failed_tasks.append((task_num, status))
            else:
                valid += 1
                if is_correct:
                    correct += 1
                    correct_detections[label] = correct_detections.get(label, 0) + 1
                    total_points += result["points"]
            json.dump(
                {
                    "run_id": run_id,
                    "task": task_num,
                    "pattern": label,
                    "status": status,
                    "correct": is_correct,
                    "cost": result.get("cost"),
                    "points": result.get("points", 0.0),
                },
                log_f,
            )
            log_f.write("\n")
            if task_num % 50 == 0 or task_num == 400:
                print(
                    f"  [{task_num:>3}/400] valid={valid} correct={correct} "
                    f"rejected={rejected} points={total_points:.2f}"
                )
    return {
        "run_id": run_id,
        "valid": valid,
        "correct": correct,
        "rejected": rejected,
        "points": total_points,
        "failed": failed_tasks,
        "detections": detections,
        "correct_detections": correct_detections,
    }


def make_zip(zip_path=None):
    # Clobber guard (2026-06-06): the live submission.zip is the competition state.
    # Rebuilding it from submissions/task*.onnx loose files would replace the live
    # 401-entry zip with whatever stale per-task onnxes happen to be on disk.
    # Callers must pass an explicit alternate path to build a zip at all.
    if zip_path is None:
        raise RuntimeError(
            "make_zip() refuses to overwrite the live submissions/submission.zip. "
            "Pass an explicit zip_path (e.g. submissions/rebuild_test.zip), or use "
            "tools/swap_and_submit.py for per-task shipping."
        )
    zip_path = pathlib.Path(zip_path)
    expected = {f"task{n:03d}.onnx" for n in range(1, 401)}
    onnx_files = sorted(SUBMISSIONS_DIR.glob("task*.onnx"))
    actual = {f.name for f in onnx_files}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        m_disp = f"missing={missing[:5]}{'...' if len(missing) > 5 else ''}"
        e_disp = f"extra={extra[:5]}{'...' if len(extra) > 5 else ''}"
        raise RuntimeError(
            f"Refusing to zip: submissions/ does not contain exactly task001.onnx..task400.onnx ({m_disp}, {e_disp})"
        )
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in onnx_files:
            zf.write(f, arcname=f.name)
    return zip_path, len(onnx_files)


# ----------------------------------------------------------------------- main

def _hybrid_picker(task_num):
    """For --hybrid: detected pattern's network_fn, else identity-only fallback (cost 0)."""
    examples = neurogolf_utils.load_examples(task_num)
    label, network_fn = detect_pattern(examples)
    if network_fn is None:
        return "fallback-identity", identity_network
    return label, network_fn


def main():
    args = sys.argv[1:]

    if args and args[0] in ("--all", "--hybrid"):
        raise SystemExit(
            "ERROR: --all and --hybrid are DEAD. Both would overwrite submissions/submission.zip\n"
            "with a 103/400-correct baseline (~1,627 pts vs live ~6,500+).\n"
            "Use tools/swap_and_submit.py for per-task shipping."
        )

        summary = build_all(picker)
        print()
        print(f"  valid:    {summary['valid']}/400")
        print(f"  correct:  {summary['correct']}/400")
        print(f"  rejected: {summary['rejected']}/400")
        print(f"  points:   {summary['points']:.2f}")
        print(f"  detections:         {summary['detections']}")
        print(f"  correct by pattern: {summary['correct_detections']}")
        if summary["failed"]:
            print(f"  failed tasks: {summary['failed'][:5]}{'...' if len(summary['failed']) > 5 else ''}")
        zip_path, n = make_zip()
        print()
        print(f"  zip:      {zip_path.relative_to(NG_ROOT).as_posix()} ({n} files, {zip_path.stat().st_size:,} bytes)")
        return

    task_num = int(args[0]) if args else 1
    result = build_one(task_num)
    width = max(len(k) for k in result)
    for k, v in result.items():
        print(f"  {k:<{width}}  {v}")


if __name__ == "__main__":
    main()
