"""verify_against_live — validator-faithful live-equivalence checking.

Default mode uses the official neurogolf_utils.convert_to_numpy benchmark and
compares the raw 1x10x30x30 output tensors exactly. That is the safe gate for
"is this prototype equivalent to live?" checks.

Legacy mode (explicit --cropped-decode) preserves the old visible-crop decode
path for debugging historical results, but it uses the old ch0=1 padding
encoding and must NOT be used to gate submit-captain requeues.

Usage:
    python tools/verify_against_live.py --proto path/to/prototype.onnx --tid 4
    python tools/verify_against_live.py --proto path/to/prototype.onnx --tid 161 --first-fail
    python tools/verify_against_live.py --proto path/to/prototype.onnx --tid 161 --cropped-decode
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile

os.environ["ORT_LOGGING_LEVEL"] = "4"
REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "data" / "competitions" / "neurogolf-2026" / "neurogolf_utils"))
NOGIT_ROOT = Path(os.environ.get("NEUROGOLF_NOGIT_ROOT", REPO / ".notongit" / "NeuroGolf_nogit"))

import numpy as np
import onnxruntime as ort

import neurogolf_utils as nu

ort.set_default_logger_severity(4)


DEFAULT_LIVE = NOGIT_ROOT / "submissions" / "alpha_b_t287v4_5653_67" / "submission.zip"
DEFAULT_DATA = Path(os.environ.get("NEUROGOLF_DATA_DIR", REPO / "data" / "competitions" / "neurogolf-2026"))

LEGACY_WARNING = (
    "LEGACY cropped-decode mode enabled; this is for debugging only and must "
    "not gate submit-captain requeues."
)


def legacy_encode(grid):
    a = np.array(grid, dtype=np.int64)
    h, w = a.shape
    out = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for c in range(10):
        out[0, c, :h, :w] = (a == c).astype(np.float32)
    out[0, 0, h:, :] = 1.0
    out[0, 0, :h, w:] = 1.0
    return out


def legacy_decode(arr, h, w):
    grid = []
    for r in range(h):
        row = []
        for c in range(w):
            vec = arr[0, :, r, c]
            idxs = np.where(np.isclose(vec, 1.0, atol=1e-4))[0]
            if len(idxs) == 1:
                row.append(int(idxs[0]))
            elif len(idxs) == 0:
                row.append(0)
            else:
                row.append(11)
        grid.append(row)
    return grid


def compare_full_tensor_outputs(proto_out, live_out, expected_out):
    return {
        "correct_proto": bool(np.array_equal(proto_out, expected_out)),
        "correct_live": bool(np.array_equal(live_out, expected_out)),
        "match_live": bool(np.array_equal(proto_out, live_out)),
    }


def compare_cropped_outputs(proto_out, live_out, expected_grid):
    h, w = len(expected_grid), len(expected_grid[0])
    proto_pred = legacy_decode(proto_out, h, w)
    live_pred = legacy_decode(live_out, h, w)
    return {
        "correct_proto": proto_pred == expected_grid,
        "correct_live": live_pred == expected_grid,
        "match_live": proto_pred == live_pred,
        "proto_pred": proto_pred,
        "live_pred": live_pred,
    }


def compare_example(proto_sess, live_sess, example, cropped_decode: bool = False):
    if cropped_decode:
        inp = legacy_encode(example["input"])
        proto_out = proto_sess.run(["output"], {"input": inp})[0]
        live_out = live_sess.run(["output"], {"input": inp})[0]
        result = compare_cropped_outputs(proto_out, live_out, example["output"])
        result.update(
            {
                "mode": "cropped",
                "input": inp,
                "proto_out": proto_out,
                "live_out": live_out,
                "expected_grid": example["output"],
            }
        )
        return result

    benchmark = nu.convert_to_numpy(example)
    if benchmark is None:
        return None
    proto_out = proto_sess.run(["output"], {"input": benchmark["input"]})[0]
    live_out = live_sess.run(["output"], {"input": benchmark["input"]})[0]
    result = compare_full_tensor_outputs(proto_out, live_out, benchmark["output"])
    result.update(
        {
            "mode": "full",
            "benchmark": benchmark,
            "proto_out": proto_out,
            "live_out": live_out,
        }
    )
    return result


def full_tensor_diff_summary(expected, actual):
    if expected.shape != actual.shape:
        return f"shape mismatch expected {expected.shape} actual {actual.shape}"
    diff = expected != actual
    if not np.any(diff):
        return "no diff"
    coords = tuple(int(x) for x in np.argwhere(diff)[0])
    return (
        f"first diff at {coords}: expected={expected[coords]!r} actual={actual[coords]!r}; "
        f"mismatching elements={int(np.count_nonzero(diff))}"
    )


def cropped_diff_summary(expected_grid, actual_grid):
    if len(expected_grid) != len(actual_grid):
        return f"row-count mismatch expected {len(expected_grid)} actual {len(actual_grid)}"
    if not expected_grid:
        return "empty grid"
    if len(expected_grid[0]) != len(actual_grid[0]):
        return f"col-count mismatch expected {len(expected_grid[0])} actual {len(actual_grid[0])}"
    for r, (exp_row, act_row) in enumerate(zip(expected_grid, actual_grid)):
        for c, (exp_cell, act_cell) in enumerate(zip(exp_row, act_row)):
            if exp_cell != act_cell:
                return f"first diff at ({r}, {c}): expected={exp_cell} actual={act_cell}"
    return "no diff"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--proto", type=Path, required=True, help="Prototype ONNX file")
    p.add_argument("--tid", type=int, required=True, help="Task ID")
    p.add_argument("--live-zip", type=Path, default=DEFAULT_LIVE)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    p.add_argument(
        "--cropped-decode",
        action="store_true",
        help=(
            "LEGACY mode: use the historical ch0=1 padding encoding and visible-"
            "crop decode. Do not use this mode to gate submit-captain requeues."
        ),
    )
    p.add_argument("--first-fail", action="store_true", help="Print details of first failing pair")
    args = p.parse_args(argv)

    # Extract live ONNX for this task.
    with ZipFile(args.live_zip) as z:
        live_blob = z.read(f"task{args.tid:03d}.onnx")
    live_tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    live_tmp.write(live_blob)
    live_tmp.close()

    # Load pairs.
    data = json.loads((args.data_dir / f"task{args.tid:03d}.json").read_text())
    pairs = data.get("train", []) + data.get("test", []) + data.get("arc-gen", [])

    proto_sess = ort.InferenceSession(str(args.proto), providers=["CPUExecutionProvider"])
    live_sess = ort.InferenceSession(live_tmp.name, providers=["CPUExecutionProvider"])

    correct_proto = 0
    correct_live = 0
    match_live = 0
    proto_live_diff = []
    total = 0

    if args.cropped_decode:
        print(LEGACY_WARNING)
        print("Mode: legacy cropped-decode (historical ch0=1 padding + visible crop)")
    else:
        print("Mode: official convert_to_numpy full-tensor equality")

    for i, example in enumerate(pairs):
        result = compare_example(proto_sess, live_sess, example, cropped_decode=args.cropped_decode)
        if result is None:
            continue
        total += 1
        if result["correct_proto"]:
            correct_proto += 1
        if result["correct_live"]:
            correct_live += 1
        if result["match_live"]:
            match_live += 1
        else:
            proto_live_diff.append((i, result))

    print(f"Task: task{args.tid:03d}")
    print(f"Total pairs (<=30x30): {total}")
    print()
    print(f"  Prototype  vs expected: {correct_proto:>4}/{total}")
    print(f"  Live       vs expected: {correct_live:>4}/{total}")
    if args.cropped_decode:
        print(
            f"  Prototype  vs LIVE (cropped decode): {match_live:>4}/{total}  "
            "<-- LEGACY ONLY; do not gate submit-captain requeues"
        )
    else:
        print(f"  Prototype  vs LIVE   : {match_live:>4}/{total}  <-- key safety check")
    print()
    if match_live < total:
        print(f"  !!  PROTOTYPE DIVERGES FROM LIVE on {total - match_live} pair(s)")
        print(f"     This is the silent-fail-trap signal. Even if proto verifies")
        print(f"     against expected on visible, the algorithm differs from the")
        print(f"     live's hidden-correct algorithm -> high LB regression risk.")
        if args.first_fail and proto_live_diff:
            i, result = proto_live_diff[0]
            example = pairs[i]
            print()
            print(f"  First diverging pair: {i} (shape {np.array(example['input']).shape})")
            if args.cropped_decode:
                print(f"  Diverges on cropped output: {cropped_diff_summary(result['proto_pred'], result['live_pred'])}")
                print(f"  Input:")
                print("  " + "\n  ".join(" ".join(f"{c:>2}" for c in row) for row in example["input"][:6]))
                print(f"  Prototype output:")
                print("  " + "\n  ".join(" ".join(f"{c:>2}" for c in row) for row in result["proto_pred"][:6]))
                print(f"  LIVE output:")
                print("  " + "\n  ".join(" ".join(f"{c:>2}" for c in row) for row in result["live_pred"][:6]))
            else:
                expected = result["benchmark"]["output"]
                print(f"  Diverges on raw tensor: {full_tensor_diff_summary(expected, result['proto_out'])}")
                print(f"  Raw LIVE difference vs proto: {full_tensor_diff_summary(result['proto_out'], result['live_out'])}")
    else:
        print(f"  OK Prototype matches LIVE on every visible+arc-gen pair.")
        print(f"     If cheaper than live, safe to ship.")

    try:
        os.unlink(live_tmp.name)
    except OSError:
        pass

    return 0 if match_live == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
