"""Run an ONNX model on a real input and dump every intermediate tensor.

Use this to debug `verify_against_live` divergence: identify WHICH intermediate
first diverges between your candidate and LIVE.

Usage:
    # Use a specific train pair as input
    python tools/agent_helpers/dump_intermediates.py model.onnx --tid 110 --pair-idx 0

    # Compare divergence between two models on the same input
    python tools/agent_helpers/dump_intermediates.py candidate.onnx --tid 110 --pair-idx 0 --compare-with live.onnx
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "competitions" / "neurogolf-2026"


def encode_grid(grid):
    """Encode a 2D color grid as [1, 10, 30, 30] F32 one-hot per NeuroGolf convention."""
    arr = np.zeros((1, 10, 30, 30), dtype=np.float32)
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            if 0 <= color < 10 and r < 30 and c < 30:
                arr[0, color, r, c] = 1.0
    return arr


def load_input(tid, pair_idx):
    """Load the pair_idx'th train+test+arc-gen pair's input from task data."""
    json_path = DATA_DIR / f"{tid}.json"
    if not json_path.exists():
        print(f"ERROR: {json_path} not found", file=sys.stderr)
        return None
    data = json.loads(json_path.read_text())
    pairs = data.get("train", []) + data.get("test", []) + data.get("arc-gen", [])
    if pair_idx >= len(pairs):
        print(f"ERROR: pair_idx {pair_idx} out of range (have {len(pairs)})", file=sys.stderr)
        return None
    return encode_grid(pairs[pair_idx]["input"])


def add_all_intermediates_as_outputs(model_path, tmp_path):
    """Modify the model so every node's output is also a graph output. Allows ORT to expose intermediates."""
    model = onnx.load(model_path)
    g = model.graph
    existing_output_names = {o.name for o in g.output}
    for node in g.node:
        for out_name in node.output:
            if out_name and out_name not in existing_output_names:
                # Add it as a graph output with unknown dtype/shape (ORT will infer)
                vi = onnx.helper.make_empty_tensor_value_info(out_name)
                g.output.append(vi)
                existing_output_names.add(out_name)
    onnx.save(model, tmp_path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model_path")
    p.add_argument("--tid", required=True, help="Task ID (e.g. 110 or task110)")
    p.add_argument("--pair-idx", type=int, default=0, help="Pair index from train+test+arc-gen")
    p.add_argument("--compare-with", default=None, help="Compare divergence against this second model")
    p.add_argument("--top-diff", type=int, default=10, help="Show top-N tensors that diverge")
    args = p.parse_args()

    tid = args.tid if args.tid.startswith("task") else f"task{int(args.tid):03d}"
    inp = load_input(tid, args.pair_idx)
    if inp is None:
        return 1

    # Modify model to expose intermediates
    tmp_a = Path("./_tmp_dump_a.onnx")
    add_all_intermediates_as_outputs(args.model_path, str(tmp_a))
    try:
        sess_a = ort.InferenceSession(str(tmp_a), providers=["CPUExecutionProvider"])
    except Exception as exc:
        print(f"ERROR: failed to load {args.model_path} with exposed intermediates: {exc}", file=sys.stderr)
        tmp_a.unlink(missing_ok=True)
        return 1

    out_names_a = [o.name for o in sess_a.get_outputs()]
    print(f"Running {args.model_path} on {tid} pair {args.pair_idx}...")
    results_a = sess_a.run(out_names_a, {"input": inp})
    intermediates_a = dict(zip(out_names_a, results_a))

    if args.compare_with:
        tmp_b = Path("./_tmp_dump_b.onnx")
        add_all_intermediates_as_outputs(args.compare_with, str(tmp_b))
        try:
            sess_b = ort.InferenceSession(str(tmp_b), providers=["CPUExecutionProvider"])
        except Exception as exc:
            print(f"ERROR: failed to load {args.compare_with}: {exc}", file=sys.stderr)
            tmp_a.unlink(missing_ok=True); tmp_b.unlink(missing_ok=True)
            return 1
        out_names_b = [o.name for o in sess_b.get_outputs()]
        results_b = sess_b.run(out_names_b, {"input": inp})
        intermediates_b = dict(zip(out_names_b, results_b))

        # Find divergences
        common = set(intermediates_a) & set(intermediates_b)
        diffs = []
        for name in common:
            a = intermediates_a[name]
            b = intermediates_b[name]
            if a.shape != b.shape:
                diffs.append((name, "SHAPE-DIFF", a.shape, b.shape, None))
                continue
            try:
                diff_pct = float(np.abs(a.astype(np.float64) - b.astype(np.float64)).sum() / max(a.size, 1))
            except Exception:
                diff_pct = float("nan")
            if not np.array_equal(a, b):
                diffs.append((name, "VAL-DIFF", a.shape, b.shape, diff_pct))

        # Sort by name (graph order). The first diverging tensor is often the root cause.
        diffs.sort()
        print(f"\n=== Divergence (top {args.top_diff}) ===")
        for name, kind, sa, sb, diff in diffs[:args.top_diff]:
            print(f"  {name:<40} {kind:<12} shape_a={sa} shape_b={sb}  mean_abs_diff={diff}")
        if not diffs:
            print("  No divergence found — both models produce identical intermediates.")
        print(f"\nTotal diverging intermediates: {len(diffs)}")

        tmp_b.unlink(missing_ok=True)
    else:
        print(f"\n=== Intermediates from {args.model_path} ({len(intermediates_a)} tensors) ===")
        for name, arr in list(intermediates_a.items())[:20]:
            print(f"  {name:<40} shape={arr.shape} dtype={arr.dtype} sum={float(arr.sum()):.4f}")
        if len(intermediates_a) > 20:
            print(f"  ... ({len(intermediates_a) - 20} more)")

    tmp_a.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
