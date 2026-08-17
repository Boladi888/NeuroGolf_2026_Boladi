"""Identify [1, C, H, W] initializers in our submission ONNXs that are channel-uniform.

These are candidates for the `_compact_mask` channel-collapse optimization
(per `Research_Archive/notes/deep-read-multi-source-onnx-solver.md`): if a tensor is
constant across the channel axis, replacing `[1, C, ...]` with `[1, 1, ...]`
relies on ONNX broadcast and reduces params by `(C-1) * (other_dims_volume)`.

Pure static analysis — no inference needed.

Usage:
    python tools/compact_mask_analyzer.py
    python tools/compact_mask_analyzer.py --submissions-dir /path/to/submissions/
    python tools/compact_mask_analyzer.py --task 050
    python tools/compact_mask_analyzer.py --min-savings 500

Outputs:
    Per-task report: which initializers in each ONNX are channel-uniform.
    Aggregate: total potential savings + per-pattern breakdown.

Cross-worktree note: defaults to `<NG_ROOT>/submissions/`. Pass --submissions-dir
to point at the main worktree if running from a Claude-Code worktree.
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

NG_ROOT = Path(__file__).resolve().parents[1]
SUBMISSIONS_DIR_DEFAULT = NG_ROOT / "submissions"


def is_uniform_along_axis(arr, axis):
    """Check if arr is constant along the given axis (>1 dim)."""
    if arr.ndim <= axis:
        return False
    if arr.shape[axis] <= 1:
        return False
    slice0 = np.take(arr, 0, axis=axis)
    for k in range(1, arr.shape[axis]):
        if not np.array_equal(np.take(arr, k, axis=axis), slice0):
            return False
    return True


def find_compact_candidates(model_path, scan_all_axes=True):
    """Return list of compact-mask candidates.

    By default scans ALL axes (not just channel axis 1) for any redundancy,
    since our builders may produce [1, 1, H, 1] tensors with H-axis redundancy
    even though the channel axis is already compact.
    """
    try:
        model = onnx.load(str(model_path))
    except Exception as e:
        return [], f"load error: {e}"
    candidates = []
    for init in model.graph.initializer:
        try:
            arr = numpy_helper.to_array(init)
        except Exception:
            continue
        if arr.ndim < 2:
            continue
        if scan_all_axes:
            axes_to_scan = range(arr.ndim)
        else:
            axes_to_scan = [1] if arr.ndim > 1 else []
        # Find the FIRST axis along which arr is uniform (largest saving wins)
        best = None
        for axis in axes_to_scan:
            if not is_uniform_along_axis(arr, axis):
                continue
            saving = (arr.shape[axis] - 1) * (arr.size // arr.shape[axis])
            if best is None or saving > best["saving_elements"]:
                compacted_shape = list(arr.shape)
                compacted_shape[axis] = 1
                best = {
                    "name": init.name,
                    "shape": tuple(arr.shape),
                    "dtype": str(arr.dtype),
                    "current_elements": int(arr.size),
                    "compacted_elements": int(np.prod(compacted_shape)),
                    "saving_elements": saving,
                    "compact_axis": axis,
                }
        if best is not None:
            candidates.append(best)
    return candidates, None


def per_pattern_breakdown(submissions_dir, runs_jsonl):
    """Return dict[task_id -> pattern] from latest runs.jsonl, if available."""
    import json
    by_run = defaultdict(dict)
    if not Path(runs_jsonl).exists():
        return {}
    with open(runs_jsonl) as f:
        for line in f:
            d = json.loads(line)
            by_run[d["run_id"]][d["task"]] = d.get("pattern", "?")
    if not by_run:
        return {}
    latest = max(by_run.keys())
    return by_run[latest]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submissions-dir", default=str(SUBMISSIONS_DIR_DEFAULT),
                    help="Path to submissions/ folder (defaults to NG_ROOT/submissions).")
    ap.add_argument("--task", type=int, help="Analyze a single task only.")
    ap.add_argument("--min-savings", type=int, default=10,
                    help="Only report initializers with saving_elements >= this (default 10).")
    ap.add_argument("--runs-jsonl", default=str(NG_ROOT / "Logs" / "runs.jsonl"),
                    help="Path to runs.jsonl for pattern attribution.")
    args = ap.parse_args()

    submissions_dir = Path(args.submissions_dir)
    if not submissions_dir.exists():
        print(f"ERROR: submissions dir not found: {submissions_dir}", file=sys.stderr)
        print("Hint: pass --submissions-dir or run from main worktree.", file=sys.stderr)
        sys.exit(1)

    pattern_map = per_pattern_breakdown(submissions_dir, args.runs_jsonl)

    if args.task:
        tasks = [args.task]
    else:
        tasks = sorted(int(p.stem.replace("task", "")) for p in submissions_dir.glob("task*.onnx"))

    per_task_findings = {}
    per_pattern_savings = defaultdict(lambda: {"hits": 0, "total_saving": 0})
    total_files = 0
    total_with_candidates = 0

    for tid in tasks:
        path = submissions_dir / f"task{tid:03d}.onnx"
        if not path.exists():
            continue
        total_files += 1
        candidates, err = find_compact_candidates(path)
        if err:
            continue
        candidates = [c for c in candidates if c["saving_elements"] >= args.min_savings]
        if candidates:
            total_with_candidates += 1
            per_task_findings[tid] = candidates
            pattern = pattern_map.get(tid, "?")
            per_pattern_savings[pattern]["hits"] += 1
            per_pattern_savings[pattern]["total_saving"] += sum(c["saving_elements"] for c in candidates)

    if args.task:
        # Single-task verbose output
        if args.task in per_task_findings:
            print(f"=== task{args.task:03d} ({len(per_task_findings[args.task])} candidate initializers) ===\n")
            for c in per_task_findings[args.task]:
                print(f"  {c['name']:<30}  shape={c['shape']}  axis={c['compact_axis']}  "
                      f"dtype={c['dtype']}  current={c['current_elements']:>6}  "
                      f"compacted={c['compacted_elements']:>6}  save={c['saving_elements']:>6}")
        else:
            print(f"task{args.task:03d}: no compact-mask candidates above threshold {args.min_savings}.")
        return

    # Aggregate report
    print(f"=== compact-mask analysis ({total_files} ONNXs scanned, {total_with_candidates} have candidates) ===\n")

    if not per_task_findings:
        print("No compact-mask candidates found above threshold.")
        return

    # Per-pattern breakdown
    print("Per-pattern breakdown (sorted by total saving):\n")
    print(f"  {'Pattern':<35} {'Hits':>5} {'Total saving (elements)':>25}")
    print(f"  {'-'*35} {'-'*5} {'-'*25}")
    for pattern, stats in sorted(per_pattern_savings.items(), key=lambda x: -x[1]["total_saving"]):
        print(f"  {pattern:<35} {stats['hits']:>5} {stats['total_saving']:>25,}")
    print()

    # Top 20 individual hits
    flat = []
    for tid, candidates in per_task_findings.items():
        for c in candidates:
            flat.append({**c, "task": tid, "pattern": pattern_map.get(tid, "?")})
    flat.sort(key=lambda x: -x["saving_elements"])

    print(f"Top 20 individual saving candidates:\n")
    print(f"  {'task':>5} {'pattern':<25} {'init_name':<30} {'shape':<25} {'axis':>4} {'save (elems)':>12}")
    print(f"  {'-'*5} {'-'*25} {'-'*30} {'-'*25} {'-'*4} {'-'*12}")
    for c in flat[:20]:
        print(f"  task{c['task']:03d} {c['pattern']:<25} {c['name'][:28]:<30} {str(c['shape']):<25} {c['compact_axis']:>4} {c['saving_elements']:>12,}")

    total_saving_elements = sum(c["saving_elements"] for c in flat)
    print(f"\nTotal saving across all candidates: {total_saving_elements:,} elements.")
    print(f"At ~4 bytes/element float, ~{total_saving_elements * 4:,} bytes (~{total_saving_elements * 4 / 1024:.1f} KB) memory savings if all are compacted.")
    print(f"Per `02_GIT.md` lane discipline, the actual implementation is Alpha-territory; this is a [B->A] recipe candidate.")


if __name__ == "__main__":
    main()
