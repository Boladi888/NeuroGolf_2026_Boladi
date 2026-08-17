"""Cumulative memory simulator for a NeuroGolf submission.

Walks all taskNNN.onnx files in a submissions directory in task-number order
and tracks a running sum of per-task intermediate memory (as reported by the
ONNX graph's value_info tensors — same accounting the Kaggle validator uses).

The hypothesis being tested: Kaggle's evaluator does NOT reset memory between
tasks, so the accumulation across 400 tasks can exceed a hidden limit and
trigger the ERROR cascade.

Usage:
    python tools/cumulative_memory_sim.py                        # default submissions/
    python tools/cumulative_memory_sim.py --submissions-dir /p  # custom path
    python tools/cumulative_memory_sim.py --threshold 500000000 # 500 MB flag threshold
    python tools/cumulative_memory_sim.py --csv out.csv         # write CSV
    python tools/cumulative_memory_sim.py --top 20              # show top-N by per-task memory
"""

import argparse
import csv
import math
import pathlib
import sys

NG_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NG_ROOT / "tools"))

from cost_predictor import predict  # noqa: E402

# Default flag threshold: 1 GB (speculative; tune empirically)
DEFAULT_THRESHOLD = 1_000_000_000

SUBMISSIONS_DIR = NG_ROOT / "submissions"


def run(submissions_dir=None, threshold=DEFAULT_THRESHOLD, csv_path=None, top_n=None):
    sdir = pathlib.Path(submissions_dir) if submissions_dir else SUBMISSIONS_DIR

    onnx_files = sorted(sdir.glob("task*.onnx"), key=lambda p: int(p.stem[4:]))
    if not onnx_files:
        print(f"No task*.onnx files found in {sdir}", file=sys.stderr)
        sys.exit(1)

    rows = []
    cumulative = 0
    flag_task = None  # first task that pushes cumulative above threshold

    for f in onnx_files:
        task_num = int(f.stem[4:])
        r = predict(f)
        if r.get("error"):
            mem = 0
            cost = 0
            pts = 0.0
            status = "ERROR"
        else:
            mem = r["memory"]
            cost = r["cost"]
            pts = r["points"]
            status = "ok"

        cumulative += mem
        flagged = cumulative >= threshold and flag_task is None
        if flagged and status == "ok":
            flag_task = task_num

        rows.append({
            "task": task_num,
            "memory": mem,
            "cost": cost,
            "points": pts,
            "cumulative": cumulative,
            "flagged": flagged,
            "status": status,
        })

    # Print table
    print(f"Cumulative memory simulator  (threshold={threshold:,} bytes = {threshold/1e9:.2f} GB)")
    print(f"Submissions dir: {sdir}")
    print(f"Tasks found: {len(rows)}")
    if flag_task is not None:
        print(f"FIRST FLAG at task{flag_task:03d} (cumulative >= {threshold:,} bytes)")
    else:
        print(f"No task exceeds threshold. Final cumulative: {cumulative:,} bytes ({cumulative/1e9:.3f} GB)")
    print()

    if top_n:
        sorted_rows = sorted(rows, key=lambda r: r["memory"], reverse=True)
        print(f"Top-{top_n} tasks by per-task memory:")
        print(f"{'task':<8} {'memory':>12} {'cost':>12} {'pts':>8}  {'cumul_at':>14}  flag?")
        print("-" * 70)
        for row in sorted_rows[:top_n]:
            flag_marker = "  <-- FIRST FLAG" if row["flagged"] else ""
            print(
                f"task{row['task']:03d}  {row['memory']:>12,} {row['cost']:>12,} {row['points']:>8.4f}"
                f"  {row['cumulative']:>14,}  {flag_marker}"
            )
        print()

    print(f"{'task':<8} {'memory':>12} {'cost':>12} {'pts':>8}  {'cumulative':>14}  note")
    print("-" * 80)
    for row in rows:
        flag_str = "<-- FIRST FLAG" if row["flagged"] else ("ERROR" if row["status"] == "ERROR" else "")
        print(
            f"task{row['task']:03d}  {row['memory']:>12,} {row['cost']:>12,} {row['points']:>8.4f}"
            f"  {row['cumulative']:>14,}  {flag_str}"
        )

    print()
    print(f"Final cumulative memory: {cumulative:,} bytes ({cumulative/1e9:.3f} GB)")

    if csv_path:
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["task", "memory", "cost", "points", "cumulative", "flagged", "status"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV written to {csv_path}")

    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--submissions-dir", default=None, help="Path to submissions/ directory")
    parser.add_argument(
        "--threshold", type=int, default=DEFAULT_THRESHOLD,
        help=f"Cumulative memory flag threshold in bytes (default: {DEFAULT_THRESHOLD:,})"
    )
    parser.add_argument("--csv", default=None, metavar="FILE", help="Write results to CSV file")
    parser.add_argument("--top", type=int, default=None, metavar="N", help="Show top-N tasks by per-task memory")
    args = parser.parse_args()

    run(
        submissions_dir=args.submissions_dir,
        threshold=args.threshold,
        csv_path=args.csv,
        top_n=args.top,
    )


if __name__ == "__main__":
    main()
