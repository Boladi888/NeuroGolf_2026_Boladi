"""Cross-reference karnak's per-task transformation labels with our truly-unsolved set.

Joins:
  - karnak `arc_primitives.csv` (400 tasks; transformation labels + complexity scores)
  - our `Logs/runs.jsonl` latest run (organic SOLVED tasks)
  - our `Logs/external_blend_*.csv` (external winners)

Outputs:
  - Per-category breakdown of the 150 truly-unsolved tasks (by Primary_Category)
  - Top 30 lowest-complexity unsolved tasks (priority targets)
  - Per-transformation breakdown (which transformations are most common in unsolved set)

Usage:
    python tools/karnak_xref.py
    python tools/karnak_xref.py --top 50
    python tools/karnak_xref.py --category Pattern_Recognition

Cross-worktree note: set NEUROGOLF_DATA_DIR if running outside main worktree.
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

NG_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(NG_ROOT / "tools"))
from mining_lib import SOLVED_TASKS, DATA_DIR  # noqa: E402

KARNAK_CSV_DEFAULT = (
    Path(
        os.environ.get(
            "NEUROGOLF_COMPETITOR_ARTIFACTS_DIR",
            str(Path(os.environ.get("NEUROGOLF_DATA_DIR", str(NG_ROOT / "data" / "competitions" / "neurogolf-2026"))).parent.parent / "competitor_artifacts"),
        )
    )
    / "karnak-task-transformation"
    / "arc_primitives.csv"
)


def load_karnak(path):
    """Return dict[tid -> row]."""
    rows = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            tid = int(row["Task_ID"].replace("task", ""))
            rows[tid] = row
    return rows


def load_organic_solved_from_runs(runs_path):
    """Return set of task ids that organic --hybrid latest run got correct."""
    if not runs_path.exists():
        return set()
    latest_id = None
    by_run = defaultdict(set)
    with open(runs_path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("correct"):
                by_run[d["run_id"]].add(d["task"])
            if latest_id is None or d["run_id"] > latest_id:
                latest_id = d["run_id"]
    return by_run.get(latest_id, set())


def load_external_blend_set():
    """Return set of task ids covered by external sources in latest blend csv."""
    candidates = sorted((NG_ROOT / "Logs").glob("external_blend_*.csv"))
    if not candidates:
        return set()
    out = set()
    with open(candidates[-1]) as f:
        for row in csv.DictReader(f):
            tid = int(row["task"])
            if row.get("winner_src", "").strip():
                out.add(tid)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--karnak-csv", default=str(KARNAK_CSV_DEFAULT))
    ap.add_argument("--top", type=int, default=30, help="Top-N lowest-complexity unsolved (default 30)")
    ap.add_argument("--category", default=None, help="Filter by Primary_Category")
    ap.add_argument("--show-solved", action="store_true", help="Also show karnak labels for SOLVED tasks (for calibration)")
    args = ap.parse_args()

    if not Path(args.karnak_csv).exists():
        print(f"ERROR: karnak CSV not found at {args.karnak_csv}", file=sys.stderr)
        print(f"Set NEUROGOLF_DATA_DIR or pass --karnak-csv. Looked at:\n  {args.karnak_csv}", file=sys.stderr)
        sys.exit(1)

    karnak = load_karnak(args.karnak_csv)
    organic_solved = SOLVED_TASKS  # Use canonical set (matches per-pattern tally)
    external_covered = load_external_blend_set()

    # Truly-unsolved = no organic AND no external
    truly_unsolved = set()
    for tid in karnak:
        if tid not in organic_solved and tid not in external_covered:
            truly_unsolved.add(tid)

    print(f"=== Coverage breakdown ({len(karnak)} total, post-PR-#30) ===\n")
    print(f"  Organic SOLVED:        {len(organic_solved)}")
    print(f"  External-covered:      {len(external_covered - organic_solved)} (outside organic SOLVED)")
    print(f"  Truly-unsolved:        {len(truly_unsolved)}")
    print()

    # Filter to truly-unsolved
    if args.category:
        unsolved_in_cat = [tid for tid in truly_unsolved if karnak[tid]["Primary_Category"] == args.category]
        print(f"=== Truly-unsolved in category '{args.category}' ({len(unsolved_in_cat)}) ===\n")
        unsolved_in_cat.sort(key=lambda t: int(karnak[t]["Estimated_Complexity"]))
        for tid in unsolved_in_cat[: args.top]:
            row = karnak[tid]
            print(
                f"  task{tid:03d} c={row['Estimated_Complexity']:>2} | {row['Primary_Category']:<25} "
                f"| size_changed={row['Grid_Size_Changed']:<5} | {row['All_Used_Transformations']}"
            )
    else:
        # Per-category breakdown of truly-unsolved
        cat_counts = Counter(karnak[tid]["Primary_Category"] for tid in truly_unsolved)
        print("=== Truly-unsolved by Primary_Category ===\n")
        cat_w = max(len(c) for c in cat_counts) if cat_counts else 20
        for cat, n in cat_counts.most_common():
            print(f"  {cat:<{cat_w}} {n:>3}")
        print()

        # Per-transformation breakdown (split by '|')
        trans_counts = Counter()
        for tid in truly_unsolved:
            for t in karnak[tid]["All_Used_Transformations"].split("|"):
                t = t.strip()
                if t:
                    trans_counts[t] += 1
        print("=== Truly-unsolved by transformation (top 20) ===\n")
        tw = max(len(t) for t in trans_counts) if trans_counts else 30
        for t, n in trans_counts.most_common(20):
            print(f"  {t:<{tw}} {n:>3}")
        print()

        # Lowest-complexity truly-unsolved
        sorted_unsolved = sorted(truly_unsolved, key=lambda t: int(karnak[t]["Estimated_Complexity"]))
        print(f"=== Top {args.top} lowest-complexity truly-unsolved (priority targets) ===\n")
        print(f"  {'task':>5} {'c':>2} | {'category':<25} | {'size_changed':<13} | transformations")
        print(f"  {'-'*5} {'-'*2} | {'-'*25} | {'-'*13} | {'-'*40}")
        for tid in sorted_unsolved[: args.top]:
            row = karnak[tid]
            print(
                f"  task{tid:03d} {row['Estimated_Complexity']:>2} | {row['Primary_Category']:<25} "
                f"| size_changed={row['Grid_Size_Changed']:<5} | {row['All_Used_Transformations'][:60]}"
            )

    if args.show_solved:
        print("\n=== Karnak labels for SOLVED tasks (calibration) ===\n")
        for tid in sorted(organic_solved):
            if tid in karnak:
                row = karnak[tid]
                print(
                    f"  task{tid:03d} c={row['Estimated_Complexity']:>2} | {row['Primary_Category']:<25} "
                    f"| {row['All_Used_Transformations']}"
                )


if __name__ == "__main__":
    main()
