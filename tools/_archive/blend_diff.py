"""Per-task organic-vs-blend comparison for displacement-target analysis.

Joins the latest `Logs/runs.jsonl` (organic --hybrid results) with the latest
`Logs/external_blend_*.csv` (external winners) to classify every task as one of:

  - organic-wins      : organic correct, blend uses our ONNX
  - external-wins     : external solved at lower cost than organic, blend uses external
  - dual-solve-margin : organic correct AND external also correct; close margin
  - external-only     : organic incorrect, external solves it
  - unsolved          : no organic, no external (blend uses identity-fallback, 0 pts)

Output: ranked tables of high-leverage targets.

Usage:
    python tools/blend_diff.py
    python tools/blend_diff.py --runs Logs/runs.jsonl --blend Logs/external_blend_<ts>.csv
    python tools/blend_diff.py --top 20 --min-pts 13.0
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS_LOG = ROOT / "Logs" / "runs.jsonl"


def latest_runs(runs_path):
    """Return per-task dict for the LATEST run_id in runs.jsonl."""
    by_run = defaultdict(dict)
    with open(runs_path) as f:
        for line in f:
            d = json.loads(line)
            by_run[d["run_id"]][d["task"]] = d
    if not by_run:
        return None, {}
    latest_id = max(by_run.keys())
    return latest_id, by_run[latest_id]


def latest_blend_csv():
    """Return the most recent Logs/external_blend_*.csv path, or None."""
    candidates = sorted((ROOT / "Logs").glob("external_blend_*.csv"))
    return candidates[-1] if candidates else None


def load_blend(blend_path):
    """Return per-task dict from external_blend CSV (only tasks with externals)."""
    out = {}
    with open(blend_path) as f:
        for row in csv.DictReader(f):
            tid = int(row["task"])
            src = row.get("winner_src", "").strip()
            if not src:
                continue  # no external candidate
            out[tid] = {
                "src": src,
                "cost": int(row["winner_cost"]) if row["winner_cost"] not in ("", "-1") else None,
                "points": float(row["winner_points"]),
            }
    return out


def classify(organic, external):
    """Return (category, blend_uses, blend_pts, organic_pts, external_pts)."""
    if organic is None:
        organic = {"correct": False, "cost": 0, "points": 0.0}
    o_pts = float(organic["points"])
    o_correct = bool(organic["correct"])
    e_pts = external["points"] if external else 0.0

    if o_correct and external is None:
        return ("organic-wins", "organic", o_pts, o_pts, 0.0)
    if o_correct and external is not None:
        if o_pts >= e_pts:
            return ("organic-wins", "organic", o_pts, o_pts, e_pts)
        # External outscores us by enough to win the blend
        if e_pts - o_pts > 0.5:
            return ("external-wins", "external", e_pts, o_pts, e_pts)
        return ("dual-solve-margin", "external", e_pts, o_pts, e_pts)
    if not o_correct and external is not None:
        return ("external-only", "external", e_pts, 0.0, e_pts)
    return ("unsolved", "fallback", 0.0, 0.0, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(RUNS_LOG))
    ap.add_argument("--blend", default=None)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--min-pts", type=float, default=10.0,
                    help="Minimum external points to be considered a 'high-value' target")
    args = ap.parse_args()

    blend_path = Path(args.blend) if args.blend else latest_blend_csv()
    if blend_path is None or not blend_path.exists():
        raise SystemExit("No external_blend CSV found in Logs/")

    run_id, organic_by_task = latest_runs(Path(args.runs))
    external_by_task = load_blend(blend_path)
    print(f"Latest organic run: {run_id}")
    print(f"Blend snapshot:     {blend_path.name}")
    print(f"Tasks in organic run: {len(organic_by_task)}")
    print(f"Tasks with external winners in blend: {len(external_by_task)}")
    print()

    rows = []
    for tid in sorted(organic_by_task):
        cat, blend_uses, blend_pts, o_pts, e_pts = classify(
            organic_by_task[tid], external_by_task.get(tid)
        )
        rows.append({
            "task": tid,
            "category": cat,
            "blend_uses": blend_uses,
            "blend_pts": blend_pts,
            "organic_pts": o_pts,
            "organic_correct": bool(organic_by_task[tid]["correct"]),
            "organic_cost": int(organic_by_task[tid]["cost"]),
            "organic_pattern": organic_by_task[tid]["pattern"],
            "external_pts": e_pts,
            "external_cost": external_by_task.get(tid, {}).get("cost"),
            "external_src": external_by_task.get(tid, {}).get("src"),
        })

    # Summary
    cat_counts = defaultdict(int)
    cat_pts = defaultdict(float)
    for r in rows:
        cat_counts[r["category"]] += 1
        cat_pts[r["category"]] += r["blend_pts"]
    print("=== Category breakdown ===")
    print(f"{'Category':<20} {'Count':>6} {'Total blend pts':>17}")
    print(f"{'-' * 20} {'-' * 6} {'-' * 17}")
    for cat in ("organic-wins", "external-wins", "dual-solve-margin", "external-only", "unsolved"):
        print(f"{cat:<20} {cat_counts[cat]:>6} {cat_pts[cat]:>17.2f}")
    total_pts = sum(cat_pts.values())
    print(f"{'TOTAL':<20} {len(rows):>6} {total_pts:>17.2f}")
    print()

    # External breakdown
    ext_src_counts = defaultdict(int)
    for r in rows:
        if r["external_src"]:
            ext_src_counts[r["external_src"]] += 1
    print("=== External-winner sources ===")
    for src, cnt in sorted(ext_src_counts.items(), key=lambda x: -x[1]):
        print(f"  {src}: {cnt}")
    print()

    # High-value displacement targets (external-only OR external-wins, sorted by external pts)
    targets = [r for r in rows
               if r["category"] in ("external-only", "external-wins")
               and r["external_pts"] >= args.min_pts]
    targets.sort(key=lambda r: -r["external_pts"])
    print(f"=== Top {args.top} high-value displacement targets (external pts >= {args.min_pts}) ===")
    print(f"{'Task':>5} {'Cat':<14} {'ExtPts':>7} {'ExtCost':>9} {'ExtSrc':<10} {'OrgPts':>7} {'OrgPattern':<25}")
    print(f"{'-' * 5} {'-' * 14} {'-' * 7} {'-' * 9} {'-' * 10} {'-' * 7} {'-' * 25}")
    for r in targets[:args.top]:
        print(
            f"{r['task']:>5} {r['category']:<14} {r['external_pts']:>7.2f} "
            f"{r['external_cost'] or 0:>9} {r['external_src']:<10} "
            f"{r['organic_pts']:>7.2f} {r['organic_pattern']:<25}"
        )
    print()
    print(f"Total external-only + external-wins (any pts): "
          f"{sum(1 for r in rows if r['category'] in ('external-only', 'external-wins'))}")
    print(f"Total external-only + external-wins (>= {args.min_pts}): {len(targets)}")
    print()

    # The TRUE displacement targets: unsolved tasks (no organic, no external).
    # Adding ANY organic solution for these directly adds to blend pts (currently 0).
    unsolved = [r for r in rows if r["category"] == "unsolved"]
    print(f"=== Unsolved tasks (organic + external both fail; {len(unsolved)} total) ===")
    print("These are pure-gain targets: any organic solve adds full pts to the blend.")
    print()
    print("  " + ", ".join(f"task{r['task']:03d}" for r in unsolved[:50]))
    if len(unsolved) > 50:
        print(f"  ... and {len(unsolved) - 50} more")

    # Cross-check against any task list passed via stdin or mining output (future use)
    print()

    # external-wins (organic correct but external beats us)
    ext_wins = [r for r in rows if r["category"] == "external-wins"]
    if ext_wins:
        print(f"=== External-wins (organic correct, external beats by >0.5 pts) ===")
        for r in ext_wins:
            print(
                f"  task{r['task']:03d}: "
                f"ext={r['external_pts']:.2f} ({r['external_src']}, cost {r['external_cost']}) "
                f"vs org={r['organic_pts']:.2f} ({r['organic_pattern']}, cost {r['organic_cost']})  "
                f"delta={r['external_pts'] - r['organic_pts']:.2f}"
            )


if __name__ == "__main__":
    main()
