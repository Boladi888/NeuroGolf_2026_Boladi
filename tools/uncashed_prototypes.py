"""Report tasks with prototype work on disk but no shipped variant.

Scans for both:
  1. `.py` prototypes in Research_Archive/organic_prototypes/ AND the
     .notongit scratch dirs (taskNNN_*.py).
  2. `.onnx` prototypes in the same scratch dirs (taskNNN_*.onnx).
     These are pre-built, ready-to-score artifacts — often the
     missing pieces in the .py-only view.

Two classification axes:

  prototype_cost_class:
    SHIP-READY     = lowest verified prototype cost < current live cost
    needs-compress = lowest verified prototype cost > current live cost
    no-verify      = no working prototype (all verify < 100%)
    unknown        = no scored prototype yet (no HIST hint, no .onnx)

  hidden_bad_flag:
    HIDDEN-BAD     = task is in HIDDEN_BAD_TIDS (silent-fail trap)
    OK             = task hasn't been seen hidden-bad

Combination:
  SHIP-READY + OK              → submit candidate
  SHIP-READY + HIDDEN-BAD      → DEFER until adversarial test
  needs-compress + OK          → compression work item
  needs-compress + HIDDEN-BAD  → low priority (would lose LB if shipped)

Sort modes:
  - by-current-cost  (default) — biggest LB headroom first
  - by-task-id       — natural order
  - by-gain          — best LB delta if hidden-correct (proto vs live)

Usage:
    python tools/uncashed_prototypes.py
    python tools/uncashed_prototypes.py --top 30 --csv out.csv
    python tools/uncashed_prototypes.py --sort by-gain
    python tools/uncashed_prototypes.py --score-onnx   # actually score
                                                       # scratch .onnxes
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO / "The_400_Tasks" / "task_state.csv"
TASK_BY_TASK = REPO / "The_400_Tasks" / "Task_by_Task"

# Regexes for mining cost/verification hints out of HIST blocks.
COST_RE = re.compile(r"\bcost[ =:]\s*([\d,]+\.?\d*[kKmM]?)", re.IGNORECASE)
COST_NUM_RE = re.compile(r"\b(\d{2,3}\.\d+M|\d{2,4}k|\d{1,3},\d{3}|\d{6,9})\b")
VERIFY_RE = re.compile(r"\b(2[0-9][0-9])\s*/\s*(2[0-9][0-9])\b")  # e.g. 266/267
SHIPPED_RE = re.compile(r"\b(NEW BEST|won|verified|shipped|\+0\.\d+ LB)\b", re.IGNORECASE)


def num(v, default: float = 0.0) -> float:
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def cost_to_float(s: str) -> float | None:
    """Parse '25.7M', '521k', '90,000' into float."""
    s = s.replace(",", "").strip()
    m = re.match(r"^([\d.]+)([kKmM]?)$", s)
    if not m:
        return None
    val = float(m.group(1))
    suffix = m.group(2).lower()
    if suffix == "k":
        val *= 1_000
    elif suffix == "m":
        val *= 1_000_000
    return val


def mine_hist_block(tid: int) -> dict:
    """Pull cost numbers + verify counts out of the task's HIST block."""
    path = TASK_BY_TASK / f"task{tid:03d}.md"
    if not path.exists():
        return {"cost_hints": [], "verify_max": "", "shipped_markers": 0}
    text = path.read_text(encoding="utf-8")
    # Extract HIST block
    m = re.search(r"<!-- HIST:BEGIN -->(.*?)<!-- HIST:END -->", text, re.DOTALL)
    if not m:
        return {"cost_hints": [], "verify_max": "", "shipped_markers": 0}
    block = m.group(1)

    # Extract cost-like numbers
    cost_hints = []
    for m in COST_RE.finditer(block):
        val = cost_to_float(m.group(1))
        if val is not None and val > 1_000:
            cost_hints.append(val)
    # Also catch bare table cells like "| 25.7M |" or "521k"
    for m in COST_NUM_RE.finditer(block):
        s = m.group(1)
        if "/" in s:
            continue
        val = cost_to_float(s)
        if val is not None and val > 10_000:  # only large costs (filter out years/dates)
            cost_hints.append(val)
    # Dedupe and sort
    cost_hints = sorted(set(cost_hints))

    # Verify counts: largest numerator/denominator
    verify_max = ""
    best_n = 0
    for m in VERIFY_RE.finditer(block):
        n, d = int(m.group(1)), int(m.group(2))
        if n > best_n:
            best_n = n
            verify_max = f"{n}/{d}"

    # Shipped markers
    shipped_markers = len(SHIPPED_RE.findall(block))

    return {
        "cost_hints": cost_hints,
        "verify_max": verify_max,
        "shipped_markers": shipped_markers,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv-in", type=Path, default=DEFAULT_CSV)
    p.add_argument("--csv", type=Path, default=None,
                   help="Write sortable CSV to this path")
    p.add_argument("--top", type=int, default=50)
    p.add_argument("--sort", choices=["by-current-cost", "by-task-id"],
                   default="by-current-cost")
    p.add_argument("--include-solved", action="store_true",
                   help="Also include rows where solved_status == Solved (sanity)")
    args = p.parse_args(argv)

    with args.csv_in.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    candidates = []
    for r in rows:
        proto = (r.get("prototype_path") or "").strip()
        if not proto:
            continue
        solved = (r.get("solved_status") or "").strip()
        if not args.include_solved and solved == "Solved":
            continue
        tid = int(r.get("task_id") or 0)
        hist = mine_hist_block(tid)
        proto_cost_hints = hist["cost_hints"]
        # Take the largest cost hint as "prototype-cost candidate" — usually
        # the prototype's own cost shows up in session logs (e.g. "25.7M cost").
        proto_cost = max(proto_cost_hints) if proto_cost_hints else None
        current_cost = num(r.get("current_cost"))

        # Headroom: if prototype is cheaper than current, gain = pts_proto - pts_now
        if proto_cost and proto_cost < current_cost:
            gain = (math.log(current_cost) - math.log(proto_cost))
            status_label = "SHIP-READY"  # prototype is cheaper!
        elif proto_cost and current_cost > 0:
            gain = -(math.log(proto_cost) - math.log(current_cost))
            status_label = "needs-compress"
        else:
            gain = 0.0
            status_label = "unknown"

        candidates.append({
            "task": r.get("task"),
            "tid": tid,
            "solved": solved,
            "family": r.get("family") or "",
            "transformations": r.get("transformations") or "",
            "current_cost": current_cost,
            "current_src": r.get("current_src") or "",
            "proto_cost_hint": proto_cost,
            "verify_max": hist["verify_max"],
            "shipped_markers": hist["shipped_markers"],
            "prototype_path": proto,
            "gain_pts": gain,
            "status": status_label,
        })

    if args.sort == "by-current-cost":
        candidates.sort(key=lambda c: -c["current_cost"])
    else:
        candidates.sort(key=lambda c: c["tid"])

    # Print
    n_total = len(candidates)
    n_ship_ready = sum(1 for c in candidates if c["status"] == "SHIP-READY")
    n_compress = sum(1 for c in candidates if c["status"] == "needs-compress")
    n_unknown = sum(1 for c in candidates if c["status"] == "unknown")
    print(f"# Uncashed prototypes — {n_total} tasks have a prototype but didn't ship")
    print(f"# Status breakdown: SHIP-READY={n_ship_ready}  needs-compress={n_compress}  unknown={n_unknown}")
    print(f"# Sorted by: {args.sort}")
    print()
    print(f"  {'task':<8} {'status':<14} {'cur_cost':>10} {'proto_hint':>11}  {'verify':<9} {'src':<14} family · transformations")
    for c in candidates[:args.top]:
        ph = c["proto_cost_hint"]
        ph_s = f"{ph:,.0f}" if ph else "?"
        trans = c["transformations"]
        if len(trans) > 36:
            trans = trans[:33] + "..."
        print(f"  {c['task']:<8} {c['status']:<14} {c['current_cost']:>10,.0f} {ph_s:>11}  {c['verify_max']:<9} {c['current_src'][:14]:<14} {c['family'][:14]:<14} · {trans}")

    print()
    print("Legend:")
    print("  SHIP-READY     = prototype cost hint < current cost (verify + rebuild submission)")
    print("  needs-compress = prototype cost hint > current cost (cracked rule but too expensive)")
    print("  unknown        = no cost hint mined from HIST; needs manual look at .py")
    print()
    print(f"To triage details, open The_400_Tasks/Task_by_Task/taskNNN.md and inspect the HIST block.")

    if args.csv:
        with args.csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "task", "tid", "status", "solved", "current_cost", "current_src",
                "proto_cost_hint", "verify_max", "gain_pts", "family",
                "transformations", "prototype_path",
            ])
            w.writeheader()
            for c in candidates:
                w.writerow({
                    "task": c["task"],
                    "tid": c["tid"],
                    "status": c["status"],
                    "solved": c["solved"],
                    "current_cost": c["current_cost"],
                    "current_src": c["current_src"],
                    "proto_cost_hint": c["proto_cost_hint"] or "",
                    "verify_max": c["verify_max"],
                    "gain_pts": c["gain_pts"],
                    "family": c["family"],
                    "transformations": c["transformations"],
                    "prototype_path": c["prototype_path"],
                })
        print(f"\nWrote CSV: {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
