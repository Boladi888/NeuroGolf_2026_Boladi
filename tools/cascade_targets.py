"""Print actionable cascade-risk targets — tasks worth writing organics for.

Filters on:
  - Task is in a top-5 cascade pool by task count (thisray4808 / ashok /
    karnak / afr1ste / needless090)
  - solved_status == 'NotSolved' (not already organic, not HiddenBad)
  - family ∈ SAFE_FAMILIES (Pattern_Recognition / Spatial_and_Geometric;
    Object_Based + Color_and_Logical filtered OUT because anchor-pattern /
    template-matching tasks tend to hit the silent-fail trap)
  - transformations contains a "safe" primitive (Filling Regions /
    Symmetry Completion / Tiling / Reflection)

Sorts by current_cost desc — biggest cost-rewrite headroom first.

Usage:
    python tools/cascade_targets.py [--csv PATH] [--top N]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO / "The_400_Tasks" / "task_state.csv"

# Top 5 cascade pools by task count (from CASCADE_ANALYSIS.md)
TARGET_POOLS = {"thisray4808", "ashok", "karnak", "afr1ste", "needless090"}

# Families with the lowest silent-fail-trap rate (per Logs/lessons.md):
# pure-global-transform tasks are hidden-correct family.
SAFE_FAMILIES = {"Pattern_Recognition", "Spatial_and_Geometric"}

# Specific transformations that are hidden-correct (per submission history):
SAFE_TRANSFORMATIONS = {"Filling Regions", "Symmetry Completion", "Tiling", "Reflection"}


def num(v, default: float = 0.0) -> float:
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--all-families", action="store_true",
                   help="Don't filter by family (include anchor-pattern tasks)")
    args = p.parse_args(argv)

    with args.csv.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    candidates = []
    for r in rows:
        src = (r.get("current_src") or "").strip()
        # Strip OVERRIDE- prefix to consolidate
        src_clean = src.replace("OVERRIDE-", "")
        if src_clean not in TARGET_POOLS:
            continue
        if (r.get("solved_status") or "") != "NotSolved":
            continue
        fam = r.get("family") or ""
        trans = r.get("transformations") or ""
        if not args.all_families and fam not in SAFE_FAMILIES:
            continue
        # Check transformations overlap with safe set
        trans_tokens = [t.strip() for t in trans.split("|")]
        has_safe = any(t in SAFE_TRANSFORMATIONS for t in trans_tokens)
        if not args.all_families and not has_safe:
            continue
        cost = num(r.get("current_cost"))
        pts = num(r.get("current_points"))
        prototype = r.get("prototype_path") or ""
        candidates.append({
            "task": r.get("task"),
            "tid": int(r.get("task_id", 0)),
            "src": src,
            "family": fam,
            "transformations": trans,
            "cost": cost,
            "points": pts,
            "prototype": prototype,
            "recipe_path": r.get("recipe_path") or "",
        })

    candidates.sort(key=lambda c: -c["cost"])

    print(f"# Cascade-risk action list (top {args.top})")
    print(f"# Filtered: source in top-5 pool, NotSolved, safe family (Pattern/Spatial), safe transformation")
    print(f"# Total candidates: {len(candidates)}")
    print()
    print(f"  {'task':<8} {'src':<14} {'cost':>10} {'pts':>7}  family · transformations")
    for c in candidates[:args.top]:
        proto = " [P]" if c["prototype"] else ""
        recipe = " [R]" if c["recipe_path"] else ""
        trans = c["transformations"]
        if len(trans) > 40:
            trans = trans[:37] + "..."
        print(f"  {c['task']:<8} {c['src']:<14} {c['cost']:>10,.0f} {c['points']:>7.2f}  "
              f"{c['family'][:14]:<14} · {trans}{proto}{recipe}")
    print()
    print("Legend: [P] = has organic_prototypes/.py · [R] = has recipe note")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
