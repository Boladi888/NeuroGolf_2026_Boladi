"""Per-task pre-mining analysis dump for a single ARC task.

The thing Beta runs FIRST when sketching a rule for an unsolved task. Eliminates
the 5-min round-trip of "load JSON, eyeball pairs, count edits, check colors".

Usage:
    python tools/task_inspect.py 042
    python tools/task_inspect.py 042 153 363       # multiple tasks
    python tools/task_inspect.py 042 --max-pairs 5

For each task, dumps:
  - SOLVED status (yes/no) per `inspect_tasks.SOLVED_TASKS`
  - Per-pair: shape (in -> out), color sets (in / out), edit-count if same shape
  - Aggregate: shape-preserved fraction, output-shape uniqueness, in-color-set / out-color-set
    consistency, output-strict-subset-of-input flag, mean edit-count
  - Op-class hints: heuristic suggestions for which ONNX op family is likely involved
    (color-perm / color-replace / pixel-perm / mirror-concat / tile / cell-edit / shape-changing / etc)
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mining_lib import SOLVED_TASKS, load_pairs


def _per_pair_summary(pair):
    inp = np.array(pair["input"])
    out = np.array(pair["output"])
    in_colors = sorted(set(int(c) for c in inp.flatten()))
    out_colors = sorted(set(int(c) for c in out.flatten()))
    edit_count = int((inp != out).sum()) if inp.shape == out.shape else None
    return {
        "in_shape": inp.shape,
        "out_shape": out.shape,
        "in_colors": in_colors,
        "out_colors": out_colors,
        "edit_count": edit_count,
        "shape_preserved": inp.shape == out.shape,
    }


def _aggregate(pairs_summary):
    n = len(pairs_summary)
    shape_pres = sum(1 for p in pairs_summary if p["shape_preserved"])
    out_shapes = Counter(p["out_shape"] for p in pairs_summary)
    in_color_sets = [tuple(p["in_colors"]) for p in pairs_summary]
    out_color_sets = [tuple(p["out_colors"]) for p in pairs_summary]
    in_color_consistent = len(set(in_color_sets)) == 1
    out_color_consistent = len(set(out_color_sets)) == 1
    out_strict_subset = all(set(p["out_colors"]).issubset(set(p["in_colors"])) for p in pairs_summary)
    in_strict_subset = all(set(p["in_colors"]).issubset(set(p["out_colors"])) for p in pairs_summary)
    edits = [p["edit_count"] for p in pairs_summary if p["edit_count"] is not None]
    return {
        "n": n,
        "shape_preserved_frac": round(shape_pres / n, 2) if n else 0.0,
        "unique_out_shapes": list(out_shapes.keys()),
        "in_color_consistent": in_color_consistent,
        "out_color_consistent": out_color_consistent,
        "out_color_strict_subset_of_in": out_strict_subset,
        "in_color_strict_subset_of_out": in_strict_subset,
        "edit_count_min": min(edits) if edits else None,
        "edit_count_max": max(edits) if edits else None,
        "edit_count_mean": round(sum(edits) / len(edits), 1) if edits else None,
    }


def _hints(agg, pairs_summary):
    """Heuristic op-class suggestions. Each suggests an ONNX family worth investigating
    given the aggregate statistics; not exhaustive, just first-pass priors.
    """
    hints = []
    n = agg["n"]
    if n == 0:
        return ["empty pairs -- task data missing or malformed"]
    sp = agg["shape_preserved_frac"]
    if sp == 1.0:
        if agg["edit_count_max"] == 0:
            hints.append("identity (every pair has zero edits)")
        elif agg["edit_count_max"] is not None and agg["edit_count_max"] <= 5:
            hints.append(f"single-cell-edit / marker family (max {agg['edit_count_max']} edits/pair)")
        if agg["out_color_strict_subset_of_in"]:
            hints.append("color-filter (output colors strict subset of input colors; some colors zeroed)")
        if agg["in_color_consistent"] and agg["out_color_consistent"]:
            in_set = set(pairs_summary[0]["in_colors"])
            out_set = set(pairs_summary[0]["out_colors"])
            if in_set == out_set:
                hints.append("color-perm / pixel-perm (color sets identical input vs output, shape preserved)")
            elif len(out_set - in_set) > 0:
                hints.append("color-replace / nonzero-recolor (output introduces new colors not in input)")
    else:
        if all(p["out_shape"][0] >= p["in_shape"][0] and p["out_shape"][1] >= p["in_shape"][1] for p in pairs_summary):
            ratios = set()
            for p in pairs_summary:
                ih, iw = p["in_shape"]
                oh, ow = p["out_shape"]
                if ih and iw and oh % ih == 0 and ow % iw == 0:
                    ratios.add((oh // ih, ow // iw))
            if ratios and len(ratios) == 1:
                kh, kw = next(iter(ratios))
                hints.append(f"small-in-large-out (uniform tile {kh}x{kw}; mirror-concat / quad-mirror / tile family)")
        if all(p["out_shape"][0] <= p["in_shape"][0] and p["out_shape"][1] <= p["in_shape"][1] for p in pairs_summary):
            hints.append("crop / extract / slice family (output strictly smaller than input)")
        if not hints:
            hints.append("shape-changing-pixel-perm / asymmetric tile / shape-derived family")

    if len(agg["unique_out_shapes"]) == 1:
        hints.append(f"fixed output shape {agg['unique_out_shapes'][0]} (constant-shape detector preconditions met)")
    if agg["in_color_consistent"]:
        hints.append("input color set consistent across pairs (color-perm / color-replace likely)")

    return hints


def inspect_task(tid, max_pairs=None):
    print(f"\n=== task{tid:03d} ===")
    is_solved = tid in SOLVED_TASKS
    print(f"SOLVED: {'YES' if is_solved else 'no'} (per inspect_tasks.SOLVED_TASKS)")
    try:
        pairs = load_pairs(tid)
    except (OSError, ValueError) as e:
        print(f"  cannot load pairs: {e}")
        return
    print(f"Pairs: {len(pairs)} total (train + test + arc-gen)")
    if not pairs:
        print("  no pairs available")
        return

    summaries = [_per_pair_summary(p) for p in pairs]

    print("\n-- Per-pair (first {}) --".format(max_pairs or len(summaries)))
    print(f"{'i':>4} {'in_shape':>10} {'out_shape':>10} {'edits':>5} {'in_colors':<25} {'out_colors':<25}")
    for i, s in enumerate(summaries[:max_pairs]):
        ic = ",".join(str(c) for c in s["in_colors"])
        oc = ",".join(str(c) for c in s["out_colors"])
        ed = str(s["edit_count"]) if s["edit_count"] is not None else "-"
        print(f"{i:>4} {str(s['in_shape']):>10} {str(s['out_shape']):>10} {ed:>5} {ic:<25} {oc:<25}")
    if max_pairs and len(summaries) > max_pairs:
        print(f"  ... and {len(summaries) - max_pairs} more pairs")

    agg = _aggregate(summaries)
    print("\n-- Aggregate --")
    for k, v in agg.items():
        print(f"  {k}: {v}")

    print("\n-- Op-class hints --")
    for h in _hints(agg, summaries):
        print(f"  - {h}")


def main():
    ap = argparse.ArgumentParser(description="Per-task pre-mining analysis dump.")
    ap.add_argument("task_ids", type=int, nargs="+", help="Task ids to inspect (e.g. 042 153)")
    ap.add_argument("--max-pairs", type=int, default=10, help="Cap number of per-pair rows printed (default 10)")
    args = ap.parse_args()
    for tid in args.task_ids:
        inspect_task(tid, max_pairs=args.max_pairs)


if __name__ == "__main__":
    main()
