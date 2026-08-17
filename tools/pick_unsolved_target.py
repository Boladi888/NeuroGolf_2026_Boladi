#!/usr/bin/env python3
"""Pick a high-tractability unsolved task for Pool B (workers 7-8).

Reads `tools/inspect_tasks.SOLVED_TASKS`, then for each unsolved task computes:
- size consistency (input shape stable, output shape stable)
- same-size (input shape == output shape)
- color count distribution
- whether output appears to be a deterministic function of input shape

Outputs a ranked list — lowest-difficulty tasks first (best tractability heuristic).
Pool B orchestrator should pick from the top of the list.

Run:
  .venv/Scripts/python tools/pick_unsolved_target.py --top 10
  .venv/Scripts/python tools/pick_unsolved_target.py --json --top 30
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import types
from collections import Counter
from typing import Any

NG_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = pathlib.Path(
    os.environ.get("NEUROGOLF_DATA_DIR", str(NG_ROOT / "data" / "competitions" / "neurogolf-2026"))
)
for stub in ("matplotlib", "matplotlib.pyplot", "IPython", "IPython.display"):
    sys.modules.setdefault(stub, types.ModuleType(stub))
sys.modules["IPython.display"].display = lambda *a, **kw: None  # type: ignore[attr-defined]
sys.modules["IPython.display"].FileLink = lambda *a, **kw: None  # type: ignore[attr-defined]
sys.modules["IPython"].display = sys.modules["IPython.display"]  # type: ignore[attr-defined]
sys.modules["matplotlib"].pyplot = sys.modules["matplotlib.pyplot"]  # type: ignore[attr-defined]

sys.path.insert(0, str(DATA_DIR / "neurogolf_utils"))
sys.path.insert(0, str(NG_ROOT))
sys.path.insert(0, str(NG_ROOT / "tools"))

import neurogolf_utils as nu  # type: ignore  # noqa: E402

nu._NEUROGOLF_DIR = DATA_DIR.as_posix() + "/"  # type: ignore[attr-defined]
from inspect_tasks import SOLVED_TASKS  # type: ignore  # noqa: E402


def grid_shape(g: list[list[int]]) -> tuple[int, int]:
    return (len(g), len(g[0]) if g else 0)


def task_features(task_id: int) -> dict[str, Any]:
    try:
        ex = nu.load_examples(task_id)
    except Exception as exc:
        return {"task": f"task{task_id:03d}", "error": str(exc)}
    pairs = list(ex.get("train", []) or []) + list(ex.get("test", []) or []) + list(ex.get("arc-gen", []) or [])
    if not pairs:
        return {"task": f"task{task_id:03d}", "error": "no pairs"}
    in_shapes = {grid_shape(p["input"]) for p in pairs}
    out_shapes = {grid_shape(p["output"]) for p in pairs}
    same_size = all(grid_shape(p["input"]) == grid_shape(p["output"]) for p in pairs)
    avg_in_cells = sum(grid_shape(p["input"])[0] * grid_shape(p["input"])[1] for p in pairs) / len(pairs)
    in_colors = {v for p in pairs for row in p["input"] for v in row}
    out_colors = {v for p in pairs for row in p["output"] for v in row}
    color_subset = out_colors.issubset(in_colors)
    # Tractability score: lower = easier. Penalize many distinct shapes and large grids.
    difficulty = len(in_shapes) * 4 + len(out_shapes) * 4 + (0 if same_size else 6) + avg_in_cells / 100
    if not color_subset:
        difficulty += 5  # output introduces new colors -> harder
    return {
        "task": f"task{task_id:03d}",
        "n_pairs": len(pairs),
        "input_shapes": sorted(list(in_shapes))[:4],
        "output_shapes": sorted(list(out_shapes))[:4],
        "same_size": same_size,
        "n_in_shapes": len(in_shapes),
        "n_out_shapes": len(out_shapes),
        "avg_in_cells": round(avg_in_cells, 1),
        "n_in_colors": len(in_colors),
        "n_out_colors": len(out_colors),
        "output_colors_subset_of_input": color_subset,
        "difficulty": round(difficulty, 2),
    }


def rank(top: int) -> list[dict[str, Any]]:
    solved = set(SOLVED_TASKS)
    rows = []
    for tid in range(1, 401):
        if tid in solved:
            continue
        rows.append(task_features(tid))
    rows = [r for r in rows if "error" not in r]
    rows.sort(key=lambda r: r["difficulty"])
    return rows[:top]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=20, help="How many candidates to print")
    parser.add_argument("--json", action="store_true", help="Emit JSON list")
    args = parser.parse_args(argv)
    rows = rank(args.top)
    if args.json:
        json.dump(rows, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    print(f"Pool B target ranking (top {args.top} of {400 - len(set(SOLVED_TASKS))} unsolved)")
    print(f"  difficulty = #shapes*4 + same_size_bonus + cells/100 + color_set_penalty")
    print(f"  {'task':8} {'diff':>5}  pairs  in_sh  out_sh same  cells  in_c  out_c  csub")
    for r in rows:
        print(
            f"  {r['task']:8} {r['difficulty']:>5.1f}  {r['n_pairs']:>5}  {r['n_in_shapes']:>5}  {r['n_out_shapes']:>5}  "
            f"{('Y' if r['same_size'] else 'N'):>4}  {r['avg_in_cells']:>5}  {r['n_in_colors']:>4}  {r['n_out_colors']:>5}  "
            f"{('Y' if r['output_colors_subset_of_input'] else 'N'):>4}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
