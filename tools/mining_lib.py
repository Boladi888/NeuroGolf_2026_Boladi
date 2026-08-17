"""Shared utilities for Beta-track candidate-pattern mining sweeps.

Use this from any new `tools/mine_<angle>.py` so per-tool boilerplate is one
import + a `CANDIDATES` dict + a `main()` that calls `run_sweep` + `print_report`.

Provides:
    DATA_DIR                          -- competition data folder
    SOLVED_TASKS                      -- re-exported from tools/inspect_tasks.py (single source of truth)
    load_pairs(task_id)               -- JSON loader returning train + test + arc-gen pairs
    iter_all_tasks()                  -- generator over all 400 task ids (sorted)
    iter_unsolved_tasks()             -- generator over (all minus SOLVED_TASKS)
    run_sweep(candidates, ...)        -- run each detector against each task; returns {name: [(tid, params)]}
    print_report(results, ...)        -- standard 2-section tabular + detailed output

Example:

    from mining_lib import iter_unsolved_tasks, run_sweep, print_report

    def detect_my_pattern(pairs):
        ...
        return {"key": value}  # match -- truthy params dict
        # or return None for no match

    CANDIDATES = {"my-pattern": detect_my_pattern}

    if __name__ == "__main__":
        targets = list(iter_unsolved_tasks())
        results = run_sweep(CANDIDATES, targets)
        print_report(results, target_n=len(targets))

Detectors may raise -- run_sweep silently swallows exceptions so one bad detector
doesn't kill the sweep. If you need to debug a detector, run it directly outside
run_sweep.
"""

import ast
import json
import os
import re
import sys
from pathlib import Path

NG_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("NEUROGOLF_DATA_DIR", str(NG_ROOT / "data" / "competitions" / "neurogolf-2026")))

sys.path.insert(0, str(NG_ROOT / "tools"))
try:
    from inspect_tasks import SOLVED_TASKS  # noqa: E402
except ModuleNotFoundError as _e:
    # `inspect_tasks` imports `neurogolf_utils` at module load (Alpha-owned, see 07_LANES.md).
    # In worktrees where data/ is not populated, that chain fails. Fall back to a regex
    # parse of the SOLVED_TASKS literal directly from inspect_tasks.py source so the single
    # source of truth is preserved without forcing data/ to be present.
    if "neurogolf_utils" not in str(_e):
        raise
    _src = (NG_ROOT / "tools" / "inspect_tasks.py").read_text()
    _m = re.search(r"SOLVED_TASKS\s*=\s*(\{[^}]+\})", _src)
    if _m is None:
        raise RuntimeError("could not parse SOLVED_TASKS from inspect_tasks.py source") from _e
    SOLVED_TASKS = ast.literal_eval(_m.group(1))


def load_pairs(task_id):
    """Load all pairs for a task: train + test + arc-gen."""
    p = DATA_DIR / f"task{task_id:03d}.json"
    with open(p) as f:
        t = json.load(f)
    return list(t.get("train", [])) + list(t.get("test", [])) + list(t.get("arc-gen", []))


def iter_all_tasks():
    """Yield all 400 task ids sorted."""
    yield from sorted(int(p.stem.replace("task", "")) for p in DATA_DIR.glob("task*.json"))


def iter_unsolved_tasks():
    """Yield unsolved task ids (all 400 minus SOLVED_TASKS)."""
    for tid in iter_all_tasks():
        if tid not in SOLVED_TASKS:
            yield tid


def run_sweep(candidates, targets=None):
    """Run each detector in `candidates` against each task in `targets`.

    candidates: dict[str, callable(pairs) -> Optional[dict]]
    targets:    iterable of task ids; defaults to iter_unsolved_tasks()

    Returns dict[name -> list[(task_id, params_dict)]] for each detector that matched.
    """
    if targets is None:
        targets = list(iter_unsolved_tasks())
    results = {name: [] for name in candidates}
    for tid in targets:
        try:
            pairs = load_pairs(tid)
        except (OSError, json.JSONDecodeError):
            continue
        for name, detect in candidates.items():
            try:
                r = detect(pairs)
                if r is not None:
                    results[name].append((tid, r))
            except Exception:
                pass
    return results


def print_report(results, target_n=None, max_sample=8, max_detail=25):
    """Print mining results in 2 sections: summary table + per-pattern detailed hits."""
    if target_n is not None:
        print(f"=== Mining report ({target_n} tasks scanned, SOLVED count: {len(SOLVED_TASKS)}) ===\n")
    else:
        print(f"=== Mining report (SOLVED count: {len(SOLVED_TASKS)}) ===\n")

    name_w = max((len(n) for n in results), default=10)
    name_w = max(name_w, 10)
    print(f"{'Pattern':<{name_w}} {'Hits':>5}  Sample task ids")
    print(f"{'-' * name_w} {'-' * 5}  {'-' * 40}")
    for name, hits in results.items():
        sample = ", ".join(str(t) for t, _ in hits[:max_sample])
        if len(hits) > max_sample:
            sample += f", ... (+{len(hits) - max_sample} more)"
        print(f"{name:<{name_w}} {len(hits):>5}  {sample}")

    print()
    print("=== Detailed hits ===")
    for name, hits in results.items():
        if not hits:
            continue
        print(f"\n--- {name} ({len(hits)} hits) ---")
        for tid, params in hits[:max_detail]:
            print(f"  task{tid:03d}: {params}")
        if len(hits) > max_detail:
            print(f"  ... and {len(hits) - max_detail} more")
