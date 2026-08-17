"""Report LUT key-space coverage for LUT-based solved tasks.

Several solved tasks rely on a learned look-up table (grid-pattern-to-color,
grid-pattern-to-grid, grid-pattern-color-to-grid, bin-nbr-lut). When the
observed key set is a small fraction of the LUT's full key space, unobserved
keys default to color 0 / all-zero output. If Kaggle's hidden private set has
a key not in our observed set, the network silently scores 0 on that task.

`silent_fail_check.py` cannot detect this — it only validates against observed
arc-gen pairs. This tool surfaces the generalization risk by reporting the
LUT coverage explicitly: if too low, recommend arc-gen enrichment via
`google/ARC-GEN`'s `arc_gen.py generate <task_num> <N>` before submission.

Usage:
    python tools/lut_density_check.py
    python tools/lut_density_check.py --threshold 0.05    # 5% coverage gate
    python tools/lut_density_check.py --tasks 104,262

Exit code:
    0  every LUT task at or above threshold.
    1  at least one LUT task has coverage < threshold.
    2  invocation / setup error.
"""

import argparse
import os
import pathlib
import sys
import types

for _name in ("matplotlib", "matplotlib.pyplot", "IPython", "IPython.display"):
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules["IPython.display"].display = lambda *a, **kw: None
sys.modules["IPython.display"].FileLink = lambda *a, **kw: None
sys.modules["IPython"].display = sys.modules["IPython.display"]
sys.modules["matplotlib"].pyplot = sys.modules["matplotlib.pyplot"]

NG_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = pathlib.Path(os.environ.get("NEUROGOLF_DATA_DIR", str(NG_ROOT / "data" / "competitions" / "neurogolf-2026")))
sys.path.insert(0, str(DATA_DIR / "neurogolf_utils"))
sys.path.insert(0, str(NG_ROOT))
sys.path.insert(0, str(NG_ROOT / "tools"))

import neurogolf_utils as nu  # noqa: E402

nu._NEUROGOLF_DIR = f"{DATA_DIR.as_posix()}/"

import build_submission as bs  # noqa: E402
from inspect_tasks import SOLVED_TASKS  # noqa: E402


def lut_coverage_for_task(task_num):
    """Return dict {detector_label, observed, key_space, coverage} or None if no LUT detector matches."""
    ex = nu.load_examples(task_num)
    pairs = ex.get("train", []) + ex.get("test", []) + ex.get("arc-gen", [])
    if not pairs:
        return None
    # Probe each LUT detector. The dispatcher's chosen one is the relevant one.
    label, _ = bs.detect_pattern(ex)

    if label == "grid-pattern-to-color":
        r = bs._detect_grid_pattern_to_color(pairs)
        if r is None:
            return None
        in_h, in_w, lut = r
        return {
            "task": task_num, "detector": label,
            "in_shape": (in_h, in_w),
            "observed": len(lut), "key_space": 2 ** (in_h * in_w),
        }
    if label == "grid-pattern-to-grid":
        r = bs._detect_grid_pattern_to_grid(pairs)
        if r is None:
            return None
        in_h, in_w, _, _, lut = r
        return {
            "task": task_num, "detector": label,
            "in_shape": (in_h, in_w),
            "observed": len(lut), "key_space": 2 ** (in_h * in_w),
        }
    if label == "grid-pattern-color-to-grid":
        r = bs._detect_grid_pattern_color_to_grid(pairs)
        if r is None:
            return None
        in_h, in_w, _, _, lut = r
        return {
            "task": task_num, "detector": label,
            "in_shape": (in_h, in_w),
            "observed": len(lut), "key_space": 2 ** (in_h * in_w) * 9,
        }
    if label == "bin-nbr-lut":
        r = bs._detect_binary_neighborhood_lut(pairs)
        if r is None:
            return None
        _, lut = r
        in_h = len(pairs[0]["input"])
        in_w = len(pairs[0]["input"][0])
        return {
            "task": task_num, "detector": label,
            "in_shape": (in_h, in_w),
            "observed": len(lut), "key_space": 512,  # bin-nbr-lut always 9-bit
        }
    return None


def run(tasks, threshold, verbose=True):
    """Return list of dicts for every LUT task; flag those below threshold."""
    rows = []
    for tn in tasks:
        info = lut_coverage_for_task(tn)
        if info is None:
            continue
        info["coverage"] = info["observed"] / info["key_space"]
        info["below_threshold"] = info["coverage"] < threshold
        rows.append(info)

    if verbose:
        print(f"LUT density check: {len(rows)} LUT-based tasks, threshold = {threshold:.2%}")
        print()
        print(f"  {'task':>5}  {'detector':<26} {'in':>7}  {'observed':>9}  {'key_space':>10}  {'coverage':>10}")
        print("  " + "-" * 75)
        for r in sorted(rows, key=lambda x: x["coverage"]):
            tag = "LOW" if r["below_threshold"] else " ok"
            print(f"  {r['task']:>5}  {r['detector']:<26} "
                  f"{str(r['in_shape']):>7}  "
                  f"{r['observed']:>9}  {r['key_space']:>10}  "
                  f"{r['coverage']:>9.2%}  {tag}")

    n_low = sum(1 for r in rows if r["below_threshold"])
    if verbose:
        print()
        if n_low == 0:
            print(f"  All {len(rows)} LUT tasks have coverage >= {threshold:.2%}. Generalization-safe.")
        else:
            risky = sorted(r["task"] for r in rows if r["below_threshold"])
            print(f"  {n_low}/{len(rows)} LUT tasks below {threshold:.2%} coverage: {risky}")
            print(f"  Mitigation: run google/ARC-GEN's `arc_gen.py generate <task> <N>` to enrich")
            print(f"              the arc-gen subset before submission, then rebuild via --hybrid.")

    return rows, n_low


def _parse_tasks(arg):
    if not arg:
        return sorted(SOLVED_TASKS)
    return sorted({int(t) for t in arg.split(",") if t.strip()})


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tasks", type=str, default="",
                        help="Comma-separated task ids (default: SOLVED_TASKS)")
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="Coverage threshold (default 0.05 = 5%%)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    tasks = _parse_tasks(args.tasks)
    _, n_low = run(tasks, args.threshold, verbose=not args.quiet)
    sys.exit(1 if n_low > 0 else 0)


if __name__ == "__main__":
    main()
