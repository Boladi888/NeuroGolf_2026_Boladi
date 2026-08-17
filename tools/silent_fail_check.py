"""Silent-fail check: per-task pass-rate against the bundled arc-gen pairs.

Defensive pre-submission harness. For each task in SOLVED_TASKS, loads the
existing submissions/taskNNN.onnx, runs onnxruntime over train + test +
arc-gen pairs from data/competitions/neurogolf-2026/taskNNN.json, reports
per-set pass counts. Flags tasks whose arc-gen pass-rate is below a
threshold (default 0.99) as silent-fail risk -- patterns that pass our
visible-pair verify but may fail Kaggle's hidden private benchmark.

The bundled arc-gen subset has ~261-267 pairs per task. neurogolf_utils.py
exposes no pair-sampling helper. For higher-N stress tests, generate locally
via google/ARC-GEN (`arc_gen.py generate <task_num> <N>`) and merge into
data/competitions/neurogolf-2026/taskNNN.json's "arc-gen" list before re-run.

Usage:
    python tools/silent_fail_check.py
    python tools/silent_fail_check.py --threshold 0.999
    python tools/silent_fail_check.py --tasks 16,53,266
    python tools/silent_fail_check.py --out Logs/sfc_pre_grape.csv

Exit code:
    0  every task passes the threshold on every pair set.
    1  at least one task flagged as silent-fail risk OR an error occurred.

Pre-requisite:
    submissions/taskNNN.onnx must exist for each task being checked. Run
    `python build_submission.py --hybrid` first if missing.
"""

import argparse
import csv
import datetime
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
SUBMISSIONS_DIR = NG_ROOT / "submissions"
LOGS_DIR = NG_ROOT / "Logs"
sys.path.insert(0, str(DATA_DIR / "neurogolf_utils"))
sys.path.insert(0, str(NG_ROOT / "tools"))

import neurogolf_utils  # noqa: E402

neurogolf_utils._NEUROGOLF_DIR = f"{DATA_DIR.as_posix()}/"

import onnxruntime  # noqa: E402

from inspect_tasks import SOLVED_TASKS  # noqa: E402


_SETS = ("train", "test", "arc-gen")

# Tasks flagged by external competitors as "verify_subset passes locally but
# Kaggle private-test scores zero" — silent-fail risk on the hidden benchmark.
# Sources:
#   the-epic-sax-ensembler.ipynb VALIDATOR_BAD_TASK_IDS: 016, 050, 276, 309, 337
#   acking-multiple-tasks-part-3.ipynb SKIP_TASKS:       294, 331, 373
#   Russian SKIP_TASKS:                                  053, 129, 266, 385
# See `Research_Archive/notes/deep-read-the-epic-sax-ensembler.md` and
# `Research_Archive/notes/deep-read-acking-multiple-tasks-part-3.md` for cross-refs.
COMPETITOR_FLAGGED_TASKS = frozenset(
    {16, 50, 53, 129, 266, 276, 294, 309, 331, 337, 373, 385}
)


def _check_one(task_num, onnx_path):
    """Return per-set (right, wrong) tuples plus error string (or None).

    Returns dict like:
        {"train": (right, wrong), "test": (...), "arc-gen": (...), "error": None}
    On error, all per-set tuples are (0, 0) and "error" is the message.
    """
    try:
        examples = neurogolf_utils.load_examples(task_num)
    except (FileNotFoundError, OSError) as e:
        return {s: (0, 0) for s in _SETS} | {"error": f"load_examples: {e}"}

    try:
        session = onnxruntime.InferenceSession(str(onnx_path))
    except Exception as e:  # noqa: BLE001 (onnxruntime raises a private exception class)
        return {s: (0, 0) for s in _SETS} | {"error": f"InferenceSession: {e}"}

    out = {"error": None}
    for s in _SETS:
        pairs = examples.get(s, [])
        if not pairs:
            out[s] = (0, 0)
            continue
        try:
            right, wrong, _ = neurogolf_utils.verify_subset(session, pairs)
        except Exception as e:  # noqa: BLE001
            return {s: (0, 0) for s in _SETS} | {"error": f"verify_subset on {s}: {e}"}
        out[s] = (right, wrong)
    return out


def _rate(right, total):
    return float(right) / total if total else 1.0


def run(tasks, threshold, out_path, verbose=True, min_arcgen_n=0):
    """Run per-task pass-rate check. Returns (rows, n_at_risk, n_errors, n_low_confidence).

    Args:
        min_arcgen_n: if > 0, tasks with fewer than this many arc-gen pairs are
            flagged as `low_confidence` (sample size insufficient to constrain
            generalization to a hidden test set), in addition to the threshold check.
    """
    rows = []
    n_at_risk = 0
    n_errors = 0
    n_low_confidence = 0

    if verbose:
        print(f"Silent-fail check: {len(tasks)} tasks, arc-gen threshold = {threshold:.4f}"
              + (f", min-arcgen-N = {min_arcgen_n}" if min_arcgen_n > 0 else ""))
        print()

    for task_num in tasks:
        onnx_path = SUBMISSIONS_DIR / f"task{task_num:03d}.onnx"
        if not onnx_path.exists():
            if verbose:
                print(f"  task{task_num:03d}: MISSING {onnx_path.name} (run build_submission.py --hybrid first)")
            n_errors += 1
            rows.append({
                "task": task_num,
                "train_right": 0, "train_total": 0,
                "test_right": 0, "test_total": 0,
                "arcgen_right": 0, "arcgen_total": 0,
                "arcgen_rate": 0.0,
                "silent_fail_risk": True,
                "low_confidence": False,
                "error": "missing onnx",
            })
            continue

        result = _check_one(task_num, onnx_path)
        if result["error"]:
            n_errors += 1
            if verbose:
                print(f"  task{task_num:03d}: ERROR {result['error']}")
            rows.append({
                "task": task_num,
                "train_right": 0, "train_total": 0,
                "test_right": 0, "test_total": 0,
                "arcgen_right": 0, "arcgen_total": 0,
                "arcgen_rate": 0.0,
                "silent_fail_risk": True,
                "low_confidence": False,
                "error": result["error"],
            })
            continue

        tr_r, tr_w = result["train"]
        te_r, te_w = result["test"]
        ag_r, ag_w = result["arc-gen"]
        ag_total = ag_r + ag_w
        ag_rate = _rate(ag_r, ag_total)
        at_risk = ag_rate < threshold or tr_w > 0 or te_w > 0
        low_conf = min_arcgen_n > 0 and ag_total < min_arcgen_n
        if at_risk:
            n_at_risk += 1
        if low_conf:
            n_low_confidence += 1

        if verbose:
            if at_risk:
                tag = "  RISK"
            elif low_conf:
                tag = " LOW-N"
            else:
                tag = "    ok"
            watch = " [WATCH]" if task_num in COMPETITOR_FLAGGED_TASKS else ""
            print(
                f"  task{task_num:03d}{tag}{watch}  "
                f"train {tr_r:>3}/{tr_r+tr_w:<3}  "
                f"test {te_r:>3}/{te_r+te_w:<3}  "
                f"arc-gen {ag_r:>4}/{ag_total:<4}  "
                f"({ag_rate*100:6.2f}%)"
            )

        rows.append({
            "task": task_num,
            "train_right": tr_r, "train_total": tr_r + tr_w,
            "test_right": te_r, "test_total": te_r + te_w,
            "arcgen_right": ag_r, "arcgen_total": ag_total,
            "arcgen_rate": round(ag_rate, 6),
            "silent_fail_risk": at_risk,
            "low_confidence": low_conf,
            "error": "",
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            "task", "train_right", "train_total", "test_right", "test_total",
            "arcgen_right", "arcgen_total", "arcgen_rate", "silent_fail_risk",
            "low_confidence", "error",
        ])
        writer.writeheader()
        writer.writerows(rows)

    if verbose:
        print()
        print(f"  CSV: {out_path.relative_to(NG_ROOT).as_posix()}")
        print(f"  At-risk tasks: {n_at_risk}/{len(tasks)}    "
              f"low-confidence: {n_low_confidence}/{len(tasks)}    "
              f"errors: {n_errors}/{len(tasks)}")
        watchlist_in_run = sorted({r["task"] for r in rows} & COMPETITOR_FLAGGED_TASKS)
        if watchlist_in_run:
            print(f"  [WATCH] external-competitor-flagged SOLVED tasks in this run "
                  f"({len(watchlist_in_run)}): {watchlist_in_run}. Even with arc-gen "
                  f"100% pass, treat as silent-fail-risk on Kaggle hidden test "
                  f"(per Research_Archive/notes/deep-read-the-epic-sax-ensembler.md +"
                  f" deep-read-acking-multiple-tasks-part-3.md). Cross-check before submit.")
        if n_at_risk == 0 and n_errors == 0 and n_low_confidence == 0:
            print(f"  All {len(tasks)} tasks pass arc-gen at >= {threshold*100:.2f}%. Submission-safe.")
        elif n_at_risk == 0 and n_errors == 0:
            low_n = sorted(r["task"] for r in rows if r.get("low_confidence"))
            print(f"  No risk-flagged tasks; LOW-N tasks need arc-gen enrichment before submission: {low_n}")
        else:
            risky = sorted(r["task"] for r in rows if r["silent_fail_risk"])
            print(f"  Hold submission. Risky tasks: {risky}")

    return rows, n_at_risk, n_errors, n_low_confidence


def _parse_tasks(arg):
    """Parse --tasks arg ('16,53,266') or fall back to SOLVED_TASKS."""
    if not arg:
        return sorted(SOLVED_TASKS)
    return sorted({int(t) for t in arg.split(",") if t.strip()})


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--threshold", type=float, default=0.99,
                        help="Minimum acceptable arc-gen pass-rate (default 0.99)")
    parser.add_argument("--min-arcgen-N", type=int, default=0,
                        help="Tasks with fewer than this many arc-gen pairs are flagged as "
                             "LOW-CONFIDENCE (sample size insufficient to constrain "
                             "generalization to a hidden test set). Default 0 (off).")
    parser.add_argument("--tasks", type=str, default="",
                        help="Comma-separated task ids (default: SOLVED_TASKS from tools/inspect_tasks.py)")
    parser.add_argument("--out", type=str, default="",
                        help="Output CSV path (default: Logs/silent_fail_<UTC-iso>.csv)")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-task prints")
    args = parser.parse_args()

    tasks = _parse_tasks(args.tasks)
    if not tasks:
        print("No tasks selected.", file=sys.stderr)
        sys.exit(2)

    if args.out:
        out_path = pathlib.Path(args.out)
        if not out_path.is_absolute():
            out_path = NG_ROOT / out_path
    else:
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
        out_path = LOGS_DIR / f"silent_fail_{stamp}.csv"

    _, n_at_risk, n_errors, n_low_conf = run(
        tasks, args.threshold, out_path,
        verbose=not args.quiet,
        min_arcgen_n=getattr(args, "min_arcgen_N"),
    )
    # Exit 0 only when no risk, no errors, and no low-confidence flagged.
    sys.exit(0 if (n_at_risk == 0 and n_errors == 0 and n_low_conf == 0) else 1)


if __name__ == "__main__":
    main()
