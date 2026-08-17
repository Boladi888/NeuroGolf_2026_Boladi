"""Blend our submissions/ with verified external ONNXs from competitor artifacts.

Pipeline:
  1. Start with `submissions/taskNNN.onnx` (built by `build_submission.py --hybrid`).
     By default, our SOLVED_TASKS are NEVER overridden — our patterns are sacred.
     Pass `--allow-cheaper-externals` to additionally override SOLVED entries
     where an external is BOTH strictly cheaper AND passes verify_subset 100%
     on visible+arc-gen. Worst case if a cheaper external silent-fails Kaggle's
     hidden test: we lose that task entirely (forfeit our organic pts on it).
  2. For each task we don't yet solve, scan candidate ONNXs from competitor
     dataset directories under `data/competitor_artifacts/`:
       a. Triage via `onnx_repair.triage()` — must be READY or REPAIRABLE-RENAME.
       b. If REPAIRABLE-RENAME, apply the I/O-rename pass.
       c. Run `neurogolf_utils.score_network` — must return non-None.
       d. Run `neurogolf_utils.verify_subset` on train + test + arc-gen — ALL pairs must pass
          (capped via `--max-pairs` for speed; default 100, use `--full` for all).
       e. Compute cost.
  3. For each unsolved task with >=1 passing candidate, pick the cheapest.
  4. Emit `submissions/blend.zip` + a CSV report at `Logs/external_blend_<UTC-iso>.csv`.

Strict policy (ethical-play stance per `03_PROJECT.md` Open bugs):
  - Banned-op (Compress, Loop, Scan, ...) and dynamic-shape candidates SKIPPED.
  - Identity placeholders SKIPPED.
  - Tool does NOT submit to Kaggle. Owner decides whether to upload.
  - Use of public Apache-2.0 / CC0 artifacts as fallback is composition, not exploit.

Default sources (in priority order — first source's candidate wins on ties):
  data/competitor_artifacts/ashok205-neuro-golf-2026/submission
  data/competitor_artifacts/karnak-task-transformation/submission
  data/competitor_artifacts/afr1ste-6285-95-artifact

Usage:
    python tools/external_blend.py                              # default blend
    python tools/external_blend.py --max-pairs 50               # cap arc-gen verify per candidate
    python tools/external_blend.py --full                       # uncapped (slow; ~265 pairs/task)
    python tools/external_blend.py --out submissions/blend.zip  # custom output
    python tools/external_blend.py --dry-run                    # report only, no zip
    python tools/external_blend.py --tasks 50,81,206            # restrict to these task ids
"""

import argparse
import csv
import datetime
import math
import os
import pathlib
import sys
import types
import zipfile

import onnx
import onnx.shape_inference

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
ARTIFACTS_DIR = pathlib.Path(os.environ.get("NEUROGOLF_COMPETITOR_ARTIFACTS_DIR", str(NG_ROOT / "data" / "competitor_artifacts")))
sys.path.insert(0, str(DATA_DIR / "neurogolf_utils"))
sys.path.insert(0, str(NG_ROOT / "tools"))

import neurogolf_utils  # noqa: E402

neurogolf_utils._NEUROGOLF_DIR = f"{DATA_DIR.as_posix()}/"

import onnxruntime  # noqa: E402

from inspect_tasks import SOLVED_TASKS  # noqa: E402
from onnx_repair import triage, repair_io_naming  # noqa: E402

DEFAULT_SOURCES = [
    ("ashok",  ARTIFACTS_DIR / "ashok205-neuro-golf-2026" / "submission"),
    ("karnak", ARTIFACTS_DIR / "karnak-task-transformation" / "submission"),
    ("afr1ste", ARTIFACTS_DIR / "afr1ste-6285-95-artifact"),
    ("needless090", ARTIFACTS_DIR / "needless090-v31"),
    ("konbu17", ARTIFACTS_DIR / "konbu17-401-v117"),
    ("thisray4808", ARTIFACTS_DIR / "thisray-4808" / "submission"),
]

NATIVE_CRASH_SKIP = {
    # Crashes the Windows onnxruntime process during verify_subset instead of
    # raising an exception we can catch.
    ("needless090", 19): "onnxruntime native crash during visible verify",
    ("thisray4808", 19): "onnxruntime native crash during visible verify",
    ("thisray4808", 137): "onnxruntime native crash during visible verify",
    ("thisray4808", 175): "onnxruntime native crash during visible verify",
    ("thisray4808", 297): "onnxruntime native crash during visible verify",
    ("afr1ste", 394): "onnxruntime native crash during visible verify",
    ("konbu17", 394): "onnxruntime native crash during visible verify",
}


_CASCADE_MEM_THRESHOLD = 200_000  # bytes; max intermediate tensor size above which we flag CASCADE-RISK


def audit_external_cascade(path):
    """Return 'CASCADE-SAFE' | 'CASCADE-RISK' | 'BANNED' for an external ONNX.

    - BANNED: score_network_v2 rejects, or shape inference fails, or dynamic dims.
    - CASCADE-RISK: cost ≤ 1000 (zero-cost trigger) OR any intermediate tensor > threshold.
    - CASCADE-SAFE: cost > 1000 AND all intermediates ≤ threshold.

    Intended for standalone auditing; _verify_candidate() integrates the same logic
    inline to avoid double-scoring.
    """
    from _score_v2 import score_network_v2  # noqa: E402
    try:
        result = score_network_v2(str(path))
    except Exception:  # noqa: BLE001
        return "BANNED"
    if result is None or result[0] is None:
        return "BANNED"
    _, mem, params = result
    cost = mem + params
    if cost < 1000:
        return "CASCADE-RISK"
    try:
        model = onnx.load(str(path))
        inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    except Exception:  # noqa: BLE001
        return "BANNED"
    max_intermediate = 0
    for vi in inferred.graph.value_info:
        if vi.name in ("input", "output"):
            continue
        if not vi.type.HasField("tensor_type"):
            continue
        dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        if any(d <= 0 for d in dims):
            return "BANNED"
        sz = 4  # rough dtype_bytes (float32)
        for d in dims:
            sz *= d
        max_intermediate = max(max_intermediate, sz)
    if max_intermediate > _CASCADE_MEM_THRESHOLD:
        return "CASCADE-RISK"
    return "CASCADE-SAFE"


def _candidate_bytes(path):
    """Triage + optional rename, return (raw_bytes, status, source_size, rename_applied)."""
    r = triage(path)
    if r["status"] == "READY":
        return pathlib.Path(path).read_bytes(), "READY", r["file_size"], False
    if r["status"] == "REPAIRABLE-RENAME":
        ok, fixed, _ = repair_io_naming(path)
        if ok:
            return fixed, "RENAMED", len(fixed), True
        return None, "RENAME-FAILED", r["file_size"], False
    return None, r["status"], r["file_size"], False


def _verify_candidate(raw_bytes, task_num, max_pairs, include_risky=False):
    """Run score_network + cascade audit + verify_subset on the candidate.

    Returns (cost_tuple, "ok") or (None, reason_str) on rejection.
    When include_risky=False (default), CASCADE-RISK and BANNED externals are
    rejected before running the expensive verify_subset.
    """
    tmp = NG_ROOT / "submissions" / f"_blend_tmp_{task_num:03d}.onnx"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    try:
        tmp.write_bytes(raw_bytes)
        try:
            if not neurogolf_utils.check_network(str(tmp)):
                return None, "check_network failed"
        except Exception as e:  # noqa: BLE001
            return None, f"check_network exception: {e}"
        from _score_v2 import score_network_v2  # noqa: E402
        try:
            result = score_network_v2(str(tmp))
        except Exception as e:  # noqa: BLE001 (onnx_tool raises ValueError, KeyError, etc.)
            return None, f"score_network exception: {e}"
        if result is None or any(v is None for v in result):
            return None, "score_network rejected"
        macs, memory, params = result
        cost = memory + params

        # Cascade safety filter (skipped if --include-risky)
        if not include_risky:
            if cost < 1000:
                return None, f"cascade-RISK (cost={cost} < 1000)"
            try:
                model = onnx.load(str(tmp))
                inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
                max_intermediate = 0
                for vi in inferred.graph.value_info:
                    if vi.name in ("input", "output"):
                        continue
                    if not vi.type.HasField("tensor_type"):
                        continue
                    dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
                    if any(d <= 0 for d in dims):
                        return None, "cascade-BANNED (dynamic intermediate shape)"
                    sz = 4
                    for d in dims:
                        sz *= d
                    max_intermediate = max(max_intermediate, sz)
                if max_intermediate > _CASCADE_MEM_THRESHOLD:
                    return None, f"cascade-RISK (max_intermediate={max_intermediate})"
            except Exception as e:  # noqa: BLE001
                return None, f"cascade-audit exception: {e}"

        # Functional verify
        try:
            session = onnxruntime.InferenceSession(raw_bytes)
        except Exception as e:  # noqa: BLE001
            return None, f"InferenceSession: {e}"
        examples = neurogolf_utils.load_examples(task_num)
        for set_name in ("train", "test", "arc-gen"):
            pairs = examples.get(set_name, [])
            if not pairs:
                continue
            sub = pairs if max_pairs is None else pairs[:max_pairs]
            try:
                right, wrong, _ = neurogolf_utils.verify_subset(session, sub)
            except Exception as e:  # noqa: BLE001
                return None, f"verify_subset {set_name}: {e}"
            if wrong > 0:
                return None, f"verify_subset {set_name} {right}/{right+wrong}"
        # 2026-05-04: cost = memory + params (MACs removed from objective).
        points = round(max(1.0, 25.0 - math.log(max(1, cost))), 4)
        return (int(cost), points, int(macs), int(memory), int(params)), "ok"
    finally:
        try:
            tmp.unlink()
        except (FileNotFoundError, OSError):
            pass


def _scan_candidates_for_task(task_num, sources, max_pairs, verbose=False, include_risky=False):
    """Return list of (source_label, file, cost_tuple) for every candidate that passes."""
    results = []
    for src_label, src_dir in sources:
        skip_reason = NATIVE_CRASH_SKIP.get((src_label, task_num))
        if skip_reason:
            if verbose:
                print(f"    [{src_label}] SKIP {skip_reason}")
            continue
        path = pathlib.Path(src_dir) / f"task{task_num:03d}.onnx"
        if not path.exists():
            continue
        raw, status, src_size, rename_applied = _candidate_bytes(path)
        if raw is None:
            if verbose:
                print(f"    [{src_label}] SKIP {status} (sz {src_size})")
            continue
        cost_tuple, reason = _verify_candidate(raw, task_num, max_pairs, include_risky=include_risky)
        if cost_tuple is None:
            if verbose:
                print(f"    [{src_label}] FAIL {reason} (sz {src_size}, status {status})")
            continue
        cost, points, macs, mem, params = cost_tuple
        results.append({
            "src": src_label,
            "raw": raw,
            "src_size": src_size,
            "renamed": rename_applied,
            "cost": cost,
            "points": points,
            "macs": macs,
            "memory": mem,
            "params": params,
        })
        if verbose:
            print(f"    [{src_label}] OK   cost={cost:>9} pts={points:.3f} (sz {src_size})")
    return results


def run(sources, max_pairs, out_zip, report_csv, target_tasks=None, dry_run=False, verbose=True,
        allow_cheaper_externals=False, include_risky=False):
    """Main blend pipeline. Returns dict of stats.

    `allow_cheaper_externals=True` extends the scan to our SOLVED_TASKS too, but
    only overrides our submission when the external is BOTH cheaper AND passes
    `verify_subset` 100% on visible+arc-gen. Worst case if a cheaper external
    silent-fails Kaggle's hidden test: we lose that task entirely (forfeit our
    organic pts on it). Enabled at owner's discretion.
    """
    if target_tasks is None:
        if allow_cheaper_externals:
            # Scan ALL 400 tasks: cheaper-and-verified externals can override SOLVED.
            target_tasks = list(range(1, 401))
        else:
            # Default: every task we don't already solve.
            target_tasks = sorted(set(range(1, 401)) - SOLVED_TASKS)

    base_zip = {}
    for n in range(1, 401):
        p = SUBMISSIONS_DIR / f"task{n:03d}.onnx"
        if p.exists():
            base_zip[n] = p.read_bytes()

    # Pre-compute our own per-task cost so we can compare on overlap.
    our_costs = {}
    if allow_cheaper_externals:
        from _score_v2 import score_network_v2  # noqa: E402
        for n in range(1, 401):
            if n not in SOLVED_TASKS:
                continue
            p = SUBMISSIONS_DIR / f"task{n:03d}.onnx"
            if not p.exists():
                continue
            try:
                m, mem, pa = score_network_v2(str(p))
                if m is not None:
                    our_costs[n] = mem + pa
            except Exception:  # noqa: BLE001
                continue

    if verbose:
        print(f"External blend: scanning {len(target_tasks)} candidate tasks across {len(sources)} sources")
        print(f"  max_pairs per arc-gen verify: {max_pairs if max_pairs else 'all (full)'}")
        print(f"  allow_cheaper_externals: {allow_cheaper_externals}")
        print(f"  cascade filter: {'DISABLED (--include-risky)' if include_risky else f'ENABLED (reject cost<1000 or intermediates>{_CASCADE_MEM_THRESHOLD} bytes)'}")
        print()

    rows = []
    n_added = 0
    n_no_candidate = 0
    n_overrode_solved = 0
    overrode_pts_delta = 0.0
    blend_pts_added = 0.0

    for tn in target_tasks:
        is_solved = tn in SOLVED_TASKS
        if is_solved and not allow_cheaper_externals:
            continue
        if verbose and tn % 25 == 0:
            print(f"  [task{tn:03d}/{len(target_tasks)}] running...", flush=True)
        cands = _scan_candidates_for_task(tn, sources, max_pairs, verbose=False, include_risky=include_risky)
        if not cands:
            if not is_solved:
                n_no_candidate += 1
                rows.append({
                    "task": tn,
                    "winner_src": "",
                    "winner_cost": -1,
                    "winner_points": 0.0,
                    "n_candidates": 0,
                    "renamed": False,
                    "macs": 0,
                    "memory": 0,
                    "params": 0,
                })
            continue
        cands.sort(key=lambda c: (c["cost"], c["src"]))  # cheapest wins; alphabetical tiebreak
        winner = cands[0]

        if is_solved:
            # Only override if STRICTLY cheaper than our organic. Equal cost ->
            # keep ours (less risk; ours is known to score on Kaggle).
            our_cost = our_costs.get(tn)
            if our_cost is None or winner["cost"] >= our_cost:
                continue
            our_pts = max(1.0, 25.0 - math.log(max(1.0, our_cost)))
            delta = winner["points"] - our_pts
            base_zip[tn] = winner["raw"]
            n_overrode_solved += 1
            overrode_pts_delta += delta
            rows.append({
                "task": tn,
                "winner_src": f"OVERRIDE-{winner['src']}",
                "winner_cost": winner["cost"],
                "winner_points": winner["points"],
                "n_candidates": len(cands),
                "renamed": winner["renamed"],
                "macs": winner["macs"],
                "memory": winner["memory"],
                "params": winner["params"],
            })
            continue

        base_zip[tn] = winner["raw"]  # standard add: filled from external
        n_added += 1
        blend_pts_added += winner["points"]
        rows.append({
            "task": tn,
            "winner_src": winner["src"],
            "winner_cost": winner["cost"],
            "winner_points": winner["points"],
            "n_candidates": len(cands),
            "renamed": winner["renamed"],
            "macs": winner["macs"],
            "memory": winner["memory"],
            "params": winner["params"],
        })

    if not dry_run:
        out_zip = pathlib.Path(out_zip)
        out_zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for n in sorted(base_zip):
                zf.writestr(f"task{n:03d}.onnx", base_zip[n])

    if report_csv:
        report_csv = pathlib.Path(report_csv)
        report_csv.parent.mkdir(parents=True, exist_ok=True)
        with report_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
                "task", "winner_src", "winner_cost", "winner_points",
                "n_candidates", "renamed", "macs", "memory", "params",
            ])
            writer.writeheader()
            writer.writerows(rows)

    if verbose:
        print()
        print(f"Blend complete:")
        print(f"  Our SOLVED_TASKS preserved:           {len(SOLVED_TASKS) - n_overrode_solved}")
        if allow_cheaper_externals:
            print(f"  Our SOLVED overridden (cheaper):     {n_overrode_solved}  (+{overrode_pts_delta:.2f} pts)")
        print(f"  Tasks with external winner added:    {n_added}")
        print(f"  Tasks with no passing candidate:     {n_no_candidate}")
        print(f"  Total tasks in blend:                {len(SOLVED_TASKS) + n_added} / 400")
        print(f"  Estimated +pts from external blend:  {blend_pts_added:.2f}")
        if not dry_run:
            print(f"  Output zip: {out_zip.relative_to(NG_ROOT).as_posix()}")
        if report_csv:
            print(f"  Report CSV: {report_csv.relative_to(NG_ROOT).as_posix()}")
        # Source attribution
        from collections import Counter
        src_counts = Counter(r["winner_src"] for r in rows if r["winner_src"])
        if src_counts:
            print(f"  Winner-source split: {dict(src_counts)}")

    return {
        "n_added": n_added,
        "n_no_candidate": n_no_candidate,
        "n_overrode_solved": n_overrode_solved,
        "overrode_pts_delta": round(overrode_pts_delta, 2),
        "blend_pts": round(blend_pts_added, 2),
    }


def _parse_tasks(arg):
    if not arg:
        return None
    return sorted({int(t) for t in arg.split(",") if t.strip()})


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--max-pairs", type=int, default=100,
                        help="Max arc-gen pairs to verify per candidate (default 100; --full to remove cap)")
    parser.add_argument("--full", action="store_true", help="Verify all pairs (slow; ~265 pairs per task)")
    parser.add_argument("--out", type=str, default="",
                        help="Output zip path (default: submissions/blend.zip)")
    parser.add_argument("--dry-run", action="store_true", help="Report only; do not write the zip")
    parser.add_argument("--tasks", type=str, default="",
                        help="Comma-separated task ids to consider (default: all unsolved 1..400)")
    parser.add_argument("--report", type=str, default="",
                        help="CSV report path (default: Logs/external_blend_<UTC-iso>.csv)")
    parser.add_argument("--allow-cheaper-externals", action="store_true",
                        help="Also override our SOLVED submission when an external is BOTH "
                             "strictly cheaper AND passes verify_subset 100%% on visible+arc-gen. "
                             "Worst case if a cheaper external silent-fails Kaggle's hidden test: "
                             "we lose that task entirely. Default OFF preserves the 'sacred' rule.")
    parser.add_argument("--include-risky", action="store_true",
                        help="Disable cascade safety filter: include CASCADE-RISK candidates "
                             "(cost < 1000 or large intermediates). Default OFF rejects them "
                             "to avoid Kaggle's cascade bug. Use only for bisection testing.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    max_pairs = None if args.full else args.max_pairs
    out_zip = pathlib.Path(args.out) if args.out else SUBMISSIONS_DIR / "blend.zip"
    if not out_zip.is_absolute():
        out_zip = NG_ROOT / out_zip
    if args.report:
        report_csv = pathlib.Path(args.report)
        if not report_csv.is_absolute():
            report_csv = NG_ROOT / report_csv
    else:
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
        report_csv = LOGS_DIR / f"external_blend_{stamp}.csv"

    target_tasks = _parse_tasks(args.tasks)
    sources = [(label, p) for label, p in DEFAULT_SOURCES if p.exists()]
    if not sources:
        print("No competitor sources found under data/competitor_artifacts/", file=sys.stderr)
        sys.exit(2)

    run(sources, max_pairs, out_zip, report_csv,
        target_tasks=target_tasks, dry_run=args.dry_run, verbose=not args.quiet,
        allow_cheaper_externals=args.allow_cheaper_externals,
        include_risky=args.include_risky)


if __name__ == "__main__":
    main()
