#!/usr/bin/env python3
"""Spread-knowledge scanner — apply every known math-equivalent shrink to
every task in submission.zip and record which ones save bytes.

Rationale: the per-task notes (see ``Logs/lessons.md``) describe shrinks
that worked on ONE task — e.g. task000's ``ones10 → scalar1`` broadcast
trick saved 99 params, and the Constant-node → graph.initializer move
(task230 lesson) saved 7+ LB across 7 tasks combined. The insight from
each is durable, but applying it to every task one-by-one isn't.

This driver:
  * Loads each ``taskNNN.onnx`` from ``submissions/submission.zip``.
  * Runs every ``SpreadPattern`` in ``tools/spread_patterns/`` against it.
  * For patterns that returned a new model, verifies byte-exact output on
    N random one-hot inputs (default 3).
  * Scores both original and modified via ``score_network_v2`` to compute
    the real ``cost_delta``.
  * Writes a JSON report + flat CSV under
    ``Logs/spread_scan_results/spread_scan_<UTC date>.{json,csv}``.

Usage::

    .venv/Scripts/python.exe tools/spread_scan.py            # scan all tasks
    .venv/Scripts/python.exe tools/spread_scan.py --tasks task000,task013
    .venv/Scripts/python.exe tools/spread_scan.py --apply    # write candidate ONNX outputs
    .venv/Scripts/python.exe tools/spread_scan.py --update-notes  # append spread-knowledge section to per-task .md
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Suppress ORT chatter before importing it elsewhere.
os.environ.setdefault("ORT_LOGGING_LEVEL", "3")

import numpy as np
import onnx

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.spread_patterns import SpreadPattern, SpreadResult, iter_patterns  # noqa: E402

SUBMISSION_ZIP = REPO_ROOT / "submissions" / "submission.zip"
RESULTS_DIR = REPO_ROOT / "Logs" / "spread_scan_results"
TASK_NOTES_DIR = REPO_ROOT / "The_400_Tasks" / "Task_by_Task"
ISOLATED_SCORE_SCRIPT = REPO_ROOT / "tools" / "spread_score_isolated.py"
ISOLATED_VERIFY_SCRIPT = REPO_ROOT / "tools" / "spread_verify_isolated.py"

DEFAULT_VERIFY_INPUTS = 3
VERIFY_TOLERANCE = 1e-3
SCORE_TIMEOUT_SEC = 60
VERIFY_TIMEOUT_SEC = 60


def _utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _enumerate_zip_tasks(zip_path: Path) -> list[str]:
    """Return sorted list of taskNNN ids in the zip."""
    if not zip_path.exists():
        raise FileNotFoundError(f"submission.zip not found at {zip_path}")
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.endswith(".onnx")]
    task_ids = sorted({Path(n).stem for n in names})
    return task_ids


def _load_model_from_zip(zip_path: Path, task_id: str) -> onnx.ModelProto:
    with zipfile.ZipFile(zip_path) as z:
        with z.open(f"{task_id}.onnx") as f:
            data = f.read()
    return onnx.load_from_string(data)


def _score_model(model: onnx.ModelProto, *, work_dir: Path) -> dict[str, Any] | None:
    """Save model to a temp file under work_dir and call the isolated
    scorer subprocess. Returns a dict on success or ``None`` on failure
    (including C-level crashes like ORT's SIGFPE on certain graphs).
    """
    tmp = work_dir / f"_spread_score_{os.getpid()}_{time.time_ns()}.onnx"
    try:
        onnx.save(model, str(tmp))
    except Exception:  # noqa: BLE001
        return None
    try:
        proc = subprocess.run(
            [sys.executable, str(ISOLATED_SCORE_SCRIPT), str(tmp)],
            capture_output=True,
            text=True,
            timeout=SCORE_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:
        return None
    except Exception:  # noqa: BLE001
        return None
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass
    if proc.returncode != 0:
        return None  # ORT/Python crash in child
    out = proc.stdout.strip().splitlines()
    if not out:
        return None
    try:
        payload = json.loads(out[-1])
    except json.JSONDecodeError:
        return None
    if "error" in payload:
        return None
    return payload


def _make_dummy_inputs(model: onnx.ModelProto, n_inputs: int, seed: int = 0) -> list[np.ndarray]:
    """Generate N random valid one-hot tensors of the model's input shape.

    NeuroGolf ONNXes take [1, 10, 30, 30] float32 one-hot encodings. We
    produce realistic inputs by randomly assigning each cell to one of
    the 10 channels.
    """
    in_node = model.graph.input[0]
    shape = []
    for d in in_node.type.tensor_type.shape.dim:
        shape.append(d.dim_value if d.dim_value > 0 else 1)
    if not shape:
        shape = [1, 10, 30, 30]
    rng = np.random.RandomState(seed)
    inputs = []
    # First input: always the "background everywhere" canonical case.
    canon = np.zeros(shape, dtype=np.float32)
    if len(shape) == 4 and shape[1] >= 1:
        canon[:, 0, :, :] = 1.0
    inputs.append(canon)
    for k in range(1, n_inputs):
        # One-hot along channel axis with random colors.
        if len(shape) == 4 and shape[1] >= 1:
            idx = rng.randint(0, shape[1], size=(shape[0], shape[2], shape[3]))
            arr = np.zeros(shape, dtype=np.float32)
            for b in range(shape[0]):
                for r in range(shape[2]):
                    for c in range(shape[3]):
                        arr[b, idx[b, r, c], r, c] = 1.0
            inputs.append(arr)
        else:
            inputs.append(rng.random(shape).astype(np.float32))
    return inputs


def _verify_byte_exact(
    original: onnx.ModelProto,
    modified: onnx.ModelProto,
    *,
    work_dir: Path,
    n_inputs: int,
    tolerance: float | None = None,
) -> tuple[bool, str]:
    """Verify both models produce equal output on N dummy inputs by
    delegating to ``spread_verify_isolated.py`` in a subprocess. This
    survives ORT C-level crashes that would otherwise terminate the
    driver (e.g. SIGFPE on certain fp16-converted graphs).
    """
    orig_path = work_dir / f"_spread_verify_orig_{os.getpid()}_{time.time_ns()}.onnx"
    mod_path = work_dir / f"_spread_verify_mod_{os.getpid()}_{time.time_ns()}.onnx"
    try:
        try:
            onnx.save(original, str(orig_path))
            onnx.save(modified, str(mod_path))
        except Exception as exc:  # noqa: BLE001
            return False, f"onnx.save failed: {exc}"
        tol_arg = "" if tolerance is None else f"{tolerance}"
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    str(ISOLATED_VERIFY_SCRIPT),
                    str(orig_path),
                    str(mod_path),
                    "--inputs",
                    str(n_inputs),
                    "--tolerance",
                    tol_arg,
                ],
                capture_output=True,
                text=True,
                timeout=VERIFY_TIMEOUT_SEC,
            )
        except subprocess.TimeoutExpired:
            return False, "verify subprocess timed out"
        except Exception as exc:  # noqa: BLE001
            return False, f"verify subprocess error: {exc}"
        if proc.returncode != 0:
            return False, f"verify subprocess crashed (exit={proc.returncode}): {proc.stderr.strip()[:200]}"
        out = proc.stdout.strip().splitlines()
        if not out:
            return False, "verify subprocess produced no output"
        try:
            payload = json.loads(out[-1])
        except json.JSONDecodeError:
            return False, f"verify subprocess non-JSON: {out[-1][:200]}"
        return bool(payload.get("ok")), str(payload.get("reason", ""))
    finally:
        for p in (orig_path, mod_path):
            try:
                p.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass


def scan_task(
    task_id: str,
    *,
    zip_path: Path = SUBMISSION_ZIP,
    work_dir: Path,
    n_verify_inputs: int = DEFAULT_VERIFY_INPUTS,
    fp16_tolerance: float = VERIFY_TOLERANCE,
    apply_dir: Path | None = None,
    patterns: list[SpreadPattern] | None = None,
) -> dict[str, Any]:
    """Run every pattern against a single task. Returns the per-task record."""
    record: dict[str, Any] = {
        "task_id": task_id,
        "scanned_at": _utc_now(),
        "original": None,
        "patterns": [],
        "error": None,
    }
    try:
        original_model = _load_model_from_zip(zip_path, task_id)
    except Exception as exc:  # noqa: BLE001
        record["error"] = f"load failed: {exc}"
        return record

    orig_score = _score_model(original_model, work_dir=work_dir)
    record["original"] = orig_score

    if patterns is None:
        patterns = list(iter_patterns())

    for pat in patterns:
        pat_rec: dict[str, Any] = {
            "name": pat.name,
            "category": pat.category,
            "verdict": "error",
            "cost_delta": None,
            "notes": "",
            "extras": {},
        }
        try:
            result = pat.apply(original_model, task_id)
        except Exception as exc:  # noqa: BLE001
            pat_rec["verdict"] = "error"
            pat_rec["notes"] = f"pattern raised {type(exc).__name__}: {exc}"
            pat_rec["extras"] = {"traceback": traceback.format_exc(limit=4)}
            record["patterns"].append(pat_rec)
            continue
        pat_rec["verdict"] = result.verdict
        pat_rec["notes"] = result.notes
        pat_rec["extras"] = result.extras

        if result.verdict == "advisory":
            pat_rec["cost_delta"] = result.cost_delta
            record["patterns"].append(pat_rec)
            continue

        if result.verdict != "applied" or result.new_model is None:
            record["patterns"].append(pat_rec)
            continue

        # Verify byte-exact (or within fp16 tolerance for fp16_selective).
        tol = fp16_tolerance if pat.name == "fp16_selective" else None
        ok, why = _verify_byte_exact(
            original_model,
            result.new_model,
            work_dir=work_dir,
            n_inputs=n_verify_inputs,
            tolerance=tol,
        )
        if not ok:
            pat_rec["verdict"] = "verify-failed"
            pat_rec["notes"] = f"{result.notes} | verify: {why}"
            record["patterns"].append(pat_rec)
            continue

        # Score the modified model.
        new_score = _score_model(result.new_model, work_dir=work_dir)
        if new_score is None:
            pat_rec["verdict"] = "score-failed"
            pat_rec["notes"] = f"{result.notes} | score_network_v2 returned None on modified model"
            record["patterns"].append(pat_rec)
            continue
        if orig_score is None:
            pat_rec["verdict"] = "score-failed"
            pat_rec["notes"] = f"{result.notes} | orig model could not be scored"
            record["patterns"].append(pat_rec)
            continue
        delta = int(new_score["cost"]) - int(orig_score["cost"])
        pat_rec["cost_delta"] = delta
        pat_rec["modified_cost"] = int(new_score["cost"])
        pat_rec["modified_points"] = new_score["points"]
        if delta == 0:
            pat_rec["verdict"] = "no-op-after-score"
            pat_rec["notes"] = f"{result.notes} | cost unchanged"
        elif delta > 0:
            pat_rec["verdict"] = "regression"
            pat_rec["notes"] = f"{result.notes} | cost INCREASED by {delta} bytes"
        else:
            pat_rec["verdict"] = "applied"

        # Optionally persist the candidate ONNX.
        if apply_dir is not None and delta < 0:
            apply_dir.mkdir(parents=True, exist_ok=True)
            candidate_path = apply_dir / f"{task_id}__{pat.name}.onnx"
            try:
                onnx.save(result.new_model, str(candidate_path))
                pat_rec["extras"]["candidate_path"] = str(candidate_path)
            except Exception as exc:  # noqa: BLE001
                pat_rec["extras"]["save_error"] = str(exc)

        record["patterns"].append(pat_rec)
    return record


def write_reports(
    records: list[dict[str, Any]],
    *,
    results_dir: Path = RESULTS_DIR,
    today: str | None = None,
) -> dict[str, Path]:
    """Persist JSON + CSV reports for a scan."""
    today = today or _utc_today()
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / f"spread_scan_{today}.json"
    csv_path = results_dir / f"spread_scan_{today}.csv"
    summary_path = results_dir / f"spread_scan_{today}_summary.md"

    json_path.write_text(json.dumps({"scanned_at": _utc_now(), "tasks": records}, indent=2), encoding="utf-8")

    rows = []
    for rec in records:
        for pat in rec.get("patterns", []):
            rows.append({
                "task_id": rec["task_id"],
                "pattern": pat["name"],
                "category": pat["category"],
                "verdict": pat["verdict"],
                "cost_delta": pat.get("cost_delta"),
                "modified_cost": pat.get("modified_cost"),
                "original_cost": (rec.get("original") or {}).get("cost"),
                "notes": pat["notes"],
            })
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["task_id", "pattern", "category", "verdict", "cost_delta", "modified_cost", "original_cost", "notes"],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Summary markdown.
    # Per-pattern table counts applied + per-pattern saved bytes (still
    # useful as a per-pattern stat; sums-across-tasks is fine because
    # each task contributes at most one "applied" row per pattern).
    # Per-task aggregates: compute BEST-per-task (a megastack ships ONE
    # ONNX per task, so per-task savings is the max applied delta on
    # that task, not the sum). The earlier version reported the sum,
    # which overstated combined savings.
    by_pattern: dict[str, dict[str, int]] = {}
    per_task_best: dict[str, int] = {}
    naive_sum_total = 0
    for r in rows:
        bp = by_pattern.setdefault(r["pattern"], {"applied": 0, "no-op": 0, "advisory": 0, "regression": 0, "verify-failed": 0, "error": 0, "skipped": 0, "saved_bytes": 0})
        v = r["verdict"]
        bp[v] = bp.get(v, 0) + 1
        if v == "applied" and r["cost_delta"] is not None and r["cost_delta"] < 0:
            saved = -int(r["cost_delta"])
            bp["saved_bytes"] += saved
            naive_sum_total += saved
            cur_best = per_task_best.get(r["task_id"], 0)
            if saved > cur_best:
                per_task_best[r["task_id"]] = saved
    best_per_task_total = sum(per_task_best.values())
    tasks_with_any_win = len(per_task_best)
    lines = [
        f"# Spread-scan summary — {today}",
        "",
        f"Tasks scanned: **{len(records)}**.",
        f"Tasks with at least one applicable pattern: **{tasks_with_any_win}**.",
        "",
        f"**Best-per-task shippable total**: **{best_per_task_total:,} bytes** "
        f"(sum of the single best applied pattern per task — what a megastack "
        f"would actually save).",
        "",
        f"_Naive sum across all (task, pattern) rows: {naive_sum_total:,} bytes._ "
        f"This OVERSTATES megastack savings because multiple patterns may "
        f"target the same task; only one applies per shipped ONNX. To compose "
        f"patterns on a single task, the compose-and-rescore step must run "
        f"sequentially — out of scope for this scan.",
        "",
        "| Pattern | Applied | No-op | Advisory | Regression | Verify-failed | Error | Skipped | Bytes saved (per-pattern sum) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, stats in sorted(by_pattern.items()):
        lines.append(
            f"| `{name}` | {stats.get('applied', 0)} | {stats.get('no-op', 0)} | "
            f"{stats.get('advisory', 0)} | {stats.get('regression', 0)} | "
            f"{stats.get('verify-failed', 0)} | {stats.get('error', 0)} | "
            f"{stats.get('skipped', 0)} | {stats.get('saved_bytes', 0):,} |"
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "summary": summary_path}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=SUBMISSION_ZIP)
    parser.add_argument("--tasks", default="", help="Comma-separated taskNNN ids; default = all tasks in the zip")
    parser.add_argument("--verify-inputs", type=int, default=DEFAULT_VERIFY_INPUTS)
    parser.add_argument("--apply", action="store_true", help="Persist candidate ONNX files (default off)")
    parser.add_argument("--apply-dir", type=Path, default=REPO_ROOT / "submissions" / "_spread_candidates")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--limit", type=int, default=0, help="Scan only the first N tasks (0 = no limit)")
    parser.add_argument("--summary-only", action="store_true", help="Print summary only; don't write reports")
    parser.add_argument(
        "--skip-tasks",
        default="task175",
        help=(
            "Comma-separated taskNNN ids to skip outright (record a 'skipped' "
            "row and move on). Default: 'task175' — known to crash ORT during "
            "profiling on its current canonical shape (worker-reported "
            "2026-05-19). Pass '' to scan everything."
        ),
    )
    args = parser.parse_args(argv)

    if args.tasks:
        task_ids = [t.strip() for t in args.tasks.split(",") if t.strip()]
    else:
        task_ids = _enumerate_zip_tasks(args.zip)
    if args.limit:
        task_ids = task_ids[: args.limit]
    skip_set = {t.strip() for t in args.skip_tasks.split(",") if t.strip()}

    work_dir = Path(tempfile.mkdtemp(prefix="spread_scan_"))
    records: list[dict[str, Any]] = []
    apply_dir = args.apply_dir if args.apply else None

    patterns = list(iter_patterns())
    pat_names = [p.name for p in patterns]
    print(f"Loaded {len(patterns)} pattern(s): {pat_names}", flush=True)

    start = time.time()
    for i, tid in enumerate(task_ids):
        t0 = time.time()
        if tid in skip_set:
            rec = {
                "task_id": tid,
                "scanned_at": _utc_now(),
                "original": None,
                "patterns": [],
                "error": f"skipped via --skip-tasks (known-bad: see CLI default)",
            }
            records.append(rec)
            print(f"  [{i+1:>3}/{len(task_ids)}] {tid}  SKIPPED", flush=True)
            continue
        try:
            rec = scan_task(
                tid,
                zip_path=args.zip,
                work_dir=work_dir,
                n_verify_inputs=args.verify_inputs,
                apply_dir=apply_dir,
                patterns=patterns,
            )
        except Exception as exc:  # noqa: BLE001
            rec = {"task_id": tid, "error": f"scan_task raised {type(exc).__name__}: {exc}", "patterns": []}
        records.append(rec)
        elapsed = time.time() - t0
        # One-line progress. `best_saved` is the MAX per-task per-pattern
        # delta, not the sum — earlier versions summed deltas across all
        # applied patterns, but only ONE pattern per task ships in a
        # megastack, so summing overstates the deliverable savings.
        # `combined_saved` is the naive sum (still useful as an upper
        # bound and for parity with old logs).
        applied_deltas = [
            -p["cost_delta"] for p in rec.get("patterns", [])
            if p.get("verdict") == "applied"
            and p.get("cost_delta") is not None
            and p["cost_delta"] < 0
        ]
        applied = len(applied_deltas)
        best_saved = max(applied_deltas) if applied_deltas else 0
        combined_saved = sum(applied_deltas)
        if applied <= 1:
            saved_str = f"saved={best_saved}"
        else:
            saved_str = f"best={best_saved} sum={combined_saved}"
        print(
            f"  [{i+1:>3}/{len(task_ids)}] {tid}  patterns_applied={applied} {saved_str} bytes ({elapsed:.1f}s)",
            flush=True,
        )

    total_elapsed = time.time() - start
    if args.summary_only:
        return 0

    paths = write_reports(records, results_dir=args.results_dir)
    print()
    print(f"Wrote: {paths['json']}")
    print(f"Wrote: {paths['csv']}")
    print(f"Wrote: {paths['summary']}")
    print(f"Total elapsed: {total_elapsed:.1f}s for {len(task_ids)} task(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
