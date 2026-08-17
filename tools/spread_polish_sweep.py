#!/usr/bin/env python3
"""Apply polish_candidate's sequential pipeline to every task in
submission.zip and write the TRUE achievable cost-delta CSV.

The original spread_scan.py reports per-pattern savings INDEPENDENTLY
(each pattern measured against the original model). When applying
multiple patterns sequentially via polish_candidate, the savings
OVERLAP — onnxsim subsumes many of safe_opt's algebraic identities,
so the sum is an overestimate.

This sweep gives the ground-truth recoverable bytes by running the
full polish pipeline on each task. Output:
  Logs/spread_scan_results/polish_sweep_<UTC date>.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("ORT_LOGGING_LEVEL", "3")

import onnx

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.polish_candidate import polish  # noqa: E402

SUBMISSION_ZIP = REPO_ROOT / "submissions" / "submission.zip"
RESULTS_DIR = REPO_ROOT / "Logs" / "spread_scan_results"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=SUBMISSION_ZIP)
    parser.add_argument("--tasks", default="")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    with zipfile.ZipFile(args.zip) as z:
        all_tasks = sorted({Path(n).stem for n in z.namelist() if n.endswith(".onnx")})
    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    else:
        tasks = all_tasks
    if args.limit:
        tasks = tasks[: args.limit]

    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"polish_sweep_{today}.csv"

    work_dir = Path(tempfile.mkdtemp(prefix="polish_sweep_"))
    rows = []
    start = time.time()
    for i, tid in enumerate(tasks):
        t0 = time.time()
        try:
            with zipfile.ZipFile(args.zip) as z:
                model = onnx.load_from_string(z.read(f"{tid}.onnx"))
            out = polish(model, work_dir=work_dir, quiet=True)
            orig_cost = (out["original_score"] or {}).get("cost", 0)
            final_cost = (out["final_score"] or {}).get("cost", 0)
            delta = int(final_cost) - int(orig_cost) if orig_cost and final_cost else 0
            applied = [s["pass"] for s in out["steps"] if s["verdict"] == "applied"]
            verify_failed = [s["pass"] for s in out["steps"] if s["verdict"] == "verify-failed"]
            row = {
                "task_id": tid,
                "orig_cost": orig_cost,
                "polished_cost": final_cost,
                "delta": delta,
                "applied_passes": ",".join(applied),
                "verify_failed_passes": ",".join(verify_failed),
            }
        except Exception as exc:
            row = {
                "task_id": tid,
                "orig_cost": "",
                "polished_cost": "",
                "delta": "",
                "applied_passes": "",
                "verify_failed_passes": f"ERROR: {type(exc).__name__}: {exc}",
            }
        rows.append(row)
        dt = time.time() - t0
        delta_str = f"{row['delta']:+d}" if isinstance(row["delta"], int) else "?"
        print(f"  [{i+1:>3}/{len(tasks)}] {tid}: delta={delta_str}  passes=[{row['applied_passes']}] ({dt:.1f}s)", flush=True)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["task_id", "orig_cost", "polished_cost", "delta", "applied_passes", "verify_failed_passes"],
        )
        writer.writeheader()
        writer.writerows(rows)

    total_savings = sum(-int(r["delta"]) for r in rows if isinstance(r["delta"], int) and r["delta"] < 0)
    tasks_with_savings = sum(1 for r in rows if isinstance(r["delta"], int) and r["delta"] < 0)
    print()
    print(f"Wrote: {out_path}")
    print(f"Tasks with sequential-pipeline savings: {tasks_with_savings}")
    print(f"TRUE achievable bytes (no double-counting): {total_savings:,}")
    print(f"Total elapsed: {time.time() - start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
