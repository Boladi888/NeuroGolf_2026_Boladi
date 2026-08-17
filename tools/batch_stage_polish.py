#!/usr/bin/env python3
"""Batch-apply polish_candidate to a list of tasks and stage the results.

Usage:
    python tools/batch_stage_polish.py task221,task252,task277,...
    python tools/batch_stage_polish.py --tag BIG_POLISH_a

Workflow per task:
  1. Extract live ONNX from submission.zip
  2. Run polish_candidate.py (all math-equivalent passes)
  3. Parse cost delta from output
  4. Verify against live (100% required)
  5. Stage in The_400_Tasks/Task_by_Task/taskNNN/staged_candidates/<candidate_id>/
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SUBMISSION_ZIP = REPO / "submissions" / "submission.zip"
TASK_BY_TASK_ROOT = REPO / "The_400_Tasks" / "Task_by_Task"
VERIFY_SCRIPT = REPO / "tools" / "verify_against_live.py"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")


def run_polish(src: Path, out: Path) -> tuple[int, str]:
    """Run polish_candidate.py. Returns (cost_delta, summary_str)."""
    r = subprocess.run(
        [sys.executable, str(REPO / "tools" / "polish_candidate.py"), str(src), "-o", str(out)],
        capture_output=True, text=True
    )
    stdout = r.stdout
    m = re.search(r"polished cost:\s*(\d+)\s*\((-?\d+) bytes\)", stdout)
    if m:
        cost_delta = int(m.group(2))
        polished_cost = int(m.group(1))
        return cost_delta, polished_cost, stdout
    return 0, None, stdout


def run_verify(proto: Path, tid: int) -> tuple[int, int, str]:
    """Run verify_against_live.py. Returns (proto_vs_live_passed, total, summary).

    The critical check is 'Prototype vs LIVE' (math equivalence).
    'Prototype vs expected' can be 0 for unsolved tasks — that's OK for shrinks.
    """
    r = subprocess.run(
        [sys.executable, str(VERIFY_SCRIPT), "--proto", str(proto), "--tid", str(tid)],
        capture_output=True, text=True
    )
    output = r.stdout + r.stderr

    # Primary check: proto matches LIVE exactly (math equivalence)
    m_live = re.search(r"Prototype\s+vs\s+LIVE\s*:\s*(\d+)/(\d+)", output, re.IGNORECASE)
    if m_live:
        return int(m_live.group(1)), int(m_live.group(2)), output.strip()

    # Fallback: first numeric pair (e.g. "passed/total")
    m = re.search(r"(\d+)/(\d+)", output)
    if m:
        return int(m.group(1)), int(m.group(2)), output.strip()
    return 0, 0, output.strip()


def stage(task_id: str, onnx_path: Path, cost_delta: int, polished_cost: int, tag: str, verify_str: str) -> Path:
    """Stage the candidate. Returns staged dir path."""
    ts = utc_now()
    cand_id = f"{task_id}_polish_{tag}_{ts}"
    dest = TASK_BY_TASK_ROOT / task_id / "staged_candidates" / cand_id
    dest.mkdir(parents=True, exist_ok=True)

    shutil.copy(onnx_path, dest / f"{task_id}.onnx")

    manifest = {
        "schema": "neurogolf.staged_submit.v1",
        "candidate_id": cand_id,
        "task_id": task_id,
        "cost_delta": cost_delta,
        "polished_cost": polished_cost,
        "pattern": "polish_sweep",
        "tag": tag,
        "verify": verify_str,
        "staged_at": datetime.now(timezone.utc).isoformat(),
        "worker": "neurogolf-worker-7",
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return dest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tasks", nargs="?", help="Comma-separated task IDs, e.g. task221,task252")
    parser.add_argument("--tag", default="t_worker7", help="Short tag embedded in candidate_id")
    parser.add_argument("--skip-verify", action="store_true", help="Skip visible verify (not recommended)")
    args = parser.parse_args()

    if args.tasks:
        task_ids = [t.strip() for t in args.tasks.split(",")]
    else:
        parser.print_help()
        return 1

    tmpdir = Path(tempfile.mkdtemp())
    print(f"Temp dir: {tmpdir}")

    try:
        # Extract all live ONNXes
        with zipfile.ZipFile(SUBMISSION_ZIP) as z:
            for tid in task_ids:
                z.extract(f"{tid}.onnx", tmpdir)

        results = []
        for tid in task_ids:
            src = tmpdir / f"{tid}.onnx"
            out = tmpdir / f"{tid}_polished.onnx"

            print(f"\n{'='*50}")
            print(f"Processing {tid}...")

            # 1. Polish
            cost_delta, polished_cost, polish_out = run_polish(src, out)
            print(f"  Polish: {cost_delta} bytes, cost={polished_cost}")
            print(f"  Detail: {[l for l in polish_out.split(chr(10)) if 'bytes' in l]}")

            if cost_delta >= 0 or not out.exists():
                print(f"  SKIP: no improvement (delta={cost_delta})")
                results.append((tid, "SKIP", cost_delta, None))
                continue

            # 2. Verify
            if args.skip_verify:
                verify_str = "SKIPPED"
                verify_ok = True
            else:
                tid_num = int(tid.replace("task", ""))
                passed, total, verify_out = run_verify(out, tid_num)
                verify_str = f"proto_vs_live={passed}/{total}"
                verify_ok = (passed == total and total > 0)
                print(f"  Verify: {verify_str} {'OK' if verify_ok else 'FAIL'}")

            if not verify_ok:
                print(f"  SKIP: verify failed {verify_str}")
                results.append((tid, "VERIFY_FAIL", cost_delta, verify_str))
                continue

            # 3. Stage
            staged_dir = stage(tid, out, cost_delta, polished_cost, args.tag, verify_str)
            print(f"  Staged: {staged_dir.name}")
            results.append((tid, "STAGED", cost_delta, verify_str))

        # Summary
        print(f"\n{'='*50}")
        print("SUMMARY:")
        total_bytes = 0
        for tid, status, delta, verify_str in results:
            print(f"  {tid}: {status} delta={delta} verify={verify_str}")
            if status == "STAGED":
                total_bytes += delta
        print(f"\nTotal bytes saved (staged): {total_bytes}")
        return 0

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
