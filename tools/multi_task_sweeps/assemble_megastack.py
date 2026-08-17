#!/usr/bin/env python3
"""Assemble a megastack from sweep results.

Reads all sweep results JSONs in Logs/multi_task_sweeps/, picks the best
candidate per task (lowest cost), and builds a submission.zip with all
winning ONNXs swapped into the canonical base.

Usage:
    .venv/Scripts/python tools/multi_task_sweeps/assemble_megastack.py \
        --base submissions/_canonical_5911_36_backup.zip \
        --out tmp_megastack/submission.zip
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, default=REPO_ROOT / "submissions" / "_canonical_5911_36_backup.zip")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "tmp_megastack" / "submission.zip")
    p.add_argument("--results-dir", type=Path, default=REPO_ROOT / "Logs" / "multi_task_sweeps")
    args = p.parse_args()

    # Aggregate best candidate per task across all sweep results
    best = {}  # task_id -> {candidate_path, cost, gain, sweep}
    total_gain = 0.0
    for json_file in args.results_dir.glob("*_results.json"):
        try:
            data = json.loads(json_file.read_text())
        except Exception:
            continue
        sweep_name = json_file.stem.replace("_results", "")
        for r in data:
            tid = r["task_id"]
            if tid not in best or r["candidate_cost"] < best[tid]["candidate_cost"]:
                best[tid] = {**r, "sweep": sweep_name}

    if not best:
        print("No candidates found in any sweep results.")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    swapped = []
    skipped = []
    with zipfile.ZipFile(args.base, "r") as zin, zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            fname = item.filename
            if fname.endswith(".onnx"):
                tid = fname[:-len(".onnx")]
                if tid in best:
                    cand_path = Path(best[tid]["candidate_path"])
                    if cand_path.exists():
                        zout.writestr(fname, cand_path.read_bytes())
                        swapped.append((tid, best[tid]))
                        total_gain += best[tid]["raw_log_e_gain"]
                        continue
                    else:
                        skipped.append(tid)
            zout.writestr(item, zin.read(item.filename))

    print(f"\nMegastack assembled: {args.out}")
    print(f"  Total task swaps: {len(swapped)}")
    print(f"  Total predicted LB gain (additive raw log_e): +{total_gain:.2f}")
    print()
    for tid, info in sorted(swapped, key=lambda x: -x[1]["raw_log_e_gain"]):
        print(f"  {tid} [{info['sweep']}]: {info['base_cost']:,} -> {info['candidate_cost']:,} (+{info['raw_log_e_gain']:.3f})")
    if skipped:
        print(f"\nSkipped (candidate file missing): {skipped}")


if __name__ == "__main__":
    raise SystemExit(main() or 0)
