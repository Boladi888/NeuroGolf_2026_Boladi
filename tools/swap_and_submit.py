#!/usr/bin/env python3
"""Swap one or more winning task ONNX files into submission.zip and (optionally) ship to Kaggle.

This is the mechanical "captain ships" step from agent_feedback_rules.md §4c:
    worker builds taskNNN_vN.onnx  ->  captain swaps into submission.zip  ->  kaggle submit  ->  poll.

Why a dedicated helper:
  * build_submission.py::make_zip rebuilds from submissions/task001..400.onnx and DROPS task000,
    so it cannot be used for a single-task swap on the live 401-entry zip.
  * The live zip must keep exactly task000.onnx..task400.onnx (401 entries). A naive re-zip that
    misses task000 silently regresses that task. This helper validates that invariant.

SAFE BY DEFAULT: with no --submit flag it writes a *staged* zip and prints the exact kaggle
command, leaving submissions/submission.zip and Kaggle untouched. Only --submit mutates the live
zip (after a timestamped backup) and calls the Kaggle CLI.

Usage:
    # stage only (non-destructive) — verify the swap, then eyeball the printed command
    python tools/swap_and_submit.py path/to/task203_v4.onnx

    # multiple tasks in one shot
    python tools/swap_and_submit.py task203_v4.onnx task181_v1.onnx

    # actually ship (backs up the live zip first, then fires + polls)
    python tools/swap_and_submit.py path/to/task203_v4.onnx --submit -m "task203 v4 ..."
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LIVE_ZIP = REPO_ROOT / "submissions" / "submission.zip"
SNAPSHOT_DIR = REPO_ROOT / "submissions" / "_submit_snapshots"
COMPETITION = "neurogolf-2026"
EXPECTED_ENTRIES = {f"task{n:03d}.onnx" for n in range(0, 401)}  # task000..task400 (401)

_TASK_RE = re.compile(r"(task\d{3})", re.IGNORECASE)


def _arcname_for(candidate: Path) -> str:
    """Infer the in-zip entry name (taskNNN.onnx) from a candidate file path."""
    m = _TASK_RE.search(candidate.name)
    if not m:
        raise SystemExit(
            f"cannot infer taskNNN from {candidate.name!r}; pass a file whose name contains taskNNN "
            f"(e.g. task203_v4.onnx) or rename it."
        )
    return f"{m.group(1).lower()}.onnx"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256(path.read_bytes())


def _load_snapshot_meta(path: Path | None) -> dict:
    if not path:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"cannot read --snapshot-meta {path}: {e}") from e


def _meta_for_task(meta: dict, task: str) -> dict:
    tasks = meta.get("tasks", {})
    if isinstance(tasks, dict):
        return tasks.get(task) or tasks.get(f"{task}.onnx") or {}
    if isinstance(tasks, list):
        for row in tasks:
            if row.get("task") == task or row.get("arcname") == f"{task}.onnx":
                return row
    return {}


def _candidate_sidecar_meta(candidate: Path) -> dict:
    side = candidate.with_suffix(".json")
    if not side.exists():
        return {}
    try:
        return json.loads(side.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_submit_snapshot(staged_zip: Path, candidates: list[Path], *, tag: str, message: str,
                          meta: dict) -> Path:
    """Copy the exact zip about to go to Kaggle plus candidate metadata."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag or "submit").strip("_") or "submit"
    out_dir = SNAPSHOT_DIR / f"{ts}_{safe_tag}"
    suffix = 1
    while out_dir.exists():
        suffix += 1
        out_dir = SNAPSHOT_DIR / f"{ts}_{safe_tag}_{suffix}"
    out_dir.mkdir(parents=True, exist_ok=False)
    snapshot_zip = out_dir / "submission.zip"
    shutil.copy2(staged_zip, snapshot_zip)
    selected = []
    for candidate in candidates:
        arc = _arcname_for(candidate)
        task = arc[:-5]
        row = _candidate_sidecar_meta(candidate)
        row.update(_meta_for_task(meta, task))
        cost = row.get("cost")
        points = row.get("points", row.get("points_if_correct"))
        selected.append({
            "task": task,
            "arcname": arc,
            "source_path": str(candidate.resolve()),
            "onnx_sha256": _sha256_file(candidate),
            "cost": cost,
            "points_if_correct": points,
            "expected_score_if_correct": points,
            "candidate_status": row.get("candidate_status", "submitted_unverified"),
        })
    manifest = {
        "schema": "neurogolf_probe_submit_snapshot_v1",
        "tag": tag,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "kaggle_message": message,
        "zip_path": "submission.zip",
        "zip_sha256": _sha256_file(snapshot_zip),
        "selected_tasks": [row["task"] for row in selected],
        "tasks": selected,
        "score_kind": "expected_score_if_correct",
        "expected_score_if_correct": meta.get("expected_score_if_correct"),
        "confirmed_kaggle_public_score": None,
        "candidate_status": "submitted_unverified",
    }
    (out_dir / "snapshot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return out_dir


def build_swapped_zip(candidates: list[Path], *, live_zip: Path, out_zip: Path) -> dict[str, tuple[int, int]]:
    """Write out_zip = live_zip with the named task entries replaced. Returns {arcname: (old, new)}."""
    swaps: dict[str, bytes] = {}
    deltas: dict[str, tuple[int, int]] = {}
    for c in candidates:
        if not c.exists():
            raise SystemExit(f"candidate not found: {c}")
        arc = _arcname_for(c)
        if arc not in EXPECTED_ENTRIES:
            raise SystemExit(f"{arc} is outside task000..task400")
        swaps[arc] = c.read_bytes()

    with zipfile.ZipFile(live_zip) as zin:
        names = zin.namelist()
        present = set(names)
        if present != EXPECTED_ENTRIES:
            missing = sorted(EXPECTED_ENTRIES - present)
            extra = sorted(present - EXPECTED_ENTRIES)
            raise SystemExit(
                f"live zip {live_zip} is not exactly task000..task400 "
                f"(missing={missing[:5]} extra={extra[:5]}); refusing to swap."
            )
        out_zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zout:
            for name in names:  # preserve original entry order
                old = zin.read(name)
                if name in swaps:
                    new = swaps[name]
                    zout.writestr(name, new)
                    deltas[name] = (len(old), len(new))
                else:
                    zout.writestr(name, old)

    # Post-write invariant check.
    with zipfile.ZipFile(out_zip) as zchk:
        if set(zchk.namelist()) != EXPECTED_ENTRIES:
            raise SystemExit(f"BUG: staged zip {out_zip} lost the task000..task400 invariant")
    return deltas


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("candidates", nargs="+", type=Path, help="taskNNN*.onnx file(s) to swap in")
    ap.add_argument("--live-zip", type=Path, default=LIVE_ZIP, help=f"live zip (default {LIVE_ZIP})")
    ap.add_argument("--out", type=Path, default=None, help="staged zip output (default submissions/_staged_submission.zip)")
    ap.add_argument("--submit", action="store_true", help="actually swap the live zip (after backup) and call the Kaggle CLI")
    ap.add_argument("-m", "--message", default="", help="Kaggle submit message (required with --submit)")
    ap.add_argument("--competition", default=COMPETITION)
    ap.add_argument("--snapshot-tag", default="", help="tag to use in the immutable submit snapshot directory")
    ap.add_argument("--snapshot-meta", type=Path, default=None, help="JSON metadata for snapshot cost/points fields")
    args = ap.parse_args()

    staged = args.out or (args.live_zip.parent / "_staged_submission.zip")
    deltas = build_swapped_zip(args.candidates, live_zip=args.live_zip, out_zip=staged)

    print(f"staged swapped zip: {staged}")
    for arc, (old, new) in sorted(deltas.items()):
        sign = "-" if new <= old else "+"
        print(f"  {arc}: {old:,} B -> {new:,} B  ({sign}{abs(new - old):,} B compressed-stream)")
    print("  (all other 400 entries copied byte-for-byte; task000..task400 invariant verified)")

    if not args.submit:
        print("\nSTAGED ONLY (live zip + Kaggle untouched). To ship:")
        print(f"  python {Path(__file__).relative_to(REPO_ROOT).as_posix()} "
              + " ".join(str(c) for c in args.candidates) + ' --submit -m "<message>"')
        print("or do it by hand:")
        print(f'  cp "{staged}" "{args.live_zip}"')
        print(f'  kaggle competitions submit -c {args.competition} -f "{args.live_zip}" -m "<message>"')
        print(f"  kaggle competitions submissions -c {args.competition}   # poll")
        return 0

    if not args.message.strip():
        raise SystemExit("--submit requires a non-empty -m/--message")

    snapshot_dir = write_submit_snapshot(
        staged,
        args.candidates,
        tag=args.snapshot_tag or args.message[:60],
        message=args.message,
        meta=_load_snapshot_meta(args.snapshot_meta),
    )
    print(f"snapshot saved: {snapshot_dir / 'snapshot_manifest.json'}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = args.live_zip.with_suffix(f".zip.bak_{ts}")
    shutil.copy2(args.live_zip, backup)
    shutil.copy2(staged, args.live_zip)
    print(f"\nbacked up live zip -> {backup.name}; swapped staged zip into {args.live_zip.name}")

    cmd = ["kaggle", "competitions", "submit", "-c", args.competition, "-f", str(args.live_zip), "-m", args.message]
    print("running:", " ".join(f'"{a}"' if " " in a else a for a in cmd))
    proc = subprocess.run(cmd, text=True, capture_output=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        print(f"\nkaggle submit returned {proc.returncode}; live zip kept the swap. "
              f"Restore with: cp {backup.name} {args.live_zip.name}")
        return proc.returncode

    print("\npolling submissions (latest first):")
    poll = subprocess.run(
        ["kaggle", "competitions", "submissions", "-c", args.competition],
        text=True, capture_output=True,
    )
    sys.stdout.write(poll.stdout)
    print("\nNEXT: append a row to Logs/submissions_ledger.csv and update "
          "The_400_Tasks/current_strategy.md once the public score lands.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
