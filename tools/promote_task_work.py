"""Promote prototype work from .notongit scratch + Research_Archive/organic_prototypes/
into per-task folders under The_400_Tasks/Task_by_Task/taskNNN/.

User's pain point: scratch dirs (.notongit/.../submissions/_alpha_a_scratch/
and _alpha_b_scratch/) hold prototype .py + .onnx files that sometimes get
deleted, losing work. Promoting to git-tracked per-task folders preserves it.

For each task with prototype work, creates:

  The_400_Tasks/Task_by_Task/taskNNN/
    prototypes/                  ← all taskNNN_*.py and taskNNN_*.onnx files
      taskNNN_<variant>.py
      taskNNN_<variant>.onnx
    README.md                    ← lists what's there, copies of HIST hints

The existing taskNNN.md stays at Task_by_Task/taskNNN.md as the navigation
index. Future runs can move it INTO the folder (if desired). For now we
keep it alongside to avoid disturbing generate_task_mds + mine_task_history.

Files are COPIED (not moved). Scratch dirs remain untouched. Future deletions
of scratch dirs are no longer destructive.

Idempotent: re-running just copies new/changed files. Existing identical
files are not overwritten.

Usage:
    python tools/promote_task_work.py           # promote everything we find
    python tools/promote_task_work.py --tid 4   # one task
    python tools/promote_task_work.py --dry-run # show what would copy
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
NOGIT_ROOT = Path(os.environ.get("NEUROGOLF_NOGIT_ROOT", REPO / ".notongit" / "NeuroGolf_nogit"))
TASK_BY_TASK = REPO / "The_400_Tasks" / "Task_by_Task"

# Source dirs we scan for task-named files
SOURCE_DIRS = [
    REPO / "Research_Archive" / "organic_prototypes",
    NOGIT_ROOT / "submissions" / "_alpha_a_scratch",
    NOGIT_ROOT / "submissions" / "_alpha_b_scratch",
]

# Files matching this pattern get promoted. Catches:
#   task004_v2_prototype.py, task110_organic.onnx, task205_numpy_reference.py, ...
TASK_RE = re.compile(r"^task(\d{3})(_.*)?\.(py|onnx)$")


def _hash(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_task_files(tid_filter: set[int] | None = None) -> dict[int, list[Path]]:
    """Returns {tid: [source_path, ...]} for all task-named files in SOURCE_DIRS."""
    out: dict[int, list[Path]] = defaultdict(list)
    for d in SOURCE_DIRS:
        if not d.exists():
            continue
        for f in sorted(d.iterdir()):
            if not f.is_file():
                continue
            m = TASK_RE.match(f.name)
            if not m:
                continue
            tid = int(m.group(1))
            if tid_filter is not None and tid not in tid_filter:
                continue
            out[tid].append(f)
    return out


def promote_one(tid: int, sources: list[Path], dry_run: bool) -> dict[str, int]:
    """Copy all source files for task tid into Task_by_Task/taskNNN/prototypes/.
    Returns counts of {copied, skipped_existing, skipped_same}."""
    counts = {"copied": 0, "skipped_existing": 0, "skipped_same": 0}
    dest_dir = TASK_BY_TASK / f"task{tid:03d}" / "prototypes"
    if not dry_run:
        dest_dir.mkdir(parents=True, exist_ok=True)
    readme_lines = [
        f"# task{tid:03d} prototypes",
        "",
        f"Files promoted by `tools/promote_task_work.py`. Originals live in",
        f"scratch dirs (`.notongit/.../_alpha_a_scratch/` or `_alpha_b_scratch/`,",
        f"or `Research_Archive/organic_prototypes/`); copies here preserve the work",
        f"against scratch-dir cleanups.",
        "",
        f"See `../task{tid:03d}.md` (or `Task_by_Task/task{tid:03d}.md`) for",
        f"the full HIST block, current cost, and shipping status.",
        "",
        "## Files",
        "",
    ]
    for src in sorted(sources, key=lambda p: p.name):
        dest = dest_dir / src.name
        # Origin marker (relative to scratch root or repo)
        try:
            origin = src.relative_to(REPO).as_posix()
        except ValueError:
            origin = src.as_posix()
        if dest.exists():
            # Skip if identical (cheap hash compare)
            try:
                if dest.stat().st_size == src.stat().st_size and _hash(dest) == _hash(src):
                    counts["skipped_same"] += 1
                    readme_lines.append(f"- `{src.name}` — unchanged (origin: `{origin}`)")
                    continue
            except OSError:
                pass
            # Otherwise overwrite (newer source wins)
            counts["skipped_existing"] += 1  # was already present, will be updated
        action = "would copy" if dry_run else "copied"
        if not dry_run:
            shutil.copy2(src, dest)
        counts["copied"] += 1
        size = src.stat().st_size
        readme_lines.append(f"- `{src.name}` ({size:,} bytes) — {action} from `{origin}`")
    readme_lines.append("")
    if not dry_run:
        readme = dest_dir.parent / "prototypes" / "README.md"
        readme.write_text("\n".join(readme_lines), encoding="utf-8")
    return counts


def parse_tid_list(s: str) -> set[int]:
    out = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if part.startswith("task"):
            part = part[4:]
        try:
            out.add(int(part))
        except ValueError:
            pass
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tid", type=str, default="", help="Comma-separated task IDs (e.g. '4,110,161'); empty = all")
    p.add_argument("--dry-run", action="store_true", help="Show what would happen without copying")
    args = p.parse_args(argv)

    tid_filter = parse_tid_list(args.tid) if args.tid else None
    task_files = collect_task_files(tid_filter)

    if not task_files:
        print(f"No task files found in: {[str(d) for d in SOURCE_DIRS]}")
        return 0

    print(f"Found prototype work for {len(task_files)} tasks "
          f"({sum(len(v) for v in task_files.values())} files total).")
    print()

    total = {"copied": 0, "skipped_existing": 0, "skipped_same": 0}
    for tid in sorted(task_files):
        counts = promote_one(tid, task_files[tid], args.dry_run)
        for k, v in counts.items():
            total[k] += v
        flags = []
        if counts["copied"]:
            flags.append(f"copied {counts['copied']}")
        if counts["skipped_existing"]:
            flags.append(f"updated {counts['skipped_existing']}")
        if counts["skipped_same"]:
            flags.append(f"same {counts['skipped_same']}")
        flag_str = ", ".join(flags) if flags else "-"
        print(f"  task{tid:03d}: {len(task_files[tid]):>2} files ({flag_str})")

    print()
    print(f"Totals: copied={total['copied']}, updated={total['skipped_existing']}, "
          f"unchanged={total['skipped_same']}")
    if args.dry_run:
        print("(dry-run; no files actually copied)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
