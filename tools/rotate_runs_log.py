"""Archive Logs/runs.jsonl when it grows past a size threshold.

Each `--hybrid` run appends 400 lines (~70 KB) to `Logs/runs.jsonl`. After ~100
runs the file is several MB and `git diff`/`git blame` start to feel sluggish.
This tool archives the current log to `Logs/runs.jsonl.<UTC-iso>.archive` and
truncates the live file to keep it growable but bounded.

Default: archive when size > 5 MB. Run manually as part of session housekeeping;
not invoked automatically by the build pipeline (rotation as a side-effect of
building would surprise users running back-to-back regressions).

Usage:
    python tools/rotate_runs_log.py                 # archive if > 5 MB; no-op otherwise
    python tools/rotate_runs_log.py --threshold-mb 10
    python tools/rotate_runs_log.py --force         # archive regardless of size
    python tools/rotate_runs_log.py --keep-tail 4000  # keep the last N lines (default 0 = empty)

Exit code:
    0  archived (or no-op because under threshold without --force).
    1  error (target missing or write failed).
"""

import argparse
import datetime
import pathlib
import shutil
import sys

NG_ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNS_LOG = NG_ROOT / "Logs" / "runs.jsonl"


def rotate(threshold_mb=5.0, force=False, keep_tail=0, verbose=True):
    """Archive runs.jsonl. Returns the archive path on rotation, None otherwise."""
    if not RUNS_LOG.exists():
        if verbose:
            print(f"  {RUNS_LOG.relative_to(NG_ROOT).as_posix()} does not exist (nothing to rotate).")
        return None

    size_bytes = RUNS_LOG.stat().st_size
    size_mb = size_bytes / (1024 * 1024)

    if not force and size_mb < threshold_mb:
        if verbose:
            print(f"  {RUNS_LOG.relative_to(NG_ROOT).as_posix()} is {size_mb:.2f} MB (< {threshold_mb} MB threshold). No rotation.")
        return None

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    archive_path = RUNS_LOG.with_suffix(f".jsonl.{stamp}.archive")
    shutil.copy2(RUNS_LOG, archive_path)

    if keep_tail > 0:
        with RUNS_LOG.open("r", encoding="utf-8") as f:
            all_lines = f.readlines()
        kept = all_lines[-keep_tail:] if len(all_lines) > keep_tail else all_lines
        with RUNS_LOG.open("w", encoding="utf-8") as f:
            f.writelines(kept)
        new_lines = len(kept)
    else:
        # Truncate to empty
        RUNS_LOG.write_text("", encoding="utf-8")
        new_lines = 0

    if verbose:
        archived_size_mb = archive_path.stat().st_size / (1024 * 1024)
        print(f"  Archived: {archive_path.relative_to(NG_ROOT).as_posix()} ({archived_size_mb:.2f} MB)")
        print(f"  Live:     {RUNS_LOG.relative_to(NG_ROOT).as_posix()} truncated to {new_lines} lines")

    return archive_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--threshold-mb", type=float, default=5.0,
                        help="Rotate only if file exceeds this many MB (default 5)")
    parser.add_argument("--force", action="store_true",
                        help="Rotate regardless of current size")
    parser.add_argument("--keep-tail", type=int, default=0,
                        help="Keep the last N lines in the live file after rotation (default 0 = empty)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    try:
        rotate(threshold_mb=args.threshold_mb, force=args.force,
               keep_tail=args.keep_tail, verbose=not args.quiet)
    except (OSError, IOError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
