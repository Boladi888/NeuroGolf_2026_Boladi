#!/usr/bin/env python3
"""Idempotently re-apply the NeuroGolf Hermes log-rotation Windows-lock patch.

Hermes' ``_ManagedRotatingFileHandler.doRollover()`` (in
``hermes-agent/hermes_logging.py``) calls ``super().doRollover()``, which goes
through ``logging.handlers.RotatingFileHandler.rotate()`` and ultimately
``os.rename(agent.log -> agent.log.1)``. On Windows, the gateway thread holds
``agent.log`` open while the cron-runner thread is trying to rotate it, so
``os.rename()`` raises ``PermissionError: [WinError 32] The process cannot
access the file because it is being used by another process``. The stdlib then
re-raises the same traceback on every subsequent emit call — gateway health
pings (~3 s cadence) and cron job stdout lines — flooding stderr.

The patch wraps ``super().doRollover()`` in a try/except that swallows the
Windows file-lock error and just keeps writing to the current file. The next
rotation attempt will succeed once the lock frees.

Every ``hermes update`` overwrites ``hermes_logging.py`` and wipes the patch.
Run this script after each update to re-apply. Idempotent.

Usage:
    python tools/patch_hermes_logging.py            # check + apply if missing
    python tools/patch_hermes_logging.py --check    # exit 0 if patched, 1 if not
    python tools/patch_hermes_logging.py --apply    # cron-compatibility flag
    python tools/patch_hermes_logging.py --json     # cron-style output
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

HERMES_LOGGING = Path.home() / "AppData" / "Local" / "hermes" / "hermes-agent" / "hermes_logging.py"
PATCH_MARKER = "NEUROGOLF_LOGROTATE_HEADLESS_PATCH_V1"

ORIGINAL_BLOCK = '''    def doRollover(self):
        super().doRollover()
        self._chmod_if_managed()'''

PATCHED_BLOCK = '''    def doRollover(self):
        # NEUROGOLF_LOGROTATE_HEADLESS_PATCH_V1
        # On Windows, the gateway thread holds agent.log open while the
        # cron-runner thread tries to rotate it. RotatingFileHandler.rotate()
        # calls os.rename(), which raises WinError 32. The stdlib then
        # re-raises on every subsequent emit, spamming tracebacks every
        # ~3 seconds. Swallow the error: keep writing to the current file
        # until the lock frees on the next rollover attempt. Re-apply via
        # tools/patch_hermes_logging.py after every `hermes update`.
        try:
            super().doRollover()
        except (PermissionError, OSError):
            if self.stream is None:
                try:
                    self.stream = self._open()
                except OSError:
                    pass
            return
        self._chmod_if_managed()'''


def is_patched(text: str) -> bool:
    return PATCH_MARKER in text


def apply_patch(text: str) -> tuple[str, str]:
    """Return (patched_text, status). status: already / applied / original-not-found."""
    if PATCH_MARKER in text:
        return text, "already"
    if ORIGINAL_BLOCK in text:
        return text.replace(ORIGINAL_BLOCK, PATCHED_BLOCK, 1), "applied"
    return text, "original-not-found"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", "--verify", dest="check", action="store_true",
                        help="Exit 0 if patched, 1 if not. Make no changes.")
    parser.add_argument("--logging-path", type=Path, default=HERMES_LOGGING,
                        help="Path to Hermes hermes_logging.py (default: %(default)s)")
    parser.add_argument("--apply", action="store_true",
                        help="Cron-launcher compatibility: default behavior already applies if missing.")
    parser.add_argument("--json", action="store_true",
                        help="Emit a one-line JSON status for cron --no-agent delivery.")
    args = parser.parse_args(argv)

    if not args.logging_path.exists():
        msg = f"ERROR: Hermes hermes_logging.py not found at {args.logging_path}"
        if args.json:
            print(json.dumps({"schema": "neurogolf.patch_hermes_logging.v1", "status": "missing-file", "path": str(args.logging_path)}))
        else:
            print(msg, file=sys.stderr)
        return 2

    text = args.logging_path.read_text(encoding="utf-8")

    if args.check:
        if is_patched(text):
            if args.json:
                print(json.dumps({"schema": "neurogolf.patch_hermes_logging.v1", "status": "present"}))
            else:
                print(f"OK: patch present in {args.logging_path}")
            return 0
        if args.json:
            print(json.dumps({"schema": "neurogolf.patch_hermes_logging.v1", "status": "missing", "path": str(args.logging_path)}))
        else:
            print(f"MISSING: patch not found in {args.logging_path}", file=sys.stderr)
        return 1

    patched_text, status = apply_patch(text)
    if status == "already":
        if args.json:
            return 0
        print(f"OK: patch already applied to {args.logging_path}")
        return 0
    if status == "original-not-found":
        msg = (
            f"ERROR: could not locate the original `doRollover` block in {args.logging_path}.\n"
            f"This likely means Hermes shipped a new version where the method changed.\n"
            f"Re-read the method and update tools/patch_hermes_logging.py "
            f"ORIGINAL_BLOCK / PATCHED_BLOCK constants."
        )
        if args.json:
            print(json.dumps({"schema": "neurogolf.patch_hermes_logging.v1", "status": "original-not-found", "path": str(args.logging_path)}))
        else:
            print(msg, file=sys.stderr)
        return 3

    backup = args.logging_path.with_suffix(args.logging_path.suffix + ".bak_pre_neurogolf_patch")
    if not backup.exists():
        shutil.copy2(args.logging_path, backup)
    args.logging_path.write_text(patched_text, encoding="utf-8")
    if args.json:
        print(json.dumps({
            "schema": "neurogolf.patch_hermes_logging.v1",
            "status": "re-applied",
            "marker": PATCH_MARKER,
            "path": str(args.logging_path),
            "backup_existed": backup.exists(),
            "warning": "hermes update appears to have wiped the log-rotation patch; auto-re-applied. If this fires repeatedly investigate the update cadence.",
        }))
    else:
        print(f"APPLIED: {PATCH_MARKER} -> {args.logging_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
