#!/usr/bin/env python3
"""Idempotently re-apply the NeuroGolf Hermes ``_cprint`` headless patch.

Hermes v0.14.0 introduced a ``_cprint("Initializing agent...")`` call that
goes through prompt_toolkit. On Windows, prompt_toolkit's Win32Output
constructor raises ``NoConsoleScreenBufferError`` when stdout is not a TTY,
which crashes every kanban worker the dispatcher spawns headlessly. The fix
is a small early-return in ``_cprint`` when stdout isn't a TTY.

Every ``hermes update`` overwrites ``cli.py`` and wipes the patch. Run this
script after each update to re-apply it. It is idempotent: if the patch is
already present, it exits 0 with no change.

Usage:
    python tools/patch_hermes_cprint.py            # check + apply if missing
    python tools/patch_hermes_cprint.py --check    # exit 0 if patched, 1 if not
    python tools/patch_hermes_cprint.py --verify   # alias for --check
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

HERMES_CLI = Path.home() / "AppData" / "Local" / "hermes" / "hermes-agent" / "cli.py"
PATCH_MARKER = "NEUROGOLF_CPRINT_HEADLESS_PATCH_V1"
# Sibling tools read this to find out when the cprint patch was applied so
# they can decide which historical task_runs failure rows are "stale" vs real.
PATCH_TIMESTAMP_FILE = Path(__file__).resolve().parent.parent / ".hermes" / "board" / "cprint_patch_applied_at.txt"

ORIGINAL_BLOCK = '''def _cprint(text: str):
    """Print ANSI-colored text through prompt_toolkit's native renderer.

    Raw ANSI escapes written via print() are swallowed by patch_stdout's
    StdoutProxy.  Routing through print_formatted_text(ANSI(...)) lets
    prompt_toolkit parse the escapes and render real colors.

    When called from a background thread while a prompt_toolkit
    ``Application`` is running (the common case for the self-improvement
    background review's ``💾 …`` summary, curator summaries, and other
    bg-thread emissions), a direct ``_pt_print`` races with the input
    area's redraw and the line can end up visually buried behind the
    prompt.  Route those cases through ``run_in_terminal`` via
    ``loop.call_soon_threadsafe``, which pauses the input area, prints
    the line above it, and redraws the prompt cleanly.
    """
    _record_output_history(text)

    try:
        from prompt_toolkit.application import get_app_or_none, run_in_terminal
    except Exception:
        _pt_print(_PT_ANSI(text))
        return'''

PATCHED_BLOCK = '''def _cprint(text: str):
    """Print ANSI-colored text through prompt_toolkit's native renderer.

    Raw ANSI escapes written via print() are swallowed by patch_stdout's
    StdoutProxy.  Routing through print_formatted_text(ANSI(...)) lets
    prompt_toolkit parse the escapes and render real colors.

    When called from a background thread while a prompt_toolkit
    ``Application`` is running (the common case for the self-improvement
    background review's ``💾 …`` summary, curator summaries, and other
    bg-thread emissions), a direct ``_pt_print`` races with the input
    area's redraw and the line can end up visually buried behind the
    prompt.  Route those cases through ``run_in_terminal`` via
    ``loop.call_soon_threadsafe``, which pauses the input area, prints
    the line above it, and redraws the prompt cleanly.

    NEUROGOLF_CPRINT_HEADLESS_PATCH_V1
    Local patch (2026-05-17): when stdout is not a TTY (Hermes dispatcher
    spawning a kanban worker headlessly on Windows), prompt_toolkit's
    Win32Output constructor raises NoConsoleScreenBufferError and the worker
    crashes immediately. Fall back to plain stdout in that case so kanban-
    dispatched workers can run. Re-apply via
    tools/patch_hermes_cprint.py after every `hermes update`.
    """
    _record_output_history(text)

    import sys as _sys
    try:
        _stdout_is_tty = bool(_sys.stdout and _sys.stdout.isatty())
    except Exception:
        _stdout_is_tty = False
    if not _stdout_is_tty:
        try:
            print(text)
        except Exception:
            pass
        return

    try:
        from prompt_toolkit.application import get_app_or_none, run_in_terminal
    except Exception:
        _pt_print(_PT_ANSI(text))
        return'''


def is_patched(text: str) -> bool:
    return PATCH_MARKER in text


def apply_patch(text: str) -> tuple[str, str]:
    """Return (patched_text, status). status is one of: already, applied, original-not-found."""
    if PATCH_MARKER in text:
        return text, "already"
    if ORIGINAL_BLOCK in text:
        return text.replace(ORIGINAL_BLOCK, PATCHED_BLOCK, 1), "applied"
    return text, "original-not-found"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", "--verify", dest="check", action="store_true",
                        help="Exit 0 if patched, 1 if not. Make no changes.")
    parser.add_argument("--cli-path", type=Path, default=HERMES_CLI,
                        help="Path to Hermes cli.py (default: %(default)s)")
    # Accept the standard cron-launcher flags so this tool can be wired into
    # ~/AppData/Local/hermes/scripts/neurogolf_dependency_serializer.py JOBS map
    # and fire every N minutes without breaking the launcher's contract.
    parser.add_argument("--apply", action="store_true",
                        help="Cron-launcher compatibility: default behavior already applies if missing.")
    parser.add_argument("--json", action="store_true",
                        help="Emit a one-line JSON status for cron --no-agent delivery.")
    args = parser.parse_args(argv)

    if not args.cli_path.exists():
        print(f"ERROR: Hermes cli.py not found at {args.cli_path}", file=sys.stderr)
        return 2

    text = args.cli_path.read_text(encoding="utf-8")

    if args.check:
        if is_patched(text):
            if args.json:
                print(json.dumps({"schema": "neurogolf.patch_hermes_cprint.v1", "status": "present"}))
            else:
                print(f"OK: patch present in {args.cli_path}")
            return 0
        if args.json:
            print(json.dumps({"schema": "neurogolf.patch_hermes_cprint.v1", "status": "missing", "cli_path": str(args.cli_path)}))
        else:
            print(f"MISSING: patch not found in {args.cli_path}", file=sys.stderr)
        return 1

    patched_text, status = apply_patch(text)
    if status == "already":
        # Ensure the timestamp file exists so downstream tools (e.g.
        # reset_stale_crash_counters) can read it. If missing, record now —
        # this is a conservative under-estimate of when it was applied but
        # is still strictly after any pre-patch crash.
        seeded_now = False
        if not PATCH_TIMESTAMP_FILE.exists():
            import time as _time
            PATCH_TIMESTAMP_FILE.parent.mkdir(parents=True, exist_ok=True)
            PATCH_TIMESTAMP_FILE.write_text(str(int(_time.time())) + "\n", encoding="utf-8")
            seeded_now = True
        if args.json:
            # Empty stdout means cron --no-agent delivers nothing. That's fine
            # for "no-op" ticks; the cron run still shows ok in the schedule list.
            return 0
        print(f"OK: patch already applied to {args.cli_path}")
        if seeded_now:
            print(f"Patch timestamp seeded at {PATCH_TIMESTAMP_FILE}")
        return 0
    if status == "original-not-found":
        print(
            f"ERROR: could not locate the original `_cprint` block in {args.cli_path}.\n"
            f"This likely means Hermes shipped a new version where the function changed.\n"
            f"Re-read the function and update tools/patch_hermes_cprint.py "
            f"ORIGINAL_BLOCK / PATCHED_BLOCK constants.",
            file=sys.stderr,
        )
        return 3
    backup = args.cli_path.with_suffix(args.cli_path.suffix + ".bak_pre_neurogolf_patch")
    if not backup.exists():
        shutil.copy2(args.cli_path, backup)
    args.cli_path.write_text(patched_text, encoding="utf-8")
    # Record the apply timestamp so tools/reset_stale_crash_counters.py can
    # detect pre-patch failures without a hardcoded constant.
    import time as _time
    PATCH_TIMESTAMP_FILE.parent.mkdir(parents=True, exist_ok=True)
    PATCH_TIMESTAMP_FILE.write_text(str(int(_time.time())) + "\n", encoding="utf-8")
    if args.json:
        # Loud signal in JSON: re-applied means hermes update wiped the patch.
        print(json.dumps({
            "schema": "neurogolf.patch_hermes_cprint.v1",
            "status": "re-applied",
            "marker": PATCH_MARKER,
            "cli_path": str(args.cli_path),
            "backup_existed": backup.exists(),
            "warning": "hermes update appears to have wiped the headless-worker patch; auto-re-applied. If this fires repeatedly investigate the update cadence.",
        }))
    else:
        print(f"APPLIED: {PATCH_MARKER} -> {args.cli_path}")
        print(f"Patch timestamp recorded at {PATCH_TIMESTAMP_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
