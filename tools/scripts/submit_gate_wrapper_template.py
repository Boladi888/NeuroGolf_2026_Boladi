#!/usr/bin/env python3
"""Cron wrapper template for tools/submit_gate_runner.py.

When you're ready to deploy submit_gate, copy this file to
``<HERMES_HOME>/scripts/submit_gate.py`` and register the
cron job (see ``docs/submit_gate_rollout.md``).

Mirrors the same pattern as ``<HERMES_HOME>/scripts/auto_triage_blocked.py``:
- Acquires the global ``neurogolf_python_control.lock`` (so it never races
  against auto_triage / dependency_serializer / autoforeman crons).
- Runs ``tools/neurogolf_smart_cleanup.py --apply --json`` first as a gate
  (returns 0 from this wrapper if cleanup fails — same protocol as the
  other control-runner wrappers).
- Then runs ``tools/submit_gate_runner.py`` in the mode the cron passes
  via arguments (``--apply --json`` for live; bare for shadow/dry-run).
- Emits the runner's JSON output only when there were applied actions
  (silent ticks don't spam the dashboard).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path("<REPO_ROOT>")
LOCK_PATH = Path("<HERMES_HOME>/neurogolf_python_control.lock")
LOCK_STALE_SECONDS = 60 * 60  # 1hr (was 20min — submit_gate mechanical-verify on N cards can exceed 20min on large batches; longer stale window prevents races where a slow-but-healthy run gets its lock stolen mid-tick. Genuine hangs still get cleaned up.)
LOCK_WAIT_SECONDS = 55.0
LOCK_RETRY_SECONDS = 0.5

CLEANUP_SCRIPT = ("tools/neurogolf_smart_cleanup.py", ["--apply", "--json"])

# Set DEPLOY_MODE = "shadow" for Phase 2 (no --apply, no kanban mutation, no
# Kaggle submit; produces a dry-run audit log only). Flip to "live" for
# Phase 3 once the shadow-mode diff against submit-captain LLM behavior has
# been reviewed.
DEPLOY_MODE = "shadow"  # "shadow" | "live"


def _job_cmd() -> list[str]:
    base = [sys.executable, str(REPO_ROOT / "tools/submit_gate_runner.py"), "--json"]
    if DEPLOY_MODE == "live":
        base.append("--apply")
    return base


class ControlLock:
    def __init__(self, path: Path = LOCK_PATH) -> None:
        self.path = path
        self.fd: int | None = None

    def __enter__(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = self._try_open()
        deadline = time.monotonic() + LOCK_WAIT_SECONDS
        while self.fd is None:
            if self._is_stale():
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
                self.fd = self._try_open()
                continue
            if time.monotonic() >= deadline:
                return False
            time.sleep(LOCK_RETRY_SECONDS)
            self.fd = self._try_open()
        return True

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self.fd is None:
            return
        os.close(self.fd)
        self.fd = None
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

    def _try_open(self) -> int | None:
        try:
            fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except (FileExistsError, PermissionError):
            return None
        os.write(fd, f"pid={os.getpid()} started={time.time():.0f}\n".encode("utf-8"))
        return fd

    def _is_stale(self) -> bool:
        try:
            age = time.time() - self.path.stat().st_mtime
        except FileNotFoundError:
            return False
        return age > LOCK_STALE_SECONDS


def main() -> int:
    cleanup_cmd = [sys.executable, str(REPO_ROOT / CLEANUP_SCRIPT[0])] + CLEANUP_SCRIPT[1]
    job_cmd = _job_cmd()
    with ControlLock() as acquired:
        if not acquired:
            # Emit a one-line JSON skip notice (matches the auto_triage
            # shared-wrapper pattern). Without this, cron last-status shows
            # "ok" even when this tick did zero work due to lock contention,
            # making it impossible to debug missed cards.
            payload = {
                "schema": "neurogolf.submit_gate.v1",
                "status": "skipped-lock-contended",
                "lock": str(LOCK_PATH),
            }
            print(json.dumps(payload))
            return 0
        cleanup_proc = subprocess.run(
            cleanup_cmd, cwd=str(REPO_ROOT), text=True, capture_output=True, check=False,
            encoding="utf-8", errors="replace",
        )
        if cleanup_proc.returncode != 0:
            return cleanup_proc.returncode
        job_proc = subprocess.run(
            job_cmd, cwd=str(REPO_ROOT), text=True, capture_output=True, check=False,
            encoding="utf-8", errors="replace",
        )

    stdout = (job_proc.stdout or "").strip()
    if job_proc.returncode != 0:
        if stdout:
            print(stdout)
        if job_proc.stderr:
            print(job_proc.stderr, file=sys.stderr)
        return job_proc.returncode
    if not stdout:
        return 0
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        print(stdout)
        return 0
    # Only print the report when there's actual work in it — silent ticks
    # keep dashboard noise low.
    if payload.get("cards_processed", 0) > 0:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
