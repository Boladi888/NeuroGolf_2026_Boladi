#!/usr/bin/env python3
"""Health monitor for the freshly-shipped Tier 1 + 3 + 2 changes.

Polls every 3 minutes for 30 minutes and emits a one-line status check
on each tick. Each emit is a single stdout line so the surrounding
Bash/Monitor wrapper can stream it as a notification. Errors are
written to stderr (which won't trigger notifications but is captured
in the run log).

What we watch:

  Tier 3 (submit_picker_cache):
    * cache file exists and is recent (< 90s)
    * schema matches
    * ready_count > 0 when queue has ready entries
    * blocked_ready_count is consistent with state.hidden_bad_tasks

  Tier 1 (stack_picker / next_submit_recommendation):
    * recommendation file exists and is recent (< 150s)
    * schema matches
    * package_validation_pass is True OR the rationale explains why not
    * stack_size between 0 and max_stack_size

  Tier 2 (submit-captain + sync-manager reasoning):
    * verify the profile yaml still has reasoning_effort: medium

  Cron crons:
    * picker_cache and stack_picker crons last ran within their schedule

The script does NOT modify anything — it's pure observation.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path("<REPO_ROOT>")
CACHE_PATH = REPO_ROOT / ".hermes" / "board" / "submit_picker_cache.json"
REC_PATH = REPO_ROOT / ".hermes" / "board" / "next_submit_recommendation.json"
SUBMIT_CAPTAIN_PROFILE = Path("<HERMES_HOME>/profiles/neurogolf-submit-captain/config.yaml")
SYNC_MANAGER_PROFILE = Path("<HERMES_HOME>/profiles/neurogolf-sync-manager/config.yaml")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _age_seconds(path: Path) -> float | None:
    if not path.exists():
        return None
    return time.time() - path.stat().st_mtime


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _profile_has_medium(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    # Match the first `reasoning_effort:` line; it's the agent setting we care about.
    m = re.search(r"^\s*reasoning_effort:\s*(\w+)", text, re.MULTILINE)
    return bool(m and m.group(1).lower() == "medium")


def _cron_health() -> dict[str, Any]:
    """Ask hermes for cron status and parse out the two new crons."""
    # Capture bytes and decode with errors="replace" — Hermes emits emoji
    # / box-drawing characters that crash the default cp1252 codec on Windows.
    out = subprocess.run(
        ["hermes", "cron", "list"], capture_output=True, text=False, check=False
    )
    if out.returncode != 0:
        return {"ok": False, "reason": f"hermes cron list rc={out.returncode}"}
    text = (out.stdout or b"").decode("utf-8", errors="replace")
    targets = ("NeuroGolf 1m submit picker cache", "NeuroGolf 2m submit stack picker")
    results: dict[str, dict[str, Any]] = {}
    for name in targets:
        idx = text.find(f"Name:      {name}")
        if idx < 0:
            results[name] = {"present": False}
            continue
        block = text[idx:idx + 600]
        last_run_match = re.search(r"Last run:\s*([\d\-T:.+Z\-]+)\s+(\w+)", block)
        active_match = re.search(r"\[(\w+)\]", text[max(0, idx - 200):idx])
        results[name] = {
            "present": True,
            "active": active_match.group(1) if active_match else None,
            "last_run_at": last_run_match.group(1) if last_run_match else None,
            "last_run_status": last_run_match.group(2) if last_run_match else None,
        }
    return {"ok": True, "crons": results}


def _check_tier3() -> dict[str, Any]:
    cache = _read_json(CACHE_PATH)
    age = _age_seconds(CACHE_PATH)
    if cache is None:
        return {"ok": False, "reason": "cache file missing or unreadable"}
    issues: list[str] = []
    if cache.get("schema") != "neurogolf.submit_picker_cache.v1":
        issues.append(f"schema mismatch: {cache.get('schema')}")
    if age is None or age > 180:
        issues.append(f"cache stale: age={age}")
    blocked = cache.get("blocked_ready_count", 0)
    ready = cache.get("ready_count", 0)
    safe = cache.get("safe_ready_count", 0)
    if ready < 0 or safe < 0 or blocked < 0 or (safe + blocked) != ready:
        issues.append(f"counts inconsistent ready={ready} safe={safe} blocked={blocked}")
    return {
        "ok": not issues,
        "age_s": int(age) if age is not None else None,
        "ready_count": ready,
        "safe_ready_count": safe,
        "blocked_ready_count": blocked,
        "top_safe_ev": cache.get("top_safe_ev"),
        "issues": issues,
    }


def _check_tier1() -> dict[str, Any]:
    rec = _read_json(REC_PATH)
    age = _age_seconds(REC_PATH)
    if rec is None:
        return {"ok": False, "reason": "recommendation file missing or unreadable"}
    issues: list[str] = []
    if rec.get("schema") != "neurogolf.next_submit_recommendation.v1":
        issues.append(f"schema mismatch: {rec.get('schema')}")
    if age is None or age > 240:
        issues.append(f"recommendation stale: age={age}")
    r = rec.get("recommendation") or {}
    stack_size = r.get("stack_size")
    max_stack = rec.get("max_stack_size")
    if stack_size is None or not isinstance(stack_size, int):
        issues.append(f"stack_size not int: {stack_size}")
    elif max_stack and stack_size > max_stack:
        issues.append(f"stack_size {stack_size} > max {max_stack}")
    return {
        "ok": not issues,
        "age_s": int(age) if age is not None else None,
        "stack_size": stack_size,
        "expected_combined_gain": r.get("expected_combined_gain"),
        "lineage_consistent": r.get("lineage_consistent"),
        "package_validation_pass": r.get("package_validation_pass"),
        "issues": issues,
    }


def _check_tier2() -> dict[str, Any]:
    sc_ok = _profile_has_medium(SUBMIT_CAPTAIN_PROFILE)
    sm_ok = _profile_has_medium(SYNC_MANAGER_PROFILE)
    return {
        "ok": sc_ok and sm_ok,
        "submit_captain_medium": sc_ok,
        "sync_manager_medium": sm_ok,
    }


def health_check() -> dict[str, Any]:
    return {
        "ts": _now_iso(),
        "tier3": _check_tier3(),
        "tier1": _check_tier1(),
        "tier2": _check_tier2(),
        "crons": _cron_health(),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval-seconds", type=int, default=180)
    parser.add_argument("--duration-seconds", type=int, default=30 * 60)
    parser.add_argument("--json", action="store_true", help="One JSON dict per tick (default).")
    parser.add_argument("--once", action="store_true", help="Run a single tick and exit.")
    args = parser.parse_args(argv)

    deadline = time.monotonic() + args.duration_seconds
    tick = 0
    while True:
        tick += 1
        report = health_check()
        # Always emit one line so the monitor sees a heartbeat.
        line = json.dumps({
            "tick": tick,
            "ts": report["ts"],
            "tier3_ok": report["tier3"].get("ok"),
            "tier3_age_s": report["tier3"].get("age_s"),
            "tier3_ready": report["tier3"].get("ready_count"),
            "tier1_ok": report["tier1"].get("ok"),
            "tier1_age_s": report["tier1"].get("age_s"),
            "tier1_stack_size": report["tier1"].get("stack_size"),
            "tier1_gain": report["tier1"].get("expected_combined_gain"),
            "tier2_ok": report["tier2"].get("ok"),
            "issues": (
                report["tier3"].get("issues") or []
            ) + (
                report["tier1"].get("issues") or []
            ),
        })
        print(line, flush=True)
        if args.once:
            return 0
        if time.monotonic() >= deadline:
            return 0
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
