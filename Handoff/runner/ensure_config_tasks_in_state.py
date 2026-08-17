#!/usr/bin/env python3
"""Merge configured HandOff runner tasks into an existing state file.

This is a repair helper for the narrow case where a live loop's config is
rerendered while the current Python runner process already has the old task list
in memory. The next runner invocation will do this naturally in Runner.load_state;
this helper keeps the state file from looking drained before that next pass.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STILL_ACTIVE = 259
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return False
        return code.value == STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def default_task_state(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": task["task"],
        "title": task["title"],
        "handoff_task_id": task.get("handoff_task_id", ""),
        "external_key": task.get("external_key"),
        "prompt_file": task["prompt_file"],
        "status": "todo",
        "worker_attempts": 0,
        "worker_session_id": None,
        "worker_agent_id": None,
        "reviewer_session_id": None,
        "reviewer_agent_id": None,
        "worker_last_output": None,
        "reviewer_last_output": None,
        "reviewer_attack_log": "",
        "reviewer_rework_prompt": "",
        "captain_packet_path": None,
        "history": [],
    }


def merge_once(config_path: Path, state_path: Path) -> list[str]:
    config = read_json(config_path)
    state = read_json(state_path)
    tasks = state.setdefault("tasks", {})
    added: list[str] = []
    for task in config.get("tasks", []):
        task_name = str(task["task"])
        if task_name not in tasks:
            tasks[task_name] = default_task_state(task)
            added.append(task_name)
        else:
            tasks[task_name]["handoff_task_id"] = task.get("handoff_task_id", tasks[task_name].get("handoff_task_id"))
            tasks[task_name]["external_key"] = task.get("external_key", tasks[task_name].get("external_key"))
            tasks[task_name]["prompt_file"] = task.get("prompt_file", tasks[task_name].get("prompt_file"))
    if added:
        state["updated_utc"] = utc_now()
        state.setdefault("repair_history", []).append(
            {"ts_utc": utc_now(), "kind": "ensure_config_tasks_in_state", "added": added}
        )
        write_json(state_path, state)
    return added


def append_log(path: Path | None, message: str) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{utc_now()} {message}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Ensure every config task exists in a runner state file.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--watch-pid", type=int)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--max-seconds", type=float, default=0.0)
    parser.add_argument("--log", type=Path)
    args = parser.parse_args()

    start = time.monotonic()
    while True:
        try:
            added = merge_once(args.config, args.state)
            if added:
                append_log(args.log, "added " + ",".join(added))
        except Exception as exc:
            append_log(args.log, f"merge failed: {exc}")

        if args.watch_pid and not pid_alive(args.watch_pid):
            try:
                added = merge_once(args.config, args.state)
                append_log(args.log, "watch pid exited; final merge added " + (",".join(added) if added else "<none>"))
            except Exception as exc:
                append_log(args.log, f"final merge failed: {exc}")
            break
        if not args.watch_pid:
            break
        if args.max_seconds and time.monotonic() - start >= args.max_seconds:
            append_log(args.log, "max seconds reached")
            break
        time.sleep(max(0.1, args.interval))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
