#!/usr/bin/env python3
"""Low-memory local monitor for the NeuroGolf HandOff runner.

Watches the runner *artifacts* (state-file freshness + task statuses), not just
process liveness, and emits on every terminal condition — complete, crashed/
interrupted, or stalled — instead of polling silently forever (the failure mode
that silently orphaned the batch31 monitor). Stamps its own pid file with a
terminal state on exit so a dead monitor is itself detectable.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path.cwd()
INTERESTING_OP_RE = re.compile(
    r"runner_worker_complete|runner_review_complete|runner_captain_complete|"
    r"handoff_sync_failed|runner_.*exception|runner_.*failed|runner_.*retry|"
    r"runner_state_corrupt|runner_terminate_attempt|runner_.*gate_cleanup|failed",
    re.I,
)
# Statuses that will not change without operator action. Used for "is the batch
# done?" detection.
TERMINAL_STATUSES = {
    "ship_packeted",
    "captain_staged",
    "drop",
    "worker_failed",
    "review_failed",
    "review_parse_failed",
    "needs_packet_fix",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_utc(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return default
    except (json.JSONDecodeError, ValueError):
        # A half-written state file (runner mid-save) should not crash the
        # monitor; treat as "no reading this tick".
        return default


def load_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.lstrip().startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def append_jsonl(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=True, separators=(",", ":")))
        f.write("\n")


def bounded_tail(path: Path, max_bytes: int = 200_000) -> list[str]:
    if not path.exists():
        return []
    size = path.stat().st_size
    with path.open("rb") as f:
        if size > max_bytes:
            f.seek(-max_bytes, os.SEEK_END)
        data = f.read()
    return data.decode("utf-8", errors="replace").splitlines()


def pid_running(pid: Any) -> bool:
    try:
        pid_int = int(pid)
    except (TypeError, ValueError):
        return False
    if pid_int <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes

            access = 0x1000  # PROCESS_QUERY_LIMITED_INFORMATION
            handle = ctypes.windll.kernel32.OpenProcess(access, False, pid_int)
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            return False
        except Exception:
            return False
    try:
        os.kill(pid_int, 0)
        return True
    except OSError:
        return False


def summarize_state(state: dict[str, Any]) -> tuple[dict[str, int], list[dict[str, Any]]]:
    counts: dict[str, int] = {}
    non_todo: list[dict[str, Any]] = []
    tasks = state.get("tasks") or {}
    for task_name in sorted(tasks):
        task = tasks[task_name]
        status = str(task.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
        if status != "todo":
            non_todo.append(
                {
                    "task": task.get("task", task_name),
                    "status": status,
                    "attempts": task.get("worker_attempts"),
                    # Codex lane uses *_agent_id; Claude lane uses assignee/reviewer.
                    "worker": task.get("worker_agent_id") or task.get("assignee"),
                    "reviewer": task.get("reviewer_agent_id") or task.get("reviewer"),
                    "packet": task.get("captain_packet_path"),
                }
            )
    return counts, non_todo


def batch_done(counts: dict[str, int]) -> bool:
    total = sum(counts.values())
    terminal = sum(v for k, v in counts.items() if k in TERMINAL_STATUSES)
    return total > 0 and terminal == total


def status_to_handoff(status: str) -> str:
    if status == "todo":
        return "todo"
    if status in {"ship_packeted", "captain_staged"}:
        return "verified"
    if status in {"drop", "rejected"}:
        return "rejected"
    if status in {"pending_review", "needs_packet_fix"}:
        return "pending_verification"
    return "in_progress"


def board_headers() -> dict[str, str]:
    headers = {"User-Agent": "Mozilla/5.0 NeuroGolf-HandoffMonitor/1.0"}
    token = load_env_file(REPO / ".env.handoff").get("HANDOFF_BOOTSTRAP_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def resolve_project_id(args: argparse.Namespace) -> str | None:
    """Project/request id used for board reads.

    Explicit ``--project-id`` wins; otherwise read it from the runner ``--config``'s
    ``project`` block — the same source the runner uses, so the monitor and runner
    can never disagree about which project the tasks live under.
    """
    explicit = getattr(args, "project_id", None)
    if explicit:
        return explicit
    config_rel = getattr(args, "config", None)
    if config_rel:
        config = read_json(REPO / config_rel, {}) or {}
        project = config.get("project") or {}
        return project.get("project_id") or project.get("request_id")
    return None


def board_assignee_id(task: dict[str, Any]) -> str | None:
    for key in ("assignee", "assigned_to", "assigned_agent", "agent_id"):
        val = task.get(key)
        if isinstance(val, str):
            return val
        if isinstance(val, dict):
            for nested_key in ("agent_id", "id", "name"):
                nested = val.get(nested_key)
                if isinstance(nested, str):
                    return nested
    return None


def fetch_board_tasks(
    project_id: str, headers: dict[str, str]
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], str | None]:
    """Read the whole project task list once and index it by id and external_key.

    The board has no working ``GET /api/v1/tasks/{id}`` route: that bare path
    returns the SPA 404 fallback (``Cannot GET /api/v1/tasks/...``) even though the
    runner's PUT to the same path works. The supported read path — the one the MCP
    ``list_tasks`` tool uses — is ``GET /api/v1/requests/{project_id}/tasks``, which
    returns full task objects (``id``, ``status``, ``external_key``) for the project.

    Returns ``(by_id, by_external_key, error)``. On any failure both indexes are
    empty and ``error`` is set — one failure for the whole pass, not one per task.
    """
    url = (
        "https://handoff.socnet.lol/api/v1/requests/"
        f"{urllib.parse.quote(project_id, safe='')}/tasks"
    )
    req = urllib.request.Request(url, method="GET", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        return {}, {}, f"HTTP {exc.code}: {text[:300]}"
    except Exception as exc:
        return {}, {}, str(exc)
    tasks = data.get("tasks") if isinstance(data, dict) else data
    if not isinstance(tasks, list):
        return {}, {}, f"unexpected board response shape: {type(tasks).__name__}"
    by_id: dict[str, dict[str, Any]] = {}
    by_key: dict[str, dict[str, Any]] = {}
    for board_task in tasks:
        if not isinstance(board_task, dict):
            continue
        board_id = board_task.get("id") or board_task.get("task_id")
        if board_id:
            by_id[str(board_id)] = board_task
        external_key = board_task.get("external_key")
        if external_key:
            by_key[str(external_key)] = board_task
    return by_id, by_key, None


def check_board_drift(state: dict[str, Any], project_id: str | None) -> list[dict[str, Any]]:
    """Compare local task state against the HandOff board.

    Reads the project task list once and looks each local task up by
    ``external_key`` first (the durable identity — ``handoff_task_id`` gets
    repointed per batch) then by ``handoff_task_id``. A board that can't be read
    yields a single ``board_read_failed`` entry for the whole pass rather than one
    per task, so an outage or a wrong endpoint is never mistaken for wholesale
    drift across every task.
    """
    if not project_id:
        return [{"kind": "board_check_unconfigured", "error": "no project_id (pass --config or --project-id)"}]
    by_id, by_key, error = fetch_board_tasks(project_id, board_headers())
    if error:
        return [{"kind": "board_read_failed", "error": error}]

    drift: list[dict[str, Any]] = []
    for task_name, task in sorted((state.get("tasks") or {}).items()):
        task_id = task.get("handoff_task_id")
        external_key = task.get("external_key")
        if not task_id and not external_key:
            continue
        board_task = by_key.get(str(external_key)) if external_key else None
        if board_task is None and task_id:
            board_task = by_id.get(str(task_id))
        local_status = str(task.get("status", "unknown"))
        expected_status = status_to_handoff(local_status)
        local_assignee = task.get("worker_agent_id") or task.get("assignee")
        if board_task is None:
            drift.append(
                {
                    "task": task.get("task", task_name),
                    "handoff_task_id": task_id,
                    "external_key": external_key,
                    "kind": "not_on_board",
                    "local_status": local_status,
                    "expected_board_status": expected_status,
                }
            )
            continue
        board_status = str(board_task.get("status", "unknown"))
        board_assignee = board_assignee_id(board_task)
        if board_status != expected_status:
            drift.append(
                {
                    "task": task.get("task", task_name),
                    "handoff_task_id": task_id,
                    "kind": "status_mismatch",
                    "local_status": local_status,
                    "expected_board_status": expected_status,
                    "board_status": board_status,
                    "local_assignee": local_assignee,
                    "board_assignee": board_assignee,
                }
            )
            continue
        if expected_status in {"in_progress", "pending_verification"} and local_assignee and board_assignee and local_assignee != board_assignee:
            drift.append(
                {
                    "task": task.get("task", task_name),
                    "handoff_task_id": task_id,
                    "kind": "assignee_mismatch",
                    "local_status": local_status,
                    "board_status": board_status,
                    "local_assignee": local_assignee,
                    "board_assignee": board_assignee,
                }
            )
    return drift


def _lane_tokens(lane: str) -> set[str]:
    lane = str(lane or "").strip()
    tokens = {lane.lower()}
    if lane.lower() == "codex":
        tokens.add("codex")
    elif lane.lower() == "codex2":
        tokens.add("codex2")
    elif lane.lower() == "claude":
        tokens.add("claude")
    return {t for t in tokens if t}


def check_gate_orphans(args: argparse.Namespace) -> list[dict[str, Any]]:
    if getattr(args, "skip_gate_orphans", False):
        return []
    snapshot_tool = REPO / "tools" / "ng_gate_queue_snapshot.py"
    if not snapshot_tool.exists():
        return []
    try:
        proc = subprocess.run(
            [
                sys.executable,
                str(snapshot_tool),
                "--json",
                "--max-rescue",
                "0",
                "--state-days",
                "7",
            ],
            cwd=str(REPO),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except Exception as exc:
        return [{"kind": "gate_snapshot_failed", "error": str(exc)}]
    if proc.returncode != 0:
        return [{"kind": "gate_snapshot_failed", "returncode": proc.returncode, "stderr": proc.stderr[-1000:]}]
    try:
        snapshot = json.loads(proc.stdout)
    except Exception as exc:
        return [{"kind": "gate_snapshot_parse_failed", "error": str(exc), "stdout": proc.stdout[-1000:]}]

    batch = str(args.batch)
    lane_tokens = _lane_tokens(getattr(args, "lane", ""))
    alerts: list[dict[str, Any]] = []
    for run in snapshot.get("runs", []):
        owner = run.get("owner_meta") or {}
        owner_lane = str(owner.get("lane_id") or owner.get("lane") or "").lower()
        owner_batch = str(owner.get("batch") or "")
        if owner_batch != batch or owner_lane not in lane_tokens:
            continue
        flags = run.get("orphan_flags") or []
        if not flags:
            continue
        alerts.append(
            {
                "kind": "gate_orphan_suspect",
                "task": run.get("task"),
                "state": run.get("state"),
                "pids": run.get("pids"),
                "flags": flags,
                "owner": owner,
                "age_s": run.get("age_s"),
                "candidate": run.get("candidate_name") or run.get("candidate"),
            }
        )
    return alerts


def format_latest(entry: dict[str, Any]) -> str:
    counts = ", ".join(f"{k}={v}" for k, v in sorted(entry["counts"].items()))
    lines = [
        f"Batch {entry['batch']} monitor check: {entry['ts_utc']}",
        f"Runner: pid={entry['runner'].get('pid')} alive={entry['runner'].get('running')} "
        f"state_age={entry.get('state_age_seconds')}s",
        f"Counts: {counts}",
    ]
    if entry.get("alert"):
        lines.append(f"ALERT: {entry['alert']}")
    if entry["non_todo"]:
        lines.append("Non-todo tasks:")
        for task in entry["non_todo"]:
            lines.append(
                f"- {task['task']}: {task['status']} attempts={task['attempts']} "
                f"worker={task['worker']} reviewer={task['reviewer']} packet={task['packet']}"
            )
    if entry.get("board_drift"):
        lines.append("Board drift:")
        for item in entry["board_drift"]:
            label = item.get("task") or item.get("kind")
            line = (
                f"- {label}: {item['kind']} local={item.get('local_status')} "
                f"board={item.get('board_status')} expected={item.get('expected_board_status')} "
                f"assignee={item.get('local_assignee')} board_assignee={item.get('board_assignee')}"
            )
            if item.get("error"):
                line += f" error={item['error']}"
            lines.append(line)
    if entry.get("gate_orphan_alerts"):
        lines.append("Gate orphan alerts:")
        for item in entry["gate_orphan_alerts"]:
            lines.append(
                f"- {item.get('task')}: {item.get('kind')} state={item.get('state')} "
                f"pids={item.get('pids')} flags={item.get('flags')} age={item.get('age_s')}"
            )
    if entry["stderr_tail"]:
        lines.append("Recent stderr:")
        lines.extend(entry["stderr_tail"][-20:])
    if entry["recent_ops"]:
        lines.append("Recent ops:")
        lines.extend(entry["recent_ops"][-20:])
    return "\n".join(lines) + "\n"


def check(args: argparse.Namespace) -> dict[str, Any]:
    state_path = REPO / args.state
    state = read_json(state_path)
    if not isinstance(state, dict):
        raise RuntimeError(f"missing or invalid state file: {state_path}")

    counts, non_todo = summarize_state(state)
    runner_pid_info = read_json(REPO / args.runner_pid_file, {}) or {}
    alive = pid_running(runner_pid_info.get("pid"))
    runner = {
        "pid": runner_pid_info.get("pid"),
        "running": alive,
        "started_utc": runner_pid_info.get("started_utc"),
        "command": runner_pid_info.get("command"),
    }

    updated = parse_utc(state.get("updated_utc"))
    state_age = int((datetime.now(timezone.utc) - updated).total_seconds()) if updated else None

    done = batch_done(counts)
    project_id = resolve_project_id(args) if args.check_board else None
    board_drift = check_board_drift(state, project_id) if args.check_board else []
    gate_orphan_alerts = check_gate_orphans(args)
    # Decide a terminal/alert condition. The runner only writes state on task
    # transitions, and a single codex call can legitimately run up to its
    # timeout, so staleness alone is NOT treated as a crash — only a dead pid
    # with work still outstanding is. Staleness is a softer "stalled" warning
    # gated well above the per-call timeout.
    alert = None
    alert_kind = None  # drives the console state tag; keeps board issues distinct from staleness
    terminal_signal = None  # "complete" | "interrupted"
    if done:
        terminal_signal = "complete"
        alert = "batch complete - all tasks terminal"
    elif not alive and runner_pid_info.get("pid") is not None:
        terminal_signal = "interrupted"
        alert = "runner process is not alive but tasks remain - crashed or interrupted"
    elif alive and state_age is not None and state_age > args.stale_seconds:
        alert = (
            f"runner alive but state unchanged for {state_age}s "
            f"(> {args.stale_seconds}s) - possibly stalled"
        )
        alert_kind = "stalled"
    # Board conditions never override a terminal/staleness alert, and a failed
    # board *read* gets its own tag rather than being collapsed into "stalled" —
    # a wrong or again-broken read endpoint must not masquerade as a stalled
    # runner (that false alarm is exactly what this fix removes). An
    # unconfigured check (no project_id) is informational only, never an alert.
    if alert is None and board_drift:
        real_drift = [d for d in board_drift if d["kind"] not in {"board_read_failed", "board_check_unconfigured"}]
        if real_drift:
            alert = "board/local drift detected"
            alert_kind = "drift"
        elif any(d["kind"] == "board_read_failed" for d in board_drift):
            alert = "board unreachable - could not read board task list"
            alert_kind = "board_unreachable"
    if alert is None and gate_orphan_alerts:
        hard_alerts = [a for a in gate_orphan_alerts if a.get("kind") == "gate_orphan_suspect"]
        if hard_alerts:
            alert = "gate orphan/mismatch suspects detected"
            alert_kind = "gate_orphans"

    stderr_path = Path(runner_pid_info.get("stderr", "")) if runner_pid_info.get("stderr") else None
    stderr_tail = bounded_tail(stderr_path)[-30:] if stderr_path else []
    recent_ops = [
        line
        for line in bounded_tail(REPO / args.operation_log)[-120:]
        if INTERESTING_OP_RE.search(line)
    ][-20:]

    entry = {
        "ts_utc": utc_now(),
        "batch": args.batch,
        "runner": runner,
        "counts": counts,
        "non_todo": non_todo,
        "state_age_seconds": state_age,
        "done": done,
        "alert": alert,
        "alert_kind": alert_kind,
        "terminal_signal": terminal_signal,
        "board_drift": board_drift,
        "gate_orphan_alerts": gate_orphan_alerts,
        "stderr_tail": stderr_tail,
        "recent_ops": recent_ops,
    }
    append_jsonl(REPO / args.log, entry)
    latest_path = REPO / args.latest
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(format_latest(entry), encoding="utf-8")
    return entry


def write_pid_file(args: argparse.Namespace, state: str = "running", reason: str = "") -> None:
    pid_path = REPO / args.pid_file
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    info = {
        "pid": os.getpid(),
        "started_utc": getattr(args, "_started_utc", utc_now()),
        "interval_seconds": args.interval_seconds,
        "log": str((REPO / args.log).resolve()),
        "latest": str((REPO / args.latest).resolve()),
        "state": state,
        "running": state == "running",
    }
    if state != "running":
        info["stopped_utc"] = utc_now()
        info["exit_reason"] = reason
    pid_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor the local HandOff runner state.")
    parser.add_argument("--batch", default="31")
    parser.add_argument("--lane", default="", help="Lane label used to filter ng_gate owner metadata.")
    parser.add_argument("--state", default="Handoff/private/runner_state_batch31.json")
    parser.add_argument("--runner-pid-file", default="Handoff/private/runner_loop.pid.json")
    parser.add_argument("--pid-file", default="Handoff/private/monitor_loop.pid.json")
    parser.add_argument("--log", default="Handoff/logs/monitor_batch31.jsonl")
    parser.add_argument("--latest", default="Handoff/logs/monitor_batch31.latest.txt")
    parser.add_argument("--operation-log", default="Handoff/logs/operations.jsonl")
    parser.add_argument("--interval-seconds", type=int, default=2400)
    parser.add_argument(
        "--stale-seconds",
        type=int,
        default=18000,
        help="Warn if the runner is alive but state has not advanced in this long "
        "(default 5h — must exceed the per-call worker timeout so the runner's own "
        "reap+retry fires first; bumped from 90m when the worker timeout went to 4h).",
    )
    parser.add_argument(
        "--exit-on-terminal",
        action="store_true",
        help="In --loop mode, stop once the batch completes or the runner dies "
        "(default: keep watching).",
    )
    parser.add_argument("--check-board", action="store_true", help="Compare local task state against HandOff board tasks.")
    parser.add_argument("--skip-gate-orphans", action="store_true", help="Do not read the ng_gate queue snapshot for alert-only orphan flags.")
    parser.add_argument(
        "--config",
        help="Runner config JSON (relative to repo). Its project.project_id/request_id "
        "is used to read the board task list for --check-board.",
    )
    parser.add_argument(
        "--project-id",
        help="Explicit HandOff project/request id for --check-board (overrides --config).",
    )
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()
    args._started_utc = utc_now()

    if args.loop:
        write_pid_file(args, "running")

    exit_reason = "stopped"
    try:
        while True:
            try:
                entry = check(args)
                tag = (
                    entry.get("terminal_signal")
                    or entry.get("alert_kind")
                    or ("stalled" if entry.get("alert") else "ok")
                )
                print(
                    f"{entry['ts_utc']} batch={entry['batch']} runner_alive={entry['runner'].get('running')} "
                    f"state={tag} counts={entry['counts']}",
                    flush=True,
                )
                if entry.get("alert"):
                    print(f"  ALERT: {entry['alert']}", file=sys.stderr, flush=True)
                if entry.get("terminal_signal") and (args.exit_on_terminal or not args.loop):
                    exit_reason = entry["terminal_signal"]
                    return 0 if entry["terminal_signal"] == "complete" else 2
            except Exception as exc:
                error_entry = {"ts_utc": utc_now(), "batch": args.batch, "error": str(exc)}
                append_jsonl(REPO / args.log, error_entry)
                (REPO / args.latest).write_text(
                    f"Monitor error at {error_entry['ts_utc']}: {exc}\n", encoding="utf-8"
                )
                print(f"monitor error: {exc}", file=sys.stderr, flush=True)

            if not args.loop:
                return 0
            time.sleep(max(60, args.interval_seconds))
    except KeyboardInterrupt:
        exit_reason = "interrupted"
        return 130
    finally:
        if args.loop:
            write_pid_file(args, "stopped", exit_reason)


if __name__ == "__main__":
    raise SystemExit(main())
