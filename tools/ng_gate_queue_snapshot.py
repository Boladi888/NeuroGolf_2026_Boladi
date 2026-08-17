#!/usr/bin/env python3
"""Read-only ng_gate queue snapshot and visible terminal watch.

Shows both active throttle lock holders and live ng_gate process trees that are
waiting before a lock is acquired. This never deletes lockfiles, changes caps,
kills processes, or starts gate work.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import pathlib
import re
import sys
import time
from collections import defaultdict
from typing import Any

from ng_harness import corpus, throttle

NG_ROOT = pathlib.Path(__file__).resolve().parents[1]
PRIVATE_DIR = NG_ROOT / "Handoff" / "private"
RESCUE_QUEUE = PRIVATE_DIR / "review_rescue_queue.jsonl"

GATE_RE = re.compile(r"tools[\\/]+ng_gate\.py\s+([A-Za-z0-9_-]+)", re.IGNORECASE)
TASK_RE = re.compile(r"--task(?:=|\s+)(\d+)")
N_RE = re.compile(r"--n(?:=|\s+)(\d+)")
CAND_RE = re.compile(r"--cand(?:=|\s+)(\"[^\"]+\"|'[^']+'|[^\s]+)")
FULL_RE = re.compile(r"(?<!\S)--full(?!\S)")
TRUTHY = {"1", "true", "yes", "on", "watch", "auto"}

try:
    import psutil
except Exception:  # noqa: BLE001
    psutil = None


def _utc_now() -> _dt.datetime:
    return _dt.datetime.now(_dt.UTC)


def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return _dt.datetime.fromtimestamp(ts, _dt.UTC).isoformat()


def _age_s(ts: float | None, now: float) -> float | None:
    if ts is None:
        return None
    return round(max(0.0, now - ts), 1)


def _fmt_age(seconds: float | int | None) -> str:
    if seconds is None:
        return "?"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    if hours < 48:
        return f"{hours}h{minutes:02d}m"
    days, hours = divmod(hours, 24)
    return f"{days}d{hours:02d}h"


def _short_path(path: str | None, width: int = 48) -> str:
    if not path:
        return ""
    clean = path.strip("\"'")
    name = pathlib.PureWindowsPath(clean).name if ("\\" in clean or ":" in clean) else pathlib.PurePosixPath(clean).name
    if name and name != clean:
        return name[:width]
    return clean[:width]


def _arg(pattern: re.Pattern[str], command_line: str) -> str:
    m = pattern.search(command_line)
    if not m:
        return ""
    return m.group(1).strip("\"'")


def _parse_gate_command(command_line: str) -> dict[str, Any]:
    mode_match = GATE_RE.search(command_line or "")
    full = bool(FULL_RE.search(command_line or ""))
    n_text = _arg(N_RE, command_line or "")
    n = corpus.MAX_N if full else int(n_text) if n_text.isdigit() else 2000
    mode = mode_match.group(1) if mode_match else ""
    task = _arg(TASK_RE, command_line or "")
    cand = _arg(CAND_RE, command_line or "")
    return {
        "mode": mode,
        "task": task,
        "candidate": cand,
        "n": n,
        "full": full,
        "expected_heavy": throttle.is_heavy(mode, n) if mode else False,
    }


def _pid_alive(pid: int | None) -> bool | None:
    if pid is None:
        return None
    if psutil is not None:
        return psutil.pid_exists(int(pid))
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:  # noqa: BLE001
        return os.name == "nt"


def _read_locks(now: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    d = throttle._DIR
    if not d.exists():
        return out
    for kind in ("heavy", "slot"):
        for f in sorted(d.glob(f"{kind}_*.lock")):
            pid = None
            try:
                pid = int(f.stem.split("_")[1])
            except Exception:  # noqa: BLE001
                pass
            try:
                text = f.read_text(encoding="utf-8", errors="replace").strip()
                parts = text.split()
            except Exception:  # noqa: BLE001
                text, parts = "", []
            label = " ".join(parts[1:-1]) if len(parts) >= 3 else (parts[1] if len(parts) >= 2 else "")
            task_match = re.search(r"\bt(\d{1,3})\b", label)
            owner = {}
            try:
                owner = json.loads(f.with_suffix(f.suffix + ".json").read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                owner = {}
            try:
                mtime = f.stat().st_mtime
            except Exception:  # noqa: BLE001
                mtime = None
            out.append({
                "kind": kind,
                "pid": pid,
                "label": label,
                "task": task_match.group(1) if task_match else "",
                "text": text,
                "file": str(f),
                "file_name": f.name,
                "mtime": _iso(mtime),
                "age_s": _age_s(mtime, now),
                "alive": _pid_alive(pid),
                "owner": owner,
            })
    return out


def _read_intents(now: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    d = throttle._DIR
    if not d.exists():
        return out
    for f in sorted(d.glob("intent_*.json")):
        try:
            meta = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            meta = {}
        try:
            pid = int(meta.get("pid") or f.stem.split("_")[1])
        except Exception:  # noqa: BLE001
            pid = None
        try:
            mtime = f.stat().st_mtime
        except Exception:  # noqa: BLE001
            mtime = None
        out.append(
            {
                "kind": "intent",
                "pid": pid,
                "file": str(f),
                "file_name": f.name,
                "mtime": _iso(mtime),
                "age_s": _age_s(mtime, now),
                "alive": _pid_alive(pid),
                "owner": meta,
            }
        )
    return out


def _list_processes(now: float) -> dict[int, dict[str, Any]]:
    if psutil is None:
        return {}
    procs: dict[int, dict[str, Any]] = {}
    attrs = ["pid", "ppid", "name", "cmdline", "create_time"]
    for proc in psutil.process_iter(attrs):
        try:
            info = proc.info
            cmd = info.get("cmdline") or []
            if isinstance(cmd, (list, tuple)):
                command_line = " ".join(str(c) for c in cmd)
            else:
                command_line = str(cmd or "")
            created = info.get("create_time")
            procs[int(info["pid"])] = {
                "pid": int(info["pid"]),
                "ppid": int(info.get("ppid") or 0),
                "name": info.get("name") or "",
                "command_line": command_line,
                "create_time": _iso(created),
                "create_ts": created,
                "age_s": _age_s(created, now),
            }
        except Exception:  # noqa: BLE001
            continue
    return procs


def _compatible(a: dict[str, Any], b: dict[str, Any]) -> bool:
    pa, pb = a.get("parsed", {}), b.get("parsed", {})
    if pa.get("task") and pb.get("task") and pa["task"] != pb["task"]:
        return False
    if pa.get("mode") and pb.get("mode") and pa["mode"] != pb["mode"]:
        return False
    ca, cb = pa.get("candidate", ""), pb.get("candidate", "")
    if ca and cb and not ca.startswith("$") and not cb.startswith("$"):
        return _short_path(ca, 200).lower() == _short_path(cb, 200).lower()
    return True


def _group_gate_processes(
    processes: dict[int, dict[str, Any]],
    locks: list[dict[str, Any]],
    intents: list[dict[str, Any]],
    now: float,
) -> list[dict[str, Any]]:
    lock_by_pid: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for lock in locks:
        if lock["pid"] is not None:
            lock_by_pid[int(lock["pid"])].append(lock)
    intent_by_pid: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for intent in intents:
        if intent["pid"] is not None:
            intent_by_pid[int(intent["pid"])].append(intent)

    matched: dict[int, dict[str, Any]] = {}
    for pid, proc in processes.items():
        cmd = proc.get("command_line", "")
        if not GATE_RE.search(cmd):
            continue
        if "ng_gate_queue_snapshot.py" in cmd:
            continue
        proc = dict(proc)
        proc["parsed"] = _parse_gate_command(cmd)
        matched[pid] = proc

    def root_for(proc: dict[str, Any]) -> int:
        current = proc
        seen: set[int] = set()
        while True:
            ppid = current.get("ppid")
            if ppid in seen or ppid not in matched:
                return int(current["pid"])
            parent = matched[ppid]
            if not _compatible(parent, current):
                return int(current["pid"])
            seen.add(int(current["pid"]))
            current = parent

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for proc in matched.values():
        grouped[root_for(proc)].append(proc)

    runs: list[dict[str, Any]] = []
    for root_pid, group in grouped.items():
        group = sorted(group, key=lambda p: (p.get("create_ts") or 0, p["pid"]))
        group_locks = []
        group_intents = []
        for proc in group:
            group_locks.extend(lock_by_pid.get(int(proc["pid"]), []))
            group_intents.extend(intent_by_pid.get(int(proc["pid"]), []))

        primary = _choose_primary(group, group_locks)
        parsed = dict(primary.get("parsed", {}))
        for proc in reversed(group):
            pp = proc.get("parsed", {})
            for key in ("task", "mode", "candidate"):
                if not parsed.get(key) and pp.get(key):
                    parsed[key] = pp[key]
        task = parsed.get("task") or _task_from_locks(group_locks) or _task_from_intents(group_intents)
        mode = parsed.get("mode", "")
        n = parsed.get("n") or 2000
        full = bool(parsed.get("full"))
        expected_heavy = bool(parsed.get("expected_heavy"))
        lock_kinds = sorted({lock["kind"] for lock in group_locks})
        lock_pids = sorted({lock["pid"] for lock in group_locks if lock["pid"] is not None})
        owner_meta = (
            next((lock.get("owner") for lock in group_locks if lock.get("owner")), {})
            or next((intent.get("owner") for intent in group_intents if intent.get("owner")), {})
        )
        started_ts = min((p.get("create_ts") or now for p in group), default=now)
        state = _classify(lock_kinds, expected_heavy)
        flags = _process_flags(group, group_locks, expected_heavy)
        runs.append({
            "task": f"{int(task):03d}" if str(task).isdigit() else str(task or ""),
            "mode": mode,
            "candidate": parsed.get("candidate", ""),
            "candidate_name": _short_path(parsed.get("candidate", "")),
            "n": n,
            "full": full,
            "expected_heavy": expected_heavy,
            "state": state,
            "root_pid": root_pid,
            "lock_pids": lock_pids,
            "pids": [int(p["pid"]) for p in group],
            "process_count": len(group),
            "started_at": _iso(started_ts),
            "age_s": _age_s(started_ts, now),
            "lock_kinds": lock_kinds,
            "locks": group_locks,
            "intents": group_intents,
            "owner_meta": owner_meta,
            "orphan_flags": flags,
            "command_line": primary.get("command_line", ""),
        })
    return sorted(runs, key=lambda r: (r["state"], r.get("task", ""), r.get("age_s") or 0))


def _choose_primary(group: list[dict[str, Any]], locks: list[dict[str, Any]]) -> dict[str, Any]:
    lock_pids = {lock["pid"] for lock in locks}
    for proc in reversed(group):
        if proc["pid"] in lock_pids:
            return proc
    for proc in reversed(group):
        cand = proc.get("parsed", {}).get("candidate", "")
        if cand and not cand.startswith("$"):
            return proc
    return group[-1]


def _task_from_locks(locks: list[dict[str, Any]]) -> str:
    for lock in locks:
        owner = lock.get("owner") or {}
        if owner.get("task_num"):
            return str(owner["task_num"])
        if owner.get("task"):
            m = re.search(r"(\d{1,3})", str(owner["task"]))
            if m:
                return m.group(1)
        if lock.get("task"):
            return str(lock["task"])
    return ""


def _task_from_intents(intents: list[dict[str, Any]]) -> str:
    for intent in intents:
        owner = intent.get("owner") or {}
        if owner.get("task_num"):
            return str(owner["task_num"])
        if owner.get("task"):
            m = re.search(r"(\d{1,3})", str(owner["task"]))
            if m:
                return m.group(1)
    return ""


def _classify(lock_kinds: list[str], expected_heavy: bool) -> str:
    if "heavy" in lock_kinds:
        return "active_heavy"
    if "slot" in lock_kinds:
        return "active_light"
    if expected_heavy:
        return "waiting_heavy"
    return "waiting_normal"


def _process_flags(group: list[dict[str, Any]], locks: list[dict[str, Any]], expected_heavy: bool) -> list[str]:
    flags: list[str] = []
    if expected_heavy and any(lock["kind"] == "slot" for lock in locks) and not any(lock["kind"] == "heavy" for lock in locks):
        flags.append("heavy_expected_without_heavy_lock")
    root = min(group, key=lambda p: p.get("create_ts") or 0)
    ppid = root.get("ppid")
    if ppid and psutil is not None and not psutil.pid_exists(int(ppid)):
        flags.append("root_parent_missing")
    return flags


def _load_runner_contexts(now: float, state_days: float | None = 7.0) -> dict[str, list[dict[str, Any]]]:
    contexts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    cutoff = None if state_days is None or state_days <= 0 else now - (state_days * 86400.0)
    for path in sorted(PRIVATE_DIR.glob("runner_state*.json")):
        if ".bak" in path.name:
            continue
        try:
            mtime = path.stat().st_mtime
        except Exception:  # noqa: BLE001
            continue
        if cutoff is not None and mtime < cutoff:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except Exception:  # noqa: BLE001
            continue
        lane = _lane_from_name(path.name)
        batch = _batch_from_name(path.name)
        for item in _walk_task_dicts(data):
            task = str(item.get("task", ""))
            if not task:
                continue
            contexts[task].append({
                "state_file": str(path),
                "state_file_name": path.name,
                "state_mtime": _iso(mtime),
                "lane": lane,
                "batch": batch,
                "task": task,
                "status": item.get("status", ""),
                "title": item.get("title", ""),
                "worker": item.get("worker_agent_id") or item.get("assignee") or "",
                "reviewer": item.get("reviewer_agent_id") or item.get("reviewer") or "",
                "worker_attempts": item.get("worker_attempts"),
                "review_failures": item.get("review_failures"),
                "worker_last_output": item.get("worker_last_output"),
                "reviewer_last_output": item.get("reviewer_last_output"),
            })
    for rows in contexts.values():
        rows.sort(key=lambda r: r.get("state_mtime") or "", reverse=True)
    return contexts


def _walk_task_dicts(obj: Any):
    if isinstance(obj, dict):
        if isinstance(obj.get("task"), str) and obj["task"].startswith("task"):
            yield obj
        for value in obj.values():
            yield from _walk_task_dicts(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _walk_task_dicts(value)


def _lane_from_name(name: str) -> str:
    lower = name.lower()
    if "codex2" in lower:
        return "Codex2"
    if "codex" in lower:
        return "Codex"
    if "claude" in lower:
        return "Claude"
    return ""


def _batch_from_name(name: str) -> str:
    m = re.search(r"_p(\d+)", name)
    return m.group(1) if m else ""


def _attach_runner_context(runs: list[dict[str, Any]], contexts: dict[str, list[dict[str, Any]]]) -> None:
    terminal_statuses = {
        "drop",
        "ship_packeted",
        "captain_staged",
        "worker_failed",
        "review_failed",
        "review_parse_failed",
        "needs_packet_fix",
        "complete",
        "stopped",
    }
    for run in runs:
        task = run.get("task", "")
        key = f"task{task}" if task and not task.startswith("task") else task
        ctxs = contexts.get(key, [])
        run["runner_contexts"] = ctxs[:3]
        run["owner"] = _owner_label(ctxs, run.get("owner_meta") or {})
        if ctxs and all(str(c.get("status", "")).lower() in terminal_statuses for c in ctxs):
            run.setdefault("orphan_flags", []).append("runner_terminal_but_process_alive")


def _owner_label(ctxs: list[dict[str, Any]], owner_meta: dict[str, Any] | None = None) -> str:
    owner_meta = owner_meta or {}
    if owner_meta:
        lane = owner_meta.get("lane_id") or owner_meta.get("lane") or "lane?"
        batch = f"p{owner_meta.get('batch')}" if owner_meta.get("batch") else ""
        role = owner_meta.get("role") or "role?"
        agent = owner_meta.get("agent") or ""
        return f"{lane} {batch} {role} {agent}".strip()
    if not ctxs:
        return "manual/unknown"
    c = ctxs[0]
    status = str(c.get("status") or "")
    if "review" in status.lower():
        actor = c.get("reviewer") or c.get("worker") or ""
    else:
        actor = c.get("worker") or c.get("reviewer") or ""
    lane = c.get("lane") or "lane?"
    status = status or "status?"
    return f"{lane} {status} {actor}".strip()


def _load_rescue_queue(now: float, limit: int = 6) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if RESCUE_QUEUE.exists():
        for line_no, line in enumerate(RESCUE_QUEUE.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            row["_line"] = line_no
            ts = _parse_iso(row.get("created_utc"))
            row["_created_ts"] = ts
            row["age_s"] = _age_s(ts, now)
            rows.append(row)
    # The rescue queue is append-only. A later resolved row for the same
    # canonical task/candidate supersedes the earlier open row, matching
    # captain_ctl.py. Canonicalizing avoids false-open rows when one record used
    # a relative path and a later resolution used an absolute path.
    latest_by_candidate: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("task", "")).lower(), _rescue_candidate_key(row.get("candidate_path")))
        latest_by_candidate[key] = row
    effective_rows = list(latest_by_candidate.values())
    open_rows = [r for r in effective_rows if str(r.get("status", "open")).lower() == "open"]
    open_rows.sort(key=lambda r: r.get("_created_ts") or 0)
    latest_by_task: dict[str, dict[str, Any]] = {}
    for row in open_rows:
        task = str(row.get("task", ""))
        if task:
            latest_by_task[task] = row
    latest = sorted(latest_by_task.values(), key=lambda r: r.get("_created_ts") or 0)
    return {
        "path": str(RESCUE_QUEUE),
        "total_rows": len(rows),
        "open_count": len(open_rows),
        "open_task_count": len(latest_by_task),
        "oldest_open": [_trim_rescue(r) for r in latest[:limit]],
        "recent_open": [_trim_rescue(r) for r in latest[-limit:]],
    }


def _rescue_candidate_key(path_text: Any) -> str:
    text = str(path_text or "")
    if not text:
        return ""
    path = pathlib.Path(text)
    if not path.is_absolute():
        path = NG_ROOT / path
    try:
        return os.path.normcase(str(path.resolve()))
    except Exception:  # noqa: BLE001
        return os.path.normcase(str(path))


def _parse_iso(text: str | None) -> float | None:
    if not text:
        return None
    try:
        return _dt.datetime.fromisoformat(str(text).replace("Z", "+00:00")).timestamp()
    except Exception:  # noqa: BLE001
        return None


def _trim_rescue(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": row.get("task", ""),
        "lane": row.get("lane", ""),
        "age_s": row.get("age_s"),
        "created_utc": row.get("created_utc"),
        "candidate_cost": row.get("candidate_cost"),
        "live_cost": row.get("live_cost"),
        "checks_missing": row.get("checks_missing", []),
        "reason": row.get("reason", ""),
    }


def gather(max_rescue: int = 6, state_days: float | None = 7.0) -> dict[str, Any]:
    now_dt = _utc_now()
    now = now_dt.timestamp()
    locks = _read_locks(now)
    intents = _read_intents(now)
    processes = _list_processes(now)
    runs = _group_gate_processes(processes, locks, intents, now)
    _attach_runner_context(runs, _load_runner_contexts(now, state_days))
    grouped_lock_pids = {pid for run in runs for pid in run.get("lock_pids", [])}
    grouped_intent_pids = {
        int(intent["pid"])
        for run in runs
        for intent in run.get("intents", [])
        if intent.get("pid") is not None
    }
    unmatched_locks = [lock for lock in locks if lock.get("pid") not in grouped_lock_pids]
    unmatched_intents = [intent for intent in intents if intent.get("pid") not in grouped_intent_pids]
    suspects = []
    for run in runs:
        if run.get("orphan_flags"):
            suspects.append({
                "kind": "process",
                "task": run.get("task"),
                "state": run.get("state"),
                "pids": run.get("pids"),
                "flags": run.get("orphan_flags"),
            })
    for lock in unmatched_locks:
        if lock.get("alive"):
            suspects.append({
                "kind": "lock_without_visible_gate_command",
                "pid": lock.get("pid"),
                "label": lock.get("label"),
                "lock": lock.get("file_name"),
                "age_s": lock.get("age_s"),
            })
    for intent in unmatched_intents:
        if intent.get("alive"):
            suspects.append({
                "kind": "intent_without_visible_gate_command",
                "pid": intent.get("pid"),
                "owner": intent.get("owner"),
                "file": intent.get("file_name"),
                "age_s": intent.get("age_s"),
            })
    return {
        "generated_at": now_dt.isoformat(),
        "throttle": {
            "dir": str(throttle._DIR),
            "normal_cap": throttle._k(),
            "normal_live": len([l for l in locks if l["kind"] == "slot" and l["alive"]]),
            "heavy_cap": throttle._heavy_k(),
            "heavy_live": len([l for l in locks if l["kind"] == "heavy" and l["alive"]]),
            "heavy_modes": sorted(throttle.HEAVY_MODES),
            "heavy_n_threshold": throttle._heavy_n(),
            "psutil": psutil is not None,
        },
        "runs": runs,
        "locks": locks,
        "intents": intents,
        "unmatched_locks": unmatched_locks,
        "unmatched_intents": unmatched_intents,
        "rescue_queue": _load_rescue_queue(now, max_rescue),
        "orphan_suspects": suspects,
    }


def print_human(snapshot: dict[str, Any]) -> None:
    throttle_info = snapshot["throttle"]
    generated = snapshot["generated_at"].replace("+00:00", "Z")
    normal_free = max(0, throttle_info["normal_cap"] - throttle_info["normal_live"])
    heavy_free = max(0, throttle_info["heavy_cap"] - throttle_info["heavy_live"])
    print(f"NG Gate Queue Watch  {generated}")
    print(f"normal {throttle_info['normal_live']}/{throttle_info['normal_cap']} used ({normal_free} free)   "
          f"heavy {throttle_info['heavy_live']}/{throttle_info['heavy_cap']} used ({heavy_free} free)")
    print(f"heavy modes={','.join(throttle_info['heavy_modes'])} n>={throttle_info['heavy_n_threshold']}  "
          f"dir={throttle_info['dir']}")
    print()
    _print_section("HEAVY SLOT HOLDERS", [r for r in snapshot["runs"] if r["state"] == "active_heavy"])
    _print_section("WAITING FULL GATES", [r for r in snapshot["runs"] if r["state"] == "waiting_heavy"])
    _print_section("LIGHT / NORMAL GATES", [r for r in snapshot["runs"] if r["state"] in {"active_light", "waiting_normal"}])
    _print_rescue(snapshot["rescue_queue"])
    _print_suspects(snapshot["orphan_suspects"])
    print()
    print("Read-only watcher. Close this window or press Ctrl+C to stop the watcher only.")


def _print_section(title: str, rows: list[dict[str, Any]]) -> None:
    print(title)
    if not rows:
        print("  (none)")
        print()
        return
    print("  task  state          mode    age      pids                 owner                                   candidate")
    for r in rows:
        pids = ">".join(str(p) for p in r.get("pids", []))[:20]
        print(f"  {r.get('task',''):<4}  {r.get('state',''):<13}  {r.get('mode',''):<6}  "
              f"{_fmt_age(r.get('age_s')):<7}  {pids:<20}  "
              f"{(r.get('owner') or '')[:38]:<38}  {r.get('candidate_name','')[:48]}")
    print()


def _print_rescue(rescue: dict[str, Any]) -> None:
    print("RESCUE QUEUE")
    print(f"  open rows={rescue['open_count']}  open tasks={rescue['open_task_count']}  path={rescue['path']}")
    rows = rescue.get("recent_open") or []
    if not rows:
        print("  (none)")
        print()
        return
    print("  recent open tasks:")
    for row in rows[-6:]:
        missing = ",".join(str(x) for x in row.get("checks_missing", []))
        print(f"  {row.get('task',''):<7} age={_fmt_age(row.get('age_s')):<7} "
              f"missing={missing[:34]:<34} reason={(row.get('reason') or '')[:60]}")
    print()


def _print_suspects(suspects: list[dict[str, Any]]) -> None:
    print("ORPHAN / MISMATCH SUSPECTS")
    if not suspects:
        print("  (none)")
        print()
        return
    for suspect in suspects:
        print("  " + json.dumps(suspect, sort_keys=True, default=str)[:180])
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description="Read-only ng_gate queue snapshot/watch.")
    ap.add_argument("--json", action="store_true", help="print one JSON snapshot")
    ap.add_argument("--watch", action="store_true", help="redraw a visible terminal watch until stopped")
    ap.add_argument("--interval", type=float, default=3.0, help="watch refresh seconds")
    ap.add_argument("--max-rescue", type=int, default=6, help="rescue rows to include in old/recent summaries")
    ap.add_argument("--state-days", type=float, default=7.0, help="recent runner-state files to join; <=0 means all")
    ap.add_argument("--no-clear", action="store_true", help="do not clear the terminal between watch frames")
    args = ap.parse_args()

    if args.watch:
        try:
            while True:
                if not args.no_clear:
                    os.system("cls" if os.name == "nt" else "clear")
                print_human(gather(args.max_rescue, args.state_days))
                time.sleep(max(1.0, args.interval))
        except KeyboardInterrupt:
            print("\nNG Gate Queue Watch stopped.")
            return 0

    snapshot = gather(args.max_rescue, args.state_days)
    if args.json:
        print(json.dumps(snapshot, indent=2, default=str))
    else:
        print_human(snapshot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
