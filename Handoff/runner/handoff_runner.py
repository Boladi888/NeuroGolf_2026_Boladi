#!/usr/bin/env python3
"""Local HandOff runner for the NeuroGolf workflow."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import threading
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from packet_extract import extract_review_json, validate_packet


REPO = Path.cwd()
UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)

# Serializes appends to shared JSONL logs (operations.jsonl) so concurrent
# worker/reviewer threads cannot interleave half-written lines.
_JSONL_LOCK = threading.Lock()


class SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO.resolve()).as_posix()
    except ValueError:
        return str(path)


def redact(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        s = re.sub(r"hot_[A-Za-z0-9_\-]+", "hot_<redacted>", value)
        s = re.sub(r"ak_[A-Za-z0-9_\-]+", "ak_<redacted>", s)
        s = re.sub(r"sk-ant-[A-Za-z0-9_\-]+", "sk-ant-<redacted>", s)
        s = re.sub(r"sk_[A-Za-z0-9_\-]+", "sk_<redacted>", s)
        s = re.sub(r"(?i)bearer\s+[A-Za-z0-9._\-]+", "Bearer <redacted>", s)
        s = re.sub(
            r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}",
            "<email_redacted>",
            s,
        )
        s = re.sub(r"0x[a-fA-F0-9]{32,}", "0x<redacted>", s)
        return s
    if isinstance(value, list):
        return [redact(x) for x in value]
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if re.search(r"(token|key|secret|password|email|wallet|address)", str(k), re.I):
                out[k] = "<redacted>"
            else:
                out[k] = redact(v)
        return out
    return value


def append_jsonl(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(redact(entry), ensure_ascii=True, separators=(",", ":"))
    with _JSONL_LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.write("\n")


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


def find_session_id_from_obj(obj: Any) -> str | None:
    if isinstance(obj, dict):
        for key in ("session_id", "conversation_id", "thread_id"):
            val = obj.get(key)
            if isinstance(val, str) and UUID_RE.search(val):
                return UUID_RE.search(val).group(0)
        for val in obj.values():
            found = find_session_id_from_obj(val)
            if found:
                return found
    elif isinstance(obj, list):
        for val in obj:
            found = find_session_id_from_obj(val)
            if found:
                return found
    return None


def find_session_id(text: str) -> str | None:
    for line in text.splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        found = find_session_id_from_obj(obj)
        if found:
            return found
    match = UUID_RE.search(text)
    return match.group(0) if match else None


class Runner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.state_lock = threading.RLock()
        self.config_path = Path(args.config)
        self.config = read_json(self.config_path, None)
        if not self.config:
            raise SystemExit(f"missing config: {self.config_path}")
        self.paths = self.config["paths"]
        self.state_path = REPO / self.paths["private_state"]
        self.raw_dir = REPO / self.paths["raw_log_dir"]
        self.operation_log = REPO / self.paths["operation_log"]
        self.prompt_dir = REPO / self.paths["prompt_dir"]
        self.template_dir = REPO / self.paths["runner_prompt_dir"]
        self.captain_inbox = REPO / self.paths["captain_lane_inbox"]
        workflow = self.config.get("workflow", {})
        self.lane = str(workflow.get("lane") or workflow.get("source_lane") or "")
        self.lane_id = str(workflow.get("lane_id") or self.lane or "lane").lower()
        self.batch_id = str(workflow.get("batch_id") or "")
        self.runner_id = str(workflow.get("runner_id") or f"neurogolf:{self.lane_id}:batch{self.batch_id}")
        try:
            self.heavy_gate_n = int(workflow.get("heavy_gate_n") or os.environ.get("NEUROGOLF_HEAVY_GATE_N") or 8000)
        except (TypeError, ValueError):
            self.heavy_gate_n = 8000
        batch_suffix = f"_p{self.batch_id}" if self.batch_id else ""
        self.process_registry_path = REPO / "Handoff" / "private" / f"process_registry_{self.lane_id}{batch_suffix}.jsonl"
        self.state = self.load_state(force=args.init_state)

    def ensure_dirs(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.captain_inbox.mkdir(parents=True, exist_ok=True)

    def load_state(self, force: bool = False) -> dict[str, Any]:
        self.ensure_dirs()
        if force and self.state_path.exists():
            backup = self.state_path.with_name(f"runner_state.{utc_stamp()}.bak.json")
            self.state_path.replace(backup)
        default_state = {
            "schema": "neurogolf.handoff.runner_state.v1",
            "created_utc": utc_now(),
            "updated_utc": utc_now(),
            "tasks": {},
            "captain": {},
        }
        try:
            state = read_json(self.state_path, default_state)
        except (json.JSONDecodeError, ValueError) as exc:
            # A truncated/corrupt state file used to raise an uncaught decode
            # error and kill startup. Back it up and start from the default so
            # the runner can recover on the next poll instead of wedging.
            if self.state_path.exists():
                bad = self.state_path.with_name(f"runner_state.corrupt.{utc_stamp()}.json")
                self.state_path.replace(bad)
                self.op(
                    "runner_state_corrupt",
                    f"Recovered from corrupt state; backed up to {rel(bad)}",
                    {"backup": rel(bad), "error": str(exc)},
                )
            state = default_state
        for t in self.config["tasks"]:
            task = t["task"]
            task_state = state["tasks"].setdefault(
                task,
                {
                    "task": task,
                    "title": t["title"],
                    "handoff_task_id": t["handoff_task_id"],
                    "external_key": t.get("external_key"),
                    "prompt_file": t["prompt_file"],
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
                },
            )
            # Config-authoritative identity: honor a repointed board id / prompt
            # from config over a stale state value (e.g. lane moved projects).
            task_state["handoff_task_id"] = t["handoff_task_id"]
            task_state["external_key"] = t.get("external_key")
            task_state["prompt_file"] = t["prompt_file"]
            task_state.setdefault("worker_agent_id", None)
            task_state.setdefault("reviewer_agent_id", None)
        self.save_state(state)
        return state

    def save_state(self, state: dict[str, Any] | None = None) -> None:
        if state is None:
            state = self.state
        with self.state_lock:
            state["updated_utc"] = utc_now()
            write_json(self.state_path, state)

    def op(self, kind: str, summary: str, data: Any = None) -> None:
        append_jsonl(
            self.operation_log,
            {
                "ts_utc": utc_now(),
                "kind": kind,
                "summary": summary,
                "data": data,
            },
        )

    def process_event(self, kind: str, data: Any = None) -> None:
        data = data or {}
        task = data.get("task") if isinstance(data, dict) else None
        role = data.get("role") if isinstance(data, dict) else None
        run_id = data.get("run_id") if isinstance(data, dict) else None
        append_jsonl(
            self.process_registry_path,
            {
                "ts_utc": utc_now(),
                "kind": kind,
                "lane": self.lane,
                "lane_id": self.lane_id,
                "batch": self.batch_id,
                "runner_id": self.runner_id,
                "task": task,
                "role": role,
                "run_id": run_id,
                "data": data,
            },
        )

    @staticmethod
    def _pid_running(pid: int | None) -> bool:
        if not pid:
            return False
        try:
            if os.name == "nt":
                proc = subprocess.run(
                    ["tasklist", "/FI", f"PID eq {int(pid)}", "/FO", "CSV", "/NH"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                return str(int(pid)) in (proc.stdout or "")
            os.kill(int(pid), 0)
            return True
        except Exception:
            return False

    @staticmethod
    def _process_table() -> list[dict[str, Any]]:
        try:
            import psutil  # type: ignore

            rows: list[dict[str, Any]] = []
            for proc in psutil.process_iter(["pid", "ppid", "name", "cmdline", "create_time"]):
                try:
                    info = proc.info
                    cmd = info.get("cmdline") or []
                    cmdline = " ".join(str(part) for part in cmd) if isinstance(cmd, (list, tuple)) else str(cmd or "")
                    rows.append(
                        {
                            "pid": int(info["pid"]),
                            "ppid": int(info.get("ppid") or 0),
                            "name": info.get("name") or "",
                            "command_line": cmdline,
                            "create_time": info.get("create_time"),
                        }
                    )
                except Exception:
                    continue
            return rows
        except Exception:
            pass

        if os.name != "nt":
            return []
        try:
            proc = subprocess.run(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-Command",
                    "Get-CimInstance Win32_Process | "
                    "Select-Object ProcessId,ParentProcessId,Name,CommandLine | ConvertTo-Json -Compress",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
            )
            if proc.returncode != 0 or not proc.stdout.strip():
                return []
            data = json.loads(proc.stdout)
            if isinstance(data, dict):
                data = [data]
            rows = []
            for item in data if isinstance(data, list) else []:
                try:
                    rows.append(
                        {
                            "pid": int(item.get("ProcessId")),
                            "ppid": int(item.get("ParentProcessId") or 0),
                            "name": item.get("Name") or "",
                            "command_line": item.get("CommandLine") or "",
                            "create_time": None,
                        }
                    )
                except Exception:
                    continue
            return rows
        except Exception:
            return []

    def _throttle_dir(self) -> Path:
        return Path(os.environ.get("NEUROGOLF_THROTTLE_DIR", str(Path(tempfile.gettempdir()) / "ng_throttle")))

    def _gate_owner_by_pid(self) -> dict[int, list[dict[str, Any]]]:
        out: dict[int, list[dict[str, Any]]] = {}
        d = self._throttle_dir()
        if not d.exists():
            return out
        for path in list(d.glob("intent_*.json")) + list(d.glob("*.lock.json")):
            try:
                meta = json.loads(path.read_text(encoding="utf-8", errors="replace"))
                pid = int(meta.get("pid") or path.stem.split("_")[1])
            except Exception:
                continue
            meta["_meta_file"] = str(path)
            out.setdefault(pid, []).append(meta)
        return out

    @staticmethod
    def _owner_task_num(owner: dict[str, Any]) -> str:
        task = owner.get("task_num") or owner.get("task") or ""
        match = re.search(r"(\d{1,3})", str(task))
        return (match.group(1).lstrip("0") if match else "") or "0"

    def _owner_matches_task(self, owner: dict[str, Any], task: str) -> bool:
        if self._owner_task_num(owner) != self.task_num(task):
            return False
        if owner.get("runner_id") and str(owner.get("runner_id")) == self.runner_id:
            return True
        owner_batch = str(owner.get("batch") or "")
        owner_lane = str(owner.get("lane_id") or owner.get("lane") or "").lower()
        lane_tokens = {self.lane_id.lower(), str(self.lane or "").lower()}
        return bool(owner_batch == self.batch_id and owner_lane in lane_tokens)

    def _task_arg_matches(self, command_line: str, task: str) -> bool:
        task_num = self.task_num(task)
        return bool(re.search(rf"--task(?:=|\s+)0*{re.escape(task_num)}\b", command_line))

    def _task_gate_processes(self, task: str, include_legacy: bool = False) -> list[dict[str, Any]]:
        owners = self._gate_owner_by_pid()
        matches: list[dict[str, Any]] = []
        seen: set[int] = set()
        for proc in self._process_table():
            pid = int(proc.get("pid") or 0)
            if not pid or pid == os.getpid() or pid in seen:
                continue
            command_line = str(proc.get("command_line") or "")
            normalized = command_line.replace("\\", "/")
            if "ng_gate.py" not in normalized:
                continue
            owner_rows = owners.get(pid, [])
            owner_match = any(self._owner_matches_task(owner, task) for owner in owner_rows)
            command_has_task = bool(re.search(r"--task(?:=|\s+)\d+", command_line))
            if owner_match and command_has_task and not self._task_arg_matches(command_line, task):
                continue
            legacy_match = include_legacy and not owner_rows and self._task_arg_matches(command_line, task)
            if owner_match or legacy_match:
                row = dict(proc)
                row["owner_meta"] = owner_rows
                row["match_kind"] = "owner" if owner_match else "legacy_task_arg"
                matches.append(row)
                seen.add(pid)
        return matches

    def _kill_pid_tree(self, pid: int, reason: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        context = context or {}
        result: dict[str, Any] = {
            "pid": pid,
            "reason": reason,
            "context": context,
            "method": "taskkill_tree" if os.name == "nt" else "kill",
        }
        try:
            if os.name == "nt":
                proc = subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=30,
                )
                result.update({"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr})
            else:
                os.kill(pid, 9)
                result.update({"returncode": 0, "stdout": "", "stderr": ""})
        except Exception as exc:
            result.update({"returncode": None, "error": str(exc)})
        time.sleep(0.5)
        result["alive_after"] = self._pid_running(pid)
        self.process_event("terminate_attempt", {**context, "pid": pid, "reason": reason, "result": result})
        self.op("runner_terminate_attempt", f"Terminate attempt for pid {pid}", {"context": context, "result": result})
        return result

    def cleanup_task_gates(self, task: str, reason: str, include_legacy: bool = False) -> dict[str, Any]:
        matches = self._task_gate_processes(task, include_legacy=include_legacy)
        sweep = {
            "task": task,
            "reason": reason,
            "include_legacy": include_legacy,
            "matches": [
                {
                    "pid": match.get("pid"),
                    "ppid": match.get("ppid"),
                    "name": match.get("name"),
                    "command_line": match.get("command_line"),
                    "match_kind": match.get("match_kind"),
                    "owner_meta": match.get("owner_meta"),
                }
                for match in matches
            ],
        }
        self.process_event("orphan_sweep", sweep)
        if not matches:
            self.process_event(
                "cleanup_verified",
                {"task": task, "reason": reason, "killed_count": 0, "remaining_owner_matches": []},
            )
            return {}
        killed = []
        for match in matches:
            killed.append(
                {
                    "process": {
                        "pid": match.get("pid"),
                        "ppid": match.get("ppid"),
                        "name": match.get("name"),
                        "command_line": match.get("command_line"),
                        "match_kind": match.get("match_kind"),
                        "owner_meta": match.get("owner_meta"),
                    },
                    "kill": self._kill_pid_tree(
                        int(match["pid"]),
                        reason,
                        {"task": task, "role": "ng_gate", "target": "orphan_gate", "match_kind": match.get("match_kind")},
                    ),
                }
            )
        remaining = self._task_gate_processes(task, include_legacy=False)
        report = {"task": task, "reason": reason, "killed": killed, "remaining_owner_matches": remaining}
        self.op("runner_task_gate_cleanup", f"Cleaned up ng_gate process(es) for {task}", report)
        self.process_event("cleanup_verified", report)
        return report

    def cleanup_terminal_task_gates(self, reason: str) -> list[dict[str, Any]]:
        terminal = {
            "ship_packeted",
            "captain_staged",
            "drop",
            "worker_failed",
            "review_failed",
            "review_parse_failed",
            "needs_packet_fix",
        }
        reports = []
        for task_state in self.ordered_task_states():
            if task_state.get("status") in terminal:
                report = self.cleanup_task_gates(task_state["task"], reason)
                if report:
                    reports.append(report)
        return reports

    def task_num(self, task: str) -> str:
        return re.sub(r"\D", "", task).lstrip("0") or "0"

    def task_prompt_path(self, task_state: dict[str, Any]) -> Path:
        return self.prompt_dir / task_state["prompt_file"]

    def template(self, name: str, values: dict[str, Any]) -> str:
        text = read_text(self.template_dir / name)
        safe_values = {k: str(v) for k, v in values.items()}

        def replace(match: re.Match[str]) -> str:
            key = match.group(1)
            return safe_values.get(key, match.group(0))

        return re.sub(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", replace, text)

    def worker_agents(self) -> list[str]:
        agents = self.config["agents"].get("workers")
        if not agents:
            agents = [self.config["agents"].get("worker")]
        return [a for a in agents if a]

    def reviewer_agents(self) -> list[str]:
        agents = self.config["agents"].get("reviewers")
        if not agents:
            agents = [self.config["agents"].get("reviewer")]
        return [a for a in agents if a]

    def task_update_headers(self, agent_id: str | None = None) -> dict[str, str]:
        env = load_env_file(REPO / ".env.handoff")
        headers = {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0 NeuroGolf-HandoffRunner/1.0"}
        token = env.get("HANDOFF_BOOTSTRAP_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"

        agent_id = agent_id or self.config["agents"].get("worker")
        if agent_id:
            agent_env = load_env_file(REPO / "Handoff" / "private" / f"{agent_id}.env")
            agent_id = agent_env.get("HANDOFF_AGENT_ID") or agent_env.get("AGENT_ID") or agent_id
            agent_key = agent_env.get("HANDOFF_AGENT_KEY") or agent_env.get("AGENT_KEY")
            if agent_key:
                headers["X-Agent-Key"] = agent_key
                headers["X-Agent-Id"] = agent_id
        return headers

    def project_id(self) -> str | None:
        project = self.config.get("project") or {}
        return project.get("project_id") or project.get("request_id")

    def find_task_id(self, obj: Any) -> str | None:
        if isinstance(obj, dict):
            for key in ("task_id", "id"):
                val = obj.get(key)
                if isinstance(val, str) and UUID_RE.fullmatch(val):
                    if key == "task_id" or "status" in obj or "title" in obj or obj.get("type") == "task":
                        return val
            for val in obj.values():
                found = self.find_task_id(val)
                if found:
                    return found
        elif isinstance(obj, list):
            for val in obj:
                found = self.find_task_id(val)
                if found:
                    return found
        return None

    def error_moved_to(self, response_text: str) -> str | None:
        try:
            obj = json.loads(response_text)
        except json.JSONDecodeError:
            return None
        moved_to = obj.get("moved_to") if isinstance(obj, dict) else None
        return moved_to if isinstance(moved_to, str) and UUID_RE.fullmatch(moved_to) else None

    def resolve_handoff_task_id(self, task_state: dict[str, Any], headers: dict[str, str]) -> str | None:
        project_id = self.project_id()
        external_key = task_state.get("external_key")
        if not project_id or not external_key:
            return None
        url = (
            "https://handoff.socnet.lol/api/v1/requests/"
            f"{urllib.parse.quote(project_id, safe='')}/tasks/resolve"
            f"?external_key={urllib.parse.quote(str(external_key), safe='')}"
        )
        req = urllib.request.Request(url, method="GET", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                response_text = resp.read().decode("utf-8", errors="replace")
            data = json.loads(response_text)
        except Exception as exc:
            self.op(
                "handoff_resolve_failed",
                f"Failed to resolve {task_state['task']} by external_key",
                {"task": task_state["task"], "external_key": external_key, "error": str(exc)},
            )
            return None
        resolved = self.find_task_id(data)
        if resolved:
            with self.state_lock:
                task_state["handoff_task_id"] = resolved
                self.save_state()
            self.op(
                "handoff_task_resolved",
                f"Resolved {task_state['task']} by external_key",
                {"task": task_state["task"], "external_key": external_key, "handoff_task_id": resolved},
            )
        return resolved

    def send_handoff_put(
        self,
        task_id: str,
        headers: dict[str, str],
        body: bytes,
        task_state: dict[str, Any],
        status: str,
    ) -> tuple[bool, int | None, str]:
        req = urllib.request.Request(
            f"https://handoff.socnet.lol/api/v1/tasks/{task_id}",
            data=body,
            method="PUT",
            headers=headers,
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                response_text = resp.read().decode("utf-8", errors="replace")
            self.op(
                "handoff_sync_complete",
                f"Synced {task_state['task']} to HandOff",
                {"task": task_state["task"], "status": status, "response": response_text},
            )
            return True, None, response_text
        except urllib.error.HTTPError as exc:
            response_text = exc.read().decode("utf-8", errors="replace")
            self.op(
                "handoff_sync_failed",
                f"Failed to sync {task_state['task']} to HandOff",
                {
                    "task": task_state["task"],
                    "status": status,
                    "handoff_task_id": task_id,
                    "http_status": exc.code,
                    "error": str(exc),
                    "response": response_text,
                },
            )
            return False, exc.code, response_text
        except urllib.error.URLError as exc:
            self.op(
                "handoff_sync_failed",
                f"Failed to sync {task_state['task']} to HandOff",
                {"task": task_state["task"], "status": status, "handoff_task_id": task_id, "error": str(exc)},
            )
            return False, None, str(exc)

    def update_handoff_task(self, task_state: dict[str, Any], status: str, result: str = "", agent_id: str | None = None, assignee: str | None = None) -> None:
        if not self.args.sync_handoff:
            return
        update_agent = agent_id or task_state.get("worker_agent_id") or self.config["agents"].get("worker")
        headers = self.task_update_headers(update_agent)
        if "Authorization" not in headers and "X-Agent-Key" not in headers:
            print("sync skipped: no HandOff bearer token or agent key found", file=sys.stderr)
            return
        task_id = task_state.get("handoff_task_id")
        if not task_id:
            task_id = self.resolve_handoff_task_id(task_state, headers)
        if not task_id:
            self.op(
                "handoff_sync_skipped",
                f"No HandOff task id for {task_state['task']}",
                {"task": task_state["task"], "external_key": task_state.get("external_key")},
            )
            return
        body_obj = {"status": status, "result": result}
        body_assignee = assignee or task_state.get("worker_agent_id")
        if body_assignee:
            body_obj["assignee"] = body_assignee
        body = json.dumps(body_obj).encode("utf-8")
        ok, http_status, response_text = self.send_handoff_put(task_id, headers, body, task_state, status)
        if ok or http_status not in {404, 410}:
            return
        moved_to = self.error_moved_to(response_text)
        resolved = moved_to or self.resolve_handoff_task_id(task_state, headers)
        if resolved and resolved != task_id:
            with self.state_lock:
                task_state["handoff_task_id"] = resolved
                self.save_state()
            self.op(
                "handoff_task_repointed",
                f"Repointed {task_state['task']} after stale board id",
                {"task": task_state["task"], "old_id": task_id, "new_id": resolved, "http_status": http_status},
            )
            self.send_handoff_put(resolved, headers, body, task_state, status)

    def status_to_handoff(self, status: str) -> str:
        if status in {"todo"}:
            return "todo"
        if status in {"ship_packeted", "captain_staged"}:
            return "verified"
        if status in {"drop", "rejected"}:
            return "rejected"
        if status in {"pending_review", "needs_packet_fix"}:
            return "pending_verification"
        return "in_progress"

    def _terminate_tree(self, proc: subprocess.Popen, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Kill the child process AND its descendants.

        `codex exec` spawns model/shell child processes; on Windows a plain
        proc.kill() leaves those orphaned (no POSIX process groups), so a timed
        out worker keeps burning CPU/RAM. taskkill /T /F kills the whole tree.
        """
        context = context or {}
        result: dict[str, Any] = {
            "pid": proc.pid,
            "method": "taskkill_tree" if os.name == "nt" else "kill",
            "context": context,
            "attempts": [],
        }
        for attempt in (1, 2):
            item: dict[str, Any] = {"attempt": attempt}
            try:
                if os.name == "nt":
                    killed = subprocess.run(
                        ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=30,
                    )
                    item.update({"returncode": killed.returncode, "stdout": killed.stdout, "stderr": killed.stderr})
                else:
                    proc.kill()
                    item.update({"returncode": 0, "stdout": "", "stderr": ""})
                try:
                    item["wait_returncode"] = proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    item["wait_timeout"] = True
            except Exception as exc:
                item.update({"returncode": None, "error": str(exc)})
                try:
                    proc.kill()
                except Exception as kill_exc:
                    item["fallback_kill_error"] = str(kill_exc)
            time.sleep(0.5)
            item["alive_after"] = self._pid_running(proc.pid)
            result["attempts"].append(item)
            if not item["alive_after"]:
                break
        result["alive_after"] = self._pid_running(proc.pid)
        self.process_event("terminate_attempt", {**context, "pid": proc.pid, "result": result})
        self.op("runner_terminate_attempt", f"Terminate attempt for pid {proc.pid}", {"context": context, "result": result})
        return result

    def codex_base_flags(self, resume: bool) -> list[str]:
        """Build the shared -c/-s/bypass flags for a codex exec invocation.

        Sandbox posture is operator-controlled so the Windows unelevated
        restricted-token sandbox (which fails closed on split/multi writable
        roots — see Handoff/runner/README.md) can be worked around without
        hand-editing ~/.codex/config.toml:
          * codex.bypass_sandbox=true  -> --dangerously-bypass-approvals-and-sandbox
          * codex.sandbox="..."        -> -s <mode> (initial call only; resume
                                          inherits the session sandbox)
          * codex.config_overrides=[]  -> extra `-c key=value` (e.g. collapse the
                                          writable set to a single root)
        """
        codex = self.config["codex"]
        flags = [
            "-c", f"model_reasoning_effort=\"{codex['reasoning_effort']}\"",
            "-c", f"approval_policy=\"{codex.get('ask_for_approval', 'never')}\"",
        ]
        for override in codex.get("config_overrides", []):
            flags += ["-c", str(override)]
        if codex.get("bypass_sandbox"):
            flags.append("--dangerously-bypass-approvals-and-sandbox")
        elif not resume:
            # `codex exec resume` does not accept -s; it inherits the session's
            # sandbox, so only the initial call sets it.
            flags += ["-s", codex.get("sandbox", "workspace-write")]
        return flags

    def _timeout_for(self, role: str) -> int:
        """Per-attempt wall-clock ceiling for a codex child, chosen by role.

        A worker does the heavy lifting (context read + several ONNX builds +
        full truth/A-B gates on a slow model), so it gets a very generous
        ceiling; a reviewer mostly re-gates a single candidate and writes a
        verdict, so it need not hold a slot as long. Both fall back to the
        single `timeout_seconds`, then to a generous default.

        The ceiling only ever bites a *hung* attempt: a productive worker
        returns the instant it finishes, so a generous ceiling costs nothing on
        the happy path — it just stops the runner from killing real work
        mid-gate (the 45-minute cap was reaping productive workers). On timeout,
        `_terminate_tree` still kills the whole codex process tree.
        """
        codex = self.config["codex"]
        base = int(codex.get("timeout_seconds", 14400))
        if role.startswith("reviewer"):
            return int(codex.get("reviewer_timeout_seconds", base))
        return int(codex.get("worker_timeout_seconds", base))

    def invoke_codex(
        self,
        role: str,
        task: str,
        prompt: str,
        session_id: str | None = None,
        resume: bool = False,
    ) -> dict[str, Any]:
        codex = self.config["codex"]
        stamp = utc_stamp()
        role_dir = self.raw_dir / task / role
        role_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = role_dir / f"{stamp}.jsonl"
        stderr_path = role_dir / f"{stamp}.stderr.txt"
        final_path = role_dir / f"{stamp}.final.md"

        if resume and session_id:
            cmd = [
                codex["executable"],
                "exec",
                "resume",
                "-m",
                codex["model"],
                *self.codex_base_flags(resume=True),
                "--json",
                "-o",
                str(final_path),
                session_id,
                "-",
            ]
        else:
            cmd = [
                codex["executable"],
                "exec",
                "-C",
                str(REPO),
                "-m",
                codex["model"],
                *self.codex_base_flags(resume=False),
                "--json",
                "-o",
                str(final_path),
                "-",
            ]

        if self.args.dry_run or not self.args.run_codex:
            print(f"DRY RUN codex {role} {task}: {' '.join(cmd)}")
            return {
                "ok": True,
                "dry_run": True,
                "session_id": session_id,
                "final_path": None,
                "stdout_path": None,
                "stderr_path": None,
                "final_text": "",
            }

        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []

        def pump(stream: Any, path: Path, chunks: list[str]) -> None:
            with path.open("w", encoding="utf-8", errors="replace") as f:
                for line in iter(stream.readline, ""):
                    chunks.append(line)
                    f.write(line)
                    f.flush()

        role_kind, _, role_agent = role.partition("_")
        run_id = f"{self.runner_id}:{task}:{role}:{stamp}"
        env = os.environ.copy()
        env.update(
            {
                "NEUROGOLF_HEAVY_GATE_N": str(self.heavy_gate_n),
                "NEUROGOLF_HANDOFF_LANE": self.lane,
                "NEUROGOLF_HANDOFF_LANE_ID": self.lane_id,
                "NEUROGOLF_HANDOFF_BATCH": self.batch_id,
                "NEUROGOLF_HANDOFF_TASK": task,
                "NEUROGOLF_HANDOFF_TASK_NUM": self.task_num(task),
                "NEUROGOLF_HANDOFF_ROLE": role_kind,
                "NEUROGOLF_HANDOFF_AGENT": role_agent or role,
                "NEUROGOLF_HANDOFF_RUNNER_ID": self.runner_id,
                "NEUROGOLF_HANDOFF_RUN_ID": run_id,
                "NEUROGOLF_HANDOFF_RAW_DIR": rel(role_dir),
            }
        )
        termination_result: dict[str, Any] | None = None
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(REPO),
            env=env,
        )
        self.process_event(
            "start",
            {
                "task": task,
                "role": role_kind,
                "agent": role_agent or role,
                "run_id": run_id,
                "pid": proc.pid,
                "cmd": cmd,
                "stdout": rel(stdout_path),
                "stderr": rel(stderr_path),
                "final": rel(final_path),
            },
        )
        threads = [
            threading.Thread(target=pump, args=(proc.stdout, stdout_path, stdout_chunks), daemon=True),
            threading.Thread(target=pump, args=(proc.stderr, stderr_path, stderr_chunks), daemon=True),
        ]
        for thread in threads:
            thread.start()
        forced_final_reap = False
        skip_stream_close = False
        try:
            if proc.stdin:
                proc.stdin.write(prompt)
                proc.stdin.close()

            timeout_seconds = self._timeout_for(role)
            deadline = time.monotonic() + timeout_seconds
            final_quiet_seconds = float(codex.get("final_file_quiet_seconds", 90))
            final_sig: tuple[int, int] | None = None
            final_stable_since: float | None = None
            while True:
                returncode = proc.poll()
                if returncode is not None:
                    break

                if final_path.exists():
                    stat = final_path.stat()
                    if stat.st_size > 0:
                        sig = (stat.st_size, stat.st_mtime_ns)
                        now = time.monotonic()
                        if sig != final_sig:
                            final_sig = sig
                            final_stable_since = now
                        elif final_stable_since is not None and now - final_stable_since >= final_quiet_seconds:
                            # Codex occasionally writes the requested final
                            # artifact but keeps the exec process alive. Reap
                            # that idle child so one completed task cannot hold
                            # a scheduler slot until the full role timeout.
                            with stderr_path.open("a", encoding="utf-8") as f:
                                f.write("\nFINAL_FILE_STABLE_REAP\n")
                            termination_result = self._terminate_tree(
                                proc,
                                {
                                    "task": task,
                                    "role": role_kind,
                                    "agent": role_agent or role,
                                    "run_id": run_id,
                                    "reason": "final_file_stable_reap",
                                },
                            )
                            # _terminate_tree already waits briefly and records
                            # the child status. A second wait here has wedged
                            # scheduler worker threads on Windows after Codex
                            # wrote a valid final artifact, leaving tasks stuck
                            # in worker_running even though their process tree
                            # was gone.
                            returncode = proc.poll()
                            if returncode is None:
                                for attempt in reversed(termination_result.get("attempts", [])):
                                    if attempt.get("wait_returncode") is not None:
                                        returncode = attempt["wait_returncode"]
                                        break
                            if returncode is None:
                                returncode = -9
                            forced_final_reap = True
                            skip_stream_close = True
                            break

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(cmd, timeout_seconds)
                time.sleep(min(5.0, max(0.2, remaining)))
        except subprocess.TimeoutExpired:
            termination_result = self._terminate_tree(
                proc,
                {
                    "task": task,
                    "role": role_kind,
                    "agent": role_agent or role,
                    "run_id": run_id,
                    "reason": "timeout",
                },
            )
            try:
                returncode = proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                returncode = proc.poll() if proc.poll() is not None else -9
            with stderr_path.open("a", encoding="utf-8") as f:
                f.write("\nTIMEOUT\n")
            for thread in threads:
                thread.join(timeout=5)
            self.process_event(
                "exit",
                {
                    "task": task,
                    "role": role_kind,
                    "agent": role_agent or role,
                    "run_id": run_id,
                    "pid": proc.pid,
                    "ok": False,
                    "error": "timeout",
                    "returncode": returncode,
                    "timeout_seconds": self._timeout_for(role),
                    "termination": termination_result,
                    "stdout": rel(stdout_path),
                    "stderr": rel(stderr_path),
                    "final": rel(final_path),
                },
            )
            return {
                "ok": False,
                "error": "timeout",
                "session_id": session_id,
                "final_path": str(final_path),
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "termination": termination_result,
                "final_text": "",
            }
        finally:
            # On Windows, closing these pipe wrappers after taskkill /T /F can
            # block even though the process tree is gone. The pump threads are
            # daemonized and the child has already been reaped, so leave them
            # alone on the forced-final path and let process teardown release
            # the handles.
            if not skip_stream_close:
                for stream in (proc.stdout, proc.stderr):
                    if stream:
                        stream.close()
        for thread in threads:
            thread.join(timeout=5)

        stdout_text = "".join(stdout_chunks)
        final_text = final_path.read_text(encoding="utf-8") if final_path.exists() else ""
        new_session = find_session_id(stdout_text) or session_id
        ok = returncode == 0 or (forced_final_reap and bool(final_text.strip()))
        proc_returncode = proc.returncode if proc.returncode is not None else returncode
        self.process_event(
            "exit",
            {
                "task": task,
                "role": role_kind,
                "agent": role_agent or role,
                "run_id": run_id,
                "pid": proc.pid,
                "returncode": proc_returncode,
                "ok": ok,
                "forced_final_reap": forced_final_reap,
                "termination": termination_result,
                "stdout": rel(stdout_path),
                "stderr": rel(stderr_path),
                "final": rel(final_path),
            },
        )
        return {
            "ok": ok,
            "returncode": proc_returncode,
            "forced_final_reap": forced_final_reap,
            "termination": termination_result,
            "session_id": new_session,
            "final_path": rel(final_path),
            "stdout_path": rel(stdout_path),
            "stderr_path": rel(stderr_path),
            "final_text": final_text,
        }

    def ordered_task_states(self) -> list[dict[str, Any]]:
        ordered = []
        for task_cfg in self.config["tasks"]:
            task_state = self.state["tasks"].get(task_cfg["task"])
            if task_state:
                ordered.append(task_state)
        return ordered

    def select_worker_task(self) -> dict[str, Any] | None:
        priority = ["forced_retry", "worker_rework", "todo"]
        tasks = self.ordered_task_states()
        for status in priority:
            for task_state in tasks:
                if task_state["status"] == status:
                    return task_state
        return None

    def select_review_task(self) -> dict[str, Any] | None:
        for task_state in self.ordered_task_states():
            if task_state["status"] == "pending_review":
                return task_state
        return None

    def claim_worker(
        self, busy_tasks: set[str], busy_agents: set[str]
    ) -> tuple[dict[str, Any], str] | None:
        """Atomically pick the next worker-eligible task + a free worker agent.

        Does NOT change task status; the scheduler adds the task to busy_tasks so
        re-selection skips it while run_worker is in flight. Returns None when no
        task is currently runnable with a free agent. Held under state_lock so it
        can't race a worker thread mutating the same state dict.
        """
        with self.state_lock:
            available = [a for a in self.worker_agents() if a not in busy_agents]
            if not available:
                return None
            for status in ("forced_retry", "worker_rework", "todo"):
                for task_state in self.ordered_task_states():
                    if task_state["task"] in busy_tasks or task_state["status"] != status:
                        continue
                    preferred = task_state.get("worker_agent_id")
                    agent = preferred if preferred in available else available[0]
                    task_state["worker_agent_id"] = agent
                    return task_state, agent
        return None

    def claim_review(
        self, busy_tasks: set[str], busy_agents: set[str]
    ) -> tuple[dict[str, Any], str] | None:
        with self.state_lock:
            available = [a for a in self.reviewer_agents() if a not in busy_agents]
            if not available:
                return None
            for task_state in self.ordered_task_states():
                if task_state["task"] in busy_tasks or task_state["status"] != "pending_review":
                    continue
                preferred = task_state.get("reviewer_agent_id")
                agent = preferred if preferred in available else available[0]
                task_state["reviewer_agent_id"] = agent
                return task_state, agent
        return None

    def run_worker(self, task_state: dict[str, Any], agent_id: str | None = None) -> str:
        task = task_state["task"]
        if agent_id:
            task_state["worker_agent_id"] = agent_id
        task_num = self.task_num(task)
        prompt_path = self.task_prompt_path(task_state)
        prior_packet = ""
        if task_state.get("worker_last_output"):
            prior_path = REPO / task_state["worker_last_output"]
            if prior_path.exists():
                prior_packet = prior_path.read_text(encoding="utf-8")

        values = {
            "task": task,
            "task_num": task_num,
            "prompt_path": rel(prompt_path),
            "prior_worker_packet": prior_packet,
            "reviewer_rework": task_state.get("reviewer_rework_prompt", ""),
            "worker_agent": task_state.get("worker_agent_id") or self.config["agents"].get("worker"),
            "reviewer_agent": task_state.get("reviewer_agent_id") or self.config["agents"].get("reviewer"),
        }

        status = task_state["status"]
        if status == "todo":
            template = "worker_initial.md"
            resume = False
            action = "initial worker pass"
        elif status == "forced_retry":
            template = "worker_forced_retry.md"
            resume = True
            action = "forced worker retry"
        elif status == "worker_rework":
            template = "worker_rework.md"
            resume = True
            action = "reviewer-directed worker rework"
        else:
            return f"No worker action for {task} status={status}"

        prompt = self.template(template, values)
        if self.args.dry_run or not self.args.run_codex:
            print(f"DRY RUN: would run {action} for {task}")
            return f"planned {action} for {task}"

        entry_status = status  # status this attempt launched from (for retry)
        with self.state_lock:
            task_state["status"] = "worker_running"
            self.save_state()
        self.update_handoff_task(
            task_state,
            "in_progress",
            f"{action} started by runner",
            agent_id=task_state.get("worker_agent_id"),
            assignee=task_state.get("worker_agent_id"),
        )

        cleanup_report: dict[str, Any] = {}
        try:
            result = self.invoke_codex(
                f"worker_{task_state.get('worker_agent_id') or 'unknown'}",
                task,
                prompt,
                session_id=task_state.get("worker_session_id"),
                resume=resume,
            )
        finally:
            cleanup_report = self.cleanup_task_gates(task, "worker_role_returned")
        if cleanup_report:
            result["gate_cleanup"] = cleanup_report

        with self.state_lock:
            task_state["worker_session_id"] = result.get("session_id")
            task_state["worker_last_output"] = result.get("final_path")
            task_state["history"].append(
                {
                    "ts_utc": utc_now(),
                    "role": "worker",
                    "action": action,
                    "result": redact(result),
                }
            )

            if not result["ok"]:
                max_failures = int(self.config["workflow"].get("worker_max_failures", 2))
                failures = int(task_state.get("worker_failures", 0)) + 1
                task_state["worker_failures"] = failures
                if failures <= max_failures:
                    # Retry: drop the (possibly hung) session so the next attempt
                    # starts fresh rather than resuming a broken/timed-out run.
                    task_state["worker_session_id"] = None
                    task_state["status"] = entry_status
                    self.save_state()
                    self.op(
                        "runner_worker_retry",
                        f"Worker failed on {task}; retry {failures}/{max_failures} queued",
                        {"task": task, "status": entry_status, "error": result.get("error")},
                    )
                    return f"worker failed on {task}; retry {failures}/{max_failures} queued"
                task_state["status"] = "worker_failed"
                self.save_state()
                self.op(
                    "runner_worker_failed",
                    f"Worker failed on {task} (no retries left after {failures})",
                    {"task": task, "result": result},
                )
                return f"worker failed on {task} (no retries left)"

            task_state["worker_attempts"] = int(task_state.get("worker_attempts", 0)) + 1
            required = int(self.config["workflow"]["worker_attempts_before_review"])
            if task_state["worker_attempts"] < required:
                task_state["status"] = "forced_retry"
                result_msg = f"attempt {task_state['worker_attempts']} complete; forced retry queued"
                handoff_status = "in_progress"
            else:
                task_state["status"] = "pending_review"
                result_msg = f"attempt {task_state['worker_attempts']} complete; reviewer queued"
                handoff_status = "pending_verification"
            self.save_state()

        self.update_handoff_task(
            task_state,
            handoff_status,
            result_msg,
            agent_id=task_state.get("worker_agent_id"),
            assignee=task_state.get("worker_agent_id"),
        )
        self.op("runner_worker_complete", f"{action} complete for {task}", {"task": task, "status": task_state["status"]})
        return result_msg

    def _review_retry_or_fail(self, task_state: dict[str, Any], terminal_status: str) -> str:
        """Bounded retry for non-fatal reviewer output problems (unparseable
        verdict JSON, or SHIP with a missing/invalid captain packet). Re-queues
        the review a few times, then settles on a terminal status so the task
        stops cycling. Must be called while holding self.state_lock.
        """
        max_failures = int(self.config["workflow"].get("reviewer_max_failures", 1))
        failures = int(task_state.get("review_failures", 0)) + 1
        task_state["review_failures"] = failures
        if failures <= max_failures:
            task_state["reviewer_session_id"] = None
            return "pending_review"
        return terminal_status

    def run_reviewer(self, task_state: dict[str, Any], agent_id: str | None = None) -> str:
        task = task_state["task"]
        if agent_id:
            task_state["reviewer_agent_id"] = agent_id
        prompt_path = self.task_prompt_path(task_state)
        task_prompt = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else ""
        worker_packet = ""
        if task_state.get("worker_last_output"):
            worker_path = REPO / task_state["worker_last_output"]
            if worker_path.exists():
                worker_packet = worker_path.read_text(encoding="utf-8")
        values = {
            "task": task,
            "task_num": self.task_num(task),
            "task_prompt": task_prompt,
            "worker_packet": worker_packet,
            "reviewer_attack_log": task_state.get("reviewer_attack_log", ""),
            "reviewer_agent": task_state.get("reviewer_agent_id") or self.config["agents"].get("reviewer"),
            "worker_agent": task_state.get("worker_agent_id") or self.config["agents"].get("worker"),
        }
        prompt = self.template("reviewer.md", values)

        if self.args.dry_run or not self.args.run_codex:
            print(f"DRY RUN: would run reviewer for {task}")
            return f"planned reviewer for {task}"

        with self.state_lock:
            task_state["status"] = "review_running"
            self.save_state()
        self.update_handoff_task(
            task_state,
            "pending_verification",
            "review started by runner",
            agent_id=task_state.get("reviewer_agent_id"),
            assignee=task_state.get("reviewer_agent_id"),
        )

        cleanup_report: dict[str, Any] = {}
        try:
            result = self.invoke_codex(
                f"reviewer_{task_state.get('reviewer_agent_id') or 'unknown'}",
                task,
                prompt,
                session_id=task_state.get("reviewer_session_id"),
                resume=bool(task_state.get("reviewer_session_id")),
            )
        finally:
            cleanup_report = self.cleanup_task_gates(task, "reviewer_role_returned")
        if cleanup_report:
            result["gate_cleanup"] = cleanup_report

        with self.state_lock:
            task_state["reviewer_session_id"] = result.get("session_id")
            task_state["reviewer_last_output"] = result.get("final_path")
            task_state["history"].append(
                {
                    "ts_utc": utc_now(),
                    "role": "reviewer",
                    "action": "review",
                    "result": redact(result),
                }
            )

            if not result["ok"]:
                max_failures = int(self.config["workflow"].get("reviewer_max_failures", 1))
                failures = int(task_state.get("review_failures", 0)) + 1
                task_state["review_failures"] = failures
                if failures <= max_failures:
                    task_state["reviewer_session_id"] = None
                    task_state["status"] = "pending_review"
                    self.save_state()
                    self.op(
                        "runner_review_retry",
                        f"Reviewer failed on {task}; retry {failures}/{max_failures} queued",
                        {"task": task, "error": result.get("error")},
                    )
                    return f"reviewer failed on {task}; retry {failures}/{max_failures} queued"
                task_state["status"] = "review_failed"
                self.save_state()
                self.op("runner_review_failed", f"Reviewer failed on {task} (no retries left)", {"task": task, "result": result})
                return f"reviewer failed on {task} (no retries left)"

            review = extract_review_json(result.get("final_text", "")) or {}
            verdict = str(review.get("verdict", "")).strip().upper().replace(" ", "_")

            if verdict == "SHIP":
                packet_path = self.write_captain_packet(task_state, review.get("captain_packet"))
                if packet_path:
                    task_state["status"] = "ship_packeted"
                    task_state["captain_packet_path"] = rel(packet_path)
                    task_state["reviewer_attack_log"] = str(review.get("attack_log", ""))
                    msg = f"reviewer SHIP; captain packet {rel(packet_path)}"
                    handoff_status = "verified"
                else:
                    task_state["status"] = self._review_retry_or_fail(task_state, "needs_packet_fix")
                    task_state["reviewer_attack_log"] = str(review.get("attack_log", ""))
                    msg = "reviewer SHIP but packet was missing/invalid"
                    handoff_status = "pending_verification"
            elif verdict == "KEEP_WORKING":
                cycles = int(task_state.get("rework_cycles", 0)) + 1
                task_state["rework_cycles"] = cycles
                max_cycles = int(self.config["workflow"].get("max_rework_cycles", 3))
                task_state["reviewer_attack_log"] = str(review.get("attack_log", ""))
                if cycles > max_cycles:
                    # Cap the worker<->reviewer rework loop. Without this the drain
                    # scheduler would spin it tight (no inter-step sleep) and burn
                    # codex calls until the batch dies.
                    task_state["status"] = "drop"
                    msg = f"reviewer KEEP_WORKING but rework cap ({max_cycles}) reached; dropping"
                    handoff_status = "rejected"
                else:
                    task_state["status"] = "worker_rework"
                    task_state["reviewer_rework_prompt"] = str(review.get("rework_prompt", ""))
                    msg = f"reviewer KEEP_WORKING; rework {cycles}/{max_cycles} queued"
                    handoff_status = "in_progress"
            elif verdict == "DROP":
                task_state["status"] = "drop"
                task_state["reviewer_attack_log"] = str(review.get("attack_log", ""))
                msg = "reviewer DROP"
                handoff_status = "rejected"
            else:
                task_state["status"] = self._review_retry_or_fail(task_state, "review_parse_failed")
                msg = "reviewer output did not include a parseable verdict JSON"
                handoff_status = "pending_verification"

            self.save_state()
            handoff_agent = task_state.get("reviewer_agent_id") or self.config["agents"].get("reviewer")
            handoff_assignee = task_state.get("worker_agent_id") if verdict == "KEEP_WORKING" else handoff_agent

        self.update_handoff_task(
            task_state,
            handoff_status,
            msg,
            agent_id=handoff_agent,
            assignee=handoff_assignee,
        )
        self.op("runner_review_complete", f"Review complete for {task}: {verdict or 'UNPARSED'}", {"task": task, "msg": msg})
        return msg

    def run_scheduler(self) -> list[str]:
        """Concurrent, self-refilling scheduler.

        Keeps up to worker_max_active workers and reviewer_max_active reviewers
        in flight, and refills a slot the instant a task finishes — so a single
        slow worker no longer holds idle slots open for the rest of the batch
        (the old run_worker_batch waited for the whole batch to drain before
        re-selecting). Workers and reviewers pipeline: a task a worker pushes to
        pending_review is picked up by a reviewer as soon as a slot frees. Drains
        until nothing is runnable, then returns.
        """
        role = self.args.role
        do_worker = role in {"all", "worker"}
        do_review = role in {"all", "reviewer"}
        worker_max = max(1, int(self.config["workflow"].get("worker_max_active", 1)))
        review_max = max(1, int(self.config["workflow"].get("reviewer_max_active", 1)))

        messages: list[str] = []
        busy_tasks: set[str] = set()
        busy_worker_agents: set[str] = set()
        busy_reviewer_agents: set[str] = set()
        meta: dict[Any, tuple[str, str, str]] = {}  # future -> (kind, task, agent)

        worker_pool = ThreadPoolExecutor(max_workers=worker_max, thread_name_prefix="howorker")
        review_pool = ThreadPoolExecutor(max_workers=review_max, thread_name_prefix="horeview")
        try:
            while True:
                if do_review:
                    while sum(1 for m in meta.values() if m[0] == "reviewer") < review_max:
                        pick = self.claim_review(busy_tasks, busy_reviewer_agents)
                        if not pick:
                            break
                        ts, agent = pick
                        busy_tasks.add(ts["task"])
                        busy_reviewer_agents.add(agent)
                        meta[review_pool.submit(self.run_reviewer, ts, agent)] = (
                            "reviewer", ts["task"], agent,
                        )
                if do_worker:
                    while sum(1 for m in meta.values() if m[0] == "worker") < worker_max:
                        pick = self.claim_worker(busy_tasks, busy_worker_agents)
                        if not pick:
                            break
                        ts, agent = pick
                        busy_tasks.add(ts["task"])
                        busy_worker_agents.add(agent)
                        meta[worker_pool.submit(self.run_worker, ts, agent)] = (
                            "worker", ts["task"], agent,
                        )

                if not meta:
                    break

                done, _ = wait(list(meta), return_when=FIRST_COMPLETED)
                for fut in done:
                    kind, task, agent = meta.pop(fut)
                    busy_tasks.discard(task)
                    if kind == "worker":
                        busy_worker_agents.discard(agent)
                    else:
                        busy_reviewer_agents.discard(agent)
                    try:
                        messages.append(f"{agent}: {fut.result()}")
                    except Exception as exc:
                        messages.append(f"{agent}: {kind} exception on {task}: {exc}")
                        self.op(
                            f"runner_{kind}_exception",
                            f"{kind} exception on {task}",
                            {"task": task, "agent_id": agent, "error": str(exc)},
                        )
        finally:
            worker_pool.shutdown(wait=True)
            review_pool.shutdown(wait=True)
            terminal_cleanup = self.cleanup_terminal_task_gates("scheduler_shutdown")
            if terminal_cleanup:
                self.op(
                    "runner_terminal_gate_cleanup",
                    "Cleaned up terminal-task ng_gate process(es) after scheduler shutdown",
                    {"reports": terminal_cleanup},
                )
        return messages

    def plan_pass(self) -> list[str]:
        """Single, non-mutating planning pass for --dry-run (no refill loop)."""
        role = self.args.role
        messages: list[str] = []
        if role in {"all", "reviewer"}:
            review_max = max(1, int(self.config["workflow"].get("reviewer_max_active", 1)))
            n = 0
            for task_state in self.ordered_task_states():
                if task_state["status"] == "pending_review" and n < review_max:
                    messages.append(self.run_reviewer(task_state))
                    n += 1
        if role in {"all", "worker"}:
            worker_max = max(1, int(self.config["workflow"].get("worker_max_active", 1)))
            n = 0
            for status in ("forced_retry", "worker_rework", "todo"):
                for task_state in self.ordered_task_states():
                    if task_state["status"] == status and n < worker_max:
                        messages.append(self.run_worker(task_state))
                        n += 1
        return messages

    @staticmethod
    def _rescue_candidate_key(value: Any) -> str:
        text = str(value or "").strip().strip("\"'")
        if not text:
            return ""
        path = Path(text)
        try:
            if not path.is_absolute():
                path = REPO / path
            return os.path.normcase(str(path.resolve()))
        except Exception:
            return os.path.normcase(str(path))

    def _rescue_row_key(self, row: dict[str, Any]) -> tuple[str, str]:
        return (
            self.task_num(str(row.get("task") or "")),
            self._rescue_candidate_key(row.get("candidate_path")),
        )

    def resolve_rescue_rows_for_ship(self, task_state: dict[str, Any], packet_path: Path) -> int:
        """Close stale rescue rows once this task has a reviewer-approved SHIP packet."""
        queue = REPO / "Handoff" / "private" / "review_rescue_queue.jsonl"
        if not queue.exists():
            return 0

        raw_rows: list[dict[str, Any]] = []
        for line in queue.read_text(encoding="utf-8-sig", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                raw_rows.append(row)

        latest: dict[tuple[str, str], dict[str, Any]] = {}
        for row in raw_rows:
            latest[self._rescue_row_key(row)] = row

        task_num = self.task_num(str(task_state.get("task") or ""))
        packet_rel = rel(packet_path)
        open_rows = [
            row
            for (row_task, _), row in latest.items()
            if row_task == task_num and str(row.get("status", "open")).lower() == "open"
        ]
        if not open_rows:
            return 0

        resolved_utc = utc_now()
        for row in open_rows:
            resolved = dict(row)
            resolved["status"] = "resolved"
            resolved["resolved_utc"] = resolved_utc
            resolved["resolution"] = f"superseded_by_runner_ship_packet {packet_rel}"
            resolved["ship_packet_path"] = packet_rel
            append_jsonl(queue, resolved)

        task = task_state.get("task")
        self.op(
            "runner_rescue_resolved",
            f"Resolved {len(open_rows)} open rescue row(s) for {task}",
            {"task": task, "packet": packet_rel, "count": len(open_rows)},
        )
        return len(open_rows)

    def write_captain_packet(self, task_state: dict[str, Any], packet: Any) -> Path | None:
        validated = validate_packet(packet if isinstance(packet, dict) else None)
        if not validated:
            return None
        packet = validated.model_dump(exclude_none=True)
        packet["task"] = task_state["task"]
        workflow = self.config.get("workflow", {})
        packet["source_lane"] = workflow["lane"]
        packet.setdefault("reviewer_verdict", "SHIP")
        # Stamp batch identity so Captain/alpha can filter packets by batch/runner
        # (p43 packets shipped with these empty — see the win-rate action reports).
        for key in ("batch_id", "runner_id"):
            if workflow.get(key) and not packet.get(key):
                packet[key] = workflow[key]
        # Backstop the ambiguous `expected_delta` with a signed cost delta
        # (candidate - live; NEGATIVE = bytes saved, same sign as expected_delta and
        # the staged-manifest cost_delta) when the reviewer gave both pieces.
        if (
            packet.get("cost_delta") is None
            and isinstance(packet.get("live_cost"), (int, float))
            and isinstance(packet.get("candidate_profile_cost"), (int, float))
        ):
            packet["cost_delta"] = int(packet["candidate_profile_cost"]) - int(packet["live_cost"])
        out = self.captain_inbox / f"{utc_stamp()}-{task_state['task']}.json"
        write_json(out, packet)
        self.op("runner_captain_packet", f"Wrote Submit Captain packet for {task_state['task']}", {"packet": rel(out)})
        try:
            self.resolve_rescue_rows_for_ship(task_state, out)
        except Exception as exc:
            self.op(
                "runner_rescue_resolve_failed",
                f"Failed to resolve rescue rows for {task_state['task']}",
                {"packet": rel(out), "error": str(exc)},
            )
        return out

    def pending_packets(self) -> list[Path]:
        packets = []
        for path in sorted(self.captain_inbox.glob("*.json")):
            data = read_json(path, {})
            if not data.get("probe_ok"):
                packets.append(path)
        return packets

    def run_captain(self) -> str:
        packets = self.pending_packets()
        batch_size = int(self.config["workflow"]["captain_batch_size"])
        if len(packets) < batch_size:
            return f"captain idle: {len(packets)}/{batch_size} packets"

        commands = [
            [sys.executable, "Agenters/Submit_Captain/captain_ctl.py", "digest", "--lane", "codex"],
            [sys.executable, "Agenters/Submit_Captain/captain_ctl.py", "probe-status"],
        ]
        if self.args.allow_kaggle_submit:
            commands.append(
                [sys.executable, "Agenters/Submit_Captain/captain_ctl.py"]
                + self.config["workflow"]["captain_safe_batch_command"]
            )

        if self.args.dry_run or not self.args.run_codex or not self.args.allow_kaggle_submit:
            print(f"DRY RUN: captain has {len(packets)} packets")
            for cmd in commands:
                print("  " + " ".join(cmd))
            if not self.args.allow_kaggle_submit:
                print("  Kaggle-affecting command held; pass -AllowKaggleSubmit when ready.")
            return f"planned captain action for {len(packets)} packets"

        stamp = utc_stamp()
        out_dir = self.raw_dir / "captain"
        out_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for idx, cmd in enumerate(commands, 1):
            proc = subprocess.run(
                cmd,
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(REPO),
            )
            stdout_path = out_dir / f"{stamp}-{idx}.stdout.txt"
            stderr_path = out_dir / f"{stamp}-{idx}.stderr.txt"
            stdout_path.write_text(proc.stdout or "", encoding="utf-8")
            stderr_path.write_text(proc.stderr or "", encoding="utf-8")
            records.append(
                {
                    "cmd": cmd,
                    "returncode": proc.returncode,
                    "stdout": rel(stdout_path),
                    "stderr": rel(stderr_path),
                }
            )
            if proc.returncode != 0:
                self.state["captain"]["status"] = "failed"
                self.state["captain"]["last_records"] = records
                self.save_state()
                self.op("runner_captain_failed", "Captain command failed", {"records": records})
                return "captain command failed"

        self.state["captain"]["status"] = "probe_started_or_completed"
        self.state["captain"]["last_records"] = records
        self.save_state()
        self.op("runner_captain_complete", "Captain commands complete", {"records": records})
        return "captain commands complete"

    def orchestrator_record(self) -> str:
        lines = []
        for task in self.state["tasks"].values():
            lines.append(
                f"{task['task']}: {task['status']} attempts={task.get('worker_attempts', 0)} "
                f"packet={task.get('captain_packet_path') or '-'}"
            )
        summary = "\n".join(lines)
        self.op("runner_orchestrator_record", "Runner state record", {"summary": summary})
        return summary

    def reconcile_orphans(self) -> None:
        """Recover transient running statuses left by a killed or wedged runner.

        The scheduler only claims runnable states, so a stale worker_running or
        review_running row can pin a slot forever after an abort. Rewind those
        rows at startup so the next drain can reclaim them.
        """
        rewound: list[str] = []
        rewound_tasks: list[str] = []
        with self.state_lock:
            for ts in self.state["tasks"].values():
                st = ts.get("status")
                if st == "worker_running":
                    ts["status"] = "todo"
                    rewound.append(f"{ts['task']}:worker_running->todo")
                    rewound_tasks.append(ts["task"])
                elif st == "review_running":
                    ts["status"] = "pending_review"
                    rewound.append(f"{ts['task']}:review_running->pending_review")
                    rewound_tasks.append(ts["task"])
            if rewound:
                self.save_state()
        for r in rewound:
            self.op("runner_orphan_rewind", f"rewound stale {r}", {"detail": r})
        for task in rewound_tasks:
            self.cleanup_task_gates(task, "startup_orphan_rewind")

    def step(self) -> str:
        role = self.args.role
        messages: list[str] = []

        # --dry-run / planning: single non-mutating pass so it can't loop on
        # statuses that never advance without a real codex run.
        if self.args.dry_run or not self.args.run_codex:
            messages.extend(self.plan_pass())
            if role in {"all", "captain"}:
                captain_msg = self.run_captain()
                if not captain_msg.startswith("captain idle"):
                    messages.append(captain_msg)
                elif role == "captain":
                    return captain_msg
            if role in {"all", "orchestrator"}:
                record = self.orchestrator_record()
                if role == "orchestrator":
                    return record
            return "\n".join(m for m in messages if m) or "runner idle (dry-run)"

        # Real run: first heal crash/abort orphans, then drain workers +
        # reviewers concurrently with refill. The captain settles whatever
        # packets accumulated.
        self.reconcile_orphans()
        if role in {"all", "worker", "reviewer"}:
            messages.extend(self.run_scheduler())

        if role in {"all", "captain"}:
            captain_msg = self.run_captain()
            if not captain_msg.startswith("captain idle"):
                messages.append(captain_msg)
            elif role == "captain":
                return captain_msg

        if role in {"all", "orchestrator"}:
            record = self.orchestrator_record()
            if role == "orchestrator":
                return record

        return "\n".join(m for m in messages if m) or "runner idle"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the NeuroGolf HandOff workflow locally.")
    p.add_argument("--config", default="Handoff/config/runner.json")
    p.add_argument("--role", choices=["all", "worker", "reviewer", "captain", "orchestrator"], default="all")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--run-codex", action="store_true")
    p.add_argument("--sync-handoff", action="store_true")
    p.add_argument("--allow-kaggle-submit", action="store_true")
    p.add_argument("--init-state", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    runner = Runner(args)
    msg = runner.step()
    print(msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())






