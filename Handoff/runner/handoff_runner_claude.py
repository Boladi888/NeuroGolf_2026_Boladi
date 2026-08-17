#!/usr/bin/env python3
"""Local HandOff runner for the NeuroGolf workflow — Claude lane.

Mirrors handoff_runner.py but drives `claude -p` worker/reviewer sessions
instead of `codex exec`, writes Submit Captain packets into the Claude lane
inbox, and syncs HandOff task status using the per-task assignee's agent key.

Safe by default: no `claude` session is launched unless --run-claude is passed;
no Kaggle-affecting captain command runs unless --allow-kaggle-submit is also
passed. Raw transcripts go under the gitignored raw_log_dir; agent keys are read
from Handoff/private/<agent>.env and never written into shareable logs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
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

# Serializes appends to shared JSONL logs (operations.jsonl is shared with the
# codex lane) so interleaved writers cannot produce half-written lines.
_JSONL_LOCK = threading.Lock()


def terminate_tree(proc: subprocess.Popen) -> None:
    """Kill the child process AND its descendants. `claude -p` spawns child
    processes; a plain kill on Windows orphans them, so a timed-out session
    keeps burning resources. taskkill /T /F kills the whole tree."""
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True,
                timeout=30,
            )
        else:
            proc.kill()
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        pass


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
        s = re.sub(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", "<email_redacted>", s)
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


def find_session_id(text: str) -> str | None:
    for line in text.splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            for key in ("session_id", "sessionId", "conversation_id"):
                val = obj.get(key)
                if isinstance(val, str) and UUID_RE.search(val):
                    return UUID_RE.search(val).group(0)
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
        self.lane = self.config["workflow"].get("lane", "Claude")
        self.state = self.load_state(force=args.init_state)

    def ensure_dirs(self) -> None:
        for d in (self.state_path.parent, self.raw_dir, self.captain_inbox):
            d.mkdir(parents=True, exist_ok=True)

    def load_state(self, force: bool = False) -> dict[str, Any]:
        self.ensure_dirs()
        if force and self.state_path.exists():
            self.state_path.replace(self.state_path.with_name(f"runner_state.{utc_stamp()}.bak.json"))
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
            ts = state["tasks"].setdefault(
                task,
                {
                    "task": task,
                    "title": t.get("title", ""),
                    "handoff_task_id": t["handoff_task_id"],
                    "external_key": t.get("external_key"),
                    "prompt_file": t["prompt_file"],
                    "bar": t.get("bar"),
                    "assignee": t.get("assignee", self.config["agents"]["workers"][0]),
                    "reviewer": t.get("reviewer", self.config["agents"]["reviewers"][0]),
                    "status": "todo",
                    "worker_attempts": 0,
                    "worker_session_id": None,
                    "reviewer_session_id": None,
                    "worker_last_output": None,
                    "reviewer_last_output": None,
                    "reviewer_attack_log": "",
                    "reviewer_rework_prompt": "",
                    "captain_packet_path": None,
                    "history": [],
                },
            )
            # Config-authoritative identity fields: if the config repoints a
            # task to a new board id (e.g. the lane moved to a different project)
            # or changes the prompt/bar, honor it over a stale state value.
            for k in ("handoff_task_id", "external_key", "prompt_file", "bar", "assignee"):
                if t.get(k) is not None:
                    ts[k] = t[k]
        self.save_state(state)
        return state

    def save_state(self, state: dict[str, Any] | None = None) -> None:
        if state is None:
            state = self.state
        with self.state_lock:
            state["updated_utc"] = utc_now()
            write_json(self.state_path, state)

    def op(self, kind: str, summary: str, data: Any = None) -> None:
        append_jsonl(self.operation_log, {"ts_utc": utc_now(), "kind": kind, "summary": summary, "data": data})

    def task_num(self, task: str) -> str:
        return re.sub(r"\D", "", task).lstrip("0") or "0"

    def task_prompt_path(self, ts: dict[str, Any]) -> Path:
        return self.prompt_dir / ts["prompt_file"]

    def template(self, name: str, values: dict[str, Any]) -> str:
        text = read_text(self.template_dir / name)
        safe = {k: str(v) for k, v in values.items()}

        def repl(m: re.Match[str]) -> str:
            return safe.get(m.group(1), m.group(0))

        return re.sub(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", repl, text)

    # ---- HandOff task status sync -------------------------------------------
    def task_update_headers(self, assignee: str) -> dict[str, str]:
        env = load_env_file(REPO / ".env.handoff")
        headers = {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0 NeuroGolf-HandoffRunner-CLD/1.0"}
        token = env.get("HANDOFF_BOOTSTRAP_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        agent_env = load_env_file(REPO / "Handoff" / "private" / f"{assignee}.env")
        key = agent_env.get("HANDOFF_AGENT_KEY") or agent_env.get("AGENT_KEY")
        if key:
            headers["X-Agent-Key"] = key
            headers["X-Agent-Id"] = agent_env.get("HANDOFF_AGENT_ID") or assignee
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

    def resolve_handoff_task_id(self, ts: dict[str, Any], headers: dict[str, str]) -> str | None:
        project_id = self.project_id()
        external_key = ts.get("external_key")
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
                f"Failed to resolve {ts['task']} by external_key",
                {"task": ts["task"], "external_key": external_key, "error": str(exc)},
            )
            return None
        resolved = self.find_task_id(data)
        if resolved:
            with self.state_lock:
                ts["handoff_task_id"] = resolved
                self.save_state()
            self.op(
                "handoff_task_resolved",
                f"Resolved {ts['task']} by external_key",
                {"task": ts["task"], "external_key": external_key, "handoff_task_id": resolved},
            )
        return resolved

    def send_handoff_put(
        self,
        task_id: str,
        headers: dict[str, str],
        body: bytes,
        ts: dict[str, Any],
        status: str,
    ) -> tuple[bool, int | None, str]:
        req = urllib.request.Request(
            f"https://handoff.socnet.lol/api/v1/tasks/{task_id}", data=body, method="PUT", headers=headers
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                txt = resp.read().decode("utf-8", errors="replace")
            self.op("handoff_sync_complete", f"Synced {ts['task']} -> {status}", {"task": ts["task"], "status": status, "response": txt[:500]})
            return True, None, txt
        except urllib.error.HTTPError as exc:
            txt = exc.read().decode("utf-8", errors="replace")
            self.op(
                "handoff_sync_failed",
                f"Sync failed {ts['task']}",
                {"task": ts["task"], "status": status, "handoff_task_id": task_id, "http_status": exc.code, "error": str(exc), "response": txt[:500]},
            )
            return False, exc.code, txt
        except urllib.error.URLError as exc:
            self.op("handoff_sync_failed", f"Sync failed {ts['task']}", {"task": ts["task"], "status": status, "handoff_task_id": task_id, "error": str(exc)})
            return False, None, str(exc)

    def update_handoff_task(self, ts: dict[str, Any], status: str, result: str = "") -> None:
        if not self.args.sync_handoff:
            return
        headers = self.task_update_headers(ts.get("assignee", ""))
        if "Authorization" not in headers and "X-Agent-Key" not in headers:
            print("sync skipped: no HandOff bearer token or agent key found", file=sys.stderr)
            return
        task_id = ts.get("handoff_task_id")
        if not task_id:
            task_id = self.resolve_handoff_task_id(ts, headers)
        if not task_id:
            self.op("handoff_sync_skipped", f"No HandOff task id for {ts['task']}", {"task": ts["task"], "external_key": ts.get("external_key")})
            return
        body = json.dumps({"status": status, "result": result}).encode("utf-8")
        ok, http_status, response_text = self.send_handoff_put(task_id, headers, body, ts, status)
        if ok or http_status not in {404, 410}:
            return
        moved_to = self.error_moved_to(response_text)
        resolved = moved_to or self.resolve_handoff_task_id(ts, headers)
        if resolved and resolved != task_id:
            with self.state_lock:
                ts["handoff_task_id"] = resolved
                self.save_state()
            self.op(
                "handoff_task_repointed",
                f"Repointed {ts['task']} after stale board id",
                {"task": ts["task"], "old_id": task_id, "new_id": resolved, "http_status": http_status},
            )
            self.send_handoff_put(resolved, headers, body, ts, status)

    # ---- claude invocation ---------------------------------------------------
    def _timeout_for(self, role: str) -> int:
        """Per-attempt wall-clock ceiling for a claude child, chosen by role.

        Workers do the heavy lifting (context read + several ONNX builds + full
        gates), so they get a very generous ceiling; reviewers mostly re-gate a
        single candidate and write a verdict. Both fall back to the single
        `timeout_seconds`, then to a generous default. The ceiling only bites a
        *hung* attempt — a productive worker returns as soon as it finishes, so
        generosity costs nothing on the happy path. On timeout, terminate_tree
        still kills the whole process tree.
        """
        cl = self.config["claude"]
        base = int(cl.get("timeout_seconds", 14400))
        if role.startswith("reviewer"):
            return int(cl.get("reviewer_timeout_seconds", base))
        return int(cl.get("worker_timeout_seconds", base))

    def invoke_claude(self, role: str, task: str, prompt: str, session_id: str | None = None, resume: bool = False) -> dict[str, Any]:
        cl = self.config["claude"]
        stamp = utc_stamp()
        role_dir = self.raw_dir / task / role
        role_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = role_dir / f"{stamp}.json"
        stderr_path = role_dir / f"{stamp}.stderr.txt"
        final_path = role_dir / f"{stamp}.final.md"

        # Per-task model override (owner 2026-07-10 "Opus fast + Fable on hard"): a task listed in
        # claude.model_overrides (task-id -> model) uses that model (e.g. "fable" for known hard-floor
        # tasks); everyone else uses claude.model. Absent the key => unchanged.
        model = (cl.get("model_overrides") or {}).get(task) or cl.get("model", "opus")
        cmd = [cl["executable"], "-p", "--output-format", "json", "--model", model]
        if cl.get("effort"):
            cmd += ["--effort", cl["effort"]]
        # Headless fast mode is OPUS-ONLY (~2.5x, identical quality, +cost). Enable via claude.fast_mode:true
        # (merged into any claude.settings dict). Skipped for non-opus models. Absent fast_mode/settings => no --settings.
        settings = dict(cl.get("settings") or {})
        if cl.get("fast_mode") and "opus" in str(model):
            settings["fastMode"] = True
        if settings:
            cmd += ["--settings", json.dumps(settings)]
        if cl.get("skip_permissions", True):
            cmd.append("--dangerously-skip-permissions")
        for d in cl.get("extra_dirs", []):
            cmd += ["--add-dir", str(d)]
        if resume and session_id:
            cmd += ["--resume", session_id]

        if self.args.dry_run or not self.args.run_claude:
            print(f"DRY RUN claude {role} {task}: {' '.join(cmd)}")
            return {"ok": True, "dry_run": True, "session_id": session_id, "final_path": None, "stdout_path": None, "stderr_path": None, "final_text": ""}

        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", cwd=str(REPO),
        )
        try:
            out, err = proc.communicate(input=prompt, timeout=self._timeout_for(role))
        except subprocess.TimeoutExpired:
            terminate_tree(proc)
            try:
                out, err = proc.communicate(timeout=30)
            except Exception:
                out, err = "", ""
            stderr_path.write_text((err or "") + "\nTIMEOUT\n", encoding="utf-8")
            return {"ok": False, "error": "timeout", "session_id": session_id, "final_path": None, "stdout_path": None, "stderr_path": rel(stderr_path), "final_text": ""}

        stdout_path.write_text(out or "", encoding="utf-8")
        stderr_path.write_text(err or "", encoding="utf-8")
        final_text = ""
        new_session = session_id
        is_error = proc.returncode != 0
        try:
            obj = json.loads(out)
            if isinstance(obj, dict):
                final_text = obj.get("result") or ""
                new_session = obj.get("session_id") or session_id
                is_error = bool(obj.get("is_error")) or is_error
        except (json.JSONDecodeError, TypeError):
            new_session = find_session_id(out or "") or session_id
            final_text = out or ""
        final_path.write_text(final_text, encoding="utf-8")
        ok = (proc.returncode == 0) and not is_error
        return {
            "ok": ok, "returncode": proc.returncode, "session_id": new_session,
            "final_path": rel(final_path), "stdout_path": rel(stdout_path),
            "stderr_path": rel(stderr_path), "final_text": final_text,
        }

    # ---- scheduling ----------------------------------------------------------
    def ordered(self) -> list[dict[str, Any]]:
        return [self.state["tasks"][t["task"]] for t in self.config["tasks"] if t["task"] in self.state["tasks"]]

    def select_worker_task(self) -> dict[str, Any] | None:
        for status in ("forced_retry", "worker_rework", "todo"):
            for ts in self.ordered():
                if ts["status"] == status:
                    return ts
        return None

    def select_review_task(self) -> dict[str, Any] | None:
        for ts in self.ordered():
            if ts["status"] == "pending_review":
                return ts
        return None

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

    def claim_worker(
        self, busy_tasks: set[str], busy_agents: set[str]
    ) -> tuple[dict[str, Any], str] | None:
        """Atomically pick the next worker-eligible task + a free worker agent.

        Does NOT change task status; the scheduler adds the task to busy_tasks so
        re-selection skips it while run_worker is in flight. Returns None when no
        task is currently runnable with a free agent. Held under state_lock so it
        can't race a worker thread mutating the same state dict. Mirrors the codex
        lane's claim_worker but keys off the Claude task's `assignee` field.
        """
        with self.state_lock:
            available = [a for a in self.worker_agents() if a not in busy_agents]
            if not available:
                return None
            for status in ("forced_retry", "worker_rework", "todo"):
                for ts in self.ordered():
                    if ts["task"] in busy_tasks or ts["status"] != status:
                        continue
                    preferred = ts.get("assignee")
                    agent = preferred if preferred in available else available[0]
                    ts["assignee"] = agent
                    return ts, agent
        return None

    def claim_review(
        self, busy_tasks: set[str], busy_agents: set[str]
    ) -> tuple[dict[str, Any], str] | None:
        """Atomically pick the next pending_review task + a free reviewer agent.

        Same contract as claim_worker (no status change; held under state_lock),
        but keys off the Claude task's `reviewer` field.
        """
        with self.state_lock:
            available = [a for a in self.reviewer_agents() if a not in busy_agents]
            if not available:
                return None
            for ts in self.ordered():
                if ts["task"] in busy_tasks or ts["status"] != "pending_review":
                    continue
                preferred = ts.get("reviewer")
                agent = preferred if preferred in available else available[0]
                ts["reviewer"] = agent
                return ts, agent
        return None

    def run_worker(self, ts: dict[str, Any], agent: str | None = None) -> str:
        task = ts["task"]
        if agent:
            with self.state_lock:
                ts["assignee"] = agent
        prior = ""
        if ts.get("worker_last_output"):
            p = REPO / ts["worker_last_output"]
            if p.exists():
                prior = p.read_text(encoding="utf-8")
        values = {
            "task": task, "task_num": self.task_num(task), "bar": ts.get("bar", "?"),
            "worker_id": ts.get("assignee", ""),
            "prompt_path": rel(self.task_prompt_path(ts)),
            "prior_worker_packet": prior,
            "reviewer_rework": ts.get("reviewer_rework_prompt", ""),
        }
        status = ts["status"]
        if status == "todo":
            template, resume, action = "worker_initial.md", False, "initial worker pass"
        elif status == "forced_retry":
            template, resume, action = "worker_forced_retry.md", True, "forced worker retry"
        elif status == "worker_rework":
            template, resume, action = "worker_rework.md", True, "reviewer-directed rework"
        else:
            return f"No worker action for {task} ({status})"

        prompt = self.template(template, values)
        if self.args.dry_run or not self.args.run_claude:
            print(f"DRY RUN: would run {action} for {task} as {ts.get('assignee')}")
            return f"planned {action} for {task}"

        with self.state_lock:
            ts["status"] = "worker_running"
            self.save_state()
        self.update_handoff_task(ts, "in_progress", f"{action} started by claude runner")
        result = self.invoke_claude("worker", task, prompt, session_id=ts.get("worker_session_id"), resume=resume)
        with self.state_lock:
            ts["worker_session_id"] = result.get("session_id")
            ts["worker_last_output"] = result.get("final_path")
            ts["history"].append({"ts_utc": utc_now(), "role": "worker", "action": action, "result": redact(result)})
            if not result["ok"]:
                max_failures = int(self.config["workflow"].get("worker_max_failures", 2))
                failures = int(ts.get("worker_failures", 0)) + 1
                ts["worker_failures"] = failures
                if failures <= max_failures:
                    # Retry from a fresh session rather than stranding the task.
                    ts["worker_session_id"] = None
                    ts["status"] = status
                    self.save_state()
                    self.op("runner_worker_retry", f"Worker failed {task}; retry {failures}/{max_failures}", {"task": task, "error": result.get("error")})
                    return f"worker failed on {task}; retry {failures}/{max_failures} queued"
                ts["status"] = "worker_failed"
                self.save_state()
                self.op("runner_worker_failed", f"Worker failed {task} (no retries left)", {"task": task, "result": redact(result)})
                return f"worker failed on {task} (no retries left)"
            ts["worker_attempts"] = int(ts.get("worker_attempts", 0)) + 1
            required = int(self.config["workflow"]["worker_attempts_before_review"])
            if ts["worker_attempts"] < required:
                ts["status"], msg, hstatus = "forced_retry", f"attempt {ts['worker_attempts']} done; forced retry queued", "in_progress"
            else:
                ts["status"], msg, hstatus = "pending_review", f"attempt {ts['worker_attempts']} done; reviewer queued", "pending_verification"
            self.save_state()
        self.update_handoff_task(ts, hstatus, msg)
        self.op("runner_worker_complete", f"{action} complete {task}", {"task": task, "status": ts["status"]})
        return msg

    def _review_retry_or_fail(self, ts: dict[str, Any], terminal_status: str) -> str:
        """Bounded retry for non-fatal reviewer output problems (unparseable
        verdict, SHIP with a bad packet). Re-queues a few times then settles."""
        max_failures = int(self.config["workflow"].get("reviewer_max_failures", 1))
        failures = int(ts.get("review_failures", 0)) + 1
        ts["review_failures"] = failures
        if failures <= max_failures:
            ts["reviewer_session_id"] = None
            return "pending_review"
        return terminal_status

    def run_reviewer(self, ts: dict[str, Any], agent: str | None = None) -> str:
        task = ts["task"]
        if agent:
            with self.state_lock:
                ts["reviewer"] = agent
        prompt_path = self.task_prompt_path(ts)
        task_prompt = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else ""
        worker_packet = ""
        if ts.get("worker_last_output"):
            p = REPO / ts["worker_last_output"]
            if p.exists():
                worker_packet = p.read_text(encoding="utf-8")
        values = {
            "task": task, "task_num": self.task_num(task), "bar": ts.get("bar", "?"), "reviewer_id": ts.get("reviewer", ""),
            "task_prompt": task_prompt, "worker_packet": worker_packet,
            "reviewer_attack_log": ts.get("reviewer_attack_log", ""),
        }
        prompt = self.template("reviewer.md", values)
        if self.args.dry_run or not self.args.run_claude:
            print(f"DRY RUN: would run reviewer for {task} as {ts.get('reviewer')}")
            return f"planned reviewer for {task}"
        with self.state_lock:
            ts["status"] = "review_running"
            self.save_state()
        self.update_handoff_task(ts, "pending_verification", "review started by claude runner")
        result = self.invoke_claude("reviewer", task, prompt, session_id=ts.get("reviewer_session_id"), resume=bool(ts.get("reviewer_session_id")))
        with self.state_lock:
            ts["reviewer_session_id"] = result.get("session_id")
            ts["reviewer_last_output"] = result.get("final_path")
            ts["history"].append({"ts_utc": utc_now(), "role": "reviewer", "action": "review", "result": redact(result)})
            if not result["ok"]:
                max_failures = int(self.config["workflow"].get("reviewer_max_failures", 1))
                failures = int(ts.get("review_failures", 0)) + 1
                ts["review_failures"] = failures
                if failures <= max_failures:
                    ts["reviewer_session_id"] = None
                    ts["status"] = "pending_review"
                    self.save_state()
                    self.op("runner_review_retry", f"Reviewer failed {task}; retry {failures}/{max_failures}", {"task": task, "error": result.get("error")})
                    return f"reviewer failed on {task}; retry {failures}/{max_failures} queued"
                ts["status"] = "review_failed"
                self.save_state()
                self.op("runner_review_failed", f"Reviewer failed {task} (no retries left)", {"task": task, "result": redact(result)})
                return f"reviewer failed on {task} (no retries left)"
            review = extract_review_json(result.get("final_text", "")) or {}
            verdict = str(review.get("verdict", "")).strip().upper().replace(" ", "_")
            if verdict == "SHIP":
                packet_path = self.write_captain_packet(ts, review.get("captain_packet"))
                if packet_path:
                    ts["status"], msg, hstatus = "ship_packeted", f"reviewer SHIP; captain packet {rel(packet_path)}", "verified"
                    ts["captain_packet_path"] = rel(packet_path)
                else:
                    ts["status"] = self._review_retry_or_fail(ts, "needs_packet_fix")
                    msg, hstatus = "reviewer SHIP but packet missing/invalid", "pending_verification"
                ts["reviewer_attack_log"] = str(review.get("attack_log", ""))
            elif verdict == "KEEP_WORKING":
                cycles = int(ts.get("rework_cycles", 0)) + 1
                ts["rework_cycles"] = cycles
                max_cycles = int(self.config["workflow"].get("max_rework_cycles", 3))
                ts["reviewer_attack_log"] = str(review.get("attack_log", ""))
                if cycles > max_cycles:
                    ts["status"], msg, hstatus = "drop", f"reviewer KEEP_WORKING but rework cap ({max_cycles}) reached; dropping", "rejected"
                else:
                    ts["status"], msg, hstatus = "worker_rework", f"reviewer KEEP_WORKING; rework {cycles}/{max_cycles} queued", "in_progress"
                    ts["reviewer_rework_prompt"] = str(review.get("rework_prompt", ""))
            elif verdict == "DROP":
                ts["status"], msg, hstatus = "drop", "reviewer DROP", "rejected"
                ts["reviewer_attack_log"] = str(review.get("attack_log", ""))
            else:
                ts["status"] = self._review_retry_or_fail(ts, "review_parse_failed")
                msg, hstatus = "reviewer output had no parseable verdict JSON", "pending_verification"
            self.save_state()
        self.update_handoff_task(ts, hstatus, msg)
        self.op("runner_review_complete", f"Review {task}: {verdict or 'UNPARSED'}", {"task": task, "msg": msg})
        return msg

    def run_scheduler(self) -> list[str]:
        """Concurrent, self-refilling scheduler.

        Keeps up to worker_max_active workers and reviewer_max_active reviewers
        in flight, and refills a slot the instant a task finishes — so a single
        slow worker no longer holds idle slots open for the rest of the batch
        (the old serial step() ran at most one worker + one reviewer per poll).
        Workers and reviewers pipeline: a task a worker pushes to pending_review
        is picked up by a reviewer as soon as a slot frees. Drains until nothing
        is runnable, then returns. Mirrors the codex lane's run_scheduler.
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
        return messages

    def plan_pass(self) -> list[str]:
        """Single, non-mutating planning pass for --dry-run (no refill loop)."""
        role = self.args.role
        messages: list[str] = []
        if role in {"all", "reviewer"}:
            review_max = max(1, int(self.config["workflow"].get("reviewer_max_active", 1)))
            n = 0
            for ts in self.ordered():
                if ts["status"] == "pending_review" and n < review_max:
                    messages.append(self.run_reviewer(ts))
                    n += 1
        if role in {"all", "worker"}:
            worker_max = max(1, int(self.config["workflow"].get("worker_max_active", 1)))
            n = 0
            for status in ("forced_retry", "worker_rework", "todo"):
                for ts in self.ordered():
                    if ts["status"] == status and n < worker_max:
                        messages.append(self.run_worker(ts))
                        n += 1
        return messages

    def write_captain_packet(self, ts: dict[str, Any], packet: Any) -> Path | None:
        validated = validate_packet(packet if isinstance(packet, dict) else None)
        if not validated:
            return None
        packet = validated.model_dump(exclude_none=True)
        packet["task"] = ts["task"]
        packet["source_lane"] = self.lane
        packet.setdefault("reviewer_verdict", "SHIP")
        # Stamp batch identity so Captain/alpha can filter packets by batch/runner
        # (p43 packets shipped with these empty — see the win-rate action reports).
        workflow = self.config.get("workflow", {})
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
        out = self.captain_inbox / f"{utc_stamp()}-{ts['task']}.json"
        write_json(out, packet)
        self.op("runner_captain_packet", f"Wrote Submit Captain packet {ts['task']}", {"packet": rel(out)})
        return out

    def pending_packets(self) -> list[Path]:
        return [p for p in sorted(self.captain_inbox.glob("*.json")) if not read_json(p, {}).get("probe_ok")]

    def run_captain(self) -> str:
        packets = self.pending_packets()
        batch = int(self.config["workflow"]["captain_batch_size"])
        if len(packets) < batch:
            return f"captain idle: {len(packets)}/{batch} packets"
        commands = [
            [sys.executable, "Agenters/Submit_Captain/captain_ctl.py", "digest", "--lane", "claude"],
            [sys.executable, "Agenters/Submit_Captain/captain_ctl.py", "probe-status"],
        ]
        if self.args.allow_kaggle_submit:
            commands.append([sys.executable, "Agenters/Submit_Captain/captain_ctl.py"] + self.config["workflow"]["captain_safe_batch_command"])
        if self.args.dry_run or not self.args.run_claude or not self.args.allow_kaggle_submit:
            print(f"DRY RUN: captain has {len(packets)} packets")
            for c in commands:
                print("  " + " ".join(c))
            if not self.args.allow_kaggle_submit:
                print("  Kaggle-affecting command held; pass --allow-kaggle-submit when ready.")
            return f"planned captain action for {len(packets)} packets"
        stamp = utc_stamp()
        out_dir = self.raw_dir / "captain"
        out_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for idx, cmd in enumerate(commands, 1):
            proc = subprocess.run(cmd, text=True, encoding="utf-8", errors="replace", stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(REPO))
            (out_dir / f"{stamp}-{idx}.stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
            (out_dir / f"{stamp}-{idx}.stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
            records.append({"cmd": cmd, "returncode": proc.returncode})
            if proc.returncode != 0:
                self.state["captain"].update({"status": "failed", "last_records": records})
                self.save_state()
                return "captain command failed"
        self.state["captain"].update({"status": "probe_started_or_completed", "last_records": records})
        self.save_state()
        return "captain commands complete"

    def orchestrator_record(self) -> str:
        lines = [
            f"{t['task']}: {t['status']} attempts={t.get('worker_attempts', 0)} packet={t.get('captain_packet_path') or '-'}"
            for t in self.state["tasks"].values()
        ]
        summary = "\n".join(lines)
        self.op("runner_orchestrator_record", "Runner state record", {"summary": summary})
        return summary

    def reconcile_orphans(self) -> None:
        """Crash/abort recovery, run at the start of every real drain: a killed runner
        leaves tasks in the transient 'running' statuses the scheduler never re-claims
        (worker_running / review_running). Rewind them to a runnable status so a restart
        drains cleanly instead of orphaning them (2026-07-11: an aborted batch62 left
        task005 stuck worker_running). Idempotent: a cleanly-drained prior invocation
        leaves no 'running' tasks, so this no-ops.
        """
        rewound: list[str] = []
        with self.state_lock:
            for ts in self.state["tasks"].values():
                st = ts.get("status")
                if st == "worker_running":
                    ts["status"] = "todo"
                    rewound.append(f"{ts['task']}:worker_running->todo")
                elif st == "review_running":
                    ts["status"] = "pending_review"
                    rewound.append(f"{ts['task']}:review_running->pending_review")
            if rewound:
                self.save_state()
        for r in rewound:
            self.op("runner_orphan_rewind", f"rewound stale {r}", {"detail": r})

    def step(self) -> str:
        role = self.args.role
        messages: list[str] = []

        # --dry-run / planning: single non-mutating pass so it can't loop on
        # statuses that never advance without a real claude run.
        if self.args.dry_run or not self.args.run_claude:
            messages.extend(self.plan_pass())
            # In normal producer mode the local/alpha drain handles captain
            # packets; don't let a packet backlog short-circuit progress unless
            # Kaggle-affecting captain actions were enabled or the captain role
            # was explicitly requested.
            if role in {"all", "captain"} and (self.args.allow_kaggle_submit or role == "captain"):
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

        # Real run: first heal crash/abort orphans (tasks stuck in a transient
        # 'running' status the scheduler never re-claims), then drain workers +
        # reviewers concurrently with slot-refill; the captain settles whatever
        # packets accumulated (same gate as above so a backlog can't short-circuit).
        self.reconcile_orphans()
        if role in {"all", "worker", "reviewer"}:
            messages.extend(self.run_scheduler())

        if role in {"all", "captain"} and (self.args.allow_kaggle_submit or role == "captain"):
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
    p = argparse.ArgumentParser(description="Run the NeuroGolf HandOff workflow locally (Claude lane).")
    p.add_argument("--config", default="Handoff/config/runner_claude.json")
    p.add_argument("--role", choices=["all", "worker", "reviewer", "captain", "orchestrator"], default="all")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--run-claude", action="store_true")
    p.add_argument("--sync-handoff", action="store_true")
    p.add_argument("--allow-kaggle-submit", action="store_true")
    p.add_argument("--init-state", action="store_true")
    return p.parse_args()


def main() -> int:
    runner = Runner(parse_args())
    print(runner.step())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
