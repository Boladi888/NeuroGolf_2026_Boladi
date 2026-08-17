#!/usr/bin/env python3
"""Render a NeuroGolf HandOff runner config from a prompt folder.

By default this is local-only: it parses Prompt_*TaskNNN*.md files, assigns
stable external keys, and writes Handoff/config/runner_<lane>_p<N>.json. Pass
--sync-board to create-or-resolve the HandOff goal/tasks before writing config.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


REPO = Path.cwd()
TASK_RE = re.compile(r"Task(\d{1,3})(?:_|-|\.|\b)", re.I)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    tmp.replace(path)


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


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO.resolve()).as_posix()
    except ValueError:
        return str(path)


def lane_profile(lane: str, batch: str) -> dict[str, str]:
    if lane == "Claude":
        return {
            "base_config": "Handoff/config/runner_claude.json",
            "config_out": f"Handoff/config/runner_claude_p{batch}.json",
            "private_state": f"Handoff/private/runner_state_claude_p{batch}.json",
            "raw_log_dir": "Handoff/logs/raw/runner_cld",
            "operation_log": "Handoff/logs/operations_claude.jsonl",
            "source_lane": "Claude",
            "lane_id": "claude",
            "engine": "claude",
        }
    if lane == "Codex2":
        base = "Handoff/config/runner_codex2_p34.json" if (REPO / "Handoff/config/runner_codex2_p34.json").exists() else "Handoff/config/runner.json"
        return {
            "base_config": base,
            "config_out": f"Handoff/config/runner_codex2_p{batch}.json",
            "private_state": f"Handoff/private/runner_state_codex2_p{batch}.json",
            "raw_log_dir": "Handoff/logs/raw/runner_cdx2",
            "operation_log": "Handoff/logs/operations_codex2.jsonl",
            "source_lane": "Codex",
            "lane_id": "codex2",
            "engine": "codex",
        }
    return {
        "base_config": "Handoff/config/runner.json",
        "config_out": f"Handoff/config/runner_codex_p{batch}.json",
        "private_state": f"Handoff/private/runner_state_codex_p{batch}.json",
        "raw_log_dir": "Handoff/logs/raw/runner",
        "operation_log": "Handoff/logs/operations_codex.jsonl",
        "source_lane": "Codex",
        "lane_id": "codex",
        "engine": "codex",
    }


def prompt_title(path: Path) -> str:
    stem = path.stem
    match = re.search(r"Task\d{1,3}[_-](.+)$", stem, re.I)
    slug = match.group(1) if match else stem
    return re.sub(r"[_-]+", " ", slug).strip() or stem


def prompt_tasks(prompt_folder: Path, excludes: set[str]) -> list[dict[str, str]]:
    tasks: list[dict[str, str]] = []
    for path in sorted(prompt_folder.glob("*.md")):
        match = TASK_RE.search(path.name)
        if not match:
            continue
        task = f"task{int(match.group(1)):03d}"
        if task in excludes:
            continue
        tasks.append({"task": task, "title": prompt_title(path), "prompt_file": path.name})
    tasks.sort(key=lambda item: int(item["task"][4:]))
    if not tasks:
        raise SystemExit(f"no Prompt_*TaskNNN*.md files found in {prompt_folder}")
    return tasks


_UUID_RE = re.compile(r"[0-9a-fA-F-]{36}")


def _is_uuid(val: Any) -> bool:
    return isinstance(val, str) and bool(_UUID_RE.fullmatch(val))


def _iter_dicts(obj: Any):
    """Yield obj and every dict nested inside it, depth-first."""
    if isinstance(obj, dict):
        yield obj
        for val in obj.values():
            yield from _iter_dicts(val)
    elif isinstance(obj, list):
        for val in obj:
            yield from _iter_dicts(val)


def extract_goal_id(data: Any) -> str | None:
    """Goal id from a POST /goals create-or-resolve response."""
    for node in _iter_dicts(data):
        for key in ("goal_id", "id"):
            val = node.get(key)
            if _is_uuid(val):
                return val
    return None


def extract_task_id(data: Any, goal_id: str | None = None) -> str | None:
    """Task id from a POST /tasks create-or-resolve response.

    The task-create response carries BOTH the task's own id and its parent
    ``goal_id``; this must return the task id and never the goal id. So prefer
    explicit task-id keys, skip the ``goal_id`` field entirely, and reject any
    value equal to the known goal id. (Regression guarded here: the old
    goal-biased ``find_id`` checked ``goal_id`` before ``id`` and stamped every
    rendered task with the goal id instead of its own task id.)
    """
    # Preferred: explicit task-id keys, never equal to the goal id.
    for node in _iter_dicts(data):
        for key in ("task_id", "id"):
            val = node.get(key)
            if _is_uuid(val) and val != goal_id:
                return val
    # Fallback: any uuid that is neither the goal id nor under a goal_id key.
    for node in _iter_dicts(data):
        for key, val in node.items():
            if key == "goal_id":
                continue
            if _is_uuid(val) and val != goal_id:
                return val
    return None


class BoardClient:
    def __init__(self, api_base: str, agent_id: str | None = None) -> None:
        self.api_base = api_base.rstrip("/")
        env = load_env_file(REPO / ".env.handoff")
        self.headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 NeuroGolf-HandoffPrep/1.0",
        }
        if env.get("HANDOFF_BOOTSTRAP_TOKEN"):
            self.headers["Authorization"] = f"Bearer {env['HANDOFF_BOOTSTRAP_TOKEN']}"
        if agent_id:
            agent_env = load_env_file(REPO / "Handoff" / "private" / f"{agent_id}.env")
            key = agent_env.get("HANDOFF_AGENT_KEY") or agent_env.get("AGENT_KEY")
            if key:
                self.headers["X-Agent-Key"] = key
                self.headers["X-Agent-Id"] = agent_env.get("HANDOFF_AGENT_ID") or agent_id

    def request(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(f"{self.api_base}{path}", data=data, method=method, headers=self.headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                text = resp.read().decode("utf-8", errors="replace")
            return json.loads(text) if text else {}
        except urllib.error.HTTPError as exc:
            text = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{method} {path} failed HTTP {exc.code}: {text[:500]}") from exc

    def create_or_resolve_goal(self, project_id: str, title: str, external_key: str) -> str:
        body = {"title": title, "external_key": external_key}
        data = self.request("POST", f"/requests/{urllib.parse.quote(project_id, safe='')}/goals", body)
        goal_id = extract_goal_id(data)
        if not goal_id:
            raise RuntimeError(f"goal response did not contain a goal id: {json.dumps(data)[:500]}")
        return goal_id

    def create_or_resolve_task(
        self,
        project_id: str,
        goal_id: str,
        title: str,
        external_key: str,
        prompt_text: str,
    ) -> str:
        body = {"title": title, "goal_id": goal_id, "external_key": external_key, "prompt": prompt_text}
        data = self.request("POST", f"/requests/{urllib.parse.quote(project_id, safe='')}/tasks", body)
        task_id = extract_task_id(data, goal_id)
        if not task_id:
            raise RuntimeError(f"task response did not contain a task id: {json.dumps(data)[:500]}")
        if goal_id and task_id == goal_id:
            raise RuntimeError(
                f"resolved task id equals goal id {goal_id} for external_key={external_key}; "
                f"refusing to stamp the goal id onto a task. response={json.dumps(data)[:500]}"
            )
        return task_id


def existing_task_ids_by_key(config: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for task in config.get("tasks", []):
        if task.get("external_key") and task.get("handoff_task_id"):
            out[str(task["external_key"])] = str(task["handoff_task_id"])
    return out


def normalize_excludes(items: list[str]) -> set[str]:
    out: set[str] = set()
    for item in items:
        digits = re.sub(r"\D", "", item)
        if digits:
            out.add(f"task{int(digits):03d}")
    return out


def normalize_runtime_config(config: dict[str, Any], engine: str) -> None:
    if engine == "codex":
        codex = config.setdefault("codex", {})
        codex["model"] = "gpt-5.5"
        codex["reasoning_effort"] = "xhigh"
        overrides = [str(item) for item in codex.get("config_overrides", [])]
        if 'service_tier="priority"' not in overrides:
            overrides.append('service_tier="priority"')
        codex["config_overrides"] = overrides
        codex["timeout_seconds"] = int(codex.get("timeout_seconds", 14400))
        codex["worker_timeout_seconds"] = int(codex.get("worker_timeout_seconds", 14400))
        codex["reviewer_timeout_seconds"] = int(codex.get("reviewer_timeout_seconds", 10800))
        workflow = config.setdefault("workflow", {})
        workflow["worker_max_active"] = 3
    elif engine == "claude":
        cl = config.setdefault("claude", {})
        cl.setdefault("model", "opus")
        cl["effort"] = "max"
        cl["timeout_seconds"] = int(cl.get("timeout_seconds", 14400))
        cl["worker_timeout_seconds"] = int(cl.get("worker_timeout_seconds", 14400))
        cl["reviewer_timeout_seconds"] = int(cl.get("reviewer_timeout_seconds", 10800))


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a NeuroGolf HandOff batch config.")
    parser.add_argument("--lane", choices=["Codex", "Codex2", "Claude"], required=True)
    parser.add_argument("--batch", required=True, help="Batch number/tag, e.g. 35.")
    parser.add_argument("--prompt-folder", required=True)
    parser.add_argument("--base-config")
    parser.add_argument("--config-out")
    parser.add_argument("--sync-board", action="store_true", help="Create-or-resolve goal/tasks on handoff.socnet.lol.")
    parser.add_argument("--api-base", default="https://handoff.socnet.lol/api/v1")
    parser.add_argument("--agent-id", help="Requester/orchestrator agent env to use for board writes; owner bearer is also used if present.")
    parser.add_argument("--goal-title")
    parser.add_argument("--goal-external-key")
    parser.add_argument("--task-external-prefix")
    parser.add_argument("--exclude", action="append", default=[], help="Task id to skip, e.g. task123. Repeatable.")
    parser.add_argument(
        "--allow-excludes",
        action="store_true",
        help="Honor --exclude. Without this, excludes are refused so prompt folders cannot be silently thinned.",
    )
    parser.add_argument(
        "--force-all-prompts",
        action="store_true",
        help="Ignore any --exclude values and render every prompt in the folder.",
    )
    parser.add_argument("--expected-count", type=int, help="Fail unless this many prompt tasks are rendered.")
    args = parser.parse_args()

    profile = lane_profile(args.lane, str(args.batch))
    base_config = REPO / (args.base_config or profile["base_config"])
    config_out = REPO / (args.config_out or profile["config_out"])
    config = read_json(base_config)
    normalize_runtime_config(config, profile["engine"])
    prompt_folder = Path(args.prompt_folder)
    if not prompt_folder.is_absolute():
        prompt_folder = REPO / prompt_folder
    requested_excludes = normalize_excludes(args.exclude)
    if args.force_all_prompts:
        if requested_excludes:
            print(
                "notice: --force-all-prompts ignoring --exclude values: "
                + ", ".join(sorted(requested_excludes)),
                file=sys.stderr,
            )
        excludes: set[str] = set()
    elif requested_excludes and not args.allow_excludes:
        raise SystemExit(
            "refusing to render a thinned prompt folder: --exclude was provided "
            "without --allow-excludes. Remove --exclude, pass --force-all-prompts, "
            "or pass --allow-excludes for an intentional audited partial batch."
        )
    else:
        excludes = requested_excludes

    all_tasks = prompt_tasks(prompt_folder, set())
    tasks = [item for item in all_tasks if item["task"] not in excludes]
    if not tasks:
        raise SystemExit(f"all prompt tasks were excluded in {prompt_folder}")
    if args.expected_count is not None and len(tasks) != args.expected_count:
        excluded_text = ", ".join(sorted(excludes)) if excludes else "<none>"
        raise SystemExit(
            f"rendered {len(tasks)} tasks, expected {args.expected_count}; "
            f"prompt_folder_tasks={len(all_tasks)} excluded={excluded_text}"
        )

    project = config.setdefault("project", {})
    project_id = project.get("project_id") or project.get("request_id")
    if not project_id:
        raise SystemExit("base config has no project.project_id/request_id")

    lane_id = profile["lane_id"]
    batch = str(args.batch)
    goal_external_key = args.goal_external_key or f"neurogolf:batch{batch}:{lane_id}"
    goal_title = args.goal_title or f"Batch {batch} rewrite prompts ({args.lane} lane)"
    task_external_prefix = args.task_external_prefix or f"neurogolf:batch{batch}:{lane_id}"

    old_ids = existing_task_ids_by_key(config)
    client = BoardClient(args.api_base, args.agent_id) if args.sync_board else None
    goal_id = project.get("goal_id") if project.get("goal_external_key") == goal_external_key else ""
    if client:
        goal_id = client.create_or_resolve_goal(str(project_id), goal_title, goal_external_key)
    project["goal_id"] = goal_id
    project["goal_external_key"] = goal_external_key

    rendered_tasks: list[dict[str, Any]] = []
    for item in tasks:
        prompt_file = prompt_folder / item["prompt_file"]
        external_key = f"{task_external_prefix}:{item['task']}"
        task_id = old_ids.get(external_key, "")
        if client:
            task_id = client.create_or_resolve_task(
                str(project_id),
                str(goal_id),
                item["title"],
                external_key,
                prompt_file.read_text(encoding="utf-8"),
            )
        row: dict[str, Any] = {
            "task": item["task"],
            "handoff_task_id": task_id,
            "external_key": external_key,
            "title": item["title"],
            "prompt_file": item["prompt_file"],
            "source_prompt_file": rel(prompt_file),
        }
        if args.lane == "Claude":
            workers = config.get("agents", {}).get("workers") or []
            reviewers = config.get("agents", {}).get("reviewers") or []
            idx = len(rendered_tasks)
            if workers:
                row["assignee"] = workers[idx % len(workers)]
            if reviewers:
                row["reviewer"] = reviewers[idx % len(reviewers)]
        rendered_tasks.append(row)

    config["tasks"] = rendered_tasks
    config["prompt_selection"] = {
        "prompt_folder_tasks": len(all_tasks),
        "rendered_tasks": len(rendered_tasks),
        "excluded_tasks": sorted(excludes),
        "force_all_prompts": bool(args.force_all_prompts),
        "allow_excludes": bool(args.allow_excludes),
        "expected_count": args.expected_count,
    }
    config.setdefault("paths", {})
    config["paths"]["prompt_dir"] = rel(prompt_folder)
    config["paths"]["private_state"] = profile["private_state"]
    config["paths"]["raw_log_dir"] = profile["raw_log_dir"]
    config["paths"]["operation_log"] = profile["operation_log"]
    config.setdefault("workflow", {})
    config["workflow"]["lane"] = profile["source_lane"]
    config["workflow"]["lane_id"] = lane_id
    config["workflow"]["source_lane"] = profile["source_lane"]
    config["workflow"]["engine"] = profile["engine"]
    config["workflow"]["batch_id"] = batch
    config["workflow"]["runner_id"] = f"neurogolf:{lane_id}:batch{batch}"

    write_json(config_out, config)
    print(f"wrote {rel(config_out)}")
    print(f"tasks={len(rendered_tasks)} sync_board={bool(client)} goal_id={goal_id or '<not set>'}")
    if not client and any(not task["handoff_task_id"] for task in rendered_tasks):
        print("warning: some handoff_task_id values are empty; run with --sync-board before --sync-handoff", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
