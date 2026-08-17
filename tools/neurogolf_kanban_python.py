#!/usr/bin/env python3
"""Small Python helpers for deterministic NeuroGolf Kanban control-plane jobs."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def read_json(path: Path, fallback: Any) -> Any:
    if not path.exists():
        return fallback
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any, *, dry_run: bool = False) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def as_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def normalize_task_id(value: Any) -> str:
    text = as_text(value)
    if not text:
        return ""
    if text.startswith("task") and text[4:].isdigit():
        return f"task{int(text[4:]):03d}"
    if text.isdigit():
        return f"task{int(text):03d}"
    return text


def card_status(card: dict[str, Any]) -> str:
    return as_text(card.get("status")).lower()


def card_assignee(card: dict[str, Any]) -> str:
    return as_text(card.get("assignee"))


def card_priority(card: dict[str, Any]) -> int:
    try:
        return int(card.get("priority") or 0)
    except (TypeError, ValueError):
        return 0


def card_created_at(card: dict[str, Any]) -> int:
    try:
        return int(card.get("created_at") or 0)
    except (TypeError, ValueError):
        return 0


def card_id(card: dict[str, Any]) -> str:
    return as_text(card.get("id"))


def card_title(card: dict[str, Any]) -> str:
    return as_text(card.get("title"))


def card_body(card: dict[str, Any]) -> str:
    return as_text(card.get("body") or card.get("description") or card.get("result"))


def card_text(card: dict[str, Any]) -> str:
    parts = [
        card.get("id"),
        card.get("title"),
        card.get("body"),
        card.get("description"),
        card.get("result"),
        card.get("assignee"),
        card.get("latest_summary"),
    ]
    return "\n".join(as_text(part) for part in parts if part is not None)


def sort_cards(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(cards, key=lambda card: (-card_priority(card), card_created_at(card), as_text(card.get("id"))))


@dataclass
class KanbanClient:
    cwd: Path
    dry_run: bool = True
    actions: list[dict[str, Any]] = field(default_factory=list)

    def _run(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        proc = subprocess.run(
            args,
            cwd=str(self.cwd),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            raise SystemExit(proc.stderr or proc.stdout or f"command failed: {' '.join(args)}")
        return proc

    def load_cards(self) -> list[dict[str, Any]]:
        proc = self._run(["hermes", "kanban", "list", "--json"])
        data = json.loads(proc.stdout)
        if not isinstance(data, list):
            raise SystemExit("hermes kanban list --json did not return a list")
        return [card for card in data if isinstance(card, dict)]

    def show(self, task_id: str) -> dict[str, Any]:
        proc = self._run(["hermes", "kanban", "show", task_id, "--json"])
        data = json.loads(proc.stdout)
        if not isinstance(data, dict):
            raise SystemExit(f"hermes kanban show {task_id} did not return an object")
        return data

    def log(self, task_id: str) -> str:
        proc = self._run(["hermes", "kanban", "log", task_id])
        return proc.stdout

    def create(
        self,
        title: str,
        *,
        body: str,
        assignee: str = "",
        workspace: str = "",
        priority: int = 0,
        created_by: str = "neurogolf-python-control",
        idempotency_key: str = "",
        max_runtime: str = "",
        max_retries: int | None = None,
    ) -> str:
        action = {
            "op": "create",
            "title": title,
            "assignee": assignee,
            "priority": priority,
            "idempotency_key": idempotency_key,
        }
        self.actions.append(action)
        if self.dry_run:
            return f"dry_create_{len(self.actions)}"
        cmd = [
            "hermes",
            "kanban",
            "create",
            title,
            "--body",
            body,
            "--priority",
            str(priority),
            "--created-by",
            created_by,
            "--json",
        ]
        if assignee:
            cmd.extend(["--assignee", assignee])
        if workspace:
            cmd.extend(["--workspace", workspace])
        if idempotency_key:
            cmd.extend(["--idempotency-key", idempotency_key])
        if max_runtime:
            cmd.extend(["--max-runtime", max_runtime])
        if max_retries is not None:
            cmd.extend(["--max-retries", str(int(max_retries))])
        proc = self._run(cmd)
        data = json.loads(proc.stdout)
        task_id = as_text(data.get("id")) if isinstance(data, dict) else ""
        if not task_id:
            raise SystemExit(f"create did not return task id: {proc.stdout}")
        action["task_id"] = task_id
        return task_id

    def comment(self, task_id: str, body: str, *, author: str = "neurogolf-python-control") -> None:
        self.actions.append({"op": "comment", "task_id": task_id, "author": author, "body": body})
        if self.dry_run:
            return
        self._run(["hermes", "kanban", "comment", task_id, "--author", author, body])

    def complete(self, task_id: str, result: str) -> None:
        self.actions.append({"op": "complete", "task_id": task_id, "result": result})
        if self.dry_run:
            return
        self._run(["hermes", "kanban", "complete", task_id, "--result", result])

    def assign(self, task_id: str, assignee: str) -> None:
        self.actions.append({"op": "assign", "task_id": task_id, "assignee": assignee})
        if self.dry_run:
            return
        self._run(["hermes", "kanban", "assign", task_id, assignee or "none"])

    def reclaim(self, task_id: str, reason: str) -> None:
        self.actions.append({"op": "reclaim", "task_id": task_id, "reason": reason})
        if self.dry_run:
            return
        self._run(["hermes", "kanban", "reclaim", task_id, "--reason", reason])

    def link(self, parent_id: str, child_id: str) -> None:
        self.actions.append({"op": "link", "parent_id": parent_id, "child_id": child_id})
        if self.dry_run:
            return
        self._run(["hermes", "kanban", "link", parent_id, child_id])

    def unlink(self, parent_id: str, child_id: str) -> None:
        self.actions.append({"op": "unlink", "parent_id": parent_id, "child_id": child_id})
        if self.dry_run:
            return
        self._run(["hermes", "kanban", "unlink", parent_id, child_id])

    def block(self, task_id: str, reason: str) -> None:
        self.actions.append({"op": "block", "task_id": task_id, "reason": reason})
        if self.dry_run:
            return
        self._run(["hermes", "kanban", "block", task_id, reason])
