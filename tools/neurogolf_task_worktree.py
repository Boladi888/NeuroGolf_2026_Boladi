#!/usr/bin/env python3
"""Create or inspect isolated git worktrees for NeuroGolf Kanban tasks.

This helper is intentionally small and local-only. It does not submit to Kaggle,
does not merge, and does not delete worktrees. It gives the orchestrator/sync
manager a deterministic way to prepare a workspace before assigning a code-
writing task.

Default worktree root:
  <REPO_ROOT>/.worktrees

Examples:
  .venv/Scripts/python tools/neurogolf_task_worktree.py create \
    --name worker1-task158 --branch worker1-task158
  .venv/Scripts/python tools/neurogolf_task_worktree.py list
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

try:
    from tools.neurogolf_paths import resolve_canonical_root, resolve_worktree_root
except ModuleNotFoundError:  # pragma: no cover
    from neurogolf_paths import resolve_canonical_root, resolve_worktree_root  # type: ignore

DEFAULT_REPO_ROOT = resolve_canonical_root(Path(__file__).resolve())
DEFAULT_WORKTREE_ROOT = resolve_worktree_root()


def run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, check=False)


def slugify(value: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    if not s:
        raise SystemExit("empty/invalid worktree name")
    return s[:96]


def git_worktree_porcelain(repo_root: Path) -> str:
    proc = run(["git", "worktree", "list", "--porcelain"], repo_root)
    if proc.returncode != 0:
        raise SystemExit(proc.stderr or proc.stdout)
    return proc.stdout


def parse_worktrees(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    cur: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            if cur:
                rows.append(cur)
                cur = {}
            continue
        if " " in line:
            k, v = line.split(" ", 1)
        else:
            k, v = line, ""
        cur[k] = v
    if cur:
        rows.append(cur)
    return rows


def plan_name(assignee: str, kanban_task_id: str, target_task_id: str = "") -> str:
    parts = [slugify(assignee), slugify(kanban_task_id)]
    target = slugify(target_task_id) if target_task_id else ""
    if target:
        parts.append(target)
    return "-".join(parts)


def plan_spec(assignee: str, kanban_task_id: str, target_task_id: str = "") -> dict[str, str]:
    name = plan_name(assignee, kanban_task_id, target_task_id)
    path = (DEFAULT_WORKTREE_ROOT / name).resolve()
    return {
        "name": name,
        "branch": name,
        "path": str(path),
        "workspace": f"dir:{path}",
    }


def create(args: argparse.Namespace) -> None:
    repo_root = args.repo_root.resolve()
    root = args.worktree_root.resolve()
    name = slugify(args.name)
    branch = slugify(args.branch or name)
    path = (root / name).resolve()
    root.mkdir(parents=True, exist_ok=True)

    try:
        path.relative_to(root)
    except ValueError:
        raise SystemExit(f"resolved worktree path escaped root: {path}")

    existing = git_worktree_porcelain(repo_root)
    path_key = str(path).replace("\\", "/").lower()
    for wt in parse_worktrees(existing):
        wt_path = wt.get("worktree", "").replace("\\", "/").lower()
        if wt_path == path_key:
            print(json.dumps({"status": "exists", "path": str(path), "branch": wt.get("branch")}, indent=2))
            return

    show_ref = run(["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"], repo_root)
    if show_ref.returncode == 0:
        cmd = ["git", "worktree", "add", str(path), branch]
    else:
        cmd = ["git", "worktree", "add", "-b", branch, str(path), args.base]
    proc = run(cmd, repo_root)
    if proc.returncode != 0:
        raise SystemExit(proc.stderr or proc.stdout)
    print(json.dumps({"status": "created", "path": str(path), "branch": branch, "base": args.base}, indent=2))


def plan(args: argparse.Namespace) -> None:
    spec = plan_spec(args.assignee, args.kanban_task_id, args.target_task_id)
    print(json.dumps(spec, indent=2))


def prepare(args: argparse.Namespace) -> None:
    spec = plan_spec(args.assignee, args.kanban_task_id, args.target_task_id)
    create_args = argparse.Namespace(
        repo_root=args.repo_root,
        worktree_root=args.worktree_root,
        name=spec["name"],
        branch=spec["branch"],
        base=args.base,
    )
    create(create_args)


def list_worktrees(args: argparse.Namespace) -> None:
    rows = parse_worktrees(git_worktree_porcelain(args.repo_root.resolve()))
    print(json.dumps(rows, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--worktree-root", type=Path, default=DEFAULT_WORKTREE_ROOT)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_create = sub.add_parser("create", help="Create a task/role worktree if missing")
    p_create.add_argument("--name", required=True, help="Directory name under worktree root, e.g. worker1-task158")
    p_create.add_argument("--branch", default=None, help="Branch name; defaults to --name")
    p_create.add_argument("--base", default="HEAD", help="Base ref for new branch (default HEAD)")
    p_create.set_defaults(func=create)

    p_plan = sub.add_parser("plan", help="Emit deterministic worktree name/path for a kanban card")
    p_plan.add_argument("--assignee", required=True)
    p_plan.add_argument("--kanban-task-id", required=True)
    p_plan.add_argument("--target-task-id", default="")
    p_plan.set_defaults(func=plan)

    p_prepare = sub.add_parser("prepare", help="Create a deterministic worktree for a kanban card if missing")
    p_prepare.add_argument("--assignee", required=True)
    p_prepare.add_argument("--kanban-task-id", required=True)
    p_prepare.add_argument("--target-task-id", default="")
    p_prepare.add_argument("--base", default="HEAD", help="Base ref for new branch (default HEAD)")
    p_prepare.set_defaults(func=prepare)

    p_list = sub.add_parser("list", help="Emit git worktree list --porcelain as JSON")
    p_list.set_defaults(func=list_worktrees)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
