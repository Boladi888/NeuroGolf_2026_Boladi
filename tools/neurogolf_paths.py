"""Shared path resolution for NeuroGolf control-plane tools.

Hermes workers may run from project-local git worktrees under
`K-NeuroGolf/.worktrees/<task-id>`. Those checkouts can contain stale copies of
`.hermes/board`, so orchestration tools should default back to the canonical
project root unless the caller explicitly overrides it.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_CANONICAL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROJECT_WORKTREE_ROOT = DEFAULT_CANONICAL_ROOT / ".worktrees"
CANONICAL_ROOT_ENV = "NEUROGOLF_CANONICAL_ROOT"
WORKTREE_ROOT_ENV = "NEUROGOLF_WORKTREE_ROOT"


def _path_candidates(start: Path | None = None) -> list[Path]:
    candidates: list[Path] = []
    if start is not None:
        candidates.append(start)
    candidates.append(Path.cwd())
    candidates.append(Path(__file__).resolve())
    return candidates


def _parents_and_self(path: Path) -> list[Path]:
    path = path.resolve()
    if path.is_file():
        path = path.parent
    return [path, *path.parents]


def _root_before_project_worktrees(path: Path) -> Path | None:
    path = path.resolve()
    parts = path.parts
    for idx, part in enumerate(parts):
        if part == ".worktrees" and idx > 0:
            candidate = Path(*parts[:idx])
            if (candidate / "The_400_Tasks").exists() and (candidate / "tools").exists():
                return candidate
    return None


def resolve_canonical_root(start: Path | None = None) -> Path:
    """Return the canonical NeuroGolf root for shared control-plane state."""

    env = os.environ.get(CANONICAL_ROOT_ENV)
    if env:
        return Path(env).expanduser().resolve()

    for candidate in _path_candidates(start):
        worktree_parent = _root_before_project_worktrees(candidate)
        if worktree_parent:
            return worktree_parent

    for candidate in _path_candidates(start):
        for parent in _parents_and_self(candidate):
            if (parent / ".worktrees").exists() and (parent / "The_400_Tasks").exists():
                return parent
            if parent == DEFAULT_CANONICAL_ROOT and parent.exists():
                return parent.resolve()

    return DEFAULT_CANONICAL_ROOT.resolve()


def resolve_worktree_root() -> Path:
    env = os.environ.get(WORKTREE_ROOT_ENV)
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_PROJECT_WORKTREE_ROOT.resolve()


def board_dir(repo_root: Path | None = None) -> Path:
    root = repo_root if repo_root is not None else resolve_canonical_root()
    return root / ".hermes" / "board"
