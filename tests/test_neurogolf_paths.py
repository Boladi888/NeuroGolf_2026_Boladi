from pathlib import Path

from tools.neurogolf_paths import resolve_canonical_root, resolve_worktree_root


def test_resolve_canonical_root_from_project_local_worktree(tmp_path: Path, monkeypatch):
    root = tmp_path / "K-NeuroGolf"
    (root / "tools").mkdir(parents=True)
    (root / "The_400_Tasks").mkdir()
    worktree = root / ".worktrees" / "t_example"
    (worktree / "tools").mkdir(parents=True)
    (worktree / "The_400_Tasks").mkdir()
    monkeypatch.delenv("NEUROGOLF_CANONICAL_ROOT", raising=False)

    assert resolve_canonical_root(worktree / "tools" / "neurogolf_preflight.py") == root.resolve()


def test_resolve_canonical_root_env_override(tmp_path: Path, monkeypatch):
    root = tmp_path / "canonical"
    root.mkdir()
    monkeypatch.setenv("NEUROGOLF_CANONICAL_ROOT", str(root))

    assert resolve_canonical_root(tmp_path / "other" / ".worktrees" / "x") == root.resolve()


def test_default_worktree_root_is_project_local(monkeypatch):
    monkeypatch.delenv("NEUROGOLF_WORKTREE_ROOT", raising=False)

    assert resolve_worktree_root().as_posix().endswith("K-NeuroGolf/.worktrees")
