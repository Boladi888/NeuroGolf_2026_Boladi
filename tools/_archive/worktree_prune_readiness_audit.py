#!/usr/bin/env python3
"""Dry-run-only helper to audit worker worktree delete-readiness.

This script is intentionally conservative: it emits reports only, never deletes.
It consolidates several checks in passes:

1) discover candidate worktree paths from filesystem + orchestrator state
2) gather metadata (git state, branch/offest, useful uncommitted files)
3) validate preservation evidence + mirror/canonical checks
4) emit JSON + markdown artifacts for human review

The output is written under the orchestrator research directory as requested by
Nikki v17: one or more timestamped checklist files plus current.md/json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


# ---------------------------
# Defaults + constants
# ---------------------------
DEFAULT_REPO_ROOT = Path(r"<REPO_ROOT>")
DEFAULT_ORCH_ROOT = Path(
    r"<NOGIT_ROOT>/orchestrator"
)
DEFAULT_WORKTREE_ROOT = Path(r"<CODEX_WORKTREES>")
CANONICAL_BEST_MANIFEST = (
    DEFAULT_ORCH_ROOT
    / "kaggle"
    / "current_best"
    / "manifest.json"
)

PRESERVED_ROOTS = [
    DEFAULT_ORCH_ROOT / "preserved" / "candidates",
    DEFAULT_ORCH_ROOT / "preserved" / "drafts",
    DEFAULT_ORCH_ROOT / "preserved" / "landed",
    DEFAULT_ORCH_ROOT / "preserved" / "failed",
]

# git status prefixes/extensions considered non-meaningful (noise)
IGNORE_STATUS_PREFIXES = (
    ".git/",
    ".pytest_cache/",
    ".mypy_cache/",
    "__pycache__/",
    ".ipynb_checkpoints/",
    ".vscode/",
    ".DS_Store",
    "Thumbs.db",
    ".coverage",
    "coverage.xml",
    "coverage.json",
    "submission_check/",
)
IGNORE_STATUS_EXTENSIONS = {
    ".tmp",
    ".swp",
    ".swo",
    ".log",
    ".orig",
    ".rej",
    ".pyc",
    ".pyo",
    ".DS_Store",
    ".coverage",
}

# If file/directory isn't this, treat as meaningful for deletion safety.
PRESERVE_OK_VERDICTS = {
    "passed_promoted",
    "passed_promoted_user_direct",
    "passed_promoted_user_direct_with_merge",
    "passed_lineage_positive_pending_integration",
    "passed_lineage_positive",
    "passed_lineage_positive_pending",
    "preserved_unsubmitted_candidate",
    "public_positive",
    "passed_promoted_from_approval",
    "public_positive_maybe",
}

STATE_KEYS_FOR_AGENT_PATH = {
    "artifact_dir",
    "landed_dir",
    "preserved_dir",
    "evidence_dir",
    "preserved_path",
    "manifest_path",
    "package_path",
    "submission_zip",
    "submission_path",
    "evidence",
    "source_package_path",
    "report",
    "worktree_path",
}

VERDICTS_ACTIVE_HINT = {"active", "assigned", "continuing", "approved_pending"}


@dataclass
class GitInfo:
    is_git: bool
    is_repo: bool
    errors: list[str] = field(default_factory=list)
    file_count: int | None = None
    path_count: int | None = None
    branch: str | None = None
    head: str | None = None
    upstream: str | None = None
    upstream_ahead: int | None = None
    upstream_behind: int | None = None
    status_line_count: int = 0
    status_meaningful_count: int = 0
    status_meaningful_examples: list[str] = field(default_factory=list)


@dataclass
class WorktreeRecord:
    path: Path
    kind: str
    agent_id: str | None = None
    owner_nickname: str | None = None
    active_assigned: bool = False
    lock_count: int = 0
    lock_tasks: list[str] = field(default_factory=list)
    current_task: str | None = None
    current_phase: str | None = None
    last_result_verdict: str | None = None
    git: dict[str, Any] = field(default_factory=dict)
    decision: str = "not_safe_yet"
    reasons: list[str] = field(default_factory=list)
    commit_refs: list[str] = field(default_factory=list)
    preserved_dirs: list[str] = field(default_factory=list)
    artifacts_mirrored: bool | None = None
    mirroring_reasons: list[str] = field(default_factory=list)

    def emit_obj(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "kind": self.kind,
            "agent_id": self.agent_id,
            "owner_nickname": self.owner_nickname,
            "active_assigned": self.active_assigned,
            "active_lock_count": self.lock_count,
            "active_lock_tasks": self.lock_tasks,
            "current_task": self.current_task,
            "current_phase": self.current_phase,
            "last_result_verdict": self.last_result_verdict,
            "decision": self.decision,
            "reasons": self.reasons,
            "commit_refs": self.commit_refs,
            "preserved_dirs": self.preserved_dirs,
            "artifacts_mirrored": self.artifacts_mirrored,
            "mirroring_reasons": self.mirroring_reasons,
            "git": self.git,
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest().upper()


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def normalize_path(value: Any) -> str | None:
    if not value:
        return None
    if isinstance(value, str):
        p = Path(value.strip()).expanduser()
        try:
            return str(p.resolve())
        except OSError:
            return str(p)
    return None


def normalize_key(value: Any) -> str | None:
    v = normalize_path(value)
    if not v:
        return None
    return str(Path(v)).replace("\\", "/").lower()


def extract_task_ids(value: Any) -> list[str]:
    if not value:
        return []
    return list({m for m in re.findall(r"task\d+", str(value).lower())})


def extract_all_ids(*values: Any) -> list[str]:
    ids: set[str] = set()
    for v in values:
        ids.update(extract_task_ids(v))
    return sorted(ids)


def run_git(cmd: list[str], cwd: Path) -> tuple[int, str, str]:
    p = subprocess.run(
        ["git", "-C", str(cwd)] + cmd,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def parse_status_meaningful(
    porcelain: str,
    ignore_prefixes: tuple[str, ...],
    ignore_extensions: set[str],
) -> tuple[int, list[str]]:
    lines = [line for line in porcelain.splitlines() if line.strip()]
    meaningful: list[str] = []
    for line in lines:
        if len(line) < 3:
            continue
        path_field = line[3:]
        # rename entries can be A -> B
        candidates = [p.strip() for p in path_field.split(" -> ")]
        skip = False
        for p_raw in candidates:
            p = p_raw.replace("\\", "/").lower()
            if any(p.startswith(prefix) for prefix in ignore_prefixes):
                skip = True
                break
            if any(p.endswith(ext) for ext in ignore_extensions):
                skip = True
                break
        if skip:
            continue
        meaningful.append(line)
    return len(meaningful), meaningful


def safe_file_uses(path: Path) -> bool:
    """Best-effort check for "safe-to-delete" non-git shells.

    Consider a non-git folder potentially safe if it contains no files and no
    potentially useful top-level content after applying ignore filters.
    """
    for child in path.iterdir():
        rel = child.name.lower()
        if rel.startswith("."):
            # Allow temp dotfiles to remain harmlessly ignored for now.
            continue
        return True
    return False


def collect_non_git_signal(path: Path) -> dict[str, Any]:
    # Lightweight scan: identify whether non-git path contains any potentially useful files
    relevant_count = 0
    sample: list[str] = []
    if not path.exists():
        return {"is_file_tree": False}

    for root, dirs, files in os.walk(path):
        root_path = Path(root)
        rel_root = root_path.relative_to(path).as_posix()
        # prune common caches
        prune = []
        for i, d in enumerate(list(dirs)):
            dl = d.lower()
            if any(
                rel_root == "."
                and dl in {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".vscode", "submission_check"}
            ):
                prune.append(d)
        for p in prune:
            dirs.remove(p)

        for f in files:
            rel = (root_path / f).relative_to(path).as_posix().replace("\\", "/")
            low = rel.lower()
            if any(low.startswith(prefix) for prefix in IGNORE_STATUS_PREFIXES):
                continue
            if any(low.endswith(ext) for ext in IGNORE_STATUS_EXTENSIONS):
                continue
            # ignore tiny lockfiles and python cache-like helpers
            if f.startswith(".") and f.endswith(".tmp"):
                continue
            relevant_count += 1
            if len(sample) < 12:
                sample.append(rel)
        if len(sample) >= 12 and relevant_count > 12:
            continue
    return {"file_count": relevant_count, "samples": sample}


def get_git_metadata(path: Path) -> GitInfo:
    rc, _, err = run_git(["rev-parse", "--is-inside-work-tree"], path)
    if rc != 0:
        ng = collect_non_git_signal(path)
        return GitInfo(
            is_git=False,
            is_repo=False,
            errors=[err.strip()] if err else [],
            file_count=ng.get("file_count", 0),
            path_count=len(list(path.iterdir())) if path.exists() else 0,
        )

    info = GitInfo(is_git=True, is_repo=True)
    rc, out, _ = run_git(["rev-parse", "--abbrev-ref", "HEAD"], path)
    info.branch = out if rc == 0 else "UNKNOWN"

    rc, out, _ = run_git(["rev-parse", "HEAD"], path)
    info.head = out if rc == 0 else None
    if info.head:
        info.commit_refs.append(info.head[:11])

    rc, out, _ = run_git(["rev-parse", "--abbrev-ref", "@{u}"], path)
    upstream = out if rc == 0 and out != "@{u}" else None
    info.upstream = upstream
    if upstream:
        if rc1 := run_git(["rev-list", "--count", f"{upstream}..HEAD"], path)[1]:
            try:
                info.upstream_ahead = int(rc1)
            except ValueError:
                info.upstream_ahead = None
        else:
            info.upstream_ahead = 0
        if rc2 := run_git(["rev-list", "--count", f"HEAD..{upstream}"], path)[1]:
            try:
                info.upstream_behind = int(rc2)
            except ValueError:
                info.upstream_behind = None
        else:
            info.upstream_behind = 0

    rc, out, _ = run_git(["status", "--porcelain=v1"], path)
    if rc == 0:
        meaningful_count, meaningful_examples = parse_status_meaningful(
            out,
            IGNORE_STATUS_PREFIXES,
            IGNORE_STATUS_EXTENSIONS,
        )
        info.status_line_count = len(out.splitlines())
        info.status_meaningful_count = meaningful_count
        info.status_meaningful_examples = meaningful_examples[:12]
        if info.upstream_ahead and info.upstream_ahead > 0:
            if rch := run_git(["rev-list", f"--max-count=1", f"{upstream}..HEAD"], path)[1]:
                # keep concise upstream refs
                pass
    return info


def build_artifact_index(preserved_roots: list[Path] | None = None) -> dict[str, list[str]]:
    index: defaultdict[str, list[str]] = defaultdict(list)
    roots = preserved_roots if preserved_roots is not None else PRESERVED_ROOTS
    for root in roots:
        if not root.exists():
            continue
        for d in root.iterdir():
            if not d.is_dir():
                continue
            key_base = d.name.lower()
            index[key_base].append(str(d))
            for m in re.findall(r"task\d+", key_base):
                index[m].append(str(d))
    return index


def collect_state_path_evidence(agent: dict[str, Any], keyset: Iterable[str]) -> list[str]:
    out: list[str] = []
    for k in keyset:
        v = agent.get(k)
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, str):
                    out.append(item)
    return [x for x in out if x]


def collect_agent_evidence(agent: dict[str, Any], preserved_index: dict[str, list[str]]) -> tuple[list[str], list[str]]:
    candidate_ids: set[str] = set()
    preserved_dirs: list[str] = []
    package_paths: list[str] = []
    verdict = None

    current_task = agent.get("current_task")
    candidate_ids.update(extract_task_ids(current_task))
    if isinstance(agent.get("task_id"), str):
        candidate_ids.update(extract_task_ids(agent.get("task_id")))

    for key in ("last_candidate", "last_result", "attempts_on_this_task"):
        blob = agent.get(key)
        if isinstance(blob, dict):
            candidate_ids.update(extract_task_ids(blob.get("task_id")))
            candidate_ids.update(extract_task_ids(blob.get("candidate_id")))
            for p in collect_state_path_evidence(
                blob,
                [
                    "artifact_dir",
                    "landed_dir",
                    "preserved_dir",
                    "evidence_dir",
                    "manifest_path",
                    "package_path",
                    "submission_zip",
                    "package",
                    "source_package_path",
                    "evidence",
                ],
            ):
                package_paths.append(p)
            if isinstance(blob.get("package_path"), str):
                package_paths.append(blob["package_path"])
            if isinstance(blob.get("candidate_id"), str):
                cand = blob["candidate_id"].lower()
                candidate_ids.add(cand)
                for tok in [cand] + re.findall(r"task\d+", cand):
                    preserved_dirs.extend(preserved_index.get(tok, []))
            if not verdict and isinstance(blob.get("verdict"), str):
                verdict = blob.get("verdict")
        elif isinstance(blob, list):
            for entry in blob:
                if isinstance(entry, dict):
                    candidate_ids.update(extract_task_ids(entry.get("task_id")))
                    candidate_ids.update(extract_task_ids(entry.get("candidate_id")))
                    package_paths.extend(
                        collect_state_path_evidence(
                            entry,
                            [
                                "artifact_dir",
                                "landed_dir",
                                "preserved_dir",
                                "evidence_dir",
                                "manifest_path",
                                "package_path",
                                "submission_path",
                                "report",
                                "path",
                            ],
                        )
                    )
                    if isinstance(entry.get("candidate_id"), str):
                        cand = str(entry["candidate_id"]).lower()
                        for tok in [cand] + re.findall(r"task\d+", cand):
                            preserved_dirs.extend(preserved_index.get(tok, []))
                    if isinstance(entry.get("task_id"), str):
                        for tok in re.findall(r"task\d+", str(entry["task_id"])):
                            preserved_dirs.extend(preserved_index.get(tok, []))

    # include explicit fields on agent root
    for v in collect_state_path_evidence(
        agent,
        [
            "worktree_path",
            "manifest_path",
            "last_status_file",
        ],
    ):
        package_paths.append(v)

    # merge with explicit candidate evidence index
    for tok in candidate_ids:
        preserved_dirs.extend(preserved_index.get(tok, []))

    # de-dupe while keeping order-ish
    norm_seen: set[str] = set()
    pres_uniq: list[str] = []
    for p in package_paths:
        n = normalize_key(p)
        if not n:
            continue
        if p not in norm_seen:
            pres_uniq.append(p)
            norm_seen.add(n)

    norm_seen.clear()
    path_uniq: list[str] = []
    for p in preserved_dirs:
        n = normalize_key(p)
        if not n:
            continue
        if n not in norm_seen:
            path_uniq.append(p)
            norm_seen.add(n)

    # if no explicit candidate mapping exists, maybe use evidence artifacts for candidate ids from current/last fields
    if not path_uniq and current_task:
        # fallback: include task directory under worktree evidence if exists
        pass

    return sorted(set(extract_task_ids(current_task))), path_uniq[:20], pres_uniq[:20], verdict


def mirror_hash_set(state: dict[str, Any], manifest: dict[str, Any]) -> set[str]:
    hashes: set[str] = set()
    candidate = state.get("canonical_best_package", {})
    direct_paths = []
    for entry in (
        candidate.get("orchestrator_path"),
        candidate.get("path"),
        candidate.get("package_path"),
        candidate.get("submission_path"),
        candidate.get("repo_package_path"),
        candidate.get("manifest_path"),
    ):
        if entry:
            direct_paths.append(entry)

    for entry in (
        manifest.get("orchestrator_package_path"),
        manifest.get("repo_package_path"),
        manifest.get("source_package_path"),
    ):
        if entry:
            direct_paths.append(entry)

    # common defaults
    direct_paths.extend(
        [
            DEFAULT_REPO_ROOT / "submissions" / "submission.zip",
            DEFAULT_ORCH_ROOT / "kaggle" / "current_best" / "submission.zip",
        ]
    )

    for p in direct_paths:
        hp = sha256_file(Path(p))
        if hp:
            hashes.add(hp)

    # include current_best manifest's explicit package hash if present
    if manifest and isinstance(manifest.get("package_sha256"), str):
        hashes.add(str(manifest["package_sha256"]).upper())
    if isinstance(candidate.get("package_sha256"), str):
        hashes.add(str(candidate.get("package_sha256")).upper())
    return hashes


def is_preserved_ok(
    evidence_paths: list[str],
    state: dict[str, Any],
    last_verdict: str | None,
    task_ids: list[str],
    agent_has_context: bool,
    allow_missing_candidate_evidence: bool = False,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []

    # If no context and no artifacts, don't hard-block:
    if not task_ids and not evidence_paths and not agent_has_context:
        return True, ["no current task context in state"]

    if allow_missing_candidate_evidence and not task_ids and not evidence_paths:
        return True, ["no current task context in state (explicitly allowed)"]

    exists = 0
    hits: list[str] = []
    for p in evidence_paths:
        pth = Path(p)
        if pth.exists():
            exists += 1
            hits.append(str(pth))
            continue
        # candidate/archive names may be absolute style; check as string alias with /\ normalized
        for alt in (str(p).replace("/", "\\"), str(p).replace("\\", "/")):
            if Path(alt).exists():
                exists += 1
                hits.append(alt)
                break
    if hits:
        reasons.append(f"preservation entries found: {len(hits)}")

    if last_verdict in PRESERVE_OK_VERDICTS:
        reasons.append(f"verdict allows preserved state ({last_verdict})")
        return True, reasons

    if exists > 0:
        return True, reasons

    if task_ids:
        reasons.append("no required preserved candidate/draft/failed evidence found")
    return False, reasons


def mirror_status(
    package_candidates: list[str],
    mirror_hashes: set[str],
) -> tuple[bool | None, list[str]]:
    if not package_candidates:
        return None, ["no package candidate path available"]

    for p in package_candidates:
        pth = Path(p)
        if pth.is_dir():
            candidate = pth / "submission.zip"
            if candidate.exists():
                h = sha256_file(candidate)
                if h and h in mirror_hashes:
                    return True, [f"mirrored: {candidate}", f"sha256={h}"]
            continue
        if pth.is_file() and pth.name.lower().endswith(".zip"):
            h = sha256_file(pth)
            if h and h in mirror_hashes:
                return True, [f"mirrored: {pth}", f"sha256={h}"]

    return False, ["package artifact hash not found in canonical/current-best mirrors"]


def index_active_locks(state: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    by_agent: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for lock_map in (state.get("active_infrastructure_task_locks", {}), state.get("active_locks", {})):
        if not isinstance(lock_map, dict):
            continue
        for task_id, lock in lock_map.items():
            if not isinstance(lock, dict):
                continue
            aid = lock.get("agent_id")
            if not aid:
                continue
            rec = {
                "task_id": task_id,
                "directive_id": lock.get("directive_id"),
                "reason": lock.get("validation_status") or lock.get("type") or lock.get("kind"),
            }
            by_agent[aid].append(rec)
    return by_agent


def discover_worktrees(
    state: dict[str, Any],
    preserved_index: dict[str, list[str]],
    mirror_hashes: set[str],
    worktree_root: Path,
    allow_missing_candidate_evidence: bool = False,
) -> tuple[list[WorktreeRecord], list[WorktreeRecord]]:
    agents = state.get("agents", {}) if isinstance(state.get("agents"), dict) else {}
    active_locks = index_active_locks(state)

    agent_by_worktree: dict[str, str] = {}
    for aid, a in agents.items():
        norm = normalize_key(a.get("worktree_path"))
        if norm:
            agent_by_worktree[norm] = aid

    safe_list: list[WorktreeRecord] = []
    not_safe: list[WorktreeRecord] = []

    # Pass A: discovered on-disk worktrees
    if not worktree_root.exists():
        return [], []

    for item in sorted(worktree_root.iterdir()):
        if not item.is_dir():
            continue
        candidate = item / "K-NeuroGolf"
        if not candidate.exists():
            if list(item.iterdir()):
                # shell dir without expected subtree and no K-NeuroGym marker
                reason = "worktree mount exists but missing K-NeuroGolf/ subtree"
                wr = WorktreeRecord(
                    path=item,
                    kind="missing-subtree",
                    reasons=[reason],
                    active_assigned=False,
                )
                not_safe.append(wr)
            continue

        norm = normalize_key(str(candidate))
        aid = agent_by_worktree.get(norm)
        a = agents.get(aid, {}) if aid else {}
        current_task = a.get("current_task") if isinstance(a, dict) else None
        owner = a.get("owner_nickname") if isinstance(a, dict) else None

        locks = active_locks.get(aid, []) if aid else []
        active = bool(locks) or bool(current_task)

        git = get_git_metadata(candidate)
        wr = WorktreeRecord(
            path=candidate,
            kind="observed",
            agent_id=aid,
            owner_nickname=owner,
            active_assigned=active,
            current_task=current_task,
            current_phase=a.get("last_phase") if isinstance(a, dict) else None,
            last_result_verdict=(
                a.get("last_result", {}).get("verdict") if isinstance(a, dict) and isinstance(a.get("last_result"), dict) else None
            ),
            git=asdict(git),
            lock_count=len(locks),
            lock_tasks=[x.get("task_id") for x in locks if isinstance(x, dict)],
        )

        if git.is_git:
            if git.branch and git.branch != "UNKNOWN":
                wr.commit_refs.append(git.branch)
            if git.head:
                wr.commit_refs.append(git.head[:10])
            if git.upstream:
                wr.commit_refs.append(f"upstream:{git.upstream}")

        task_ids, preserved_dirs, package_paths, verdict = collect_agent_evidence(
            a,
            preserved_index,
        ) if isinstance(a, dict) else ([], [], [], None)
        wr.preserved_dirs = preserved_dirs

        preserved_ok, preserve_reasons = is_preserved_ok(
            preserved_dirs + package_paths,
            state,
            verdict,
            task_ids,
            bool(task_ids or a),
        )
        wr.reasons.extend(preserve_reasons)

        # Mirror checks
        mirrored, mirror_reasons = mirror_status(package_paths, mirror_hashes)
        wr.artifacts_mirrored = mirrored
        wr.mirroring_reasons = mirror_reasons

        if not git.is_repo:
            if not git.file_count:
                wr.reasons.append("non-git empty worktree shell")
            else:
                wr.reasons.append("non-git with potentially useful workspace files")
                if git.file_count and git.path_count:
                    wr.reasons.append(
                        f"non-git top-level files={git.file_count}; subpaths={git.path_count}"
                    )
        else:
            if git.status_meaningful_count > 0:
                wr.reasons.append(
                    f"meaningful uncommitted or untracked files ({git.status_meaningful_count})"
                )
                for line in git.status_meaningful_examples[:6]:
                    wr.reasons.append(f"status: {line}")
            if git.upstream_ahead and git.upstream_ahead > 0:
                wr.reasons.append(f"local branch has {git.upstream_ahead} commits ahead of upstream")
            if git.upstream_behind and git.upstream_behind > 0:
                wr.reasons.append(
                    f"local branch is {git.upstream_behind} commits behind upstream"
                )
            if not preserved_ok and verdict not in PRESERVE_OK_VERDICTS:
                wr.reasons.append("required preserved/candidate evidence not confirmed")
            if mirrored is False:
                wr.reasons.extend(mirror_reasons)

        if active:
            wr.reasons.append("agent has active assignment/lock in state")

        if not wr.reasons:
            wr.decision = "safe_to_delete"
            safe_list.append(wr)
        else:
            wr.decision = "not_safe_yet"
            not_safe.append(wr)

    # Pass B: mapped state paths absent on disk
    known_worktree_paths = {
        normalize_key(str(i.path)) for i in safe_list + not_safe
    }

    for aid, agent in agents.items():
        p = normalize_path(agent.get("worktree_path"))
        if not p:
            continue
        norm = normalize_key(p)
        if norm in known_worktree_paths:
            continue
        # missing path entirely
        state_task = agent.get("current_task")
        owner = agent.get("owner_nickname") or aid
        locks = active_locks.get(aid, [])
        active = bool(locks) or bool(state_task) or aid in {
            "Lily",
            "Valerie",
            "codex-68ea",
        }

        # preserve evidence pass (for retired/outbound workers)
        task_ids, preserved_dirs, package_paths, verdict = collect_agent_evidence(
            agent,
            preserved_index,
        )
        wr = WorktreeRecord(
            path=Path(p),
            kind="mapped-missing",
            agent_id=aid,
            owner_nickname=owner,
            active_assigned=active,
            current_task=state_task,
            current_phase=agent.get("last_phase"),
            last_result_verdict=agent.get("last_result", {}).get("verdict") if isinstance(agent.get("last_result"), dict) else None,
            preserved_dirs=preserved_dirs,
            lock_count=len(locks),
            lock_tasks=[x.get("task_id") for x in locks if isinstance(x, dict)],
        )

        preserved_ok, preserve_reasons = is_preserved_ok(
            preserved_dirs + package_paths,
            state,
            wr.last_result_verdict,
            task_ids,
            bool(task_ids or agent),
        )
        mirrored, mirror_reasons = mirror_status(package_paths, mirror_hashes)
        wr.artifacts_mirrored = mirrored
        wr.mirroring_reasons = mirror_reasons

        wr.reasons.append("mapped worktree path missing from filesystem")
        wr.reasons.extend(preserve_reasons)

        if not package_paths:
            wr.reasons.append("no package evidence candidates were discoverable for mirrored provenance")
        elif mirrored is False:
            wr.reasons.extend(mirror_reasons)

        has_git_or_provenance = bool(wr.commit_refs)
        # commit/branch metadata cannot be collected once path is missing;
        # capture that as explicit rationale for strict reviewability.
        if not has_git_or_provenance and not package_paths:
            wr.reasons.append("commit/branch provenance unavailable: mapped worktree path missing")
        elif not has_git_or_provenance and package_paths:
            wr.reasons.append("commit/branch provenance unavailable (path missing on disk)")

        if active:
            wr.reasons.append("agent has active assignment/lock in state")
            wr.decision = "not_safe_yet"
            not_safe.append(wr)
        elif not preserved_ok:
            wr.decision = "not_safe_yet"
            not_safe.append(wr)
        elif package_paths and mirrored is not True:
            wr.reasons.append("mirror/canonical evidence unavailable for mapped package evidence")
            wr.decision = "not_safe_yet"
            not_safe.append(wr)
        else:
            wr.decision = "safe_to_delete"
            safe_list.append(wr)

    # Deduplicate by normalized path
    def dedupe(items: list[WorktreeRecord]) -> list[WorktreeRecord]:
        out = []
        seen = set()
        for r in items:
            n = normalize_key(str(r.path))
            if not n:
                continue
            if n in seen:
                continue
            seen.add(n)
            out.append(r)
        return out

    return dedupe(safe_list), dedupe(not_safe)


def markdown_report(
    state: dict[str, Any],
    safe: list[WorktreeRecord],
    not_safe: list[WorktreeRecord],
    summary: dict[str, Any],
    ts: str,
    policy_notes: list[str],
) -> str:
    out = []
    out.append("# Worktree delete-readiness audit")
    out.append(f"Generated: {ts}")
    out.append("")
    out.append("Current policy: dry-run only. Mark safe only when active assignment is absent and no residual useful state remains.")
    out.append("")
    out.append(f"- Safe to delete: {len(safe)}")
    out.append(f"- Not safe yet: {len(not_safe)}")
    out.append(f"- Known agents in state: {summary['total_agents_in_state']}")
    out.append(f"- Active locks seen: {summary['active_locks']}")
    out.append("")
    if policy_notes:
        out.append("## Notes")
        for note in policy_notes:
            out.append(f"- {note}")
        out.append("")

    out.append("## Safe to delete")
    out.append("")
    if not safe:
        out.append("- None")
    else:
        for item in sorted(safe, key=lambda r: str(r.path).lower()):
            owner = item.owner_nickname or item.agent_id or "(unknown)"
            out.append(f"- `{item.path}` agent={owner}")
            if item.commit_refs:
                out.append(f"  - commits: {', '.join(item.commit_refs)}")
            if item.git.get("branch"):
                out.append(f"  - branch: {item.git.get('branch')} upstream={item.git.get('upstream') or 'none'}")
            if item.preserved_dirs:
                out.append("  - preserved evidence: " + "; ".join(item.preserved_dirs[:2]))
            if item.artifacts_mirrored is not None:
                out.append(f"  - mirrored: {item.artifacts_mirrored}")

    out.append("")
    out.append("## Not safe yet")
    out.append("")
    if not not_safe:
        out.append("- None")
    else:
        for item in sorted(not_safe, key=lambda r: str(r.path).lower()):
            owner = item.owner_nickname or item.agent_id or "(unknown)"
            active = "ACTIVE" if item.active_assigned else "INACTIVE"
            out.append(f"- `{item.path}` ({active}) agent={owner}")
            if item.current_task:
                out.append(f"  - current_task: {item.current_task}")
            if item.current_phase:
                out.append(f"  - current_phase: {item.current_phase}")
            if item.lock_count:
                out.append(f"  - active_locks: {item.lock_count}")
            for r in item.reasons:
                out.append(f"  - {r}")
    return "\n".join(out)


def serialize_records(items: list[WorktreeRecord]) -> list[dict[str, Any]]:
    return [i.emit_obj() for i in sorted(items, key=lambda x: str(x.path).lower())]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--orchestrator-root", type=Path, default=DEFAULT_ORCH_ROOT)
    parser.add_argument("--worktree-root", type=Path, default=DEFAULT_WORKTREE_ROOT)
    parser.add_argument(
        "--state-json",
        type=Path,
        default=None,
        help="orchestrator state.json (defaults to <orchestrator-root>/state.json)",
    )
    parser.add_argument(
        "--current-best-manifest",
        type=Path,
        default=None,
        help="canonical current_best manifest JSON",
    )
    parser.add_argument(
        "--allow-missing-candidate-evidence",
        action="store_true",
        help="Allow no-context shell worktrees to be marked safe when no state/agent context exists",
    )
    args = parser.parse_args()

    repo_root = args.repo_root
    orch_root = args.orchestrator_root
    state_json = args.state_json or (orch_root / "state.json")
    manifest_path = args.current_best_manifest or (orch_root / "kaggle" / "current_best" / "manifest.json")
    worktree_root = args.worktree_root

    state = read_json(state_json)
    if not isinstance(state, dict):
        raise RuntimeError(f"Could not load state.json from {state_json}")

    preserved_roots = [
        orch_root / "preserved" / "candidates",
        orch_root / "preserved" / "drafts",
        orch_root / "preserved" / "landed",
        orch_root / "preserved" / "failed",
    ]

    preserved_index = build_artifact_index(preserved_roots)
    manifest = read_json(manifest_path) or {}
    mirror_hashes = mirror_hash_set(state, manifest)

    safe, not_safe = discover_worktrees(
        state,
        preserved_index,
        mirror_hashes,
        worktree_root,
        args.allow_missing_candidate_evidence,
    )

    ts = utc_now()
    canonical_best = state.get("canonical_best_package", {}) if isinstance(state.get("canonical_best_package"), dict) else {}

    summary = {
        "generated_from": str(state_json),
        "canonical_best_package": canonical_best.get("candidate_id"),
        "canonical_best_score": canonical_best.get("score"),
        "active_locks": len(state.get("active_locks", {})) + len(state.get("active_infrastructure_task_locks", {})),
        "active_task_locks": list(state.get("active_locks", {}).keys()),
        "total_agents_in_state": len(state.get("agents", {})) if isinstance(state.get("agents"), dict) else 0,
        "safe_to_delete": len(safe),
        "not_safe_yet": len(not_safe),
        "preserved_index_entries": sum(len(v) for v in preserved_index.values()),
        "mirror_reference_hashes": len(mirror_hashes),
    }

    report_root = orch_root / "research"
    report_root.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": ts,
        "source_paths": {
            "repo_root": str(repo_root),
            "orchestrator_root": str(orch_root),
            "state_json": str(state_json),
            "worktree_root": str(worktree_root),
            "kaggle_current_best_manifest": str(manifest_path),
        },
        "policy": {
            "dry_run": True,
            "safe_only_when_no_active_assignment": True,
            "required_preservation_checks": "preserved candidate/drafts/failed or explicit evidence path",
            "meaningful_change_filters": {
                "ignore_prefixes": list(IGNORE_STATUS_PREFIXES),
                "ignore_extensions": sorted(IGNORE_STATUS_EXTENSIONS),
            },
            "commits_must_be_pushed": True,
            "can_be_mirrored_to_current_best": True,
        },
        "summary": summary,
        "safe_to_delete": serialize_records(safe),
        "not_safe_yet": serialize_records(not_safe),
    }

    notes = [
        "No deletions were performed; outputs are audit-only.",
        "Conservative mode: any active lock/task, uncommitted meaningful changes, unpushed commits, or missing preservation evidence blocks deletion.",
    ]
    if summary["canonical_best_package"]:
        notes.append(f"Canonical best from state: {summary['canonical_best_package']}")
    if summary["active_locks"]:
        notes.append("Active orchestrator locks were detected; safe deletions require no active lock/assignment for the worktree owner.")

    ts_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_json = report_root / f"worktree_prune_readiness_audit_{ts_id}.json"
    out_md = report_root / f"worktree_prune_readiness_audit_{ts_id}.md"

    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    out_md.write_text(
        markdown_report(state, safe, not_safe, summary, ts, notes) + "\n",
        encoding="utf-8",
    )

    current = report_root / "worktree_prune_readiness_current.json"
    current_md = report_root / "worktree_prune_readiness_current.md"
    current.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    current_md.write_text(
        markdown_report(state, safe, not_safe, summary, ts, notes) + "\n",
        encoding="utf-8",
    )

    # Backward compatibility: keep previous filename for existing dashboards/readers
    latest_ts = out_json.name
    print(json.dumps(
        {
            "status": "completed",
            "generated_at": ts,
            "json": str(out_json),
            "markdown": str(out_md),
            "safe_to_delete": len(safe),
            "not_safe_yet": len(not_safe),
            "state_path": str(state_json),
            "artifact_manifest": latest_ts,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
