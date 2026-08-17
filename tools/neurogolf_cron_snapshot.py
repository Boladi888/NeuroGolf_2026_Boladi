#!/usr/bin/env python3
"""Read-only NeuroGolf cron snapshot for fast Hermes watchdog jobs.

Hermes cron can inject script stdout into an agent prompt, or run a script in
`--no-agent` mode. This helper gives each cron job a compact live state packet
so the prompt can stay narrow and guard-heavy.

Mode is inferred from the script filename when no explicit `--mode` is passed,
which lets the same file be copied into ~/.hermes/scripts under several names.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

CANONICAL_REPO_ROOT = Path(os.environ.get("NEUROGOLF_CANONICAL_ROOT", Path(__file__).resolve().parent.parent))
if CANONICAL_REPO_ROOT.exists():
    sys.path.insert(0, str(CANONICAL_REPO_ROOT))

try:
    from tools.neurogolf_board_policy import ACTIVE_CARD_STATUSES, DEFAULT_QUEUE_WORKERS, audit_worker_queue
    from tools.neurogolf_control_config import primary_reviewer_profile, reviewer_profiles, specialist_profiles
    from tools.neurogolf_paths import board_dir, resolve_canonical_root
except ModuleNotFoundError:  # pragma: no cover
    from neurogolf_board_policy import ACTIVE_CARD_STATUSES, DEFAULT_QUEUE_WORKERS, audit_worker_queue  # type: ignore
    from neurogolf_control_config import primary_reviewer_profile, reviewer_profiles, specialist_profiles  # type: ignore
    from neurogolf_paths import board_dir, resolve_canonical_root  # type: ignore


REPO_ROOT = resolve_canonical_root(Path(__file__).resolve())
BOARD_DIR = board_dir(REPO_ROOT)
SUBMIT_QUEUE = BOARD_DIR / "submit_queue.json"
CANDIDATE_REGISTRY = BOARD_DIR / "candidate_registry.json"
TASK_OUTCOME_REGISTRY = BOARD_DIR / "task_outcome_registry.json"
TASK_STATE = REPO_ROOT / "The_400_Tasks" / "task_state.csv"
TASK_STATUS_XLSX = REPO_ROOT / "The_400_Tasks" / "NeuroGolf_Task_Status.xlsx"
TASK_BACKLOG_CURSOR = BOARD_DIR / "task_backlog_cursor.json"
MANIFEST = REPO_ROOT / "submissions" / "current_best_manifest.json"
PACKAGE = REPO_ROOT / "submissions" / "submission.zip"
SUBMIT_RUNTIME_CONFIG = BOARD_DIR / "submit_runtime_config.json"
SPECIALISTS = specialist_profiles()


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    # `text=True` without an explicit encoding defaults to locale.getpreferredencoding(),
    # which on Windows is cp1252. Kanban output regularly contains UTF-8 bytes (emoji,
    # em-dashes, curly quotes from worker handoffs) that cp1252 can't decode — on
    # Python 3.14 this raises UnicodeDecodeError in the reader thread and leaves
    # proc.stdout as None, which then crashes downstream `.strip()` calls and silently
    # gates every dependent cron job (auto_triage, dependency-serializer, autoforeman)
    # via the smart_cleanup pre-check in scripts/auto_triage_blocked.py:155-160.
    return subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=False,
        encoding="utf-8",
        errors="replace",
    )


def read_json(path: Path, fallback: Any) -> Any:
    if not path.exists():
        return fallback
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def format_est_minute(value: Any) -> str:
    if not value:
        return ""
    text = str(value).strip()
    try:
        if text.endswith("Z"):
            parsed = datetime.fromisoformat(text[:-1] + "+00:00")
        else:
            parsed = datetime.fromisoformat(text)
        return parsed.astimezone().strftime("%Y-%m-%d %H:%M EST")
    except ValueError:
        return text[:16].replace("T", " ") + " EST"


def task_sort_key(task_id: str) -> tuple[int, str]:
    match = re.search(r"task(\d{3})", task_id or "")
    if match:
        return int(match.group(1)), task_id
    return 9999, task_id


def infer_mode(argv0: str) -> str:
    stem = Path(argv0).stem.lower()
    if "human" in stem or "digest" in stem:
        return "human"
    if "sync" in stem or "repo" in stem or "github" in stem:
        return "sync"
    if "submit" in stem:
        return "submit"
    if "blocked" in stem:
        return "blocked"
    if "backlog" in stem or "foreman" in stem:
        return "backlog"
    if "analyst" in stem:
        return "analyst"
    return "watchdog"


def load_cards() -> list[dict[str, Any]]:
    proc = run(["hermes", "kanban", "list", "--json"])
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout)
    text = proc.stdout.strip()
    return json.loads(text) if text else []


def summarize_manifest() -> dict[str, Any]:
    manifest = read_json(MANIFEST, {})
    best = manifest.get("current_best") if isinstance(manifest.get("current_best"), dict) else manifest
    entries = manifest.get("entries", []) if isinstance(manifest, dict) else []
    last_result = best.get("last_submit_result") if isinstance(best.get("last_submit_result"), dict) else None
    if last_result is None and isinstance(entries, list):
        best_id = best.get("candidate_id")
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if best_id and entry.get("candidate_id") != best_id:
                continue
            maybe_result = entry.get("last_submit_result")
            if isinstance(maybe_result, dict):
                last_result = maybe_result
                break
    last_result = last_result or {}
    actual_sha = ""
    sha_matches = None
    if PACKAGE.exists():
        actual_sha = hashlib.sha256(PACKAGE.read_bytes()).hexdigest().upper()
        expected_sha = (best.get("package_sha256") or best.get("submission_sha256") or "").upper()
        sha_matches = bool(expected_sha and actual_sha == expected_sha)
    return {
        "candidate_id": best.get("candidate_id"),
        "score": best.get("score"),
        "score_checked_at": best.get("score_checked_at") or manifest.get("updated_at"),
        "package_sha256": actual_sha,
        "manifest_sha_matches_package": sha_matches,
        "changed_entries": best.get("changed_entries") or best.get("changed_tasks"),
        "last_winning_submit": {
            "submitted_at": last_result.get("submitted_at"),
            "submitted_at_est": format_est_minute(last_result.get("submitted_at")),
            "observed_complete_at": last_result.get("observed_complete_at"),
            "observed_complete_at_est": format_est_minute(last_result.get("observed_complete_at")),
            "public_score": last_result.get("public_score"),
            "delta_vs_lineage_base": last_result.get("delta_vs_lineage_base"),
            "attempt_kind": last_result.get("attempt_kind"),
            "synced_at": last_result.get("synced_at"),
            "sync_status": last_result.get("sync_status"),
        },
    }


def compact_card(card: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": card.get("id"),
        "status": card.get("status"),
        "assignee": card.get("assignee") or "",
        "priority": card.get("priority"),
        "title": card.get("title"),
    }


def summarize_board(cards: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = Counter((card.get("status") or "") for card in cards)
    assignee_status_counts: dict[str, Counter[str]] = defaultdict(Counter)
    active_cards = []
    for card in cards:
        assignee = card.get("assignee") or ""
        status = card.get("status") or ""
        assignee_status_counts[assignee][status] += 1
        if status in ACTIVE_CARD_STATUSES:
            active_cards.append(card)

    active_by_assignee: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for card in active_cards:
        active_by_assignee[card.get("assignee") or ""].append(compact_card(card))

    return {
        "total_cards": len(cards),
        "status_counts": dict(sorted(status_counts.items())),
        "active_counts": {status: status_counts.get(status, 0) for status in sorted(ACTIVE_CARD_STATUSES)},
        "active_by_assignee": dict(sorted(active_by_assignee.items())),
        "queue_workers": list(DEFAULT_QUEUE_WORKERS),
        "idle_queue_workers": [
            worker for worker in DEFAULT_QUEUE_WORKERS if not active_by_assignee.get(worker)
        ],
        "specialist_active": {
            specialist: active_by_assignee.get(specialist, []) for specialist in SPECIALISTS
        },
        "todo_or_blocked": [
            compact_card(card)
            for card in cards
            if card.get("status") in {"todo", "blocked"}
        ][:20],
    }


def summarize_submit_queue() -> dict[str, Any]:
    doc = read_json(SUBMIT_QUEUE, {})
    queue = doc.get("queue", doc if isinstance(doc, list) else [])
    if not isinstance(queue, list):
        queue = []
    known_best = doc.get("current_best_manifest") if isinstance(doc, dict) else {}
    if not isinstance(known_best, dict):
        known_best = {}

    statuses = Counter((entry.get("status") or "") for entry in queue if isinstance(entry, dict))
    ready = [
        entry
        for entry in queue
        if isinstance(entry, dict) and entry.get("submit_ready") is True
    ]
    ready_ids = [
        str(entry.get("candidate_id") or entry.get("queue_id") or entry.get("task_id") or "")
        for entry in ready
    ]
    ready_key = hashlib.sha256("\n".join(sorted(ready_ids)).encode("utf-8")).hexdigest()[:12]

    expected_gain = 0.0
    missing_gain = 0
    for entry in ready:
        value = entry.get("expected_lb_point_gain")
        if value is None:
            value = entry.get("expected_point_gain")
        if value is None and isinstance(entry.get("approval_packet"), dict):
            value = entry["approval_packet"].get("expected_lb_point_gain")
            if value is None:
                value = entry["approval_packet"].get("expected_point_gain")
        try:
            if value is None:
                missing_gain += 1
            else:
                expected_gain += float(value)
        except (TypeError, ValueError):
            missing_gain += 1

    return {
        "path": str(SUBMIT_QUEUE),
        "queue_total": len(queue),
        "status_counts": dict(sorted(statuses.items())),
        "known_public_best": {
            "candidate_id": known_best.get("candidate_id"),
            "score": known_best.get("score"),
            "previous_global_score": known_best.get("previous_global_score"),
            "delta_vs_previous_global": known_best.get("delta_vs_previous_global"),
            "package_sha256": known_best.get("package_sha256"),
            "changed_entries": known_best.get("changed_entries") or [],
            "sync_status": known_best.get("sync_status"),
            "synced_at": known_best.get("synced_at"),
        },
        "submit_ready_count": len(ready),
        "submit_ready_key": f"ng-submit-captain-{ready_key}" if ready else "",
        "submit_ready_expected_gain_sum": expected_gain,
        "submit_ready_missing_expected_gain": missing_gain,
        "submit_ready_preview": [
            {
                "task_id": entry.get("task_id"),
                "candidate_id": entry.get("candidate_id"),
                "status": entry.get("status"),
                "expected_lb_point_gain": entry.get("expected_lb_point_gain") or entry.get("expected_point_gain"),
            }
            for entry in ready[:12]
        ],
    }


def task_ids_from_text(*values: Any) -> set[str]:
    found: set[str] = set()
    for value in values:
        if not value:
            continue
        found.update(match.group(0) for match in re.finditer(r"task\d{3}", str(value)))
    return found


def live_task_ids(cards: list[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for card in cards:
        if card.get("status") not in ACTIVE_CARD_STATUSES:
            continue
        ids.update(task_ids_from_text(card.get("title"), card.get("body"), card.get("description")))
    return ids


def submit_queue_task_ids() -> set[str]:
    doc = read_json(SUBMIT_QUEUE, {})
    queue = doc.get("queue", doc if isinstance(doc, list) else [])
    ids: set[str] = set()
    if not isinstance(queue, list):
        return ids
    active_submit_statuses = {
        "ready",
        "approved-for-submit-captain",
        "awaiting-submit-credit-or-stack",
        "submitted-complete-promote-pending",
        "submitted-complete-promote-sync-pending",
    }
    for entry in queue:
        if not isinstance(entry, dict):
            continue
        if entry.get("submit_ready") is True or entry.get("status") in active_submit_statuses:
            ids.update(task_ids_from_text(entry.get("task_id"), entry.get("candidate_id")))
            for changed in entry.get("changed_task_ids") or []:
                ids.update(task_ids_from_text(changed))
    return ids


def read_task_rows() -> list[dict[str, str]]:
    if not TASK_STATE.exists():
        return []
    with TASK_STATE.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return sorted(rows, key=lambda row: task_sort_key(row.get("task") or ""))


def summarize_task_progress(
    cards: list[dict[str, Any]],
    *,
    preview_limit: int = 24,
    consider_limit: int = 80,
) -> dict[str, Any]:
    cursor = read_json(TASK_BACKLOG_CURSOR, {})
    rows = read_task_rows()
    task_ids = [row.get("task") or "" for row in rows if row.get("task")]
    next_task = str(cursor.get("next_task_id") or cursor.get("next_task") or (task_ids[0] if task_ids else ""))
    task_to_index = {task_id: index for index, task_id in enumerate(task_ids)}
    start_index = task_to_index.get(next_task, 0)
    active_ids = live_task_ids(cards)
    submit_ids = submit_queue_task_ids()

    preview: list[dict[str, Any]] = []
    considered = 0
    skipped_counts: Counter[str] = Counter()
    window_end_task_id = next_task
    if rows:
        ordered = rows[start_index:] + rows[:start_index]
        for row in ordered:
            task_id = row.get("task") or ""
            if not task_id:
                continue
            considered += 1
            window_end_task_id = task_id
            skip_reason = ""
            if task_id in active_ids:
                skip_reason = "active-board-card"
            elif task_id in submit_ids:
                skip_reason = "submit-queue-covered"
            status = row.get("status") or row.get("best_known_status") or ""
            if not skip_reason and status in {"tested_public_positive_current_best", "tested_public_positive_current_best_stack_member"}:
                skip_reason = "already-current-best-positive"
            if skip_reason:
                skipped_counts[skip_reason] += 1
            try:
                current_cost = float(row.get("current_cost") or 0)
            except ValueError:
                current_cost = 0.0
            try:
                current_points = float(row.get("current_points") or 0)
            except ValueError:
                current_points = 0.0
            if not skip_reason:
                preview.append(
                    {
                        "task": task_id,
                        "current_cost": int(current_cost) if current_cost.is_integer() else current_cost,
                        "current_points": round(current_points, 3),
                        "status": status,
                        "next_action": (row.get("next_action") or "")[:120],
                    }
                )
            if len(preview) >= preview_limit or considered >= consider_limit:
                break

    status_counts = Counter((row.get("status") or "(blank)") for row in rows)
    return {
        "source_csv": str(TASK_STATE),
        "source_xlsx": str(TASK_STATUS_XLSX),
        "source_xlsx_exists": TASK_STATUS_XLSX.exists(),
        "cursor_path": str(TASK_BACKLOG_CURSOR),
        "cursor_exists": TASK_BACKLOG_CURSOR.exists(),
        "next_task_id": next_task,
        "pass_count": cursor.get("pass_count", 0),
        "last_advanced_at": cursor.get("last_advanced_at"),
        "rows": len(rows),
        "status_counts": dict(status_counts.most_common(8)),
        "active_task_ids": sorted(active_ids, key=task_sort_key),
        "submit_queue_task_ids": sorted(submit_ids, key=task_sort_key),
        "numeric_preview": preview,
        "considered_until_preview": considered,
        "window_end_task_id": window_end_task_id,
        "skipped_counts": dict(sorted(skipped_counts.items())),
        "policy": "scan task000..task400 in numeric order from cursor; use high-EV as priority within the window, not as a reason to forget earlier tasks",
    }


def compact_worker_pool(workers: list[str]) -> str:
    expected = [f"neurogolf-worker-{index}" for index in range(1, 9)]
    if workers == expected:
        return "neurogolf-worker-1..8"
    return ",".join(workers)


def preview_tasks(items: list[dict[str, Any]], limit: int = 8) -> str:
    tasks = [str(item.get("task") or "") for item in items[:limit] if item.get("task")]
    if not tasks:
        return "(none)"
    suffix = "..." if len(items) > limit else ""
    return ",".join(tasks) + suffix


def summarize_submit_mode() -> dict[str, Any]:
    now = datetime.now().astimezone()
    refresh = now.replace(hour=21, minute=0, second=0, microsecond=0)
    if now < refresh:
        return {
            "mode": "pre-refresh-burst",
            "until": refresh.strftime("%Y-%m-%d %H:%M EST"),
            "rule": "drain worthy submit-ready candidates aggressively before the 21:00 EST refresh; keep submit-captain as sole submitter and obey daily cap/no-duplicate gates",
        }
    return {
        "mode": "post-refresh-awaiting-human-cap",
        "until": "",
        "rule": "burst window has passed; use submit_runtime_config.json plus submit_credit_state.json for the approved post-refresh hourly cap",
    }


def summarize_submit_runtime_config() -> dict[str, Any]:
    doc = read_json(SUBMIT_RUNTIME_CONFIG, {})
    if not isinstance(doc, dict):
        doc = {}
    try:
        max_per_hour = int(doc.get("max_submits_per_hour") or 0)
    except (TypeError, ValueError):
        max_per_hour = 0
    enabled = bool(doc.get("post_refresh_auto_submit_enabled")) and max_per_hour > 0
    return {
        "path": str(SUBMIT_RUNTIME_CONFIG),
        "post_refresh_auto_submit_enabled": bool(doc.get("post_refresh_auto_submit_enabled")),
        "max_submits_per_hour": max_per_hour,
        "post_refresh_auto_submit_allowed": enabled,
    }


def summarize_registries() -> dict[str, Any]:
    registry = read_json(CANDIDATE_REGISTRY, {})
    outcome = read_json(TASK_OUTCOME_REGISTRY, {})
    entries = registry.get("entries", registry if isinstance(registry, list) else [])
    tasks = outcome.get("tasks", {})
    return {
        "candidate_registry_entries": len(entries) if isinstance(entries, list) else 0,
        "task_outcome_registry_tasks": len(tasks) if isinstance(tasks, dict) else 0,
    }


def dry_run_dispatch() -> dict[str, Any]:
    proc = run(["hermes", "kanban", "dispatch", "--dry-run", "--json"])
    if proc.returncode != 0:
        return {"error": proc.stderr or proc.stdout}
    try:
        return json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return {"raw": proc.stdout}


def summarize_git() -> dict[str, Any]:
    branch = run(["git", "branch", "--show-current"])
    status = run(["git", "status", "--porcelain"])
    remotes = run(["git", "remote", "-v"])
    return {
        "branch": (branch.stdout or "").strip(),
        "dirty_entry_count": len([line for line in (status.stdout or "").splitlines() if line.strip()]),
        "remote_preview": [line for line in (remotes.stdout or "").splitlines()[:4]],
    }


def summarize_worktrees() -> dict[str, Any]:
    proc = run(["git", "worktree", "list", "--porcelain"])
    rows = []
    current: dict[str, str] = {}
    for line in (proc.stdout or "").splitlines():
        if not line.strip():
            if current:
                rows.append(current)
                current = {}
            continue
        key, _, value = line.partition(" ")
        current[key] = value
    if current:
        rows.append(current)
    project_prefix = (REPO_ROOT / ".worktrees").resolve().as_posix().lower()
    project_rows = [
        row for row in rows if row.get("worktree", "").replace("\\", "/").lower().startswith(project_prefix)
    ]
    return {
        "total_git_worktrees": len(rows),
        "project_local_worktrees": len(project_rows),
        "project_local_preview": [row.get("worktree") for row in project_rows[:12]],
    }


def diagnostics_text() -> str:
    proc = run(["hermes", "kanban", "diagnostics"])
    return (proc.stdout or proc.stderr).strip()


def build_snapshot(mode: str) -> dict[str, Any]:
    cards = load_cards()
    board = summarize_board(cards)
    queue_audit = audit_worker_queue(cards)
    submit = summarize_submit_queue()
    current_best = summarize_manifest()
    submit_mode = summarize_submit_mode()
    submit_runtime_config = summarize_submit_runtime_config()
    known_public_best = submit.get("known_public_best") or {}
    synced_score = as_float(current_best.get("score"))
    known_score = as_float(known_public_best.get("score"))
    if known_score is not None and synced_score is not None:
        known_public_best["delta_vs_synced"] = round(known_score - synced_score, 6)
        known_public_best["is_ahead_of_synced"] = known_score > synced_score
    else:
        known_public_best["delta_vs_synced"] = None
        known_public_best["is_ahead_of_synced"] = False
    diagnostics = diagnostics_text()
    dispatch = dry_run_dispatch()
    reviewer_active_count = sum(
        len(board["specialist_active"].get(profile, []))
        for profile in reviewer_profiles()
    )

    recommendations: list[dict[str, Any]] = []
    blocked_count = board["active_counts"].get("blocked", 0)
    post_refresh_hold = (
        submit_mode.get("mode") == "post-refresh-awaiting-human-cap"
        and not submit_runtime_config.get("post_refresh_auto_submit_allowed")
    )
    if submit["submit_ready_count"] and not board["specialist_active"]["neurogolf-submit-captain"] and not post_refresh_hold:
        recommendations.append(
            {
                "action": "create_submit_captain_card",
                "idempotency_key": submit["submit_ready_key"],
                "reason": "submit_ready candidates exist and submit-captain has no active card",
            }
        )
    if submit["submit_ready_count"] >= 6:
        recommendations.append(
            {
                "action": "route_submit_backlog_for_optimization",
                "idempotency_key": f"{submit['submit_ready_key']}-opt-return",
                "reason": "6+ submit-ready candidates are backed up; keep draining highest EV, but return low/missing-gain or under-tested candidates to original workers for further optimization",
            }
        )
    if (
        blocked_count
        and not board["specialist_active"]["neurogolf-blocked-55"]
        and not reviewer_active_count
    ):
        blocked_ids = [
            str(card.get("id") or "")
            for card in cards
            if card.get("status") == "blocked"
        ]
        digest = hashlib.sha256("\n".join(sorted(blocked_ids)).encode("utf-8")).hexdigest()[:12]
        recommendations.append(
            {
                "action": "create_blocked_cleanup_card",
                "idempotency_key": f"ng-blocked-cleanup-{digest}",
                "reason": f"{blocked_count} blocked cards and no active blocked-cleanup owner",
            }
        )
    if queue_audit["idle_queue_workers"] and not queue_audit["active_violations"]:
        recommendations.append(
            {
                "action": "consider_backlog_assignment",
                "idle_queue_workers": queue_audit["idle_queue_workers"],
                "reason": "Alpha queue workers have no active assigned lane; assign only one runnable head, keep extra work unassigned or dependency-gated",
            }
        )
    if queue_audit.get("unassigned_backlog_bank_count", 0) < 16:
        recommendations.append(
            {
                "action": "top_up_unassigned_backlog_bank",
                "current_bank": queue_audit.get("unassigned_backlog_bank_count", 0),
                "target_bank": 16,
                "reason": "concrete unassigned worker-card bank is below target",
            }
        )
    if queue_audit["active_violations"]:
        recommendations.append(
            {
                "action": "serialize_queue_before_more_worker_work",
                "violations": queue_audit["active_violations"],
            }
        )
    if known_public_best.get("is_ahead_of_synced"):
        recommendations.insert(
            0,
            {
                "action": "deterministic_current_best_sync_required",
                "candidate_id": known_public_best.get("candidate_id"),
                "score": known_public_best.get("score"),
                "package_sha256": known_public_best.get("package_sha256"),
                "reason": "submit_queue records a higher public-positive result than canonical submissions/current_best_manifest.json",
                "tool": "tools/neurogolf_auto_sync_current_best.py",
            },
        )

    now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M EST")
    return {
        "mode": mode,
        "checked_at": now,
        "repo_root": str(REPO_ROOT),
        "current_best": current_best,
        "board": board,
        "queue_audit": queue_audit,
        "submit_queue": submit,
        "submit_mode": submit_mode,
        "submit_runtime_config": submit_runtime_config,
        "task_progress": summarize_task_progress(cards),
        "registries": summarize_registries(),
        "git": summarize_git(),
        "worktrees": summarize_worktrees(),
        "dispatch_dry_run": dispatch,
        "diagnostics": diagnostics,
        "recommendations": recommendations,
        "read_only": True,
    }


def render_text(snapshot: dict[str, Any]) -> str:
    if snapshot["mode"] == "human":
        return render_human_digest(snapshot)

    board = snapshot["board"]
    submit = snapshot["submit_queue"]
    best = snapshot["current_best"]
    known_best = submit.get("known_public_best") or {}
    queue_audit = snapshot["queue_audit"]
    lines = [
        f"NeuroGolf cron snapshot mode={snapshot['mode']} checked={snapshot['checked_at']}",
        f"repo={snapshot['repo_root']}",
        f"score={best.get('score')} candidate={best.get('candidate_id')} sha={str(best.get('package_sha256') or '')[:12]} sha_ok={best.get('manifest_sha_matches_package')}",
        f"known_public_best={known_best.get('score')} candidate={known_best.get('candidate_id')} sync={known_best.get('sync_status')} sha={str(known_best.get('package_sha256') or '')[:12]} delta_vs_synced={known_best.get('delta_vs_synced')}",
        f"board active={board['active_counts']} total={board['total_cards']}",
        f"idle_queue_workers={','.join(queue_audit['idle_queue_workers']) or '(none)'} lane_free_workers={','.join(queue_audit['lane_free_workers']) or '(none)'} queue_workers={','.join(board['queue_workers'])}",
        f"queue_audit_ok={queue_audit['ok']} lane_conflicts={len(queue_audit['lane_context_violations'])} dispatch_risks={len(queue_audit['dispatch_risk_violations'])} assigned_ready_backlog={queue_audit['pending_assignment_backlog_count']} parked_queue={sum(len(v) for v in queue_audit['parked_by_assignee'].values())} unassigned_bank={queue_audit['unassigned_backlog_bank_count']}",
        f"submit_ready={submit['submit_ready_count']} submit_key={submit['submit_ready_key']} queue_total={submit['queue_total']} missing_expected_gain={submit['submit_ready_missing_expected_gain']}",
        f"submit_mode={snapshot['submit_mode']['mode']} rule={snapshot['submit_mode']['rule']}",
        f"task_scan_next={snapshot['task_progress']['next_task_id']} pass={snapshot['task_progress']['pass_count']} preview={','.join(item['task'] for item in snapshot['task_progress']['numeric_preview'][:8])}",
        f"dispatch_dry_run={json.dumps(snapshot['dispatch_dry_run'], sort_keys=True)}",
    ]
    if snapshot["recommendations"]:
        lines.append("recommendations:")
        for item in snapshot["recommendations"]:
            lines.append(f"- {json.dumps(item, sort_keys=True)}")
    else:
        lines.append("recommendations: none")
    if snapshot["diagnostics"]:
        lines.append("diagnostics:")
        lines.append(snapshot["diagnostics"])
    lines.append("active_specialists:")
    for name, cards in board["specialist_active"].items():
        lines.append(f"- {name}: {json.dumps(cards, ensure_ascii=False)}")
    lines.append("todo_or_blocked_preview:")
    for card in board["todo_or_blocked"][:12]:
        lines.append(f"- {json.dumps(card, ensure_ascii=False)}")
    return "\n".join(lines)


def render_human_digest(snapshot: dict[str, Any]) -> str:
    board = snapshot["board"]
    best = snapshot["current_best"]
    submit = snapshot["submit_queue"]
    known_best = submit.get("known_public_best") or {}
    queue = snapshot["queue_audit"]
    git = snapshot["git"]
    worktrees = snapshot["worktrees"]
    specialists = board["specialist_active"]
    last_win = best.get("last_winning_submit") or {}
    task_progress = snapshot["task_progress"]
    submit_mode = snapshot["submit_mode"]
    lines = [
        f"NeuroGolf {snapshot['checked_at']}",
        f"Score: {best.get('score')} synced | Best: {best.get('candidate_id')} | sha_ok={best.get('manifest_sha_matches_package')}",
        f"Known public best: {known_best.get('score') or '?'} | candidate={known_best.get('candidate_id') or '?'} | sync={known_best.get('sync_status') or '?'} | sha={str(known_best.get('package_sha256') or '')[:8] or '?'} | delta_vs_synced={known_best.get('delta_vs_synced')}",
        f"Last win: submitted={last_win.get('submitted_at_est') or '?'} | complete={last_win.get('observed_complete_at_est') or '?'} | delta={last_win.get('delta_vs_lineage_base')}",
        f"Board: running={board['active_counts'].get('running', 0)} ready={board['active_counts'].get('ready', 0)} todo={board['active_counts'].get('todo', 0)} blocked={board['active_counts'].get('blocked', 0)}",
        f"Workers: pool={compact_worker_pool(board['queue_workers'])}; idle={','.join(queue['idle_queue_workers']) or '(none)'}; lane_free={','.join(queue['lane_free_workers']) or '(none)'}",
        f"Queue safety: ok={queue['ok']} lane_conflicts={len(queue['lane_context_violations'])} dispatch_risks={len(queue['dispatch_risk_violations'])} assigned-ready={queue['pending_assignment_backlog_count']} parked={sum(len(v) for v in queue['parked_by_assignee'].values())} unassigned-bank={queue['unassigned_backlog_bank_count']}",
        f"Submit: mode={submit_mode['mode']} cap={snapshot['submit_runtime_config']['max_submits_per_hour']}/h enabled={snapshot['submit_runtime_config']['post_refresh_auto_submit_allowed']} ready={submit['submit_ready_count']} queue_total={submit['queue_total']} missing_expected_gain={submit['submit_ready_missing_expected_gain']}",
        f"Task scan: next={task_progress['next_task_id']} pass={task_progress['pass_count']} rows={task_progress['rows']} source=task_state.csv/xlsx preview={preview_tasks(task_progress['numeric_preview'])}",
        f"Specialists: submit={len(specialists['neurogolf-submit-captain'])} blocked55={len(specialists['neurogolf-blocked-55'])} reviewers={sum(len(specialists.get(profile, [])) for profile in reviewer_profiles())}/{len(reviewer_profiles())} primary={primary_reviewer_profile()} analyst={len(specialists['neurogolf-board-analyst'])}",
        f"Git: branch={git['branch']} dirty_entries={git['dirty_entry_count']}",
        f"Worktrees: project_local={worktrees['project_local_worktrees']} total_git={worktrees['total_git_worktrees']}",
    ]
    if snapshot["recommendations"]:
        lines.append("Recommended control actions:")
        for item in snapshot["recommendations"][:6]:
            lines.append(f"- {item.get('action')}: {item.get('reason') or item.get('idempotency_key') or item}")
    if queue["active_violations"]:
        lines.append("Serialization problems:")
        for item in queue["active_violations"][:6]:
            reason = item.get("reason", "worker lane serialization")
            lines.append(f"- {item['assignee']}: {','.join(item['card_ids'])} ({','.join(item['statuses'])}) {reason}")
    if submit["submit_ready_preview"]:
        lines.append("Submit-ready preview:")
        for entry in submit["submit_ready_preview"][:8]:
            lines.append(f"- {entry.get('task_id')}: {entry.get('candidate_id')} [{entry.get('status')}]")
    if snapshot["diagnostics"]:
        first_diag = "\n".join(snapshot["diagnostics"].splitlines()[:8])
        lines.append("Diagnostics:")
        lines.append(first_diag)
    lines.append("Minimal rebase policy: workers/reviewers verify deltas on original valid base; submit-captain rebases/stacks only for real submit assembly or conflict diagnosis.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["watchdog", "submit", "blocked", "backlog", "analyst", "human", "sync"], default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--always", action="store_true", help="For watchdog mode, print even when no action is recommended")
    args = parser.parse_args()

    mode = args.mode or infer_mode(sys.argv[0])
    snapshot = build_snapshot(mode)

    if args.json:
        text = json.dumps(snapshot, indent=2)
    else:
        text = render_text(snapshot)

    # In no-agent watchdog mode, empty stdout means no delivery. Agent-backed
    # jobs need the snapshot every time, so only watchdog may go silent.
    if mode == "watchdog" and not args.always and not snapshot["recommendations"]:
        return 0
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
