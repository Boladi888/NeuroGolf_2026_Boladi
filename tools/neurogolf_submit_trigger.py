#!/usr/bin/env python3
"""Create at most one submit-captain card when deterministic gates allow it."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.neurogolf_cron_snapshot import build_snapshot
from tools.neurogolf_kanban_python import KanbanClient

SUBMIT_CAPTAIN = "neurogolf-submit-captain"
SUBMIT_RUNTIME_CONFIG = REPO_ROOT / ".hermes" / "board" / "submit_runtime_config.json"
SUBMIT_CREDIT_STATE = REPO_ROOT / ".hermes" / "board" / "submit_credit_state.json"
SUBMIT_WAIT_MARKERS = (
    "waiting for 10-minute spacing",
    "waiting for rate limit",
    "post-refresh rate limit",
    "hourly cap",
    "submit credit",
)
DEFAULT_SUBMIT_RUNTIME_CONFIG = {
    "schema": "neurogolf.submit_runtime_config.v1",
    "post_refresh_auto_submit_enabled": False,
    "max_submits_per_hour": 0,
    "notes": "Post-refresh submits require explicit runtime config: post_refresh_auto_submit_enabled=true and max_submits_per_hour>0.",
}


def load_submit_runtime_config(path: Path = SUBMIT_RUNTIME_CONFIG) -> dict[str, Any]:
    if not path.exists():
        return dict(DEFAULT_SUBMIT_RUNTIME_CONFIG)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return dict(DEFAULT_SUBMIT_RUNTIME_CONFIG)
    merged = dict(DEFAULT_SUBMIT_RUNTIME_CONFIG)
    merged.update(data)
    return merged


def load_submit_credit_state(path: Path = SUBMIT_CREDIT_STATE) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def post_refresh_auto_submit_allowed(config: dict[str, Any]) -> bool:
    try:
        max_per_hour = int(config.get("max_submits_per_hour") or 0)
    except (TypeError, ValueError):
        max_per_hour = 0
    return bool(config.get("post_refresh_auto_submit_enabled")) and max_per_hour > 0


def _parse_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    try:
        if text.endswith("Z"):
            return datetime.fromisoformat(text[:-1] + "+00:00")
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed
    except ValueError:
        return None


def submit_credit_skip_reasons(
    *,
    submit_config: dict[str, Any],
    credit_state: dict[str, Any] | None,
    post_refresh_mode: bool,
    now: datetime | None = None,
) -> list[str]:
    if not post_refresh_mode or not post_refresh_auto_submit_allowed(submit_config):
        return []
    if not credit_state:
        return []
    now = now or datetime.now(timezone.utc)
    credits = credit_state.get("credits") if isinstance(credit_state.get("credits"), dict) else {}
    policy = credit_state.get("policy") if isinstance(credit_state.get("policy"), dict) else {}
    try:
        available = int(credits.get("available") or 0)
    except (TypeError, ValueError):
        available = 0
    try:
        daily_used = int(credits.get("daily_used") or 0)
    except (TypeError, ValueError):
        daily_used = 0
    try:
        daily_cap = int(policy.get("daily_submit_cap") or 0)
    except (TypeError, ValueError):
        daily_cap = 0
    try:
        max_per_hour = int(submit_config.get("max_submits_per_hour") or 0)
    except (TypeError, ValueError):
        max_per_hour = 0

    skip_reasons: list[str] = []
    if available <= 0:
        skip_reasons.append("submit credits exhausted")
    if daily_cap > 0 and daily_used >= daily_cap:
        skip_reasons.append("daily submit cap reached")
    last_submit = _parse_timestamp(credits.get("last_submit_at"))
    if last_submit and max_per_hour > 0:
        interval_seconds = int(3600 / max_per_hour)
        elapsed = (now.astimezone(timezone.utc) - last_submit.astimezone(timezone.utc)).total_seconds()
        if elapsed < interval_seconds:
            wait_seconds = max(1, int(interval_seconds - elapsed))
            skip_reasons.append(f"post-refresh rate limit waiting {wait_seconds}s")
    return skip_reasons


def _active_submit_cards(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    board = snapshot.get("board") or {}
    specialists = board.get("specialist_active") or {}
    cards = specialists.get(SUBMIT_CAPTAIN) or []
    return cards if isinstance(cards, list) else []


def _active_submit_card_ids(snapshot: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for card in _active_submit_cards(snapshot):
        task_id = str(card.get("id") or "")
        if task_id:
            ids.append(task_id)
    return ids


def _known_best_ahead(snapshot: dict[str, Any]) -> bool:
    submit = snapshot.get("submit_queue") or {}
    known = submit.get("known_public_best") or {}
    return bool(known.get("is_ahead_of_synced"))


def _worker_pid_from_detail(detail: dict[str, Any]) -> int | None:
    runs = detail.get("runs") if isinstance(detail.get("runs"), list) else []
    for run in reversed(runs):
        if not isinstance(run, dict):
            continue
        value = run.get("worker_pid")
        try:
            if value:
                return int(value)
        except (TypeError, ValueError):
            continue
    events = detail.get("events") if isinstance(detail.get("events"), list) else []
    for event in reversed(events):
        if not isinstance(event, dict):
            continue
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        value = payload.get("pid")
        try:
            if value:
                return int(value)
        except (TypeError, ValueError):
            continue
    return None


def plan_waiting_submit_cleanup(
    detail: dict[str, Any],
    *,
    snapshot: dict[str, Any] | None = None,
    now: float | None = None,
    stale_after_seconds: int = 25 * 60,
) -> dict[str, Any] | None:
    task = detail.get("task") if isinstance(detail.get("task"), dict) else {}
    task_id = str(task.get("id") or detail.get("id") or "")
    if not task_id or task.get("assignee") != SUBMIT_CAPTAIN or task.get("status") != "running":
        return None

    events = detail.get("events") if isinstance(detail.get("events"), list) else []
    notes: list[str] = []
    last_event_at = 0
    for event in events:
        if not isinstance(event, dict):
            continue
        try:
            last_event_at = max(last_event_at, int(event.get("created_at") or 0))
        except (TypeError, ValueError):
            pass
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        note = str(payload.get("note") or "")
        if note:
            notes.append(note)
    combined = "\n".join(notes).lower()
    reason = ""
    try:
        started_at = int(task.get("started_at") or 0)
    except (TypeError, ValueError):
        started_at = 0
    if snapshot:
        current_best = snapshot.get("current_best") if isinstance(snapshot.get("current_best"), dict) else {}
        last_win = current_best.get("last_winning_submit") if isinstance(current_best.get("last_winning_submit"), dict) else {}
        synced_at = _parse_timestamp(last_win.get("synced_at") or last_win.get("observed_complete_at"))
        synced_ts = int(synced_at.timestamp()) if synced_at else 0
        task_text = f"{task.get('title') or ''}\n{task.get('body') or ''}".lower()
        changed_entries = current_best.get("changed_entries") if isinstance(current_best.get("changed_entries"), list) else []
        changed_tasks = [
            str(entry).lower().replace(".onnx", "")
            for entry in changed_entries
            if str(entry).lower().startswith("task")
        ]
        if started_at and synced_ts >= started_at and any(task_id in task_text for task_id in changed_tasks):
            reason = "submit-captain public result is already complete and deterministic best sync promoted it"
    if any(marker in combined for marker in SUBMIT_WAIT_MARKERS):
        reason = "submit-captain is holding the lane while waiting on a Python-owned submit rate/credit gate"
    elif not reason:
        now = now if now is not None else time.time()
        reference_at = max(started_at, last_event_at)
        if reference_at and now - reference_at >= stale_after_seconds:
            reason = f"submit-captain has been silent for >= {stale_after_seconds // 60} minutes; closing stale lane for cron retry"
    if not reason:
        return None

    return {
        "op": "complete_waiting_submit_captain",
        "task_id": task_id,
        "title": task.get("title"),
        "worker_pid": _worker_pid_from_detail(detail),
        "reason": reason,
    }


def plan_submit_trigger(
    snapshot: dict[str, Any],
    *,
    allow_post_refresh: bool = False,
    submit_config: dict[str, Any] | None = None,
    credit_state: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    submit = snapshot.get("submit_queue") or {}
    mode = snapshot.get("submit_mode") or {}
    queue = snapshot.get("queue_audit") or {}
    current_best = snapshot.get("current_best") or {}
    submit_config = dict(DEFAULT_SUBMIT_RUNTIME_CONFIG if submit_config is None else submit_config)
    config_allows_post_refresh = post_refresh_auto_submit_allowed(submit_config)
    ready_count = int(submit.get("submit_ready_count") or 0)
    skip_reasons: list[str] = []

    if ready_count <= 0:
        skip_reasons.append("no submit-ready queue entries")
    if _active_submit_cards(snapshot):
        skip_reasons.append("active submit-captain card exists")
    post_refresh_mode = mode.get("mode") == "post-refresh-awaiting-human-cap"
    if post_refresh_mode and not (allow_post_refresh or config_allows_post_refresh):
        skip_reasons.append("post-refresh auto-submit disabled or hourly cap unset")
    skip_reasons.extend(
        submit_credit_skip_reasons(
            submit_config=submit_config,
            credit_state=credit_state,
            post_refresh_mode=post_refresh_mode,
            now=now,
        )
    )
    if queue.get("ok") is False:
        skip_reasons.append("queue safety is not ok")
    if _known_best_ahead(snapshot):
        skip_reasons.append("known public best is ahead of canonical sync; deterministic auto-sync must run first")
    if current_best.get("manifest_sha_matches_package") is False:
        skip_reasons.append("canonical current best package sha does not match manifest")

    if skip_reasons:
        return {
            "schema": "neurogolf.submit_trigger.v1",
            "ok": False,
            "actions": [],
            "skip_reasons": skip_reasons,
            "ready_count": ready_count,
            "submit_mode": mode.get("mode"),
            "submit_config": {
                "post_refresh_auto_submit_enabled": bool(submit_config.get("post_refresh_auto_submit_enabled")),
                "max_submits_per_hour": submit_config.get("max_submits_per_hour"),
            },
        }

    ready_preview = submit.get("submit_ready_preview") or []
    lines = [
        "PYTHON SUBMIT TRIGGER",
        "",
        f"Current synced score: {current_best.get('score')}",
        f"Current best candidate: {current_best.get('candidate_id')}",
        f"Submit-ready count: {ready_count}",
        f"Submit-ready key: {submit.get('submit_ready_key')}",
        f"Submit mode: {mode.get('mode')}",
        f"Post-refresh hourly cap: {submit_config.get('max_submits_per_hour')}",
        "",
        "Instructions:",
        "- Submit-captain is the only role allowed to submit.",
        "- Build from canonical current best and the ready queue; workers/reviewers do not rebase.",
        "- Drain highest expected value first.",
        "- Do not sleep or hold this card while waiting for the post-refresh hourly cap, credits, Kaggle completion polling beyond the active submit, or another external timer. If the next submit is time-gated, write a heartbeat/summary and complete; Python cron will reopen a fresh card when the gate allows it.",
        "- If 6+ candidates are backed up, submit worthy high-EV items and route low/missing-gain or under-tested candidates back to their origin worker.",
        "- After public result, update submit_queue.json/ledger; deterministic best sync promotes any positive public result.",
        "- Do not copy or mirror winning packages into legacy .notongit current_best/root/submissions zip paths; Python sync defaults to repo-local canonical artifacts only.",
        "",
        "Ready preview:",
    ]
    for entry in ready_preview[:12]:
        lines.append(
            f"- {entry.get('task_id')}: {entry.get('candidate_id')} status={entry.get('status')} expected_gain={entry.get('expected_lb_point_gain')}"
        )

    title = f"submit-captain drain {ready_count} ready candidate(s) on {current_best.get('score')}"
    # Idempotency-key freshness window: Hermes's `--idempotency-key` deduplicates
    # against ANY card with that key, including `done` cards. If the ready-queue
    # set doesn't change for hours, the base key stays identical, so every cron
    # tick just returns the old (already-completed) card id and no new submit-
    # captain card is created — submissions stall. Bucketing into 15-minute
    # windows by floored epoch division means cron jitter (jobs that fire a
    # few seconds before/after a wall-clock minute boundary) can never straddle
    # two adjacent windows the way `%H + minute//10` could near minute 9 vs 10.
    base_key = submit.get("submit_ready_key") or ""
    epoch_minutes = int(now.astimezone(timezone.utc).timestamp() // 60)
    freshness_bucket = epoch_minutes // 15  # changes every 15 wall-clock minutes
    idempotency_key = f"{base_key}-{freshness_bucket}" if base_key else ""
    return {
        "schema": "neurogolf.submit_trigger.v1",
        "ok": True,
        "ready_count": ready_count,
        "submit_mode": mode.get("mode"),
        "submit_config": {
            "post_refresh_auto_submit_enabled": bool(submit_config.get("post_refresh_auto_submit_enabled")),
            "max_submits_per_hour": submit_config.get("max_submits_per_hour"),
        },
        "actions": [
            {
                "op": "create_submit_captain",
                "title": title,
                "body": "\n".join(lines),
                "assignee": SUBMIT_CAPTAIN,
                "priority": 95,
                "idempotency_key": idempotency_key,
            }
        ],
        "skip_reasons": [],
    }


def apply_submit_trigger_plan(client: KanbanClient, plan: dict[str, Any]) -> None:
    for action in plan.get("actions", []):
        if action.get("op") == "create_submit_captain":
            client.create(
                str(action["title"]),
                body=str(action["body"]),
                assignee=str(action["assignee"]),
                workspace="worktree",
                priority=int(action["priority"]),
                created_by="neurogolf-submit-trigger",
                idempotency_key=str(action.get("idempotency_key") or ""),
                max_runtime="30m",
            )
        elif action.get("op") == "complete_waiting_submit_captain":
            task_id = str(action["task_id"])
            pid = action.get("worker_pid")
            if pid:
                try:
                    os.kill(int(pid), signal.SIGTERM)
                    action["terminated_pid"] = int(pid)
                except OSError as exc:
                    action["terminate_error"] = str(exc)
                    if os.name == "nt":
                        proc = subprocess.run(
                            ["taskkill", "/PID", str(int(pid)), "/T", "/F"],
                            text=True,
                            capture_output=True,
                            check=False,
                        )
                        action["taskkill_returncode"] = proc.returncode
            client.comment(
                task_id,
                f"Python submit trigger is closing this card: {action['reason']}. "
                "The cron trigger will create a fresh submit-captain card when deterministic gates allow it.",
            )
            client.complete(task_id, str(action["reason"]))


def render_text(report: dict[str, Any]) -> str:
    if not report.get("actions"):
        return "no submit card created: " + "; ".join(report.get("skip_reasons") or ["no action"])
    action = report["actions"][0]
    if action["op"] == "complete_waiting_submit_captain":
        return f"close waiting submit-captain card {action['task_id']}: {action['reason']}"
    return f"create submit-captain card {action['idempotency_key']} for {report['ready_count']} ready candidate(s)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--snapshot-json", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--allow-post-refresh", action="store_true")
    parser.add_argument("--submit-config", type=Path, default=SUBMIT_RUNTIME_CONFIG)
    parser.add_argument("--credit-state-json", type=Path, default=SUBMIT_CREDIT_STATE)
    args = parser.parse_args()

    if args.snapshot_json:
        snapshot = json.loads(args.snapshot_json.read_text(encoding="utf-8"))
    else:
        snapshot = build_snapshot("submit")
    if not isinstance(snapshot, dict):
        raise SystemExit("expected snapshot JSON object")

    submit_config = load_submit_runtime_config(args.submit_config)
    credit_state = load_submit_credit_state(args.credit_state_json)
    report = plan_submit_trigger(
        snapshot,
        allow_post_refresh=args.allow_post_refresh,
        submit_config=submit_config,
        credit_state=credit_state,
    )
    client = KanbanClient(args.repo_root, dry_run=not args.apply)
    if not report.get("actions") and "active submit-captain card exists" in report.get("skip_reasons", []):
        cleanup_actions: list[dict[str, Any]] = []
        for task_id in _active_submit_card_ids(snapshot):
            detail = client.show(task_id)
            action = plan_waiting_submit_cleanup(detail, snapshot=snapshot)
            if action:
                cleanup_actions.append(action)
        if cleanup_actions:
            report["actions"] = cleanup_actions
            report["skip_reasons"] = []
    if args.apply:
        apply_submit_trigger_plan(client, report)
        report["applied_actions"] = client.actions
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(render_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
