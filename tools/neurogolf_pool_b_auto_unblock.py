#!/usr/bin/env python3
"""Auto-unblock Pool B cards that were force-blocked by the Hermes dispatcher.

Workers 7 and 8 (Pool B / Claude Code manual /loop) have profiles configured
with a closed port. The Hermes gateway dispatcher tries to spawn them when
their cards transition to ``ready``, the API call fails instantly, and after
``max_retries`` crashes the kanban auto-blocks the card with a
``repeated_crashes`` diagnostic. The Pool B foreman now passes
``--max-retries 1`` so this happens after a single crash, not two.

A blocked card can't be claimed by Claude Code's ``/loop`` without first being
unblocked. That's an extra manual step every cycle. This cron reverses it
automatically: scan diagnostics for blocked Pool B cards whose only error is
``repeated_crashes``, unblock them, and let the dispatcher and Claude Code
race for the next ready window.

Loop dynamics, with cron @ every 3m:
  * t=0..60s   card `ready`; dispatcher may fire and crash → blocks
  * t=60s..3m  card `blocked`
  * t=3m       this cron unblocks → ready
  * Claude Code's /loop (every 10m, or dynamic) eventually claims during a
    ready window; from there the manual session owns it.

Idempotency: each card can be auto-unblocked at most ``--max-unblocks`` times
(default 8). The count is tracked via comments tagged with
``[neurogolf-pool-b-auto-unblock]`` — once we've posted that many, the card
is left blocked and the operator must investigate.

Run:
    python tools/neurogolf_pool_b_auto_unblock.py            # dry-run
    python tools/neurogolf_pool_b_auto_unblock.py --apply
    python tools/neurogolf_pool_b_auto_unblock.py --apply --json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.neurogolf_board_policy import POOL_B_SOLVER_HUNT_WORKERS

COMMENT_MARKER = "[neurogolf-pool-b-auto-unblock]"
PID_NOT_ALIVE_RE = re.compile(r"pid\s+\d+\s+not alive", re.IGNORECASE)
# Hermes emits `repeated_crashes` on the first round of dispatcher fail-fast
# and `repeated_failures` on subsequent rounds (after one unblock+recrash
# cycle the counter is named `consecutive_failures` instead of
# `consecutive_crashes`). Both signal the same underlying condition for
# Pool B closed-port workers, so we accept either.
DISPATCH_CRASH_KINDS = frozenset({"repeated_crashes", "repeated_failures"})


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_hermes(args: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["hermes", "kanban", *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        check=False,
    )


def _stdout_utf8(proc: subprocess.CompletedProcess[bytes]) -> str:
    return (proc.stdout or b"").decode("utf-8", errors="replace")


def diagnostics_json() -> list[dict[str, Any]]:
    proc = _run_hermes(["diagnostics", "--json"])
    if proc.returncode != 0:
        return []
    try:
        data = json.loads(_stdout_utf8(proc))
    except json.JSONDecodeError:
        return []
    return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []


def card_show(task_id: str) -> str:
    proc = _run_hermes(["show", task_id])
    if proc.returncode != 0:
        return ""
    return _stdout_utf8(proc)


def is_pool_b_dispatch_crash(entry: dict[str, Any]) -> bool:
    """True iff the diagnostics entry is for a blocked Pool B card whose only
    error is the dispatcher's closed-port crash signature."""
    if entry.get("status") != "blocked":
        return False
    if entry.get("assignee") not in POOL_B_SOLVER_HUNT_WORKERS:
        return False
    diags = entry.get("diagnostics") or []
    if not diags:
        return False
    for diag in diags:
        if not isinstance(diag, dict):
            return False
        if diag.get("kind") not in DISPATCH_CRASH_KINDS:
            return False
        last_error = ""
        data = diag.get("data")
        if isinstance(data, dict):
            last_error = str(data.get("last_error") or "")
        if not PID_NOT_ALIVE_RE.search(last_error):
            return False
        # repeated_failures with most_recent_outcome != crashed could be a
        # real worker failure (e.g. exit code from a legitimate run), not
        # the dispatcher's closed-port no-op. Only auto-unblock when the
        # most recent outcome is `crashed` (or unspecified, which matches
        # the first-round `repeated_crashes` shape).
        if isinstance(data, dict):
            outcome = str(data.get("most_recent_outcome") or "").lower()
            if outcome and outcome != "crashed":
                return False
    return True


def auto_unblock_count(task_id: str) -> int:
    """How many auto-unblock comments are already on the card."""
    text = card_show(task_id)
    if not text:
        return 0
    return text.count(COMMENT_MARKER)


# When the dispatcher hasn't tried to spawn the card for this long, the
# ping-pong loop has cooled down. The 8/8 cap (which exists to prevent
# infinite cron→dispatcher→cron loops) can safely reset because the
# loop is no longer active. This is what prevents the "worker-5 was
# offline for 4 hours, the cron hit the cap, manual intervention needed
# when worker-5 came back online" failure mode.
DISPATCHER_COOLDOWN_SECONDS = 30 * 60  # 30 minutes


def diagnostic_cooldown_satisfied(
    entry: dict[str, Any],
    *,
    now_ts: float | None = None,
    cooldown_seconds: int = DISPATCHER_COOLDOWN_SECONDS,
) -> bool:
    """True iff the dispatcher hasn't crashed this card recently.

    Reads ``diagnostic.last_seen_at`` (UNIX seconds) from the entry.
    Cooldown is satisfied iff we have AT LEAST ONE diagnostic with a
    parseable timestamp AND all parseable timestamps are older than
    ``cooldown_seconds``. Diagnostics without timestamps don't count
    in either direction — they're conservatively ignored so legacy
    data falls back to the original cap behavior (return False).
    """
    import time as _time
    now = now_ts if now_ts is not None else _time.time()
    parseable_timestamps: list[float] = []
    for diag in entry.get("diagnostics") or []:
        if not isinstance(diag, dict):
            continue
        last_seen = diag.get("last_seen_at")
        try:
            ts = float(last_seen)
        except (TypeError, ValueError):
            continue
        parseable_timestamps.append(ts)
    if not parseable_timestamps:
        # No timestamp data — fall back to conservative cap behavior.
        return False
    # If the MOST RECENT crash is older than the cooldown, the loop has
    # cooled down. (Old crashes are fine; what matters is "no recent
    # crash" = no active ping-pong.)
    most_recent = max(parseable_timestamps)
    return (now - most_recent) >= cooldown_seconds


# A manual review-required block from a worker takes priority over the
# closed-port crash diagnostic. Workers post the JSON handoff comment and
# block with `--reason "review-required: ..."` once they have a real
# candidate; the dispatcher's stale crash diagnostic is still attached but
# the block is INTENTIONAL. Without this guard, the auto-unblock cron
# repeatedly unblocks a card that the worker is trying to hand off, the
# dep serializer demotes the now-ready card behind something else, and
# auto-triage misses the candidate.
REVIEW_REQUIRED_RE = re.compile(r"review[- ]required\s*:", re.IGNORECASE)
WORKER_OWNER_RE = re.compile(r'"?worker_owner"?\s*:\s*"neurogolf-worker-\d+"', re.IGNORECASE)


def has_pending_handoff(task_id: str) -> bool:
    """Return True iff the card looks like it carries a real worker handoff
    that the auto-unblock cron must not touch.

    Two complementary signals — either is enough to skip:
      * A `review-required:` reason or comment from a worker. This is the
        explicit hand-off marker workers post when blocking.
      * A `"worker_owner": "neurogolf-worker-N"` field inside any comment
        body. This is the JSON handoff schema; its presence means a worker
        has already deposited a real candidate.
    """
    text = card_show(task_id)
    if not text:
        return False
    return bool(REVIEW_REQUIRED_RE.search(text) or WORKER_OWNER_RE.search(text))


def plan_unblocks(
    *,
    diags: list[dict[str, Any]] | None = None,
    max_unblocks: int = 8,
) -> dict[str, Any]:
    diags = diags if diags is not None else diagnostics_json()
    actions: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for entry in diags:
        if not is_pool_b_dispatch_crash(entry):
            continue
        task_id = str(entry.get("task_id") or "")
        if not task_id:
            continue
        if has_pending_handoff(task_id):
            skipped.append(
                {
                    "task_id": task_id,
                    "assignee": entry.get("assignee"),
                    "reason": (
                        "card has a review-required reason or worker_owner JSON handoff; "
                        "this is an intentional manual block. Auto-triage / route-repair / "
                        "dependency-serializer own the next step, NOT auto-unblock."
                    ),
                }
            )
            continue
        prior = auto_unblock_count(task_id)
        if prior >= max_unblocks:
            # Cap reached. Original behavior: stay blocked, require human.
            # New behavior (2026-05-18 fix for worker-5 offline-then-online
            # case): if the dispatcher has been quiet for the cooldown
            # window, the loop is dormant and the cap can safely reset.
            # Cooldown means "no new crash in last 30 min" — which happens
            # naturally when:
            #   * The Pool B worker was offline (no /loop, no dispatcher
            #     spawn race) for hours.
            #   * The user paused the dispatcher for a manual session.
            #   * The Hermes gateway was restarted.
            # In all cases, the original ping-pong concern no longer
            # applies — give the card another shot at recovery.
            if diagnostic_cooldown_satisfied(entry):
                actions.append(
                    {
                        "op": "unblock_pool_b_card",
                        "task_id": task_id,
                        "assignee": entry.get("assignee"),
                        "prior_count": prior,
                        "cooldown_reset": True,
                        "comment": (
                            f"{COMMENT_MARKER} unblocked at {_utc_now()} (cooldown-reset; "
                            f"prior count {prior} >= cap {max_unblocks}, but no dispatcher "
                            f"crash in last {DISPATCHER_COOLDOWN_SECONDS // 60} min — loop "
                            "dormant). If this re-triggers the dispatcher and ping-pongs "
                            "again, the cap will catch it within "
                            f"{max_unblocks * 3} min and a human investigates."
                        ),
                    }
                )
                continue
            skipped.append(
                {
                    "task_id": task_id,
                    "assignee": entry.get("assignee"),
                    "reason": (
                        f"prior auto-unblock count {prior} >= {max_unblocks} AND dispatcher "
                        "still active (recent crash); leaving blocked for manual review"
                    ),
                }
            )
            continue
        actions.append(
            {
                "op": "unblock_pool_b_card",
                "task_id": task_id,
                "assignee": entry.get("assignee"),
                "prior_count": prior,
                "comment": (
                    f"{COMMENT_MARKER} unblocked at {_utc_now()} (attempt {prior + 1}/{max_unblocks}). "
                    "Dispatcher crash from closed-port worker config is expected; this restores "
                    "the card to ready so the Claude Code /loop session can claim it. After "
                    f"{max_unblocks} consecutive auto-unblocks, the card stays blocked and the "
                    "operator must investigate."
                ),
            }
        )
    return {
        "schema": "neurogolf.pool_b_auto_unblock.v1",
        "actions": actions,
        "skipped": skipped,
        "action_count": len(actions),
        "skipped_count": len(skipped),
    }


def apply_unblocks(plan: dict[str, Any]) -> list[dict[str, Any]]:
    applied: list[dict[str, Any]] = []
    for action in plan.get("actions", []):
        task_id = str(action["task_id"])
        # Order matters: post the comment BEFORE the unblock so the marker
        # is durable even if unblock fails. The comment also gives us the
        # idempotency signal for next tick.
        comment_proc = _run_hermes(
            ["comment", task_id, "--author", "neurogolf-pool-b-auto-unblock", str(action["comment"])]
        )
        comment_ok = comment_proc.returncode == 0
        unblock_proc = _run_hermes(["unblock", task_id])
        unblock_ok = unblock_proc.returncode == 0
        applied.append(
            {
                **action,
                "comment_ok": comment_ok,
                "unblock_ok": unblock_ok,
                "msg": _stdout_utf8(unblock_proc).strip()[:200],
            }
        )
    return applied


def render_text(report: dict[str, Any]) -> str:
    lines = [
        f"actions={report['action_count']} skipped={report['skipped_count']}",
    ]
    for action in report["actions"]:
        lines.append(
            f"- unblock {action['task_id']} ({action.get('assignee')}); attempt {action.get('prior_count', 0) + 1}"
        )
    for item in report["skipped"][:12]:
        lines.append(f"- skipped {item.get('task_id')}: {item.get('reason')}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--max-unblocks",
        type=int,
        default=8,
        help="Hard cap on how many times this cron will unblock the same card before giving up.",
    )
    args = parser.parse_args(argv)

    plan = plan_unblocks(max_unblocks=args.max_unblocks)
    if args.apply:
        plan["applied"] = apply_unblocks(plan)

    if args.json:
        json.dump(plan, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print(render_text(plan))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
