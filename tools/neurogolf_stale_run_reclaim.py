#!/usr/bin/env python3
"""Reassign running cards whose current run has gone silent past a threshold.

The Hermes dispatcher spawns an agent process for each card in `running`
status. When the underlying LLM session is half-alive (API quota, closed
provider, transient network) the process exists but emits no heartbeats —
the card stays `running` indefinitely and blocks the assignee's lane.

This watchdog:
  - lists running cards,
  - finds the active run (no end timestamp) for each,
  - measures time since the active run's most recent "agent-alive" event
    (heartbeat / claim_extended / commented / spawned),
  - if that's > --stale-seconds (default 480 = 8 min) AND the assignee is a
    Hermes-dispatcher profile, reassigns the card via `hermes kanban reassign
    --reclaim` to a Pool B /loop worker (for worker cards) or the peer
    reviewer profile (for reviewer cards).

Targets are imported from `tools/neurogolf_board_policy.py` so a single
toggle there propagates to this watchdog automatically.

Run:
  .venv/Scripts/python tools/neurogolf_stale_run_reclaim.py               # dry-run
  .venv/Scripts/python tools/neurogolf_stale_run_reclaim.py --apply       # mutate

Cron contract: emits `--json` per AUTOMATION.md when --json is set, suitable
for `hermes cron create ... --no-agent --deliver local`.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.neurogolf_board_policy import (
    POOL_A_DRAINING_WORKERS,
    POOL_A_SHRINK_WORKERS,
    POOL_B_SOLVER_HUNT_WORKERS,
)

# Default: a card whose active run has emitted no agent-alive event in the
# last 8 minutes is treated as stuck. The user picked 8 min because the
# Hermes dispatcher's own stale-lock reclaim runs around 30-40 min, which
# is too patient for a half-alive LLM session.
DEFAULT_STALE_SECONDS = 8 * 60

# Profiles the Hermes dispatcher actually spawns. Cards on these can go
# silent if the bound LLM session is broken. The watchdog only acts on
# cards assigned to one of these.
DISPATCHER_PROFILES: tuple[str, ...] = (
    tuple(POOL_A_SHRINK_WORKERS)
    + tuple(POOL_A_DRAINING_WORKERS)
    + ("neurogolf-reviewer", "neurogolf-reviewer-2")
)

# Pool B workers are manual Claude Code /loop sessions. Stuck worker cards
# get routed to whichever of these has the fewest active cards.
LOOP_WORKER_TARGETS: tuple[str, ...] = POOL_B_SOLVER_HUNT_WORKERS

# Reviewer cards swap to the peer profile. If both reviewers are stuck the
# watchdog will ping-pong; that's acceptable because each flip is at least
# --stale-seconds apart, and the operator's dashboard surfaces persistent
# reviewer stalls separately.
REVIEWER_PEERS: dict[str, str] = {
    "neurogolf-reviewer": "neurogolf-reviewer-2",
    "neurogolf-reviewer-2": "neurogolf-reviewer",
}

# Event kinds that indicate the agent is still alive on the current run.
# `claim_extended` is included because the dispatcher only extends a claim
# after seeing the worker pid is alive. `commented` covers cases where the
# agent posts a comment but skips the dedicated heartbeat event. `spawned`
# and `claimed` are the run's birth events — useful as a fallback floor
# when no later activity exists yet.
AGENT_ALIVE_EVENT_KINDS: frozenset[str] = frozenset(
    {"heartbeat", "claim_extended", "commented", "spawned", "claimed"}
)


def run_hermes(args: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["hermes", "kanban", *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        check=False,
    )


def _stdout_utf8(proc: subprocess.CompletedProcess[bytes]) -> str:
    return (proc.stdout or b"").decode("utf-8", errors="replace")


def _stderr_utf8(proc: subprocess.CompletedProcess[bytes]) -> str:
    return (proc.stderr or b"").decode("utf-8", errors="replace")


def hermes_list_running() -> list[dict[str, Any]]:
    proc = run_hermes(["list", "--status", "running", "--json"])
    if proc.returncode != 0:
        return []
    try:
        return [c for c in json.loads(_stdout_utf8(proc)) if isinstance(c, dict)]
    except json.JSONDecodeError:
        return []


def hermes_show_json(card_id: str) -> dict[str, Any] | None:
    proc = run_hermes(["show", "--json", card_id])
    if proc.returncode != 0:
        return None
    try:
        data = json.loads(_stdout_utf8(proc))
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def find_active_run(runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick the in-flight run from a `show --json` runs list.

    The active run is the one without `ended_at`. If multiple match (shouldn't
    happen but be defensive), return the one with the latest `started_at`.
    """
    if not isinstance(runs, list):
        return None
    candidates = [r for r in runs if isinstance(r, dict) and r.get("ended_at") in (None, 0)]
    if not candidates:
        return None
    candidates.sort(key=lambda r: int(r.get("started_at") or 0), reverse=True)
    return candidates[0]


def latest_alive_timestamp(show_data: dict[str, Any]) -> int | None:
    """Return the unix ts of the most recent agent-alive event on the active run.

    Falls back to the active run's `started_at` when no later event matches.
    Returns None when there's no active run at all.
    """
    runs = show_data.get("runs") or []
    active = find_active_run(runs)
    if active is None:
        return None
    active_run_id = active.get("id")
    floor_ts = int(active.get("started_at") or 0) or None
    latest_ts: int | None = None
    for ev in show_data.get("events") or []:
        if not isinstance(ev, dict):
            continue
        if ev.get("run_id") != active_run_id:
            continue
        if ev.get("kind") not in AGENT_ALIVE_EVENT_KINDS:
            continue
        ts = ev.get("created_at")
        try:
            ts_i = int(ts) if ts is not None else None
        except (TypeError, ValueError):
            continue
        if ts_i is None:
            continue
        if latest_ts is None or ts_i > latest_ts:
            latest_ts = ts_i
    if latest_ts is None:
        return floor_ts
    return latest_ts


def pick_loop_worker(
    running_cards: list[dict[str, Any]],
    *,
    pending_reassignments: dict[str, str] | None = None,
    targets: tuple[str, ...] = LOOP_WORKER_TARGETS,
) -> str | None:
    """Pick the /loop worker with the fewest active cards.

    Accounts for cards we've already decided to reassign in this run so the
    watchdog doesn't dump two stuck cards on the same idle /loop worker.
    """
    if not targets:
        return None
    load: dict[str, int] = {w: 0 for w in targets}
    for c in running_cards:
        a = c.get("assignee") or ""
        if a in load:
            load[a] += 1
    if pending_reassignments:
        for new_assignee in pending_reassignments.values():
            if new_assignee in load:
                load[new_assignee] += 1
    return min(load.items(), key=lambda kv: (kv[1], kv[0]))[0]


def reassignment_target(
    card: dict[str, Any],
    running_cards: list[dict[str, Any]],
    *,
    pending_reassignments: dict[str, str] | None = None,
) -> str | None:
    """Decide who a stuck card should be reassigned to. None = leave alone."""
    assignee = card.get("assignee") or ""
    if assignee in REVIEWER_PEERS:
        return REVIEWER_PEERS[assignee]
    if assignee in DISPATCHER_PROFILES:
        return pick_loop_worker(running_cards, pending_reassignments=pending_reassignments)
    return None


def plan_reassignments(
    running_cards: list[dict[str, Any]],
    *,
    stale_seconds: int,
    now_ts: int,
    show_fetcher=hermes_show_json,
) -> list[dict[str, Any]]:
    """Per-card verdict pass; pure-data so it's unit-testable."""
    actions: list[dict[str, Any]] = []
    pending: dict[str, str] = {}
    for card in running_cards:
        cid = card.get("id") or ""
        assignee = card.get("assignee") or ""
        if not cid:
            continue
        if assignee not in DISPATCHER_PROFILES:
            continue
        data = show_fetcher(cid)
        if data is None:
            continue
        latest = latest_alive_timestamp(data)
        if latest is None:
            continue
        age = now_ts - latest
        if age < stale_seconds:
            continue
        target = reassignment_target(card, running_cards, pending_reassignments=pending)
        if not target or target == assignee:
            actions.append({
                "card": cid, "assignee": assignee, "age_s": int(age),
                "verdict": "stale-no-target",
            })
            continue
        pending[cid] = target
        actions.append({
            "card": cid, "assignee": assignee, "age_s": int(age),
            "verdict": "reassign", "target": target,
        })
    return actions


def apply_reassignment(action: dict[str, Any], stale_seconds: int) -> dict[str, Any]:
    reason = (
        f"stale-run watchdog: active run had no agent-alive event for "
        f"{action['age_s']}s (threshold {stale_seconds}s); reassigning "
        f"from {action['assignee']} to {action['target']}."
    )
    proc = run_hermes([
        "reassign", action["card"], action["target"], "--reclaim",
        "--reason", reason,
    ])
    return {
        "reassign_rc": proc.returncode,
        "reassign_stderr": _stderr_utf8(proc).strip()[:200],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Mutate Hermes (default: dry-run).")
    parser.add_argument("--json", action="store_true",
                        help="Emit a compact JSON report on stdout (cron --no-agent contract).")
    parser.add_argument("--stale-seconds", type=int, default=DEFAULT_STALE_SECONDS,
                        help=f"Stale-activity threshold (default {DEFAULT_STALE_SECONDS}).")
    args = parser.parse_args(argv)

    now_ts = int(time.time())
    running = hermes_list_running()
    actions = plan_reassignments(running, stale_seconds=args.stale_seconds, now_ts=now_ts)

    applied: list[dict[str, Any]] = []
    if args.apply:
        for action in actions:
            if action["verdict"] != "reassign":
                continue
            result = apply_reassignment(action, args.stale_seconds)
            merged = {**action, **result}
            applied.append(merged)

    if args.json:
        mutating = [a for a in applied if a.get("reassign_rc") == 0]
        payload = {
            "schema": "neurogolf.stale_run_reclaim.v1",
            "action_count": len(mutating),
            "found_stale": sum(1 for a in actions if a["verdict"] == "reassign"),
            "applied_actions": [
                {
                    "op": "reassign-stale-run",
                    "task_id": a["card"],
                    "from": a["assignee"],
                    "to": a["target"],
                    "age_s": a["age_s"],
                    "result": "ok" if a.get("reassign_rc") == 0 else "failed",
                }
                for a in applied
            ],
        }
        json.dump(payload, sys.stdout)
        sys.stdout.write("\n")
    else:
        if not actions:
            print(f"No stale running cards (checked {len(running)} running, threshold {args.stale_seconds}s).")
        else:
            for a in actions:
                line = (
                    f"  {a['card']:14}  assignee={a['assignee']:25s}  "
                    f"age={a['age_s']:>5}s  verdict={a['verdict']}"
                )
                if a["verdict"] == "reassign":
                    line += f" -> {a['target']}"
                print(line)
            print(f"\nmode: {'APPLY' if args.apply else 'DRY-RUN'}  "
                  f"(stale threshold {args.stale_seconds}s)")
            if args.apply:
                for r in applied:
                    print(f"  applied {r['card']}: rc={r.get('reassign_rc')} "
                          f"{r.get('reassign_stderr', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
