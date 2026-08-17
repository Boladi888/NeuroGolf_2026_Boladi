#!/usr/bin/env python3
"""Pool B autoforeman — feeds unsolved-task new-solver hunt cards to workers 7-8.

Pool A (workers 1-4) does current-live shrink on already-solved tasks via
``neurogolf_backlog_autoforeman.py``. Pool B is for the structurally different
job: build a first-ever working ONNX for one of the ~246 tasks not in
``tools.inspect_tasks.SOLVED_TASKS``. Cost floor doesn't apply — the candidate
is competing against the global zero-padding baseline (cost 0, points 0), so
any genuinely-working model is an improvement.

Workers 7 & 8 (Claude Code Sonnet & Opus) are driven by a manual ``/loop``
session; the Hermes dispatcher is configured to fail-fast on them. The cards
this script creates are picked up by the playbook's ``list --status ready``
poll.

Selection policy (deliberately dumb):
  * pick the lowest-numbered unsolved task that is not already claimed by
    any active card on any worker (Pool A or Pool B), and not on the
    short-term "tried recently and failed" skip list embedded in the body
    of any per-task notes file.
  * round-robin across idle Pool B workers (workers 7-8 by default).
  * cap at ``create_limit`` new cards per run.

Idempotency key: ``ng-pool-b-{worker}-{task_id}-{YYYYMMDD}`` — one re-attempt
slot per worker per task per day. Hermes refuses duplicate idempotency keys
so the same key never produces two live cards.

Run:
    python tools/neurogolf_pool_b_autoforeman.py            # dry-run
    python tools/neurogolf_pool_b_autoforeman.py --apply
    python tools/neurogolf_pool_b_autoforeman.py --apply --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.inspect_tasks import SOLVED_TASKS
from tools.neurogolf_board_policy import (
    ACTIVE_CARD_STATUSES,
    DEFAULT_QUEUE_WORKERS,
    POOL_B_SOLVER_HUNT_WORKERS,
    audit_worker_queue,
    canonical_target_claims,
)
from tools.neurogolf_kanban_python import (
    KanbanClient,
    as_text,
    card_assignee,
    card_id,
    card_status,
    card_title,
    read_json,
)
from tools.neurogolf_preflight import build_worker_preflight

BOARD_DIR = REPO_ROOT / ".hermes" / "board"
REGISTRY_JSON = BOARD_DIR / "candidate_registry.json"
SUBMIT_QUEUE_JSON = BOARD_DIR / "submit_queue.json"
STATE_JSON = BOARD_DIR / "state.json"
TASK_OUTCOME_JSON = BOARD_DIR / "task_outcome_registry.json"
TASK_BY_TASK_DIR = REPO_ROOT / "The_400_Tasks" / "Task_by_Task"
SUBMISSION_ZIP = REPO_ROOT / "submissions" / "submission.zip"

POOL_B_METHOD_FAMILY = "new-solver solver-hunt unsolved-task first-ever-working-onnx"
POOL_B_LINEAGE_BASE = "zero-padding-baseline"

# A task is treated as "actually solved" when its submission.zip entry is
# larger than this many bytes. Empirically the zero-pad-style minimal
# solvers in the canonical zip cluster at 158-353 bytes; the next tier
# (real but small solvers) starts at ~443 bytes. 400 is the natural break.
# Override via --zero-pad-max-bytes if a future build_submission.py changes
# the baseline size.
ZERO_PAD_MAX_BYTES_DEFAULT = 400


def _utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def all_unsolved_task_ids() -> list[str]:
    """Tasks Pool B should target — the ones NOT in the human-curated
    ``inspect_tasks.SOLVED_TASKS`` list.

    "Unsolved" here means "not in the curated SOLVED list," NOT "no working
    solver exists." Many of these tasks DO have real solvers in
    ``submissions/submission.zip``; Pool B's job is to find a BETTER one
    (different family, smaller, more correct on hidden test). The card
    body is generated from ``current_solver_info`` so the worker knows
    what they're actually competing against.
    """
    return [f"task{i:03d}" for i in sorted(set(range(400)) - SOLVED_TASKS)]


def current_solver_info(
    task_id: str,
    zip_path: Path = SUBMISSION_ZIP,
    *,
    zero_pad_max_bytes: int = ZERO_PAD_MAX_BYTES_DEFAULT,
) -> dict[str, Any]:
    """Inspect submission.zip to tell the worker what they're up against.

    Returns ``{"exists": bool, "size": int, "is_zero_pad_tier": bool}``.
    ``is_zero_pad_tier`` distinguishes "real working solver to beat" from
    "tiny baseline (probably constant-output / zero-pad) to replace."
    """
    if not zip_path.exists():
        return {"exists": False, "size": 0, "is_zero_pad_tier": True}
    try:
        with zipfile.ZipFile(zip_path) as z:
            try:
                info = z.getinfo(f"{task_id}.onnx")
            except KeyError:
                return {"exists": False, "size": 0, "is_zero_pad_tier": True}
            size = info.file_size
            return {
                "exists": True,
                "size": size,
                "is_zero_pad_tier": size <= zero_pad_max_bytes,
            }
    except zipfile.BadZipFile:
        return {"exists": False, "size": 0, "is_zero_pad_tier": True}


def task_id_from_int(n: int) -> str:
    return f"task{n:03d}"


_TASK_NUMBER_RE = re.compile(r"task0*(\d{1,3})")


def _task_int_from_card(card: dict[str, Any]) -> int | None:
    for field in ("title", "body"):
        text = as_text(card.get(field))
        match = _TASK_NUMBER_RE.search(text)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                continue
    return None


def actively_claimed_task_ids(cards: list[dict[str, Any]]) -> set[str]:
    """Tasks currently held by any active card on any worker."""
    out: set[str] = set()
    workers = set(DEFAULT_QUEUE_WORKERS)
    for card in cards:
        if not isinstance(card, dict):
            continue
        if card_status(card) not in ACTIVE_CARD_STATUSES:
            continue
        owner = card_assignee(card)
        if owner not in workers:
            continue
        n = _task_int_from_card(card)
        if n is not None:
            out.add(task_id_from_int(n))
    # Also include the canonical claims helper for cards where the task id
    # lives in a less obvious place (e.g. repair cards whose body names the
    # original taskNNN). The helper returns task -> claim_info.
    out.update(canonical_target_claims(cards).keys())
    return out


# How many recent "tried and didn't improve" public submits should disqualify
# a task from being picked again? 1 is the right threshold for Pool B — the
# task explicitly failed the hidden-test gate, so a different family or a
# substantially-better local cost is needed (which Pool B's existing tooling
# can't easily verify). When in doubt, the worker can mark the task notes
# with `pool-b-revive:` to override this exhaustion (not currently used).
EXHAUSTED_PUBLIC_NEGATIVE_THRESHOLD = 1


def _is_exhaustion_signal_status(status: str) -> bool:
    """A submit_queue status that indicates 'tried on Kaggle, no LB gain'.

    Hermes appends contextual suffixes to status strings (e.g.
    ``submitted-failed-public-negative-on-5755.62`` or
    ``submitted-failed-public-negative-stack-isolation``), so we substring-
    match the canonical exhaustion signals rather than using an exact set.

    Explicit successes are excluded — ``submitted-complete-promote-synced``
    (the candidate IS the current LB best) and
    ``submitted-complete-promote-superseded`` (the candidate promoted to LB,
    then a later candidate replaced it — that's two successes in a row, not
    exhaustion).
    """
    if not status:
        return False
    s = status.lower()
    # Anything matching these substrings is a failed-on-hidden signal.
    return (
        "public-negative" in s
        or "no-improve" in s
        or "no-promote" in s  # catches "no-promote" and "no-promote-tie"
        or "public-flat" in s
        or "no-requeue" in s
    )


def _is_fresh_candidate_status(status: str) -> bool:
    """A status indicating an active/pending candidate that hasn't yet been
    classified as exhausted — so the task should NOT be marked exhausted
    even if it has prior public-negative entries.

    ``ready``/``awaiting`` are the obvious cases; ``returned-to-worker-*``
    and ``returned-remediation-*`` indicate the worker has been asked to
    rework a candidate, which is still active work.
    """
    if not status:
        return False
    s = status.lower()
    return (
        "ready" in s
        or "awaiting" in s
        or "returned-to-worker" in s
        or "returned-remediation" in s
    )


def submitted_candidate_history_for_task(
    task_id: str,
    queue_doc: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Return all submit_queue entries for ``task_id`` that have been
    submitted to Kaggle (or returned/superseded) — sorted most-recent
    first. Used by the picker to enrich exhausted-task card bodies with
    actual prior-attempt evidence.
    """
    if not isinstance(queue_doc, dict):
        return []
    queue = queue_doc.get("queue") or []
    if not isinstance(queue, list):
        return []
    hits: list[dict[str, Any]] = []
    for entry in queue:
        if not isinstance(entry, dict):
            continue
        if entry.get("task_id") != task_id:
            continue
        status = (entry.get("status") or entry.get("submit_queue_status") or "").lower()
        # Anything that's a terminal-or-near-terminal outcome counts.
        if status.startswith("submitted") or "negative" in status or "superseded" in status:
            hits.append({
                "candidate_id": entry.get("candidate_id") or "",
                "status": status,
                "priority_score": entry.get("priority_score"),
            })
    # No real timestamp on entries, so use insertion order (queue is
    # append-only) and reverse to most-recent-first.
    hits.reverse()
    return hits


def task_ids_exhausted_by_public_history(queue_doc: dict[str, Any] | None) -> set[str]:
    """Tasks whose recent submit history shows public-negative or public-no-
    improve results AND no fresh approved candidate queued.

    Pool B keeps re-picking the lowest-numbered unsolved task. If task000
    has been submitted twice and both came back PUBLIC-TIE, the picker
    will keep handing it to fresh workers forever — wasting their time on
    a task that's effectively exhausted within the current LB ceiling.

    A task is "exhausted" iff:
      (a) at least ``EXHAUSTED_PUBLIC_NEGATIVE_THRESHOLD`` entries match
          ``_is_exhaustion_signal_status`` (substring match — Hermes
          appends contextual suffixes), AND
      (b) no entry matches ``_is_fresh_candidate_status`` (ready /
          awaiting / returned-to-worker / returned-remediation — any of
          which indicate an active candidate that hasn't yet been tried
          on the new canonical).

    Workers can override via the per-task notes file (``pool-b-revive:``
    marker would bypass this — but the override is optional; mostly the
    operator updates SOLVED_TASKS to mark the task done-good-enough).
    """
    if not isinstance(queue_doc, dict):
        return set()
    queue = queue_doc.get("queue") or []
    if not isinstance(queue, list):
        return set()
    from collections import defaultdict
    per_task_negative: dict[str, int] = defaultdict(int)
    per_task_has_fresh: dict[str, bool] = defaultdict(bool)
    for entry in queue:
        if not isinstance(entry, dict):
            continue
        task_id = entry.get("task_id")
        if not task_id:
            continue
        status = (entry.get("status") or entry.get("submit_queue_status") or "").lower()
        if _is_exhaustion_signal_status(status):
            per_task_negative[task_id] += 1
        if _is_fresh_candidate_status(status):
            per_task_has_fresh[task_id] = True
    exhausted: set[str] = set()
    for task_id, negatives in per_task_negative.items():
        if negatives >= EXHAUSTED_PUBLIC_NEGATIVE_THRESHOLD and not per_task_has_fresh.get(task_id):
            exhausted.add(task_id)
    return exhausted


def per_task_notes_say_dead(task_id: str) -> str:
    """Return a non-empty reason if the task's notes file marks it dead/skip.

    Pool B is deliberately permissive — only DEAD / DO-NOT-RETRY signals from
    the task's own notes file count. Cost / soft-defer policy is Pool A's job.
    """
    notes_path = TASK_BY_TASK_DIR / f"{task_id}.md"
    if not notes_path.exists():
        return ""
    try:
        text = notes_path.read_text(encoding="utf-8", errors="replace").lower()
    except OSError:
        return ""
    for marker in (
        "pool-b-skip:",
        "solver-hunt-skip:",
        "do not pool-b mine",
        "do not solver-hunt",
        "dead: not solvable",
    ):
        if marker in text:
            return f"per-task notes contain '{marker}'"
    return ""


def load_preflight_docs() -> dict[str, Any]:
    return {
        "registry_doc": read_json(REGISTRY_JSON, {}),
        "queue_doc": read_json(SUBMIT_QUEUE_JSON, {}),
        "state_doc": read_json(STATE_JSON, {}),
        "task_outcome_doc": read_json(TASK_OUTCOME_JSON, {}),
    }


def pool_b_preflight_for(
    *,
    task_id: str,
    assignee: str,
    cards: list[dict[str, Any]],
    docs: dict[str, Any],
    current_solver: dict[str, Any],
) -> dict[str, Any]:
    if current_solver["exists"] and not current_solver["is_zero_pad_tier"]:
        lineage_base = f"current-best in submission.zip ({current_solver['size']} bytes)"
        new_evidence = (
            f"Pool B (curated-SOLVED-list complement). submission.zip already contains "
            f"task{task_id[-3:]}.onnx at {current_solver['size']} bytes — a real working "
            f"solver. Goal: beat its cost OR replace with a different family that's "
            f"smaller / more correct on hidden test."
        )
    else:
        lineage_base = POOL_B_LINEAGE_BASE
        new_evidence = (
            "Pool B new-solver hunt. Task currently has no working ONNX in "
            "submission.zip (zero-pad-tier or missing). Any working candidate "
            "is novel evidence."
        )
    return build_worker_preflight(
        task_id=task_id,
        assignee=assignee,
        method_family=POOL_B_METHOD_FAMILY,
        lineage_base_candidate_id=lineage_base,
        new_evidence=new_evidence,
        cards=cards,
        registry_doc=docs.get("registry_doc") if isinstance(docs.get("registry_doc"), dict) else {},
        queue_doc=docs.get("queue_doc") if isinstance(docs.get("queue_doc"), dict) else {},
        state_doc=docs.get("state_doc") if isinstance(docs.get("state_doc"), dict) else {},
        task_outcome_doc=docs.get("task_outcome_doc") if isinstance(docs.get("task_outcome_doc"), dict) else {},
        repo_root=REPO_ROOT,
        self_kanban_task_id="",
    )


def _exhaustion_section(
    task_id: str,
    submitted_history: list[dict[str, Any]],
) -> list[str]:
    """Render the "this task is exhausted by public-test failures; you need
    a fundamentally different algorithm family" framing for an exhausted
    task. Includes prior candidate IDs + their Kaggle outcomes so the
    worker can audit what's already been tried."""
    lines = [
        "",
        "============================================================",
        f"  EXHAUSTED TASK NOTICE: prior Kaggle attempts on {task_id}",
        "============================================================",
        "",
        "This task has been submitted to Kaggle one or more times. ALL prior",
        "attempts returned PUBLIC-TIE / PUBLIC-NO-IMPROVE / PUBLIC-NEGATIVE.",
        "The local score_network_v2 gain DID NOT translate into a real LB",
        "improvement — meaning the hidden test doesn't reward the shrink",
        "pattern this task's existing solver(s) use.",
        "",
        "Prior submitted candidates (most recent first):",
    ]
    if not submitted_history:
        lines.append("  (queue history lookup returned no entries — check submit_queue.json)")
    for hit in submitted_history[:6]:
        cid = hit.get("candidate_id") or "?"
        status = hit.get("status") or "?"
        lines.append(f"  - {cid[:70]}")
        lines.append(f"      status: {status}")
    if len(submitted_history) > 6:
        lines.append(f"  ... and {len(submitted_history) - 6} more")
    lines.extend([
        "",
        "**What this task NEEDS now:**",
        "  1. A FUNDAMENTALLY DIFFERENT algorithm family — NOT another shrink of",
        "     the same Conv/Slice/Pad lane that's already been tried.",
        "     Examples: switch from gather-based decode to matmul, from",
        "     stride-based shift to ScatterND, from per-channel max to einsum",
        "     contraction, etc. The shape of the computation must differ.",
        "  2. Local evidence that the NEW family generalizes beyond what the",
        "     visible test set rewards. Cross-check against adversarial",
        "     perturbations AND against synthetic arc-gen variations.",
        "  3. Strong adversarial parity (>= 99% match vs live on the official",
        "     encoder, NOT just the cropped-decode shortcut).",
        "",
        "**What this task does NOT need:**",
        "  - Another -50 cost shrink of the same family that's already tied.",
        "  - A different fp16/fp32 cast of an existing tied submission.",
        "  - A minor restructuring (e.g., 3 ops collapsed to 2) without",
        "    changing the underlying computation.",
        "",
        "If you don't have a fundamentally-different idea in mind, BLOCK with",
        "reason 'human-eyes: task exhausted — needs novel algorithm family;",
        "no fresh idea available' and ask the operator to either retire the",
        "task (add `pool-b-skip:` to its notes) or provide guidance.",
        "============================================================",
        "",
    ])
    return lines


def card_body_for(
    *,
    task_id: str,
    assignee: str,
    preflight: dict[str, Any],
    current_solver: dict[str, Any],
    is_exhausted: bool = False,
    submitted_history: list[dict[str, Any]] | None = None,
) -> str:
    if current_solver["exists"] and not current_solver["is_zero_pad_tier"]:
        lineage_line = f"lineage_base: current-best in submission.zip ({current_solver['size']} bytes)"
        goal_lines = [
            f"- Current state: submission.zip already contains {task_id}.onnx at "
            f"{current_solver['size']} bytes — a real working solver, NOT zero-pad.",
            f"- Goal: improve on the current solver. Either lower its score_network_v2 cost",
            "  while staying correct, OR replace it with a different family that handles",
            "  hidden-test edge cases the current one might miss. A candidate that ties on",
            "  cost but with cleaner / more general structure is also valuable.",
            "- This task is NOT in the curated `inspect_tasks.SOLVED_TASKS` set, meaning",
            "  the human-side review hasn't yet blessed the current solver — that's why",
            "  it lives in Pool B and not Pool A's shrink cursor.",
        ]
    elif current_solver["exists"]:
        lineage_line = (
            f"lineage_base: zero-pad-tier baseline ({current_solver['size']} bytes — "
            f"probably constant-output / trivial)"
        )
        goal_lines = [
            f"- Current state: submission.zip has {task_id}.onnx at "
            f"{current_solver['size']} bytes — tiny enough to be a constant-output or "
            f"zero-pad placeholder.",
            "- Goal: build a real working ONNX. The current entry is the baseline to beat.",
        ]
    else:
        lineage_line = "lineage_base: zero-padding-baseline (no entry in submission.zip)"
        goal_lines = [
            "- Current state: no entry for this task in submission.zip.",
            "- Goal: produce the FIRST working ONNX for this task. There is no",
            "  current-best to beat — only the zero-padding baseline.",
        ]
    lines = [
        f"task_id: {task_id}",
        f"assignee_target: {assignee}",
        f"pool: B (curated-SOLVED-list complement; existing solver may still need improvement)",
        lineage_line,
        f"method_family: {POOL_B_METHOD_FAMILY}",
    ]
    if is_exhausted:
        lines.extend(_exhaustion_section(task_id, submitted_history or []))
    lines.extend([
        "",
        "Worker instructions (Pool B):",
        *goal_lines,
        "- Inspect the task with: .venv/Scripts/python tools/inspect_tasks.py <NNN>",
        f"- Read existing notes: The_400_Tasks/Task_by_Task/{task_id}.md",
        "- Build candidates in The_400_Tasks/Task_by_Task/<task>/prototypes/.",
        "- Local gates before handoff (block --reason 'review-required: ...'):",
        "  * verify_against_live --tid NNN  passes on all train pairs",
        "  * score_network_v2 cost reported (no floor; honest number)",
        "  * adversarial perturbations >=95% where applicable",
        "  * at least 3 documented attempts in the per-task notes file",
        "- Hand-off JSON must include `worker_owner` — see the KanbanWorkerLoop",
        "  playbook for the full JSON shape. The auto-triage cron reads that JSON",
        "  to decide review-card vs bank.",
        "",
        f"Title schema: this card was created as `[worker {task_id} {assignee} solver-hunt]`.",
        "When handing off, do NOT rewrite this title; just block with a reason.",
    ])
    footer = preflight.get("required_card_footer")
    if isinstance(footer, list):
        lines.append("")
        lines.append("Preflight:")
        lines.extend(str(item) for item in footer)
    return "\n".join(lines)


def plan_pool_b(
    cards: list[dict[str, Any]],
    *,
    docs: dict[str, Any] | None = None,
    workers: tuple[str, ...] = POOL_B_SOLVER_HUNT_WORKERS,
    create_limit: int = 4,
    max_per_worker: int = 1,
    unsolved_override: list[str] | None = None,
    today: str | None = None,
) -> dict[str, Any]:
    docs = docs or {}
    today = today or _utc_today()
    audit = audit_worker_queue(cards, queue_workers=workers)
    idle_workers = list(audit.get("idle_queue_workers") or [])
    unsolved = list(unsolved_override) if unsolved_override is not None else all_unsolved_task_ids()
    claimed = actively_claimed_task_ids(cards)
    # Read submit_queue from docs so callers can override in tests.
    queue_doc = (docs or {}).get("queue_doc")
    if not isinstance(queue_doc, dict):
        queue_doc = read_json(SUBMIT_QUEUE_JSON, {})
    exhausted = task_ids_exhausted_by_public_history(queue_doc)
    # Deprioritize (don't skip) tasks that have failed public-test: fresh
    # tasks come first, exhausted tasks come last. When an exhausted task
    # does get picked, the card body tells the worker that prior attempts
    # came back public-tie / public-no-improve, so they should try a
    # FUNDAMENTALLY DIFFERENT algorithm family rather than another shrink
    # variant. The worker still has agency — they can skip if they don't
    # have a different idea ready.
    not_claimed = [t for t in unsolved if t not in claimed]
    fresh = [t for t in not_claimed if t not in exhausted]
    deprioritized = [t for t in not_claimed if t in exhausted]
    available = fresh + deprioritized

    actions: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    used_per_worker: dict[str, int] = {w: 0 for w in workers}
    available_iter = iter(available)
    remaining_budget = max(0, create_limit)

    for worker in idle_workers:
        if remaining_budget <= 0:
            break
        if used_per_worker.get(worker, 0) >= max_per_worker:
            continue
        # Drain through `available` until we find a task that's allowed,
        # logging skips along the way so the operator can see why a task
        # was passed over.
        chosen = ""
        chosen_preflight: dict[str, Any] = {}
        chosen_solver_info: dict[str, Any] = {}
        while True:
            try:
                task_id = next(available_iter)
            except StopIteration:
                break
            dead_reason = per_task_notes_say_dead(task_id)
            if dead_reason:
                skipped.append({"task_id": task_id, "reason": dead_reason})
                continue
            # `pool-b-skip-exhausted:` lets a worker permanently opt out of
            # picking up exhausted tasks. Default: pick them (with framing)
            # because the operator wants the LB upside if a different
            # algorithm family is available.
            solver_info = current_solver_info(task_id)
            preflight = pool_b_preflight_for(
                task_id=task_id,
                assignee=worker,
                cards=cards,
                docs=docs,
                current_solver=solver_info,
            )
            if preflight.get("verdict") == "DENY":
                skipped.append(
                    {
                        "task_id": task_id,
                        "assignee": worker,
                        "reason": "preflight DENY",
                        "deny_reasons": preflight.get("deny_reasons"),
                    }
                )
                continue
            chosen = task_id
            chosen_preflight = preflight
            chosen_solver_info = solver_info
            break
        if not chosen:
            break  # ran out of unsolved tasks; nothing else to do this run
        chosen_is_exhausted = chosen in exhausted
        chosen_history = (
            submitted_candidate_history_for_task(chosen, queue_doc)
            if chosen_is_exhausted else []
        )
        actions.append(
            {
                "op": "create_pool_b_card",
                "task_id": chosen,
                "assignee": worker,
                "title": f"[worker {chosen} {worker} solver-hunt] new-solver candidate hunt",
                "body": card_body_for(
                    task_id=chosen,
                    assignee=worker,
                    preflight=chosen_preflight,
                    current_solver=chosen_solver_info,
                    is_exhausted=chosen_is_exhausted,
                    submitted_history=chosen_history,
                ),
                "priority": 55,
                "idempotency_key": f"ng-pool-b-{worker}-{chosen}-{today}",
            }
        )
        used_per_worker[worker] = used_per_worker.get(worker, 0) + 1
        remaining_budget -= 1

    return {
        "schema": "neurogolf.pool_b_autoforeman.v1",
        "actions": actions,
        "skipped": skipped,
        "queue_ok": audit.get("ok"),
        "idle_workers_start": list(audit.get("idle_queue_workers") or []),
        "pool_b_workers": list(workers),
        "unsolved_total": len(unsolved),
        "unsolved_claimed_total": sum(1 for t in unsolved if t in claimed),
        "unsolved_available_total": len(available),
        "action_count": len(actions),
        "skipped_count": len(skipped),
        "today_bucket": today,
    }


def apply_pool_b_plan(client: KanbanClient, plan: dict[str, Any]) -> None:
    for action in plan.get("actions", []):
        if action.get("op") != "create_pool_b_card":
            continue
        client.create(
            str(action["title"]),
            body=str(action["body"]),
            assignee=str(action["assignee"]),
            workspace="worktree",
            priority=int(action["priority"]),
            created_by="neurogolf-pool-b-autoforeman",
            idempotency_key=str(action["idempotency_key"]),
            max_runtime="6h",
            # Pool B workers (7-8) point at a closed port — the Hermes
            # dispatcher will try to spawn them, fail instantly, and the
            # consecutive-crash counter will trip. max_retries=1 means the
            # card blocks on the FIRST crash instead of after 2, halving
            # the spam. The pool-b auto-unblock cron then bounces it back
            # to ready so the manual Claude Code /loop session can claim it.
            max_retries=1,
        )


def render_text(report: dict[str, Any]) -> str:
    lines = [
        (
            f"actions={report['action_count']} skipped={report['skipped_count']} "
            f"queue_ok={report['queue_ok']} unsolved_total={report['unsolved_total']} "
            f"unsolved_available={report['unsolved_available_total']} "
            f"idle_workers={report['idle_workers_start']}"
        ),
    ]
    for action in report["actions"]:
        lines.append(f"- {action['op']} {action['task_id']} -> {action.get('assignee') or 'bank'}")
    for item in report["skipped"][:12]:
        lines.append(f"- skipped {item.get('task_id')}: {item.get('reason')}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--kanban-json", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--create-limit", type=int, default=4)
    parser.add_argument("--max-per-worker", type=int, default=1)
    args = parser.parse_args(argv)

    client = KanbanClient(args.repo_root, dry_run=not args.apply)
    if args.kanban_json:
        cards = json.loads(args.kanban_json.read_text(encoding="utf-8"))
    else:
        cards = client.load_cards()
    if not isinstance(cards, list):
        raise SystemExit("expected Kanban JSON list")

    report = plan_pool_b(
        [card for card in cards if isinstance(card, dict)],
        docs=load_preflight_docs(),
        create_limit=args.create_limit,
        max_per_worker=args.max_per_worker,
    )
    if args.apply:
        apply_pool_b_plan(client, report)
        report["applied_actions"] = client.actions
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(render_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
