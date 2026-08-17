"""submit_gate orchestrator: lock, audit log, main entry point.

The runner ties the pure decision logic (``submit_gate.py``), the mechanical
verification (``submit_gate_mechanical.py``), and the Kaggle wrapper
(``submit_gate_kaggle.py``) into a single ``main()`` invocation suitable for
the existing 2-min cron wrapper at ``<HERMES_HOME>/scripts/``.

Per-run flow:
1. Acquire the global ``neurogolf_python_control.lock``. If held by another
   tick, exit 0 silently — next cron tick retries.
2. List ``blocked`` cards with ``review-required`` handoff JSON.
3. For each: load pass state, mechanical-check, classify, apply verdict.
4. Write an audit log row per card to ``Logs/submit_gate_<UTC>.md``.
5. Release the lock.

The runner is intentionally thin — every heavy lift is in a tested submodule.
The integration tests in ``test_submit_gate_runner.py`` cover the wiring;
the unit tests in the submodules cover the leaf behaviors.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.algorithm_family_directive import normalize_family
from tools.auto_triage_blocked import (
    Handoff,
    _stderr_utf8,
    _stdout_utf8,
    parse_handoff,
    run_hermes,
)
from tools.submit_gate import (
    DEFAULT_GAIN_FLOOR,
    DEFAULT_MAX_ATTEMPTS_PER_PASS,
    DeferredForBudget,
    DeferredForRateLimit,
    DeferredForStaleLineage,
    GiveUpForPass,
    PassState,
    Retry,
    Submit,
    VerifyResult,
    classify_handoff,
    normalized_family_for_handoff,
)
from tools.submit_gate_kaggle import (
    SubmitOutcome,
    load_budget_state,
    load_lineage_state,
    submit_with_safety,
)
from tools.submit_gate_mechanical import MechanicalCheckInput, run_mechanical_checks
from tools.task_outcome_registry_io import (
    bump_pass_state,
    load_registry,
    read_pass_state,
    record_known_attempt,
    save_registry,
)


# ============================================================================
# Constants
# ============================================================================


CONTROL_LOCK_PATH = Path(os.environ.get("NEUROGOLF_CONTROL_LOCK", Path.home() / ".hermes" / "neurogolf_python_control.lock"))
LOCK_STALE_SECONDS = 20 * 60
LOCK_WAIT_SECONDS = 55.0
LOCK_RETRY_SECONDS = 0.5

DEFAULT_REGISTRY_PATH = REPO_ROOT / ".hermes" / "board" / "task_outcome_registry.json"
# Per tools/neurogolf_cron_snapshot.py:47 — the actual cursor with pass_count
# lives in task_backlog_cursor.json, not state.json. The two share a directory
# but state.json is a different document (workflow_refs, agents, etc.).
DEFAULT_STATE_PATH = REPO_ROOT / ".hermes" / "board" / "task_backlog_cursor.json"
DEFAULT_AUDIT_DIR = REPO_ROOT / "Logs"


# ============================================================================
# Control lock — same shape as scripts/auto_triage_blocked.py wrapper
# ============================================================================


class ControlLock:
    """File-based lock honoring the existing neurogolf_python_control protocol."""

    def __init__(
        self,
        path: Path = CONTROL_LOCK_PATH,
        *,
        stale_seconds: int = LOCK_STALE_SECONDS,
        wait_seconds: float = LOCK_WAIT_SECONDS,
        retry_seconds: float = LOCK_RETRY_SECONDS,
    ) -> None:
        self.path = path
        self.stale_seconds = stale_seconds
        self.wait_seconds = wait_seconds
        self.retry_seconds = retry_seconds
        self.fd: int | None = None

    def __enter__(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = self._try_open()
        deadline = time.monotonic() + self.wait_seconds
        while self.fd is None:
            if self._is_stale():
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
                self.fd = self._try_open()
                continue
            if time.monotonic() >= deadline:
                return False
            time.sleep(self.retry_seconds)
            self.fd = self._try_open()
        return True

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self.fd is None:
            return
        os.close(self.fd)
        self.fd = None
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

    def _try_open(self) -> int | None:
        try:
            fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except (FileExistsError, PermissionError):
            return None
        payload = f"pid={os.getpid()} started={time.time():.0f}\n".encode("utf-8")
        os.write(fd, payload)
        return fd

    def _is_stale(self) -> bool:
        try:
            age = time.time() - self.path.stat().st_mtime
        except FileNotFoundError:
            return False
        return age > self.stale_seconds


# ============================================================================
# Audit log
# ============================================================================


@dataclass
class AuditRow:
    """One processed card -> one row in the audit log."""

    card_id: str
    task_id: str
    attempts_before: int
    predicted_gain: float | None
    verdict_kind: str
    verdict_reason: str
    submit_invoked: bool
    notes: str = ""
    handoff_summary: dict[str, Any] = field(default_factory=dict)


def write_audit_log(
    rows: list[AuditRow],
    *,
    out_dir: Path = DEFAULT_AUDIT_DIR,
    now: datetime | None = None,
    dry_run: bool = False,
) -> Path:
    """Write rows to Logs/submit_gate_<UTC>.md. Returns the log path.

    Even dry-runs write logs — that's how shadow-mode rollout produces a
    diff against the LLM submit-captain's behavior.
    """
    now = now or datetime.now(timezone.utc)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = now.strftime("%Y%m%dT%H%M%SZ")
    suffix = "_dryrun" if dry_run else ""
    out_path = out_dir / f"submit_gate_{timestamp}{suffix}.md"

    lines = [
        f"# submit_gate  {now.isoformat().replace('+00:00', 'Z')}",
        "",
        f"Mode: **{'DRY-RUN' if dry_run else 'APPLY'}**",
        f"Cards processed: {len(rows)}",
        "",
        "## Verdicts",
        "",
        "| card | task | attempts | predicted_gain | verdict | submit? | reason |",
        "|------|------|----------|----------------|---------|---------|--------|",
    ]
    for row in rows:
        gain_str = f"{row.predicted_gain:+.4f}" if row.predicted_gain is not None else "(none)"
        lines.append(
            f"| `{row.card_id}` | {row.task_id} | {row.attempts_before} | "
            f"{gain_str} | {row.verdict_kind} | "
            f"{'yes' if row.submit_invoked else 'no'} | {row.verdict_reason} |"
        )
    lines.append("")
    lines.append("## Handoff extracts (forensics)")
    lines.append("")
    for row in rows:
        lines.append(f"### `{row.card_id}` — {row.task_id}")
        lines.append("```json")
        lines.append(json.dumps(row.handoff_summary, indent=2, default=str))
        lines.append("```")
        if row.notes:
            lines.append(f"Notes: {row.notes}")
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


# ============================================================================
# Per-card processing
# ============================================================================


@dataclass
class CardContext:
    """Bundle of paths and per-card derived state for one processing iteration."""

    card_id: str
    task_id: str
    handoff: Handoff
    candidate_zip_path: Path | None
    current_best_zip_path: Path | None
    pass_state: PassState


def list_blocked_review_required() -> list[dict[str, Any]]:
    """Return blocked cards that have a review-required reason. Reuses
    auto_triage_blocked's hermes wrappers."""
    proc = run_hermes(["list", "--status", "blocked", "--json"])
    if proc.returncode != 0:
        return []
    try:
        raw = json.loads(_stdout_utf8(proc))
    except json.JSONDecodeError:
        return []
    if not isinstance(raw, list):
        return []
    return [
        card for card in raw
        if isinstance(card, dict)
        and "review-required" in (card.get("latest_summary") or card.get("title") or "").lower()
    ]


def process_card(
    card: dict[str, Any],
    *,
    registry: dict[str, Any],
    current_pass: int,
    candidate_zip_resolver: Callable[[dict[str, Any], Handoff], Path | None],
    current_best_zip_path: Path | None,
    score_fn: Callable | None = None,
    verify_fn: Callable | None = None,
    submit_fn: Callable | None = None,
    verify_after_fn: Callable | None = None,
    notifier_fn: Callable | None = None,
    budget_loader: Callable[[], Any] | None = None,
    lineage_loader: Callable[[Handoff], Any] | None = None,
    kanban_writer: "KanbanWriter | None" = None,
    gain_floor: float = DEFAULT_GAIN_FLOOR,
    max_attempts_per_pass: int = DEFAULT_MAX_ATTEMPTS_PER_PASS,
    dry_run: bool = True,
) -> AuditRow:
    """Process one blocked card end-to-end.

    Returns an AuditRow describing what happened. Mutates ``registry`` in place
    (caller saves it atomically once per cron tick).

    ``candidate_zip_resolver`` is the only non-trivial wiring — it figures out
    where the candidate's .zip lives given the kanban card + handoff. In
    production it inspects the worktree path on the card; in tests it returns
    a path directly.
    """
    card_id = card.get("id", "")
    handoff = parse_handoff(card_id)

    if handoff.parse_error:
        return AuditRow(
            card_id=card_id,
            task_id=handoff.task_id or "?",
            attempts_before=0,
            predicted_gain=None,
            verdict_kind="UnparseableHandoff",
            verdict_reason=f"could not parse handoff: {handoff.parse_error}",
            submit_invoked=False,
            handoff_summary={"raw_excerpt": handoff.raw_text[:200]},
        )

    task_id = handoff.task_id or "?"
    pass_state_dict = read_pass_state(registry, task_id, current_pass)
    pass_state = PassState(
        attempts_this_pass=pass_state_dict["attempts_this_pass"],
        attempted_families=pass_state_dict["attempted_families"],
        current_pass=current_pass,
    )

    candidate_zip_path = candidate_zip_resolver(card, handoff)
    if candidate_zip_path is None:
        verify = VerifyResult.failed(
            "candidate zip path could not be resolved from card / handoff"
        )
    else:
        verify = run_mechanical_checks(
            MechanicalCheckInput(
                handoff=handoff,
                candidate_zip_path=candidate_zip_path,
                current_best_zip_path=current_best_zip_path,
                task_id=task_id,
            ),
            score_fn=score_fn,
            verify_fn=verify_fn,
        )

    budget = budget_loader() if budget_loader else None
    lineage = lineage_loader(handoff) if lineage_loader else None

    verdict = classify_handoff(
        handoff,
        verify=verify,
        pass_state=pass_state,
        budget=budget,
        lineage=lineage,
        gain_floor=gain_floor,
        max_attempts_per_pass=max_attempts_per_pass,
    )

    submit_invoked = False
    notes_parts: list[str] = []
    submit_outcome: SubmitOutcome | None = None
    verdict_kind = verdict.__class__.__name__

    if isinstance(verdict, Submit):
        if not dry_run and submit_fn is not None and candidate_zip_path is not None:
            submit_outcome = submit_with_safety(
                handoff,
                candidate_zip_path,
                submit_fn=submit_fn,
                verify_after_fn=verify_after_fn,
                notifier_fn=notifier_fn,
            )
            submit_invoked = True
            notes_parts.append(f"submit_status={submit_outcome.status}")
            if submit_outcome.status == "ok":
                family = normalized_family_for_handoff(handoff)
                bump_pass_state(registry, task_id, current_pass, family=family)
                record_known_attempt(
                    registry,
                    task_id,
                    candidate=handoff.candidate_id,
                    result=f"submitted, predicted +{handoff.expected_point_gain:.4f}",
                    evidence=f"submit_gate audit log {datetime.now(timezone.utc).isoformat()}",
                )
                # Kanban side effects: complete the worker card; create retry card
                # if forced exploration (also_retry=True).
                if kanban_writer is not None:
                    kanban_writer.complete_blocked(
                        card_id=card_id,
                        result=f"submit_gate: submitted ({verdict.reason})",
                    )
                    if verdict.also_retry:
                        attempts_now = registry["tasks"][task_id]["attempts_this_pass"]
                        families_now = registry["tasks"][task_id]["attempted_families"]
                        retry_spec = build_retry_card_spec(
                            original_card=card,
                            handoff=handoff,
                            verdict_reason=verdict.reason,
                            directive=verdict.retry_directive or "",
                            retry_attempt_index=attempts_now + 1,
                            attempted_families=families_now,
                        )
                        kanban_writer.create_retry(retry_spec)
                        notes_parts.append(f"retry_card_chain={retry_spec['chain_key']}")
        else:
            # dry-run or no submit_fn wired — still record what would happen
            if not dry_run:
                notes_parts.append("submit_fn not wired; dry-run semantics applied")
    elif isinstance(verdict, Retry):
        if not dry_run:
            family = normalized_family_for_handoff(handoff)
            bump_pass_state(registry, task_id, current_pass, family=family)
            record_known_attempt(
                registry,
                task_id,
                candidate=handoff.candidate_id or "(no candidate)",
                result=f"retry: {verdict.reason}",
                evidence=f"submit_gate audit log {datetime.now(timezone.utc).isoformat()}",
            )
            if kanban_writer is not None:
                attempts_now = registry["tasks"][task_id]["attempts_this_pass"]
                families_now = registry["tasks"][task_id]["attempted_families"]
                retry_spec = build_retry_card_spec(
                    original_card=card,
                    handoff=handoff,
                    verdict_reason=verdict.reason,
                    directive=verdict.directive,
                    retry_attempt_index=attempts_now + 1,
                    attempted_families=families_now,
                )
                kanban_writer.create_retry(retry_spec)
                kanban_writer.complete_blocked(
                    card_id=card_id,
                    result=f"submit_gate: retry directive issued ({verdict.reason})",
                )
                notes_parts.append(f"retry_card_chain={retry_spec['chain_key']}")
    elif isinstance(verdict, GiveUpForPass):
        if not dry_run:
            family = normalized_family_for_handoff(handoff)
            bump_pass_state(registry, task_id, current_pass, family=family)
            record_known_attempt(
                registry,
                task_id,
                candidate=handoff.candidate_id or "(no candidate)",
                result=f"give-up-for-pass: {verdict.last_attempt_reason}",
                evidence=f"submit_gate audit log {datetime.now(timezone.utc).isoformat()}",
            )
            entry = registry.setdefault("tasks", {}).setdefault(task_id, {})
            entry["last_attempt_summary"] = verdict.notes_summary
            if kanban_writer is not None:
                kanban_writer.complete_blocked(
                    card_id=card_id,
                    result=f"submit_gate: give-up-for-pass ({verdict.notes_summary})",
                )
    elif isinstance(verdict, DeferredForStaleLineage):
        # Lineage stale — route back to worker with new-base directive. No
        # attempt consumed because it's a coordination issue.
        if not dry_run and kanban_writer is not None:
            retry_spec = build_retry_card_spec(
                original_card=card,
                handoff=handoff,
                verdict_reason=verdict.reason,
                directive=verdict.directive,
                retry_attempt_index=pass_state.attempts_this_pass + 1,
                attempted_families=pass_state.attempted_families,
            )
            kanban_writer.create_retry(retry_spec)
            kanban_writer.complete_blocked(
                card_id=card_id,
                result=f"submit_gate: stale-lineage rebase directive ({verdict.reason})",
            )
            notes_parts.append(f"stale_lineage_retry_chain={retry_spec['chain_key']}")
    # DeferredForBudget / DeferredForRateLimit: leave card blocked, next tick
    # will retry. No kanban mutation here.

    return AuditRow(
        card_id=card_id,
        task_id=task_id,
        attempts_before=pass_state.attempts_this_pass,
        predicted_gain=handoff.expected_point_gain,
        verdict_kind=verdict_kind,
        verdict_reason=verdict.reason,
        submit_invoked=submit_invoked,
        notes="; ".join(notes_parts),
        handoff_summary={
            "candidate_id": handoff.candidate_id,
            "package_sha": handoff.package_sha[:16] + "..." if handoff.package_sha else "",
            "algorithm_family": normalize_family(handoff.algorithm_family),
            "expected_point_gain": handoff.expected_point_gain,
            "lineage_base": handoff.lineage_base,
            "changed_entries": handoff.changed_entries,
        },
    )


# ============================================================================
# main()
# ============================================================================


def read_current_pass(state_path: Path = DEFAULT_STATE_PATH) -> int:
    """Read ``pass_count`` from .hermes/board/state.json (0 when missing)."""
    if not state_path.exists():
        return 0
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    if not isinstance(data, dict):
        return 0
    raw = data.get("pass_count")
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw)
    return 0


class KanbanWriter:
    """Injectable interface for kanban mutations.

    Default implementation shells out to ``hermes kanban``; tests stub by
    creating a subclass or a duck-typed object that records calls.
    """

    def create_retry(self, spec: dict[str, Any]) -> dict[str, Any]:
        """Create a new kanban card from a build_retry_card_spec result.

        Returns ``{"id": <new_id>, "ok": bool, "error": str}``. Idempotent
        via the chain_key — if the kanban already has a card with this
        chain_key in an active state, ``hermes kanban create`` returns the
        existing id (per --idempotency-key behavior).
        """
        cmd = [
            "create",
            spec["title"],
            "--body", spec["body"],
            "--priority", str(spec.get("priority", 70)),
            "--idempotency-key", spec["chain_key"],
            "--created-by", "tools/submit_gate.py",
            "--json",
        ]
        parent_id = spec.get("parent_id")
        if parent_id:
            cmd.extend(["--parent", parent_id])
        proc = run_hermes(cmd)
        out = _stdout_utf8(proc)
        if proc.returncode != 0:
            return {"id": "", "ok": False, "error": out[:200]}
        try:
            data = json.loads(out)
        except json.JSONDecodeError:
            return {"id": "", "ok": False, "error": f"non-JSON output: {out[:200]}"}
        return {"id": data.get("id", ""), "ok": True, "error": ""}

    def complete_blocked(self, *, card_id: str, result: str) -> dict[str, Any]:
        """Mark a blocked card as completed with the given result message.

        Caller is responsible for sequencing — we expect to be called AFTER
        any retry card creation, so the lane frees up exactly when its
        successor is queued.
        """
        proc = run_hermes(["complete", card_id, "--result", result])
        if proc.returncode != 0:
            return {"ok": False, "error": _stderr_utf8(proc)[:200]}
        return {"ok": True, "error": ""}


def build_retry_card_spec(
    *,
    original_card: dict[str, Any],
    handoff: Handoff,
    verdict_reason: str,
    directive: str,
    retry_attempt_index: int,
    attempted_families: list[str],
) -> dict[str, Any]:
    """Produce the hermes-kanban-create spec for a retry card.

    Body includes the original card's id as ``source_worker_task_id``,
    the directive text built by ``algorithm_family_directive.build_directive``,
    the family blacklist, and the retry-attempt-index. ``route_repair_cards.py``
    picks up the new card via its parent_id link and assigns it back to the
    original worker (or whichever worker the dep-serializer logic decides).

    The chain_key uses the original card id + retry index so multiple retries
    on the same task get distinct cards rather than colliding via idempotency.
    """
    task_id = handoff.task_id
    candidate_id = handoff.candidate_id or "(no candidate)"
    chain_key = f"retry:{original_card.get('id', 'x')}:attempt{retry_attempt_index}"
    # IMPORTANT: title must match tools/route_repair_cards.py REPAIR_TITLE_PATTERNS
    # so that script routes the unassigned retry card back to the original
    # worker. ``[repair task<NNN>]`` is the established pattern; we tack on
    # the retry index so it's clear this is a submit_gate retry, not a
    # public-negative repair.
    title = f"[repair {task_id}] submit_gate retry-{retry_attempt_index} from {original_card.get('id', '?')}"[:200]
    blacklist_str = ", ".join(f"`{f}`" for f in attempted_families) if attempted_families else "(none yet)"
    body_lines = [
        f"task_id: {task_id}",
        f"source_worker_task_id: {original_card.get('id', '')}",
        f"source_candidate_id: {candidate_id}",
        f"retry_attempt_index: {retry_attempt_index}",
        f"do_not_use_families: {blacklist_str}",
        f"submit_gate_reason: {verdict_reason}",
        "",
        directive,
    ]
    return {
        "title": title,
        "body": "\n".join(body_lines),
        "priority": 70,
        "parent_id": original_card.get("id", ""),
        "chain_key": chain_key,
        # No --assignee here: route_repair_cards routes by parent on the next
        # dep-serializer tick. This keeps the worker-routing logic centralized.
    }


def default_candidate_zip_resolver(card: dict[str, Any], handoff: Handoff) -> Path | None:
    """Try standard locations for the candidate zip.

    Order:
        1. Worktree at ``card.workspace_path / submissions / *.zip``
        2. The_400_Tasks/Task_by_Task/<task>/artifacts/<candidate_id>/*.zip
        3. None (gives up; mechanical check fails with a clean error)
    """
    candidate_id = handoff.candidate_id
    task_id = handoff.task_id

    workspace_path = card.get("workspace_path")
    if workspace_path:
        wp = Path(workspace_path)
        if wp.exists():
            submissions = wp / "submissions"
            if submissions.is_dir():
                # Prefer a zip whose name mentions the candidate_id
                for zip_path in submissions.glob("*.zip"):
                    if candidate_id and candidate_id in zip_path.name:
                        return zip_path
                # Fall back to any .zip in the submissions folder
                zips = list(submissions.glob("*.zip"))
                if zips:
                    return zips[0]

    if task_id and candidate_id:
        artifacts = REPO_ROOT / "The_400_Tasks" / "Task_by_Task" / task_id / "artifacts" / candidate_id
        if artifacts.is_dir():
            for zip_path in artifacts.glob("*.zip"):
                return zip_path

    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Mutate state (kanban, registry, Kaggle). Default is dry-run.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--gain-floor", type=float, default=DEFAULT_GAIN_FLOOR)
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS_PER_PASS)
    parser.add_argument("--json", action="store_true", help="Emit JSON summary on stdout for cron consumption.")
    parser.add_argument("--no-lock", action="store_true", help="Skip the control lock (test harness only).")
    args = parser.parse_args(argv)

    dry_run = not args.apply
    now = datetime.now(timezone.utc)

    if not args.no_lock:
        lock_ctx = ControlLock()
    else:
        lock_ctx = _NullContext()

    with lock_ctx as acquired:
        if not acquired:
            # Couldn't get the lock within timeout — exit silently per the
            # established control-runner protocol.
            if args.json:
                json.dump({"schema": "neurogolf.submit_gate.v1", "skipped": "lock_contention"}, sys.stdout)
            return 0

        current_pass = read_current_pass(args.state)
        registry = load_registry(args.registry)
        registry_snapshot_for_rollback = json.loads(json.dumps(registry))

        cards = list_blocked_review_required()
        current_best_zip = REPO_ROOT / "submissions" / "submission.zip"
        current_best_zip_path = current_best_zip if current_best_zip.exists() else None

        # Real KanbanWriter for production; tests inject their own.
        kanban_writer = KanbanWriter() if not dry_run else None
        rows: list[AuditRow] = []
        try:
            for card in cards:
                row = process_card(
                    card,
                    registry=registry,
                    current_pass=current_pass,
                    candidate_zip_resolver=default_candidate_zip_resolver,
                    current_best_zip_path=current_best_zip_path,
                    score_fn=None,  # heavy ORT loop; production wiring fills this in
                    verify_fn=None,
                    submit_fn=None,  # production wiring fills in the Kaggle CLI invocation
                    verify_after_fn=None,
                    kanban_writer=kanban_writer,
                    budget_loader=load_budget_state,
                    lineage_loader=load_lineage_state,
                    gain_floor=args.gain_floor,
                    max_attempts_per_pass=args.max_attempts,
                    dry_run=dry_run,
                )
                rows.append(row)
        except Exception:
            # Roll back any in-memory registry mutations on unexpected error
            registry = registry_snapshot_for_rollback
            raise
        finally:
            if not dry_run:
                # Atomic write of accumulated mutations
                try:
                    save_registry(args.registry, registry)
                except Exception:
                    # If we can't write registry, leak the kanban mutations
                    # rather than crash the cron tick — they'll be visible in
                    # the next dry-run audit log diff.
                    pass

        log_path = write_audit_log(rows, out_dir=args.audit_dir, now=now, dry_run=dry_run)

    if args.json:
        json.dump(
            {
                "schema": "neurogolf.submit_gate.v1",
                "mode": "APPLY" if not dry_run else "DRY-RUN",
                "cards_processed": len(rows),
                "verdict_counts": _count_verdicts(rows),
                "log_path": str(log_path.relative_to(REPO_ROOT)) if log_path.is_relative_to(REPO_ROOT) else str(log_path),
            },
            sys.stdout,
        )
        sys.stdout.write("\n")
    else:
        print(f"submit_gate: {len(rows)} cards processed, log at {log_path.name}")

    return 0


def _count_verdicts(rows: list[AuditRow]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.verdict_kind] = counts.get(row.verdict_kind, 0) + 1
    return counts


class _NullContext:
    """Replacement for ControlLock when --no-lock is set (testing only)."""

    def __enter__(self) -> bool:
        return True

    def __exit__(self, *_args: object) -> None:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
