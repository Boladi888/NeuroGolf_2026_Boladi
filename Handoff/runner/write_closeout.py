#!/usr/bin/env python3
"""Generate a runner batch's completion sentinel + closeout report deterministically.

The next batch's prep should be able to read ONE file (the completion sentinel)
instead of cross-reading pid / state / monitor text / ops-log / the process
table to answer "is this batch done, and what happened?".

Mandated by the HandOffz skill:
  Handoff/skills/handoff-workflows/references/batch-closeout.md
  Handoff/skills/handoff-workflows/references/setup-checklist.md
    ("Batch-Scoped PIDs And Completion Sentinel" lists the exact sentinel fields)
and the proposal
  Agenters/_proposals/neurogolf_onnx_win_actions_handoff_cleanup_2026-07-03.md
    ("Make Runner Closeout The Source Of Truth For Next Batch").

Interface:
  python Handoff/runner/write_closeout.py --lane <Claude|Codex|Codex2> --batch N
         [--config PATH] [--apply] [--force]

Default is DRY-RUN: prints the sentinel JSON and the closeout markdown it WOULD
write, and writes nothing. --apply writes the sentinel (and, if absent, the
closeout markdown). It refuses to write unless the runner state shows the batch
is genuinely terminal (every task terminal AND the pid file state=stopped with
an exit_reason set), unless --force.

Reads only; the only mutations happen under --apply, and even then it never
overwrites a hand-written closeout markdown.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # keep em-dashes/unicode from mojibaking on a cp1252 (PowerShell) console
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ---------------------------------------------------------------------------
# Terminal-status vocabulary.
#
# Kept byte-identical to handoff_monitor.TERMINAL_STATUSES so "is the batch
# done?" is decided the same way here and in the monitor. We import it when the
# monitor is importable and fall back to an inline copy otherwise (e.g. when this
# script is run from a directory where the sibling import does not resolve). If
# the two ever drift, verify_terminal_vocab() prints a warning.
# ---------------------------------------------------------------------------
_FALLBACK_TERMINAL_STATUSES = {
    "ship_packeted",
    "captain_staged",
    "drop",
    "worker_failed",
    "review_failed",
    "review_parse_failed",
    "needs_packet_fix",
}

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from handoff_monitor import TERMINAL_STATUSES as _MONITOR_TERMINAL_STATUSES  # type: ignore

    TERMINAL_STATUSES = set(_MONITOR_TERMINAL_STATUSES)
    _TERMINAL_SOURCE = "handoff_monitor.TERMINAL_STATUSES"
except Exception:  # pragma: no cover - import fallback
    TERMINAL_STATUSES = set(_FALLBACK_TERMINAL_STATUSES)
    _TERMINAL_SOURCE = "inline fallback (handoff_monitor import failed)"

# Statuses that indicate a SHIP-ish outcome (a captain packet was produced) vs a
# DROP-ish outcome (no packet). Used only for the human-facing SHIP/DROP tally in
# the closeout markdown; the sentinel carries the full per-status breakdown.
SHIP_STATUSES = {"ship_packeted", "captain_staged"}
DROP_STATUSES = {"drop", "rejected"}

REPO = Path.cwd()


# ---------------------------------------------------------------------------
# Small IO helpers (utf-8-sig so PowerShell-written BOM'd JSON still parses --
# the pid file is written by the PS launcher and carries a BOM).
# ---------------------------------------------------------------------------
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_json_or_none(path: Path) -> Any:
    try:
        return read_json(path)
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, ValueError):
        return None


def lane_lower(lane: str) -> str:
    """Map a lane label to its lowercase file-path token.

    Claude -> claude, Codex -> codex, Codex2 -> codex2. This matches the config
    `lane_id`, the runner-state filename, and the pid-file basename convention.
    """
    return lane.strip().lower()


def first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Input resolution. Everything is anchored at the repo root (cwd) so the tool
# behaves identically no matter which subdir the caller runs it from.
# ---------------------------------------------------------------------------
def resolve_inputs(lane: str, batch: int, config_override: str | None) -> dict[str, Any]:
    ll = lane_lower(lane)

    state_path = REPO / "Handoff" / "private" / f"runner_state_{ll}_p{batch}.json"

    # Prefer a batch-scoped pid file (the setup-checklist audit ideal); fall back
    # to the lane-global convenience pointer. Record which one we used and its
    # own `batch` field so a stale lane-global pid (already rolled to a later
    # batch) is detectable rather than silently trusted.
    pid_candidates = [
        REPO / "Handoff" / "private" / f"runner_loop.{ll}.p{batch}.pid.json",
        REPO / "Handoff" / "private" / f"runner_loop.{ll}.pid.json",
    ]
    pid_path = first_existing(pid_candidates) or pid_candidates[0]

    if config_override:
        config_path = Path(config_override)
        if not config_path.is_absolute():
            config_path = REPO / config_path
    else:
        config_path = REPO / "Handoff" / "config" / f"runner_{ll}_p{batch}.json"

    # Monitor latest text + jsonl (naming: monitor_<lane>_batch<N>.latest.txt).
    monitor_latest = REPO / "Handoff" / "logs" / f"monitor_{ll}_batch{batch}.latest.txt"
    monitor_jsonl = REPO / "Handoff" / "logs" / f"monitor_{ll}_batch{batch}.jsonl"

    closeout_md = REPO / "Handoff" / "logs" / f"{ll}_p{batch}_closed.md"
    sentinel_path = REPO / "Handoff" / "private" / "completed" / f"{ll}_p{batch}.complete.json"

    return {
        "lane_lower": ll,
        "state_path": state_path,
        "pid_path": pid_path,
        "pid_is_batch_scoped": pid_path.name == f"runner_loop.{ll}.p{batch}.pid.json",
        "config_path": config_path,
        "monitor_latest": monitor_latest,
        "monitor_jsonl": monitor_jsonl,
        "closeout_md": closeout_md,
        "sentinel_path": sentinel_path,
    }


def rel(path: Path) -> str:
    """Repo-relative POSIX string for a path, for readable sentinel/report values."""
    try:
        return path.resolve().relative_to(REPO.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


# ---------------------------------------------------------------------------
# Derivations.
# ---------------------------------------------------------------------------
def summarize_tasks(state: dict[str, Any]) -> dict[str, Any]:
    """Return terminal counts, per-task rows, packet paths, and terminality."""
    tasks = state.get("tasks") or {}
    counts: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    packets: list[str] = []
    non_terminal: list[str] = []

    for name in sorted(tasks):
        task = tasks[name]
        status = str(task.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1

        packet = task.get("captain_packet_path")
        if packet:
            packets.append(packet)

        rows.append(
            {
                "task": task.get("task", name),
                "title": task.get("title"),
                "status": status,
                "worker": task.get("worker_agent_id") or task.get("assignee"),
                "reviewer": task.get("reviewer_agent_id") or task.get("reviewer"),
                "packet": packet,
            }
        )

        if status not in TERMINAL_STATUSES:
            non_terminal.append(f"{task.get('task', name)}={status}")

    total = sum(counts.values())
    all_terminal = total > 0 and not non_terminal

    ship_count = sum(v for k, v in counts.items() if k in SHIP_STATUSES)
    drop_count = sum(v for k, v in counts.items() if k in DROP_STATUSES)

    return {
        "counts": counts,
        "rows": rows,
        "packets": packets,
        "non_terminal": non_terminal,
        "total": total,
        "all_terminal": all_terminal,
        "ship_count": ship_count,
        "drop_count": drop_count,
    }


def process_registry_path(lane: str, batch: int) -> Path:
    return REPO / "Handoff" / "private" / f"process_registry_{lane_lower(lane)}_p{batch}.jsonl"


def read_process_registry(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            row = {"kind": "parse_error", "line_no": line_no, "line": line[:500]}
        rows.append(row)
    return rows


def summarize_process_registry(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    terminate_attempts: list[dict[str, Any]] = []
    orphan_sweeps: list[dict[str, Any]] = []
    cleanup_events: list[dict[str, Any]] = []
    final_cleanup_events: list[dict[str, Any]] = []
    for row in rows:
        kind = str(row.get("kind") or "unknown")
        counts[kind] = counts.get(kind, 0) + 1
        if kind == "terminate_attempt":
            terminate_attempts.append(row)
        elif kind == "orphan_sweep":
            orphan_sweeps.append(row)
        elif kind == "cleanup_verified":
            cleanup_events.append(row)
            data = row.get("data") or {}
            if data.get("reason") == "scheduler_shutdown":
                final_cleanup_events.append(row)

    latest_cleanup = (final_cleanup_events or cleanup_events)[-1] if (final_cleanup_events or cleanup_events) else None
    cleanup_failures = []
    failure_source = final_cleanup_events or cleanup_events
    for event in failure_source:
        data = event.get("data") or {}
        remaining = data.get("remaining_owner_matches") or []
        if remaining:
            cleanup_failures.append(
                {
                    "ts_utc": event.get("ts_utc"),
                    "task": event.get("task") or data.get("task"),
                    "remaining_count": len(remaining),
                }
            )
    return {
        "event_counts": counts,
        "event_total": len(rows),
        "terminate_attempt_count": len(terminate_attempts),
        "orphan_sweep_count": len(orphan_sweeps),
        "cleanup_verified_count": len(cleanup_events),
        "final_cleanup_verified_count": len(final_cleanup_events),
        "latest_cleanup": latest_cleanup,
        "cleanup_failures": cleanup_failures[-10:],
        "has_cleanup_audit": bool(final_cleanup_events),
    }


def build_sentinel(lane: str, batch: int, paths: dict[str, Any], state: dict[str, Any],
                   pid: dict[str, Any] | None, config: dict[str, Any] | None,
                   summary: dict[str, Any], process_summary: dict[str, Any],
                   registry_path: Path) -> dict[str, Any]:
    """Build the completion-sentinel dict with the exact fields the checklist lists.

    Fields we cannot derive from (state / pid / config) are set to null and given
    an adjacent `*_note` explaining why, so a consumer never mistakes "unknown"
    for "zero".
    """
    workflow = (config or {}).get("workflow") or {}
    project = (config or {}).get("project") or {}

    goal_id = project.get("goal_id")
    runner_id = workflow.get("runner_id")

    pid_state = (pid or {}).get("state")
    exit_reason = (pid or {}).get("exit_reason")
    started_utc = (pid or {}).get("started_utc")
    stopped_utc = (pid or {}).get("stopped_utc")
    pid_batch = (pid or {}).get("batch")

    # If the pid file is lane-global, its `batch` field is the only guard that it
    # actually describes THIS batch. Surface a mismatch instead of hiding it.
    pid_batch_matches = None
    if pid is not None:
        pid_batch_matches = str(pid_batch) == str(batch)

    sentinel: dict[str, Any] = {
        "schema": "neurogolf.handoff.completion_sentinel.v1",
        "generated_utc": utc_now(),
        "generated_by": "Handoff/runner/write_closeout.py",
        "lane": lane,
        "batch": batch,
        "goal_id": goal_id,
        "runner_id": runner_id,
        "state": pid_state,
        "exit_reason": exit_reason,
        "started_utc": started_utc,
        "stopped_utc": stopped_utc,
        "state_path": rel(paths["state_path"]),
        "config_path": rel(paths["config_path"]),
        "pid_path": rel(paths["pid_path"]),
        "pid_is_batch_scoped": paths["pid_is_batch_scoped"],
        "pid_batch_field": pid_batch,
        "pid_batch_matches_request": pid_batch_matches,
        "latest_monitor_path": rel(paths["monitor_latest"]) if paths["monitor_latest"].exists() else None,
        "monitor_jsonl_path": rel(paths["monitor_jsonl"]) if paths["monitor_jsonl"].exists() else None,
        "closeout_path": rel(paths["closeout_md"]),
        "closeout_exists": paths["closeout_md"].exists(),
        "terminal_counts": summary["counts"],
        "terminal_total": summary["total"],
        "all_tasks_terminal": summary["all_terminal"],
        "ship_count": summary["ship_count"],
        "drop_count": summary["drop_count"],
        "non_terminal_tasks": summary["non_terminal"],
        "packet_paths": summary["packets"],
        "tasks": summary["rows"],
        "process_registry_path": rel(registry_path),
        "process_registry_exists": registry_path.exists(),
        "process_registry_summary": process_summary,
        # --- Fields not derivable from the three inputs; explicit null + reason.
        "sync_retry_count": None,
        "sync_retry_note": (
            "not derivable from runner_state/pid/config; the sync-retry queue lives "
            "outside these inputs (see references/api-and-mcp-notes.md). Fill from the "
            "retry queue when wiring this into the live runner drained-exit path."
        ),
        "orphan_cleanup_result": process_summary.get("latest_cleanup"),
        "orphan_cleanup_note": (
            "derived from the append-only process registry"
            if process_summary.get("has_cleanup_audit")
            else "no scheduler_shutdown cleanup_verified event found in the append-only process registry"
        ),
        # Provenance for the terminal-status vocabulary used for the counts.
        "terminal_status_vocab_source": _TERMINAL_SOURCE,
        "terminal_statuses": sorted(TERMINAL_STATUSES),
    }
    return sentinel


def build_closeout_md(lane: str, batch: int, sentinel: dict[str, Any],
                      summary: dict[str, Any]) -> str:
    """A machine-generated closeout, modeled on claude_p46_closed.md's shape.

    Written only when no hand-written closeout exists (the human version is
    always richer -- cross-lane captain actions, staging narrative, integrity
    reconciles -- so we never clobber it).
    """
    lines: list[str] = []
    lines.append(f"# {lane} lane — batch {batch} close-out (machine-generated)")
    lines.append("")
    lines.append(
        f"_Generated {sentinel['generated_utc']} by `Handoff/runner/write_closeout.py`._ "
        f"This is a deterministic derivation from the runner state + pid + config; "
        f"a hand-written closeout, if present, supersedes it and is never overwritten."
    )
    lines.append("")
    lines.append(f"**State (pid):** `{sentinel['state']}` / `exit_reason={sentinel['exit_reason']}`")
    lines.append(f"**Opened (UTC):** {sentinel['started_utc']}")
    lines.append(f"**Closed (UTC):** {sentinel['stopped_utc']}")
    lines.append(
        f"**Config:** `{sentinel['config_path']}`  ·  goal `{sentinel['goal_id']}`  ·  "
        f"runner `{sentinel['runner_id']}`  ·  pid `{sentinel['pid_path']}`"
    )
    lines.append(f"**State file:** `{sentinel['state_path']}`")
    if sentinel["latest_monitor_path"]:
        lines.append(f"**Latest monitor:** `{sentinel['latest_monitor_path']}`")
    lines.append("")

    lines.append("## Terminal outcome")
    counts_str = ", ".join(f"{k}={v}" for k, v in sorted(summary["counts"].items()))
    lines.append(f"All-terminal: **{summary['all_terminal']}**  ·  counts: {counts_str}")
    lines.append(f"Produced **{summary['ship_count']} SHIP / {summary['drop_count']} DROP** "
                 f"({summary['total']} tasks).")
    lines.append("")
    lines.append("| Task | Title | Status | Packet |")
    lines.append("|---|---|---|---|")
    for row in summary["rows"]:
        packet = row["packet"] or "—"
        title = row["title"] or ""
        lines.append(f"| {row['task']} | {title} | {row['status']} | {packet} |")
    lines.append("")

    lines.append("## Packets")
    if summary["packets"]:
        for pkt in summary["packets"]:
            lines.append(f"- `{pkt}`")
    else:
        lines.append("- none")
    lines.append("")

    lines.append("## Process cleanup audit")
    process_summary = sentinel.get("process_registry_summary") or {}
    lines.append(f"- Registry: `{sentinel.get('process_registry_path')}` "
                 f"(exists={sentinel.get('process_registry_exists')})")
    lines.append(f"- Events: {process_summary.get('event_counts', {})}")
    lines.append(f"- Cleanup verified events: {process_summary.get('cleanup_verified_count', 0)}")
    lines.append(f"- Final scheduler-shutdown cleanup events: {process_summary.get('final_cleanup_verified_count', 0)}")
    if process_summary.get("cleanup_failures"):
        lines.append(f"- Remaining owner-matched processes after cleanup: {process_summary['cleanup_failures']}")
    lines.append("")

    lines.append("## Not auto-derivable (see sentinel `*_note` fields)")
    lines.append(f"- **sync-retry count:** {sentinel['sync_retry_count']} — {sentinel['sync_retry_note']}")
    lines.append(f"- **orphan-cleanup result:** {sentinel['orphan_cleanup_result']} — {sentinel['orphan_cleanup_note']}")
    lines.append("")
    lines.append(f"Completion sentinel: `{rel(Path(sentinel['closeout_path']))}` companion at "
                 f"`Handoff/private/completed/{lane_lower(lane)}_p{batch}.complete.json`.")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Terminality gate.
# ---------------------------------------------------------------------------
def check_terminal(summary: dict[str, Any], pid: dict[str, Any] | None,
                   batch: int, process_summary: dict[str, Any]) -> list[str]:
    """Return a list of reasons the batch is NOT safe to close. Empty == safe."""
    problems: list[str] = []

    if summary["total"] == 0:
        problems.append("runner state has zero tasks")
    if summary["non_terminal"]:
        problems.append("non-terminal tasks: " + ", ".join(summary["non_terminal"]))

    if pid is None:
        problems.append("pid file missing/unreadable (cannot confirm state=stopped)")
    else:
        pid_state = pid.get("state")
        exit_reason = pid.get("exit_reason")
        if pid_state != "stopped":
            problems.append(f"pid state is {pid_state!r}, expected 'stopped'")
        if not exit_reason:
            problems.append("pid exit_reason is empty/unset")
        if "batch" in pid and str(pid.get("batch")) != str(batch):
            problems.append(
                f"pid file batch={pid.get('batch')!r} does not match --batch {batch} "
                f"(lane-global pid may have rolled to a later batch)"
            )

    if not process_summary.get("has_cleanup_audit"):
        problems.append("no final process-registry cleanup audit recorded (missing scheduler_shutdown cleanup_verified event)")
    elif process_summary.get("cleanup_failures"):
        problems.append("process-registry cleanup audit still has owner-matched remaining processes")

    return problems


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Write a runner batch's completion sentinel + closeout report."
    )
    ap.add_argument("--lane", required=True, choices=["Claude", "Codex", "Codex2"])
    ap.add_argument("--batch", required=True, type=int)
    ap.add_argument("--config", default=None, help="Override config path (default derived from lane/batch).")
    ap.add_argument("--apply", action="store_true", help="Write the sentinel (and closeout if absent). Default is DRY-RUN.")
    ap.add_argument("--force", action="store_true", help="Write even if the batch is not detectably terminal.")
    args = ap.parse_args(argv)

    paths = resolve_inputs(args.lane, args.batch, args.config)

    # --- Load inputs. State is mandatory; pid/config are read-if-present so we
    #     can still emit a useful dry-run and clear diagnostics when they are absent.
    if not paths["state_path"].exists():
        print(f"ERROR: runner state not found: {rel(paths['state_path'])}", file=sys.stderr)
        return 2
    try:
        state = read_json(paths["state_path"])
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"ERROR: runner state is not valid JSON ({rel(paths['state_path'])}): {exc}", file=sys.stderr)
        return 2

    pid = read_json_or_none(paths["pid_path"])
    config = read_json_or_none(paths["config_path"])

    if config is None:
        print(f"WARNING: config not found/unreadable: {rel(paths['config_path'])} "
              f"(goal_id/runner_id will be null)", file=sys.stderr)
    if pid is None:
        print(f"WARNING: pid file not found/unreadable: {rel(paths['pid_path'])} "
              f"(state/exit_reason/started/stopped will be null)", file=sys.stderr)

    summary = summarize_tasks(state)
    registry_path = process_registry_path(args.lane, args.batch)
    process_rows = read_process_registry(registry_path)
    process_summary = summarize_process_registry(process_rows)
    sentinel = build_sentinel(args.lane, args.batch, paths, state, pid, config, summary, process_summary, registry_path)
    closeout_md = build_closeout_md(args.lane, args.batch, sentinel, summary)

    problems = check_terminal(summary, pid, args.batch, process_summary)

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"=== write_closeout {mode} :: lane={args.lane} batch={args.batch} ===")
    print(f"state:   {rel(paths['state_path'])}")
    print(f"pid:     {rel(paths['pid_path'])} "
          f"({'batch-scoped' if paths['pid_is_batch_scoped'] else 'lane-global'})")
    print(f"config:  {rel(paths['config_path'])}")
    print(f"sentinel target: {rel(paths['sentinel_path'])}")
    print(f"closeout target: {rel(paths['closeout_md'])} (exists={paths['closeout_md'].exists()})")
    print(f"process registry: {rel(registry_path)} (events={process_summary['event_total']}, cleanup_verified={process_summary['cleanup_verified_count']})")
    print()

    if problems:
        print("TERMINALITY: NOT terminal —")
        for p in problems:
            print(f"  - {p}")
    else:
        print("TERMINALITY: terminal (all tasks terminal; pid stopped with exit_reason).")
    print()

    print("----- COMPLETION SENTINEL (would write) -----")
    print(json.dumps(sentinel, indent=2, ensure_ascii=False))
    print()
    print("----- CLOSEOUT MARKDOWN (would write only if none hand-written) -----")
    print(closeout_md)

    if not args.apply:
        print("DRY-RUN: nothing written. Re-run with --apply to write the sentinel"
              + (" and closeout." if not paths["closeout_md"].exists() else " (closeout already exists; left as-is)."))
        return 0

    # --- APPLY path -------------------------------------------------------
    if problems and not args.force:
        print("REFUSING to write: batch is not detectably terminal (see above). "
              "Re-run with --force to override.", file=sys.stderr)
        return 3

    paths["sentinel_path"].parent.mkdir(parents=True, exist_ok=True)
    paths["sentinel_path"].write_text(
        json.dumps(sentinel, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"WROTE sentinel: {rel(paths['sentinel_path'])}")

    if paths["closeout_md"].exists():
        print(f"KEPT existing closeout (not overwritten): {rel(paths['closeout_md'])}")
    else:
        paths["closeout_md"].write_text(closeout_md, encoding="utf-8")
        print(f"WROTE closeout: {rel(paths['closeout_md'])}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
