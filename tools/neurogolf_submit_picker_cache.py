#!/usr/bin/env python3
"""Pre-compute a cached view of the submit queue for the submit-captain.

The submit-captain agent currently spends ~11 minutes per card thinking
through the same questions every time:

  * Which queue entries are ``ready`` and approved for submit?
  * Which of those have hidden-bad SHAs or come from failed-public-negative
    lineages?
  * What's the EV ranking?
  * Which entries can be stacked together (i.e. don't touch the same
    ``changed_task_ids``)?

All of this is deterministic — just JSON lookups across
``.hermes/board/submit_queue.json`` and ``.hermes/board/state.json``.
This cron pre-computes the answers and writes them to
``.hermes/board/submit_picker_cache.json`` every minute, so the
submit-captain can read a ready-to-use index instead of reasoning from
raw state files. The follow-on stack-picker tool (Tier 1) uses this same
cache as its input.

The cache is ADVISORY — it is NOT consumed by any other Python control
plane tool. Submit-captain (LLM agent) reads it as a hint, decides what
to actually submit, and remains the only role allowed to call
``kaggle competitions submit``. If the cache disagrees with reality, the
agent overrides.

Output schema: see ``write_cache()`` below.

Usage:
    python tools/neurogolf_submit_picker_cache.py            # dry-run
    python tools/neurogolf_submit_picker_cache.py --apply    # write cache
    python tools/neurogolf_submit_picker_cache.py --apply --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BOARD_DIR = REPO_ROOT / ".hermes" / "board"
SUBMIT_QUEUE_JSON = BOARD_DIR / "submit_queue.json"
STATE_JSON = BOARD_DIR / "state.json"
CURRENT_BEST_MANIFEST = REPO_ROOT / "submissions" / "current_best_manifest.json"
TASK_OUTCOME_JSON = BOARD_DIR / "task_outcome_registry.json"
OUTPUT_PATH = BOARD_DIR / "submit_picker_cache.json"

SCHEMA_VERSION = "neurogolf.submit_picker_cache.v1"

# Statuses that indicate a candidate has already been submitted and got a
# public-negative result. Listing the queue entry's own SHA blocks future
# resubmits of the same package.
FAILED_STATUS_RE = re.compile(
    r"submitted-failed-public-negative|public-negative-isolated|hidden-bad",
    re.IGNORECASE,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _normalize_sha(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().lower()


def _normalize_task_id(value: Any) -> str:
    if isinstance(value, int):
        return f"task{value:03d}"
    if isinstance(value, str):
        s = value.strip().lower()
        if s.startswith("task"):
            return s
        if s.isdigit():
            return f"task{int(s):03d}"
    return ""


def collect_hidden_bad_task_ids(state: dict[str, Any]) -> list[str]:
    """``state.hidden_bad_tasks`` is a list of integer task numbers; return
    them as canonical ``taskNNN`` strings."""
    raw = state.get("hidden_bad_tasks") or []
    out: list[str] = []
    seen: set[str] = set()
    for value in raw:
        task_id = _normalize_task_id(value)
        if task_id and task_id not in seen:
            out.append(task_id)
            seen.add(task_id)
    return sorted(out)


def collect_failed_public_negative_shas(state: dict[str, Any], queue: list[dict[str, Any]]) -> list[str]:
    """Union of SHAs that have already failed on Kaggle. Pulled from:
      * ``state.failed_validations[*].package_sha256``
      * ``submit_queue[*].package_sha256`` where ``status`` matches the
        failed-public-negative / hidden-bad regex
    """
    shas: set[str] = set()
    for entry in state.get("failed_validations") or []:
        if isinstance(entry, dict):
            sha = _normalize_sha(entry.get("package_sha256"))
            if sha:
                shas.add(sha)
    for entry in queue:
        if not isinstance(entry, dict):
            continue
        status = str(entry.get("status") or entry.get("submit_queue_status") or "").lower()
        if FAILED_STATUS_RE.search(status):
            sha = _normalize_sha(entry.get("package_sha256"))
            if sha:
                shas.add(sha)
    return sorted(shas)


def _ev_for_entry(entry: dict[str, Any]) -> float:
    """Best available EV signal — `priority_score` is what reviewers set on
    approval; ``expected_point_gain`` (or ``expected_gain``) is the
    fallback from the worker's handoff JSON."""
    for key in ("priority_score", "expected_point_gain", "expected_gain"):
        value = entry.get(key)
        try:
            f = float(value)
        except (TypeError, ValueError):
            continue
        if f != 0:  # 0.0 is the default-when-missing sentinel; skip
            return f
    return 0.0


def _changed_tasks_for_entry(entry: dict[str, Any]) -> list[str]:
    raw = entry.get("changed_task_ids") or []
    if isinstance(raw, str):
        raw = [raw]
    out: list[str] = []
    seen: set[str] = set()
    for value in raw:
        task_id = _normalize_task_id(value)
        if task_id and task_id not in seen:
            out.append(task_id)
            seen.add(task_id)
    # Fall back to single-task-id if changed_task_ids is empty.
    if not out:
        single = _normalize_task_id(entry.get("task_id"))
        if single:
            out.append(single)
    return out


def is_ready_for_submit(entry: dict[str, Any]) -> bool:
    """``ready`` (reviewer approved + waiting for credits/stack) or
    ``awaiting-submit-credit-or-stack`` (parked waiting for the burst
    window)."""
    if not isinstance(entry, dict):
        return False
    status = str(entry.get("status") or entry.get("submit_queue_status") or "").lower()
    return status in ("ready", "awaiting-submit-credit-or-stack")


def build_ready_index(
    queue: list[dict[str, Any]],
    *,
    hidden_bad_tasks: set[str],
    failed_shas: set[str],
) -> list[dict[str, Any]]:
    """Project the ready queue into compact entries for the picker, sorted
    by EV (descending)."""
    rows: list[dict[str, Any]] = []
    for entry in queue:
        if not is_ready_for_submit(entry):
            continue
        ev = _ev_for_entry(entry)
        changed = _changed_tasks_for_entry(entry)
        package_sha = _normalize_sha(entry.get("package_sha256"))
        hidden_bad_task_hit = sorted(set(changed) & hidden_bad_tasks)
        failed_sha_hit = bool(package_sha and package_sha in failed_shas)
        # package_path lives at top level for some entries and nested under
        # approval_packet for others (reviewer-approved candidates). Pull
        # from whichever is populated so the picker can validate on disk.
        package_path = (
            entry.get("package_path")
            or (entry.get("approval_packet") or {}).get("package_path")
            or ""
        )
        # SHA may also be nested in approval_packet for some entries.
        if not package_sha:
            package_sha = _normalize_sha(
                (entry.get("approval_packet") or {}).get("package_sha256")
            )
            failed_sha_hit = bool(package_sha and package_sha in failed_shas)
        rows.append(
            {
                "candidate_id": entry.get("candidate_id") or "",
                "task_id": _normalize_task_id(entry.get("task_id")),
                "queue_id": entry.get("queue_id") or "",
                "status": entry.get("status") or entry.get("submit_queue_status") or "",
                "ev": ev,
                "changed_task_ids": changed,
                "package_sha256": package_sha,
                "package_path": package_path,
                "lineage_base_candidate_id": entry.get("lineage_base_candidate_id") or "",
                "lineage_base_score": entry.get("lineage_base_score"),
                "reviewer_disposition": entry.get("reviewer_disposition") or "",
                "approval_mode": entry.get("approval_mode") or "",
                "worker_owner": entry.get("worker_owner") or "",
                "return_worker": entry.get("return_worker") or "",
                "risk_flags": list(entry.get("risk_flags") or []),
                # Picker hints — pre-flagged so the agent / Tier 1 picker
                # doesn't have to re-derive them.
                "hidden_bad_task_hit": hidden_bad_task_hit,
                "failed_sha_hit": failed_sha_hit,
                "safe_to_submit": (not hidden_bad_task_hit) and (not failed_sha_hit),
            }
        )
    rows.sort(key=lambda r: (-r["ev"], r["candidate_id"]))
    return rows


def build_current_best_summary(manifest: dict[str, Any] | None) -> dict[str, Any]:
    if not manifest:
        return {}
    return {
        "score": manifest.get("score"),
        "candidate_id": manifest.get("candidate_id"),
        "task_id": manifest.get("task_id"),
        "package_path": manifest.get("package_path"),
        "package_sha256": _normalize_sha(manifest.get("package_sha256")),
        "promoted_at": manifest.get("promoted_at"),
        "lineage_base_score": manifest.get("lineage_base_score"),
    }


def build_cache(
    *,
    queue_doc: dict[str, Any] | None,
    state_doc: dict[str, Any] | None,
    manifest_doc: dict[str, Any] | None,
    task_outcome_doc: dict[str, Any] | None,
    now_iso: str | None = None,
) -> dict[str, Any]:
    queue = (queue_doc or {}).get("queue") if isinstance(queue_doc, dict) else None
    queue = queue if isinstance(queue, list) else []
    state = state_doc if isinstance(state_doc, dict) else {}
    manifest = manifest_doc if isinstance(manifest_doc, dict) else {}
    hidden_bad_list = collect_hidden_bad_task_ids(state)
    failed_shas_list = collect_failed_public_negative_shas(state, queue)
    hidden_bad_set = set(hidden_bad_list)
    failed_shas_set = set(failed_shas_list)
    ready = build_ready_index(
        queue,
        hidden_bad_tasks=hidden_bad_set,
        failed_shas=failed_shas_set,
    )
    safe_ready = [r for r in ready if r["safe_to_submit"]]
    total_safe_ev = round(sum(r["ev"] for r in safe_ready), 6)
    top_safe_ev = round(max((r["ev"] for r in safe_ready), default=0.0), 6)
    return {
        "schema": SCHEMA_VERSION,
        "computed_at": now_iso or _utc_now(),
        "current_best": build_current_best_summary(manifest),
        "hidden_bad_task_ids": hidden_bad_list,
        "failed_public_negative_shas": failed_shas_list,
        "ready_candidates_by_ev": ready,
        "ready_count": len(ready),
        "safe_ready_count": len(safe_ready),
        "blocked_ready_count": len(ready) - len(safe_ready),
        "top_safe_ev": top_safe_ev,
        "total_safe_ev": total_safe_ev,
        "task_outcome_registry_updated_at": (
            (task_outcome_doc or {}).get("updated_at") if isinstance(task_outcome_doc, dict) else None
        ),
    }


def write_cache(cache: dict[str, Any], path: Path = OUTPUT_PATH) -> None:
    """Atomic write via temp-then-replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(cache, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write the cache file (default: dry-run).")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args(argv)

    cache = build_cache(
        queue_doc=_read_json(SUBMIT_QUEUE_JSON),
        state_doc=_read_json(STATE_JSON),
        manifest_doc=_read_json(CURRENT_BEST_MANIFEST),
        task_outcome_doc=_read_json(TASK_OUTCOME_JSON),
    )

    if args.apply:
        write_cache(cache, args.output)

    if args.json:
        # Compact summary for cron --no-agent delivery.
        payload = {
            "schema": SCHEMA_VERSION,
            "applied": args.apply,
            "computed_at": cache["computed_at"],
            "ready_count": cache["ready_count"],
            "safe_ready_count": cache["safe_ready_count"],
            "blocked_ready_count": cache["blocked_ready_count"],
            "top_safe_ev": cache["top_safe_ev"],
            "total_safe_ev": cache["total_safe_ev"],
        }
        json.dump(payload, sys.stdout)
        sys.stdout.write("\n")
    else:
        print(
            f"ready={cache['ready_count']}  safe={cache['safe_ready_count']}  "
            f"blocked={cache['blocked_ready_count']}  top_ev=+{cache['top_safe_ev']}  "
            f"total_safe_ev=+{cache['total_safe_ev']}"
        )
        print(f"mode: {'APPLY' if args.apply else 'DRY-RUN'}  output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
