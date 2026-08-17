#!/usr/bin/env python3
"""Reconstruct submit_queue entries for orphan candidates.

Reads ``orphan_candidates.json`` (produced by
``neurogolf_orphan_candidates.py``) and rebuilds the minimum-viable
submit_queue entry for each, then writes those entries into
``submit_queue.json::queue``. Submit-captain then sees them on its next
queue scan and can pick them up.

Safety guards (all reasons a candidate is SKIPPED):

  * Package file doesn't exist on disk.
  * Package SHA mismatches the file on disk.
  * Package zip is malformed or has wrong entry count.
  * Risk flags include ``failed_public_negative`` or ``hidden-bad``.
  * candidate_id is already in submit_queue (no double-write).
  * ``--require-staged-candidates`` is on and the package lives in a worker
    worktree (cleanup risk; observed for task098 in this batch).
  * ``--exclude-task`` matches the task_id (manual blocklist).

This tool is dry-run by default. ``--apply`` writes to submit_queue.json
atomically (temp-then-replace) after taking a timestamped backup.

Usage:
    python tools/neurogolf_promote_orphans.py            # dry-run, text
    python tools/neurogolf_promote_orphans.py --apply
    python tools/neurogolf_promote_orphans.py --apply \\
        --require-staged-candidates --exclude-task task379
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BOARD_DIR = REPO_ROOT / ".hermes" / "board"
REGISTRY_PATH = BOARD_DIR / "candidate_registry.json"
SUBMIT_QUEUE_PATH = BOARD_DIR / "submit_queue.json"
ORPHAN_REPORT_PATH = BOARD_DIR / "orphan_candidates.json"
CURRENT_BEST_MANIFEST = REPO_ROOT / "submissions" / "current_best_manifest.json"

EXPECTED_ZIP_ENTRY_COUNT = 401  # matches stack_picker's constant

# Default stale-lineage threshold. Reviewer-approved candidates whose
# ``lineage_base_score`` is more than this many LB points BELOW the
# current canonical's ``score`` get refused unless ``--allow-stale-lineage``
# is set. Real example that motivated this: task388 was reviewer-approved
# at expected +3.27 against ``task141_v4_fp16_pos_on_5743_97`` (old base),
# but by the time the orphan-promoter saw it, current best was at 5755.05
# (+11.08 LB ahead). The candidate landed -15.27 on Kaggle because the
# +3.27 estimate was for a baseline that no longer exists.
STALE_LINEAGE_LB_THRESHOLD = 5.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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


def _file_sha256(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().lower()


def validate_orphan(orphan: dict[str, Any]) -> dict[str, Any]:
    """Return ``{"ok": bool, "messages": [...]}``. ``ok`` is True iff the
    package on disk is healthy enough to enqueue for the submit-captain.
    """
    messages: list[str] = []
    pkg_path_str = (orphan.get("package_path") or "").strip()
    if not pkg_path_str:
        return {"ok": False, "messages": ["no package_path in orphan record"]}
    pkg_path = Path(pkg_path_str)
    if not pkg_path.exists():
        return {"ok": False, "messages": [f"package not on disk: {pkg_path}"]}
    expected_sha = _normalize_sha(orphan.get("package_sha256"))
    if expected_sha:
        actual_sha = _file_sha256(pkg_path)
        if actual_sha != expected_sha:
            messages.append(f"SHA mismatch: expected {expected_sha[:12]}.. got {actual_sha[:12]}..")
            return {"ok": False, "messages": messages}
    try:
        with zipfile.ZipFile(pkg_path) as z:
            names = z.namelist()
    except zipfile.BadZipFile:
        return {"ok": False, "messages": [f"package is not a valid zip: {pkg_path}"]}
    onnx_count = sum(1 for n in names if n.endswith(".onnx"))
    # Two valid shapes:
    #   (a) Full canonical: 401 entries (task000..task400). Submit-captain
    #       submits as-is.
    #   (b) Delta: 1 entry named taskNNN.onnx matching the candidate's
    #       task_id. Submit-captain merges this single ONNX onto the
    #       canonical current best before submitting.
    # Anything else is malformed.
    task_id = orphan.get("task_id") or ""
    expected_delta_entry = f"{task_id}.onnx" if task_id else ""
    if len(names) == EXPECTED_ZIP_ENTRY_COUNT and onnx_count == EXPECTED_ZIP_ENTRY_COUNT:
        shape = "full"
    elif len(names) == 1 and expected_delta_entry and names[0] == expected_delta_entry:
        shape = "delta"
    elif len(names) == 1 and names[0].endswith(".onnx"):
        # Delta package but the single entry doesn't match the task_id.
        messages.append(
            f"delta package's single entry {names[0]!r} does not match task_id {task_id!r}"
        )
        return {"ok": False, "messages": messages}
    else:
        messages.append(
            f"package has {len(names)} entries ({onnx_count} onnx); expected either "
            f"{EXPECTED_ZIP_ENTRY_COUNT} (full canonical) or 1 (delta = {expected_delta_entry!r})"
        )
        return {"ok": False, "messages": messages}
    return {"ok": True, "messages": messages, "shape": shape}


def reconstruct_queue_entry(
    orphan: dict[str, Any],
    *,
    registry_entry: dict[str, Any] | None = None,
    now_iso: str | None = None,
) -> dict[str, Any]:
    """Build a minimum-viable submit_queue entry from the orphan + its
    registry entry. The submit-captain only requires a subset of the
    full schema (status=ready + candidate_id + package metadata + EV);
    we mirror the standard shape so the agent sees a familiar entry.
    """
    now_iso = now_iso or _utc_now()
    reg = registry_entry or {}
    cid = orphan["candidate_id"]
    task_id = orphan.get("task_id") or reg.get("task_id")
    pkg_path = orphan.get("package_path") or (reg.get("approval_packet") or {}).get("package_path")
    pkg_sha = orphan.get("package_sha256") or (reg.get("approval_packet") or {}).get("package_sha256")
    ev = orphan.get("expected_point_gain") or 0.0
    return {
        "queue_id": f"submit-{_utc_compact()}-{cid}",
        "status": "ready",
        "candidate_id": cid,
        "task_id": task_id,
        "reviewer_disposition": orphan.get("reviewer_disposition") or "approved-for-submit-captain",
        "approval_mode": orphan.get("approval_mode") or "auto-rate-limited",
        "priority_score": ev,
        "lineage_base_candidate_id": orphan.get("lineage_base_candidate_id"),
        "lineage_base_score": reg.get("lineage_base_score"),
        "worker_owner": orphan.get("worker_owner"),
        "return_worker": orphan.get("worker_owner"),
        "changed_task_ids": reg.get("changed_task_ids") or ([task_id] if task_id else []),
        "risk_flags": list(orphan.get("risk_flags") or []) + ["promoted-from-orphan-2026-05-18"],
        "preflight_status": "pass",
        "package_path": pkg_path,
        "package_sha256": pkg_sha,
        "approval_packet": reg.get("approval_packet") or {
            "candidate_id": cid,
            "task_id": task_id,
            "reviewer_disposition": orphan.get("reviewer_disposition"),
            "approval_mode": orphan.get("approval_mode"),
            "package_path": pkg_path,
            "package_sha256": pkg_sha,
        },
        "promoted_from_orphan": {
            "promoted_at": now_iso,
            "promoted_by": "tools/neurogolf_promote_orphans.py",
            "reason": "candidate_registry had submit_queue_status=ready but submit_queue had no matching candidate_id",
        },
    }


def stale_lineage_reason(
    orphan: dict[str, Any],
    registry_entry: dict[str, Any] | None,
    *,
    current_best_score: float | None,
    threshold_lb: float = STALE_LINEAGE_LB_THRESHOLD,
) -> str:
    """Return a non-empty skip reason if the candidate's lineage_base
    score is more than ``threshold_lb`` below current canonical. Empty
    string means the lineage is fresh enough OR we can't tell.

    The reviewer's expected_point_gain is computed against the candidate's
    lineage_base_score, not against current canonical. When current
    canonical has moved up substantially since the review, the gain
    estimate is for a baseline that no longer exists — the candidate may
    be a net regression against the new baseline even though it was a
    legitimate improvement against the old one.
    """
    if current_best_score is None:
        return ""
    reg = registry_entry or {}
    base_score = reg.get("lineage_base_score")
    if base_score is None:
        # Try to pull from the orphan record (some fields are duplicated).
        base_score = orphan.get("lineage_base_score")
    try:
        base = float(base_score)
    except (TypeError, ValueError):
        return ""
    gap = float(current_best_score) - base
    if gap > threshold_lb:
        return (
            f"stale lineage: candidate was reviewer-approved against "
            f"lineage_base_score={base} but current best is {current_best_score} "
            f"(gap +{gap:.2f} LB > threshold {threshold_lb} LB). The reviewer's "
            f"expected-gain estimate is for a baseline that no longer exists."
        )
    return ""


def plan_promotions(
    *,
    orphan_report: dict[str, Any],
    registry_doc: dict[str, Any] | None,
    queue_doc: dict[str, Any] | None,
    manifest_doc: dict[str, Any] | None = None,
    require_staged_candidates: bool = True,
    allow_stale_lineage: bool = False,
    stale_lineage_threshold_lb: float = STALE_LINEAGE_LB_THRESHOLD,
    exclude_tasks: set[str] | None = None,
) -> dict[str, Any]:
    """Decide which orphans to promote. Each orphan is either ``promote``
    or ``skip`` with a recorded reason. No side effects."""
    exclude_tasks = exclude_tasks or set()
    registry = registry_doc if isinstance(registry_doc, dict) else {}
    queue = queue_doc if isinstance(queue_doc, dict) else {}
    reg_by_cid = {
        e.get("candidate_id"): e
        for e in (registry.get("entries") or [])
        if isinstance(e, dict) and e.get("candidate_id")
    }
    queue_cids = {
        e.get("candidate_id")
        for e in (queue.get("queue") or [])
        if isinstance(e, dict) and e.get("candidate_id")
    }
    current_best_score: float | None = None
    if isinstance(manifest_doc, dict):
        try:
            current_best_score = float(manifest_doc.get("score"))
        except (TypeError, ValueError):
            current_best_score = None
    decisions: list[dict[str, Any]] = []
    promote_entries: list[dict[str, Any]] = []
    for orphan in orphan_report.get("orphans") or []:
        cid = orphan.get("candidate_id")
        task_id = orphan.get("task_id")
        skip_reasons: list[str] = []
        if cid in queue_cids:
            skip_reasons.append("already in submit_queue (no-op)")
        if task_id in exclude_tasks:
            skip_reasons.append(f"task_id {task_id} in --exclude-task list")
        risk_flags = [str(rf).lower() for rf in (orphan.get("risk_flags") or [])]
        if any("failed_public_negative" in rf or "hidden-bad" in rf or "hidden_bad" in rf for rf in risk_flags):
            skip_reasons.append("risk flag suggests hidden-bad / failed public negative")
        if require_staged_candidates and not orphan.get("package_in_staged_candidates"):
            loc = "worktree" if orphan.get("package_in_worktree") else "other"
            skip_reasons.append(f"package_path not in staged_candidates (location={loc}); pass --no-require-staged-candidates to override")
        validation = validate_orphan(orphan)
        if not validation["ok"]:
            skip_reasons.append("validation failed: " + "; ".join(validation["messages"]))
        if not allow_stale_lineage:
            stale_reason = stale_lineage_reason(
                orphan,
                reg_by_cid.get(cid),
                current_best_score=current_best_score,
                threshold_lb=stale_lineage_threshold_lb,
            )
            if stale_reason:
                skip_reasons.append(stale_reason)
        if skip_reasons:
            decisions.append({
                "candidate_id": cid,
                "task_id": task_id,
                "expected_point_gain": orphan.get("expected_point_gain"),
                "decision": "skip",
                "reasons": skip_reasons,
            })
            continue
        entry = reconstruct_queue_entry(orphan, registry_entry=reg_by_cid.get(cid))
        promote_entries.append(entry)
        decisions.append({
            "candidate_id": cid,
            "task_id": task_id,
            "expected_point_gain": orphan.get("expected_point_gain"),
            "decision": "promote",
            "reasons": ["registry says ready, queue missing, all gates pass"],
            "queue_id": entry["queue_id"],
        })
    promote_total_ev = round(sum(e.get("priority_score") or 0 for e in promote_entries), 6)
    return {
        "decisions": decisions,
        "promote_entries": promote_entries,
        "promote_count": len(promote_entries),
        "skip_count": sum(1 for d in decisions if d["decision"] == "skip"),
        "promote_total_ev": promote_total_ev,
    }


def apply_promotions(plan: dict[str, Any], queue_path: Path = SUBMIT_QUEUE_PATH) -> dict[str, Any]:
    """Mutate the submit_queue: append the planned entries, take a backup
    of the previous file before overwriting."""
    if plan["promote_count"] == 0:
        return {"applied_count": 0, "backup_path": None}
    queue_doc = _read_json(queue_path)
    if not isinstance(queue_doc, dict) or not isinstance(queue_doc.get("queue"), list):
        raise SystemExit(f"submit_queue at {queue_path} is missing or malformed; refusing to write")
    backup_path = queue_path.with_suffix(queue_path.suffix + f".bak.{_utc_compact()}.promote_orphans")
    shutil.copy2(queue_path, backup_path)
    queue_doc["queue"].extend(plan["promote_entries"])
    queue_doc["updated_at"] = _utc_now()
    queue_doc.setdefault("notes", []).append(
        f"{queue_doc['updated_at']} tools/neurogolf_promote_orphans.py promoted "
        f"{plan['promote_count']} orphan candidate(s) into queue "
        f"(combined EV +{plan['promote_total_ev']}). Backup: {backup_path.name}"
    )
    tmp = queue_path.with_suffix(queue_path.suffix + ".tmp")
    tmp.write_text(json.dumps(queue_doc, indent=2) + "\n", encoding="utf-8")
    tmp.replace(queue_path)
    return {"applied_count": plan["promote_count"], "backup_path": str(backup_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="Mutate submit_queue.json (default: dry-run).")
    parser.add_argument("--json", action="store_true",
                        help="Emit JSON instead of text.")
    parser.add_argument("--orphan-report", type=Path, default=ORPHAN_REPORT_PATH,
                        help="Orphan report file (built by neurogolf_orphan_candidates.py).")
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    parser.add_argument("--queue", type=Path, default=SUBMIT_QUEUE_PATH)
    parser.add_argument("--require-staged-candidates", action=argparse.BooleanOptionalAction, default=True,
                        help="Require the package to live in task-local staged_candidates/ (default: on).")
    parser.add_argument("--allow-stale-lineage", action="store_true",
                        help=(
                            "Allow candidates whose lineage_base_score is more than "
                            f"{STALE_LINEAGE_LB_THRESHOLD} LB below current canonical. Default: refuse — "
                            "reviewer's expected-gain estimate is for an old baseline and may produce "
                            "a public-negative submit (observed: task388 reviewer-approved at +3.27 "
                            "against 5743.97 base actually lost -15.27 against 5755.05 current)."
                        ))
    parser.add_argument("--stale-lineage-threshold-lb", type=float, default=STALE_LINEAGE_LB_THRESHOLD,
                        help="LB-gap threshold for stale-lineage detection (default: %(default)s).")
    parser.add_argument("--manifest", type=Path, default=CURRENT_BEST_MANIFEST,
                        help="Path to current_best_manifest.json (default: %(default)s).")
    parser.add_argument("--exclude-task", action="append", default=[],
                        help="Task id to exclude (can be passed multiple times).")
    args = parser.parse_args(argv)

    orphan_report = _read_json(args.orphan_report)
    if not orphan_report or "orphans" not in orphan_report:
        # Build inline as a fallback.
        from tools.neurogolf_orphan_candidates import build_report
        orphan_report = build_report(
            registry_doc=_read_json(args.registry),
            queue_doc=_read_json(args.queue),
        )

    plan = plan_promotions(
        orphan_report=orphan_report,
        registry_doc=_read_json(args.registry),
        queue_doc=_read_json(args.queue),
        manifest_doc=_read_json(args.manifest),
        require_staged_candidates=args.require_staged_candidates,
        allow_stale_lineage=args.allow_stale_lineage,
        stale_lineage_threshold_lb=args.stale_lineage_threshold_lb,
        exclude_tasks=set(args.exclude_task),
    )

    apply_result = {"applied_count": 0, "backup_path": None}
    if args.apply:
        apply_result = apply_promotions(plan, args.queue)

    if args.json:
        summary = {
            "schema": "neurogolf.promote_orphans.v1",
            "applied": args.apply,
            "promote_count": plan["promote_count"],
            "skip_count": plan["skip_count"],
            "promote_total_ev": plan["promote_total_ev"],
            "applied_count": apply_result["applied_count"],
            "backup_path": apply_result["backup_path"],
            "decisions": plan["decisions"],
        }
        json.dump(summary, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print(f"plan: promote={plan['promote_count']}  skip={plan['skip_count']}  ev=+{plan['promote_total_ev']}")
        for d in plan["decisions"]:
            ev = d.get("expected_point_gain") or 0
            marker = "PROMOTE" if d["decision"] == "promote" else "skip"
            print(f"  {marker:8}  +{ev:6.3f}  {d['task_id']:10}  {d['candidate_id'][:55]}")
            for r in d.get("reasons") or []:
                print(f"           - {r}")
        print(f"mode: {'APPLY' if args.apply else 'DRY-RUN'}  applied_count={apply_result['applied_count']}")
        if apply_result["backup_path"]:
            print(f"backup: {apply_result['backup_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
