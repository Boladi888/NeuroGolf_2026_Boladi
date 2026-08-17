#!/usr/bin/env python3
"""Deterministically promote an exact staged NeuroGolf package to current best.

This is intentionally boring machinery: it refuses to copy anything unless the
source package hash equals the caller-provided SHA256, then updates the package
mirrors and structured JSON breadcrumbs. LLM sync-manager cards should call
this script instead of hand-editing/copying current-best artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_REPO_ROOT = Path(os.environ.get("NEUROGOLF_CANONICAL_ROOT", Path(__file__).resolve().parent.parent))
REPO_ROOT_FROM_FILE = Path(__file__).resolve().parent.parent
if str(REPO_ROOT_FROM_FILE) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_FROM_FILE))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def parse_task_id(value: str | None) -> int | None:
    text = (value or "").strip().lower()
    if text.startswith("task"):
        text = text[4:]
    return int(text) if text.isdigit() else None


def read_json(path: Path, fallback: Any) -> Any:
    if not path.exists():
        return fallback
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any, *, dry_run: bool, actions: list[str]) -> None:
    actions.append(f"write_json {path}")
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write (2026-05-20 fix): the original ``path.write_text()`` was
    # non-atomic — a SIGKILL or power loss mid-write would truncate the
    # destination, corrupting current_best_manifest.json / submit_queue.json
    # / candidate_registry.json. Downstream readers (dashboard, auto_sync,
    # submit_gate) would then either crash on JSONDecodeError or fall
    # through to stale legacy fields. Pattern mirrors task_outcome_registry_io
    # and the other atomic writers across tools/.
    serialized = json.dumps(data, indent=2) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(serialized)
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            # fsync can fail on some non-POSIX FS (network shares); the
            # subsequent os.replace is still atomic, just less durable.
            pass
    os.replace(tmp, path)


def write_text(path: Path, data: str, *, dry_run: bool, actions: list[str]) -> None:
    actions.append(f"write_text {path}")
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data, encoding="utf-8")


def copy_verified_package(source: Path, dest: Path, *, dry_run: bool, actions: list[str]) -> None:
    actions.append(f"copy {source} -> {dest}")
    if dry_run:
        return
    # Skip self-copy: when the operator pre-places the canonical package
    # at submissions/submission.zip and passes that same path as
    # ``--package-path``, shutil.copy2 on Windows raises PermissionError
    # ("file in use") because CopyFile2 refuses src == dst. Real-world
    # trigger: manual-recovery syncs (2026-05-20) where the canonical zip
    # is already correct and only the metadata needs catching up.
    try:
        if source.resolve() == dest.resolve():
            actions[-1] = f"copy {source} -> {dest} (skipped: same path)"
            return
    except OSError:
        # resolve() failures on either side: fall through to the normal
        # copy and surface whatever real error it raises.
        pass
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def as_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def effective_orchestrator_current_best(args: argparse.Namespace) -> Path | None:
    if not getattr(args, "sync_legacy_mirrors", False):
        return None
    configured = getattr(args, "orchestrator_current_best", None)
    if configured:
        return Path(configured).resolve()
    return None


def effective_main_local_dir(args: argparse.Namespace) -> Path | None:
    if not getattr(args, "sync_legacy_mirrors", False):
        return None
    configured = getattr(args, "main_local_dir", None)
    if configured:
        return Path(configured).resolve()
    return None


def dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path.resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def package_target_paths(
    repo_root: Path,
    *,
    orchestrator_current_best: Path | None,
    main_local_dir: Path | None,
) -> list[Path]:
    targets = [
        repo_root / "submissions" / "submission.zip",
        repo_root / "repo" / "submissions" / "submission.zip",
    ]
    if orchestrator_current_best:
        targets.append(orchestrator_current_best)
    if main_local_dir and main_local_dir.exists():
        targets.append(main_local_dir / "submission.zip")
        if (main_local_dir / "submissions").exists():
            targets.append(main_local_dir / "submissions" / "submission.zip")
    return dedupe_paths(targets)


def manifest_entry(args: argparse.Namespace, synced_at: str) -> dict[str, Any]:
    last_submit_result = {
        "submitted_at": args.submitted_at,
        "observed_complete_at": args.completed_at,
        "result_status": "complete",
        "public_score": args.score,
        "delta_vs_lineage_base": args.delta,
        "package_sha256": args.package_sha256,
        "stack_composition": split_csv(args.stack_composition),
        "attempt_kind": args.attempt_kind,
        "synced_at": synced_at,
        "sync_status": "synced",
    }
    return {
        "candidate_id": args.candidate_id,
        "score": args.score,
        "previous_global_score": args.previous_score,
        "delta_vs_previous_global": args.delta,
        "task_id": args.task_id,
        "package_sha256": args.package_sha256,
        "changed_entries": split_csv(args.changed_entries),
        "source_package_path": str(args.package_path),
        "sync_status": "synced",
        "synced_at": synced_at,
        "last_submit_result": last_submit_result,
        "worker_owner": "neurogolf-submit-captain",
        "followup_worker": "neurogolf-sync-manager",
        "reviewer_disposition": "approved-for-submit-captain",
        "approval_mode": "auto-rate-limited",
        "predicted_delta": args.expected_gain,
        "expected_point_gain": args.expected_gain,
        "tests_run": [
            f"sha256({args.package_path}) == {args.package_sha256}",
            f"Kaggle COMPLETE public score {args.score}",
        ],
        "gates": {
            "package_hash": "pass",
            "kaggle_public_validation": "pass",
        },
        "package_path": str(args.package_path),
        "lineage_base_score": args.previous_score,
        "changed_task_ids": [entry.removesuffix(".onnx") for entry in split_csv(args.changed_entries)],
        "risk_summary": args.risk_summary,
        "submit_ready": False,
        "submit_queue_status": "submitted-complete-promote-synced",
        "notes": args.sync_note,
        "sync_task_id": args.sync_task_id,
        "updated_at": synced_at,
    }


def update_v2_manifest(doc: dict[str, Any], entry: dict[str, Any], synced_at: str, source_manifest: Path) -> dict[str, Any]:
    updated = dict(doc)
    entries = updated.get("entries")
    if not isinstance(entries, list):
        entries = []
    kept = [
        item
        for item in entries
        if not (isinstance(item, dict) and item.get("candidate_id") == entry["candidate_id"])
    ]
    updated["current_best"] = entry
    updated["entries"] = [entry] + kept
    updated["updated_at"] = synced_at
    updated["source_manifest"] = str(source_manifest)
    updated.setdefault("schema", "neurogolf.candidate_registry.v2-draft")
    # Keep top-level v1-legacy fields in sync with the nested current_best
    # (2026-05-20 fix). Original v2 schema only wrote ``current_best`` /
    # ``entries`` and left top-level ``score`` / ``candidate_id`` / ``task_id``
    # / etc. STALE. Downstream consumers that still consult the top-level
    # (legacy tools, occasional readers, my submit_gate_kaggle before the
    # priority fix) would silently see the old score after a v2 promote.
    # Real-world impact (megastack153 incident, 2026-05-20): top-level
    # said 5835.3 while nested said 5861.43, and submit_gate's lineage-
    # stale gate read the wrong number. Writing both at once keeps every
    # reader consistent regardless of which field they prefer.
    updated["score"] = entry["score"]
    updated["candidate_id"] = entry["candidate_id"]
    updated["task_id"] = entry.get("task_id")
    updated["lineage_base_score"] = entry.get("previous_global_score")
    updated["previous_global_score"] = entry.get("previous_global_score")
    updated["delta_vs_lineage"] = entry.get("delta_vs_previous_global")
    updated["delta_vs_previous_global"] = entry.get("delta_vs_previous_global")
    return updated


def update_flat_manifest(doc: dict[str, Any], entry: dict[str, Any], synced_at: str) -> dict[str, Any]:
    updated = dict(doc)
    updated.update(
        {
            "score": entry["score"],
            "candidate_id": entry["candidate_id"],
            "task_id": entry["task_id"],
            "lineage_base_score": entry["previous_global_score"],
            "previous_global_score": entry["previous_global_score"],
            "delta_vs_lineage": entry["delta_vs_previous_global"],
            "delta_vs_previous_global": entry["delta_vs_previous_global"],
            "submitted_at": entry["last_submit_result"].get("submitted_at"),
            "promoted_at": synced_at,
            "package_path": "submissions/submission.zip",
            "package_sha256": entry["package_sha256"],
            "changed_members": entry["changed_entries"],
            "stack_composition": entry["last_submit_result"].get("stack_composition") or [],
            "source_package_path": entry["source_package_path"],
            "sync_note": entry["notes"],
        }
    )
    return updated


def update_submit_queue(path: Path, entry: dict[str, Any], *, dry_run: bool, actions: list[str]) -> None:
    doc = read_json(path, {})
    if not isinstance(doc, dict) or not doc:
        # Initialize minimal valid doc instead of silently returning
        # (2026-05-20 fix). The original ``return`` skipped if the file
        # was a non-dict (list / null). And read_json returns ``{}`` for
        # missing files, which is technically a dict but lacks the
        # required schema / queue / submit_history scaffolding — the
        # function then "wrote" only ``current_best_manifest`` without
        # the surrounding invariant structure. The auto_sync validator
        # would then fail with "no submit_queue entry matches" forever.
        # Both cases now produce a fresh skeleton.
        doc = {
            "schema": "neurogolf.submit_queue.v1",
            "queue": [],
            "submit_history": [],
            "notes": "Initialized by neurogolf_sync_current_best.update_submit_queue because the existing file was missing, empty, or not a dict.",
        }
    # Ensure required scaffolding even when the doc was non-empty but
    # missing the queue/history sections (e.g. a manifest-only dict).
    doc.setdefault("schema", "neurogolf.submit_queue.v1")
    doc.setdefault("queue", [])
    doc.setdefault("submit_history", [])
    doc["updated_at"] = entry["synced_at"]
    doc["current_best_manifest"] = {
        "candidate_id": entry["candidate_id"],
        "score": entry["score"],
        "previous_global_score": entry["previous_global_score"],
        "delta_vs_previous_global": entry["delta_vs_previous_global"],
        "package_sha256": entry["package_sha256"],
        "changed_entries": entry["changed_entries"],
        "sync_status": "synced",
        "synced_at": entry["synced_at"],
    }
    queue = doc.get("queue")
    matched_in_queue = False
    if isinstance(queue, list):
        for item in queue:
            if not isinstance(item, dict):
                continue
            actual = item.get("actual_result") if isinstance(item.get("actual_result"), dict) else {}
            execution = item.get("submit_execution") if isinstance(item.get("submit_execution"), dict) else {}
            package_match = (
                item.get("candidate_id") == entry["candidate_id"]
                or actual.get("stack_candidate_id") == entry["candidate_id"]
                or actual.get("package_sha256") == entry["package_sha256"]
                or execution.get("package_sha256") == entry["package_sha256"]
            )
            if not package_match:
                continue
            matched_in_queue = True
            item["status"] = "submitted-complete-promote-synced"
            item["submit_queue_status"] = "submitted-complete-promote-synced"
            item["submit_ready"] = False
            item["updated_at"] = entry["synced_at"]
            sync = item.get("sync_followup") if isinstance(item.get("sync_followup"), dict) else {}
            sync.update(
                {
                    "sync_status": "synced",
                    "synced_at": entry["synced_at"],
                    "sync_completed_at": entry["synced_at"],
                    "sync_task_id": entry.get("sync_task_id"),
                }
            )
            item["sync_followup"] = sync

    # Invariant repair (2026-05-19): the validator at
    # tools/neurogolf_auto_sync_current_best.py:325 requires the candidate
    # referenced in current_best_manifest to be findable in either queue[]
    # or submit_history[]. Historically the submit-captain LLM maintained
    # those two arrays manually before invoking sync; with submit-captain
    # off, no Python tool wrote them and the auto-sync cron started failing
    # on every newly-promoted candidate (e.g. alternates31_on_5800_80).
    # Fix: this function — the canonical post-submit promoter — now
    # ALWAYS appends to submit_history[] and ALSO appends a synthetic
    # queue[] entry when none matched. Both writes are idempotent
    # (skip if same candidate_id + sha already present).
    _ensure_submit_history_entry(doc, entry)
    if not matched_in_queue:
        _append_synthetic_queue_entry(doc, entry)

    write_json(path, doc, dry_run=dry_run, actions=actions)


def _ensure_submit_history_entry(doc: dict[str, Any], entry: dict[str, Any]) -> None:
    """Append a submit_history record for ``entry``. Idempotent — does
    nothing if a record with the same candidate_id + package_sha256 already
    exists.

    Schema mirrors the existing 39 entries in the live submit_queue.json:
        at, status, stage_id, candidate_id, package_sha256, submitted_at,
        expected_point_gain, lineage_base_score, kanban_task_id (optional).
    """
    history = doc.setdefault("submit_history", [])
    if not isinstance(history, list):
        return
    cand_id = entry.get("candidate_id")
    sha = (entry.get("package_sha256") or "").upper()
    for row in history:
        if not isinstance(row, dict):
            continue
        if row.get("candidate_id") == cand_id and (row.get("package_sha256") or "").upper() == sha:
            return  # already recorded
    submitted_at = entry.get("last_submit_result", {}).get("submitted_at") or entry.get("synced_at")
    expected_gain = entry.get("expected_point_gain")
    if expected_gain is None:
        expected_gain = entry.get("predicted_delta")
    history.append(
        {
            "at": entry.get("synced_at"),
            "status": "submitted-complete-promote-synced",
            "stage_id": cand_id,
            "candidate_id": cand_id,
            "package_sha256": entry.get("package_sha256"),
            "submitted_at": submitted_at,
            "expected_point_gain": expected_gain,
            "actual_point_gain": entry.get("delta_vs_previous_global"),
            "lineage_base_score": entry.get("previous_global_score"),
            "kaggle_score": entry.get("score"),
            "source": "neurogolf_sync_current_best.update_submit_queue",
        }
    )


def _append_synthetic_queue_entry(doc: dict[str, Any], entry: dict[str, Any]) -> None:
    """Add a synthetic queue[] entry for a promoted candidate that wasn't
    already present. This restores the validator's invariant (every
    current_best_manifest must point to a findable queue entry) when the
    upstream submitter — historically submit-captain LLM, currently
    whatever process stages the multi-task stacks — bypasses the queue.

    Idempotent: if a queue entry with matching candidate_id or sha already
    exists, do nothing.
    """
    queue = doc.setdefault("queue", [])
    if not isinstance(queue, list):
        return
    cand_id = entry.get("candidate_id")
    sha = (entry.get("package_sha256") or "").upper()
    for item in queue:
        if not isinstance(item, dict):
            continue
        if item.get("candidate_id") == cand_id:
            return
        existing_sha = (item.get("package_sha256") or "").upper()
        if existing_sha and existing_sha == sha:
            return
    synthetic = {
        "candidate_id": cand_id,
        "package_sha256": entry.get("package_sha256"),
        "task_id": entry.get("task_id"),
        "score": entry.get("score"),
        "previous_global_score": entry.get("previous_global_score"),
        "delta_vs_previous_global": entry.get("delta_vs_previous_global"),
        "changed_entries": entry.get("changed_entries", []),
        "status": "submitted-complete-promote-synced",
        "submit_queue_status": "submitted-complete-promote-synced",
        "submit_ready": False,
        "expected_point_gain": entry.get("expected_point_gain") or entry.get("predicted_delta"),
        "lineage_base_score": entry.get("previous_global_score"),
        "updated_at": entry.get("synced_at"),
        "sync_followup": {
            "sync_status": "synced",
            "synced_at": entry.get("synced_at"),
            "sync_completed_at": entry.get("synced_at"),
            "sync_task_id": entry.get("sync_task_id"),
        },
        "actual_result": {
            "candidate_id": cand_id,
            "package_sha256": entry.get("package_sha256"),
            "public_score": entry.get("score"),
            "result_status": "complete",
        },
        "submit_execution": {
            "package_sha256": entry.get("package_sha256"),
            "package_path": entry.get("source_package_path") or entry.get("package_path"),
        },
        "notes": (
            "Synthesized by neurogolf_sync_current_best.update_submit_queue because "
            "no pre-existing queue entry matched the promoted candidate. Restores the "
            "validator invariant when the upstream submitter bypasses the queue."
        ),
    }
    queue.append(synthetic)


def append_ledger(path: Path, args: argparse.Namespace, *, dry_run: bool, actions: list[str]) -> None:
    ledger_id = args.ledger_id or f"submit-captain-{args.candidate_id}"
    if path.exists():
        text = path.read_text(encoding="utf-8")
        if ledger_id in text or args.candidate_id in text or args.package_sha256[:8] in text:
            actions.append(f"ledger already contains {ledger_id}; skip")
            return
    row = {
        "date": (args.completed_at or args.submitted_at or utc_now())[:10],
        "number": "n/a",
        "fruit": ledger_id,
        "kind": args.ledger_description
        or f"submit-captain {args.candidate_id} changed={'+'.join(split_csv(args.changed_entries))} sha={args.package_sha256[:8]}",
        "local_prediction": f"+{args.expected_gain:g} vs {args.previous_score:g}" if args.expected_gain is not None else "",
        "kaggle_score": f"{args.score:g}",
        "verdict": args.ledger_verdict
        or f"COMPLETE +{args.delta:g} vs {args.previous_score:g}",
        "source": args.ledger_source or f"kanban {args.sync_task_id or 'deterministic-sync'}",
    }
    actions.append(f"append_ledger {path} {ledger_id}")
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["date", "number", "fruit", "kind", "local_prediction", "kaggle_score", "verdict", "source"],
        )
        if not exists or path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(row)


def current_score(repo_root: Path) -> float | None:
    manifest = read_json(repo_root / "submissions" / "current_best_manifest.json", {})
    if not isinstance(manifest, dict):
        return None
    best = manifest.get("current_best") if isinstance(manifest.get("current_best"), dict) else manifest
    try:
        return float(best.get("score"))
    except (TypeError, ValueError):
        return None


def promote(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    args.package_path = args.package_path.resolve()
    args.package_sha256 = args.package_sha256.upper()
    orchestrator_current_best = effective_orchestrator_current_best(args)
    main_local_dir = effective_main_local_dir(args)
    actions: list[str] = []
    canonical_package = repo_root / "submissions" / "submission.zip"

    if not args.package_path.exists():
        raise SystemExit(f"package not found: {args.package_path}")
    actual_sha = sha256_file(args.package_path)
    if actual_sha != args.package_sha256:
        raise SystemExit(f"package sha mismatch: expected {args.package_sha256}, got {actual_sha}")

    existing_score = current_score(repo_root)
    if existing_score is not None and args.score < existing_score and not args.allow_score_regression:
        raise SystemExit(f"refusing score regression: current={existing_score}, requested={args.score}")

    task_state_entries: list[str] = []
    if not getattr(args, "skip_task_state_sync", False):
        try:
            from tools.neurogolf_sync_task_state import infer_changed_task_entries

            task_state_entries = infer_changed_task_entries(canonical_package, args.package_path, split_csv(args.changed_entries))
        except Exception as exc:  # noqa: BLE001
            actions.append(f"task_state_diff_failed: {exc}")

    synced_at = args.synced_at or utc_now()
    entry = manifest_entry(args, synced_at)

    package_targets = package_target_paths(
        repo_root,
        orchestrator_current_best=orchestrator_current_best,
        main_local_dir=main_local_dir,
    )
    for target in package_targets:
        copy_verified_package(args.package_path, target, dry_run=args.dry_run, actions=actions)

    root_manifest_path = repo_root / "submissions" / "current_best_manifest.json"
    root_manifest = read_json(root_manifest_path, {"schema": "neurogolf.candidate_registry.v2-draft", "entries": []})
    if isinstance(root_manifest, dict) and ("current_best" in root_manifest or "entries" in root_manifest):
        root_manifest = update_v2_manifest(root_manifest, entry, synced_at, root_manifest_path)
    elif isinstance(root_manifest, dict):
        root_manifest = update_flat_manifest(root_manifest, entry, synced_at)
    write_json(root_manifest_path, root_manifest, dry_run=args.dry_run, actions=actions)

    repo_manifest_path = repo_root / "repo" / "submissions" / "current_best_manifest.json"
    if repo_manifest_path.parent.exists():
        repo_manifest = read_json(repo_manifest_path, {})
        repo_manifest = update_flat_manifest(repo_manifest if isinstance(repo_manifest, dict) else {}, entry, synced_at)
        write_json(repo_manifest_path, repo_manifest, dry_run=args.dry_run, actions=actions)

    if orchestrator_current_best:
        orchestrator_manifest_path = orchestrator_current_best.with_name("manifest.json")
        orchestrator_manifest = read_json(
            orchestrator_manifest_path,
            {"schema": "neurogolf.candidate_registry.v2-draft", "entries": []},
        )
        if isinstance(orchestrator_manifest, dict) and (
            "current_best" in orchestrator_manifest or "entries" in orchestrator_manifest
        ):
            orchestrator_manifest = update_v2_manifest(orchestrator_manifest, entry, synced_at, root_manifest_path)
        else:
            orchestrator_manifest = update_flat_manifest(
                orchestrator_manifest if isinstance(orchestrator_manifest, dict) else {},
                entry,
                synced_at,
            )
        write_json(orchestrator_manifest_path, orchestrator_manifest, dry_run=args.dry_run, actions=actions)
        write_text(
            orchestrator_current_best.with_name("sha256.txt"),
            f"{args.package_sha256}  submission.zip\n",
            dry_run=args.dry_run,
            actions=actions,
        )

    if main_local_dir and (main_local_dir / "submissions").exists():
        main_local_manifest_path = main_local_dir / "submissions" / "current_best_manifest.json"
        main_local_manifest = read_json(
            main_local_manifest_path,
            {"schema": "neurogolf.candidate_registry.v2-draft", "entries": []},
        )
        if isinstance(main_local_manifest, dict) and (
            "current_best" in main_local_manifest or "entries" in main_local_manifest
        ):
            main_local_manifest = update_v2_manifest(main_local_manifest, entry, synced_at, root_manifest_path)
        else:
            main_local_manifest = update_flat_manifest(
                main_local_manifest if isinstance(main_local_manifest, dict) else {},
                entry,
                synced_at,
            )
        write_json(main_local_manifest_path, main_local_manifest, dry_run=args.dry_run, actions=actions)

    registry_path = repo_root / ".hermes" / "board" / "candidate_registry.json"
    if registry_path.exists():
        registry = read_json(registry_path, {"schema": "neurogolf.candidate_registry.v2-draft", "entries": []})
        if isinstance(registry, dict):
            registry = update_v2_manifest(registry, entry, synced_at, root_manifest_path)
            write_json(registry_path, registry, dry_run=args.dry_run, actions=actions)

    submit_queue_path = repo_root / ".hermes" / "board" / "submit_queue.json"
    if submit_queue_path.exists():
        update_submit_queue(submit_queue_path, entry, dry_run=args.dry_run, actions=actions)

    append_ledger(repo_root / "Logs" / "submissions_ledger.csv", args, dry_run=args.dry_run, actions=actions)

    task_state_sync: dict[str, Any] | None = None
    if getattr(args, "skip_task_state_sync", False):
        task_state_sync = {"ok": True, "skipped": True, "reason": "--skip-task-state-sync"}
        actions.append("task_state_sync skipped by CLI flag")
    else:
        try:
            from tools.neurogolf_sync_task_state import sync_task_state

            task_state_sync = sync_task_state(
                repo_root=repo_root,
                package_path=canonical_package if canonical_package.exists() and not args.dry_run else args.package_path,
                changed_entries=task_state_entries or split_csv(args.changed_entries),
                candidate_id=args.candidate_id,
                primary_task_id=parse_task_id(args.task_id),
                public_score=args.score,
                previous_score=args.previous_score,
                delta=args.delta,
                completed_at=args.completed_at or synced_at,
                mark_shipped=True,
                dry_run=args.dry_run,
                run_enrich=not getattr(args, "skip_task_state_enrich", False),
            )
            if task_state_sync.get("skipped"):
                actions.append(f"task_state_sync skipped: {task_state_sync.get('reason')}")
            else:
                actions.append(
                    "task_state_sync updated="
                    + ",".join(f"task{int(tid):03d}" for tid in task_state_sync.get("updated_task_ids", []))
                )
        except Exception as exc:  # noqa: BLE001
            task_state_sync = {"ok": False, "error": str(exc), "changed_entries": task_state_entries}
            actions.append(f"task_state_sync failed: {exc}")

    final_hashes = {}
    target_hash_audit = {}
    for target in package_targets:
        current_hash = sha256_file(target) if target.exists() else None
        target_hash_audit[str(target)] = {
            "current_sha256": current_hash,
            "planned_sha256": args.package_sha256,
            "would_change": current_hash != args.package_sha256,
            "written": not args.dry_run,
        }
        if target.exists():
            final_hashes[str(target)] = current_hash
        elif args.dry_run:
            final_hashes[str(target)] = "(dry-run not written)"

    return {
        "ok": True,
        "dry_run": args.dry_run,
        "candidate_id": args.candidate_id,
        "score": args.score,
        "package_sha256": args.package_sha256,
        "verified_source_sha256": actual_sha,
        "previous_current_score": existing_score,
        "synced_at": synced_at,
        "actions": actions,
        "final_hashes": final_hashes,
        "target_hash_audit": target_hash_audit,
        "task_state_sync": task_state_sync,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--package-path", type=Path, required=True)
    parser.add_argument("--package-sha256", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--score", type=float, required=True)
    parser.add_argument("--previous-score", type=float, required=True)
    parser.add_argument("--delta", type=float, required=True)
    parser.add_argument("--expected-gain", type=as_float, default=None)
    parser.add_argument("--changed-entries", required=True, help="Comma-separated zip entries, e.g. task221.onnx,task303.onnx")
    parser.add_argument("--submitted-at", default="")
    parser.add_argument("--completed-at", default="")
    parser.add_argument("--attempt-kind", default="stack")
    parser.add_argument("--task-id", default="stack")
    parser.add_argument("--stack-composition", default="")
    parser.add_argument("--risk-summary", default="")
    parser.add_argument("--sync-note", default="Deterministic exact-package current-best sync.")
    parser.add_argument("--sync-task-id", default="")
    parser.add_argument("--ledger-id", default="")
    parser.add_argument("--ledger-description", default="")
    parser.add_argument("--ledger-verdict", default="")
    parser.add_argument("--ledger-source", default="")
    parser.add_argument("--synced-at", default="")
    parser.add_argument(
        "--orchestrator-current-best",
        type=Path,
        default=None,
        help="Optional legacy mirror path for an orchestrator current-best zip; ignored unless --sync-legacy-mirrors is set.",
    )
    parser.add_argument(
        "--main-local-dir",
        type=Path,
        default=None,
        help="Optional legacy local directory whose current-best package mirrors should be kept in sync; ignored unless --sync-legacy-mirrors is set.",
    )
    parser.add_argument(
        "--sync-legacy-mirrors",
        action="store_true",
        help="Opt in to updating legacy .notongit current-best mirrors. Default is repo-local canonical artifacts only.",
    )
    parser.add_argument("--allow-score-regression", action="store_true")
    parser.add_argument(
        "--skip-task-state-sync",
        action="store_true",
        help="Do not refresh The_400_Tasks task_state.csv / NeuroGolf_Task_Status.xlsx after promotion.",
    )
    parser.add_argument(
        "--skip-task-state-enrich",
        action="store_true",
        help="Patch task costs/status only; skip the derived workbook enrichment pass.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = promote(args)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
