#!/usr/bin/env python3
"""Auto-promote NeuroGolf public-positive current-best packages.

This script is meant for a no-agent Hermes cron. It reads the structured
submit queue, checks whether the queue's public-positive current best is ahead
of the canonical current-best artifacts, and if so delegates to
neurogolf_sync_current_best.py with exact package/hash fields.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


DEFAULT_REPO_ROOT = Path(os.environ.get("NEUROGOLF_CANONICAL_ROOT", Path(__file__).resolve().parent.parent))
if str(DEFAULT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEFAULT_REPO_ROOT))

from tools.neurogolf_sync_current_best import (  # noqa: E402
    current_score,
    effective_main_local_dir,
    effective_orchestrator_current_best,
    promote,
    sha256_file,
    utc_now,
)


def read_json(path: Path, fallback: Any) -> Any:
    if not path.exists():
        return fallback
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def first_nonempty(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", []):
            return value
    return None


def resolve_path(repo_root: Path, value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    return repo_root / path


def current_manifest(repo_root: Path) -> dict[str, Any]:
    doc = read_json(repo_root / "submissions" / "current_best_manifest.json", {})
    if not isinstance(doc, dict):
        return {}
    best = doc.get("current_best")
    return best if isinstance(best, dict) else doc


def canonical_package_sha(repo_root: Path) -> str | None:
    path = repo_root / "submissions" / "submission.zip"
    if not path.exists():
        return None
    return sha256_file(path)


def package_sha_or_none(path: Path) -> str | None:
    if not path.exists():
        return None
    return sha256_file(path)


def best_from_manifest(path: Path) -> dict[str, Any]:
    doc = read_json(path, {})
    if not isinstance(doc, dict):
        return {}
    best = doc.get("current_best")
    return best if isinstance(best, dict) else doc


def package_mirror_paths(
    repo_root: Path,
    *,
    orchestrator_current_best: Path | None,
    main_local_dir: Path | None,
) -> list[Path]:
    paths = [repo_root / "submissions" / "submission.zip"]
    repo_mirror = repo_root / "repo" / "submissions" / "submission.zip"
    if repo_mirror.exists() or repo_mirror.parent.exists():
        paths.append(repo_mirror)
    if orchestrator_current_best and (orchestrator_current_best.exists() or orchestrator_current_best.parent.exists()):
        paths.append(orchestrator_current_best)
    if main_local_dir and main_local_dir.exists():
        root_zip = main_local_dir / "submission.zip"
        if root_zip.exists():
            paths.append(root_zip)
        submissions_zip = main_local_dir / "submissions" / "submission.zip"
        if submissions_zip.exists() or submissions_zip.parent.exists():
            paths.append(submissions_zip)
    return paths


def manifest_mirror_paths(
    repo_root: Path,
    *,
    orchestrator_current_best: Path | None,
    main_local_dir: Path | None,
) -> list[Path]:
    paths = [repo_root / "submissions" / "current_best_manifest.json"]
    repo_mirror = repo_root / "repo" / "submissions" / "current_best_manifest.json"
    if repo_mirror.exists() or repo_mirror.parent.exists():
        paths.append(repo_mirror)
    if orchestrator_current_best and orchestrator_current_best.parent.exists():
        paths.append(orchestrator_current_best.with_name("manifest.json"))
    if main_local_dir and (main_local_dir / "submissions").exists():
        paths.append(main_local_dir / "submissions" / "current_best_manifest.json")
    return paths


def mirror_state(
    repo_root: Path,
    *,
    expected_candidate_id: str,
    expected_score: float | None,
    expected_sha: str,
    orchestrator_current_best: Path | None,
    main_local_dir: Path | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    packages: list[dict[str, Any]] = []
    stale = False
    for path in package_mirror_paths(
        repo_root,
        orchestrator_current_best=orchestrator_current_best,
        main_local_dir=main_local_dir,
    ):
        actual_sha = package_sha_or_none(path)
        ok = actual_sha == expected_sha
        stale = stale or not ok
        packages.append({"path": str(path), "sha256": actual_sha, "ok": ok})

    manifests: list[dict[str, Any]] = []
    for path in manifest_mirror_paths(
        repo_root,
        orchestrator_current_best=orchestrator_current_best,
        main_local_dir=main_local_dir,
    ):
        best = best_from_manifest(path)
        actual_sha = str(best.get("package_sha256") or "").upper() or None
        actual_score = as_float(best.get("score"))
        actual_candidate = str(best.get("candidate_id") or "") or None
        ok = (
            path.exists()
            and actual_sha == expected_sha
            and actual_candidate == expected_candidate_id
            and (expected_score is None or actual_score == expected_score)
        )
        stale = stale or not ok
        manifests.append(
            {
                "path": str(path),
                "candidate_id": actual_candidate,
                "score": actual_score,
                "package_sha256": actual_sha,
                "ok": ok,
            }
        )
    return packages, manifests, stale


def queue_entries(doc: dict[str, Any]) -> list[dict[str, Any]]:
    queue = doc.get("queue")
    if not isinstance(queue, list):
        return []
    return [entry for entry in queue if isinstance(entry, dict)]


def submit_history_entries(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Return submit_history[] from the queue doc — empty list on missing.

    Used as a fallback by ``best_matching_entry`` so the validator accepts
    a candidate when ``queue[]`` was bypassed by an upstream submitter but
    ``submit_history[]`` was populated (or when ``update_submit_queue``
    synthesized history without a matching queue entry).
    """
    hist = doc.get("submit_history")
    if not isinstance(hist, list):
        return []
    return [entry for entry in hist if isinstance(entry, dict)]


def matches_known_best(entry: dict[str, Any], candidate_id: str, package_sha: str) -> bool:
    actual = as_dict(entry.get("actual_result"))
    execution = as_dict(entry.get("submit_execution"))
    candidates = {
        str(entry.get("candidate_id") or ""),
        str(actual.get("stack_candidate_id") or ""),
        str(actual.get("candidate_id") or ""),
    }
    shas = {
        str(entry.get("package_sha256") or "").upper(),
        str(actual.get("package_sha256") or "").upper(),
        str(execution.get("package_sha256") or "").upper(),
    }
    return candidate_id in candidates or package_sha.upper() in shas


def best_matching_entry(entries: list[dict[str, Any]], candidate_id: str, package_sha: str) -> dict[str, Any] | None:
    matches = [entry for entry in entries if matches_known_best(entry, candidate_id, package_sha)]
    if not matches:
        return None
    for entry in matches:
        execution = as_dict(entry.get("submit_execution"))
        if execution.get("package_path") and str(execution.get("package_sha256") or "").upper() == package_sha.upper():
            return entry
    return matches[0]


def changed_entries_from(known: dict[str, Any], actual: dict[str, Any], execution: dict[str, Any], entry: dict[str, Any]) -> list[str]:
    for value in (
        known.get("changed_entries"),
        actual.get("stack_changed_entries"),
        actual.get("changed_entries"),
        execution.get("changed_entries"),
        entry.get("changed_entries"),
        entry.get("changed_task_ids"),
    ):
        items = as_list(value)
        if items:
            return [str(item if str(item).endswith(".onnx") else f"{item}.onnx") for item in items]
    return []


def stack_composition_from(actual: dict[str, Any], entry: dict[str, Any], candidate_id: str) -> list[str]:
    for value in (
        actual.get("stack_composition"),
        actual.get("stack_candidate_ids"),
        as_dict(entry.get("stack_plan")).get("stack_candidate_ids"),
    ):
        items = as_list(value)
        if items:
            return [str(item) for item in items]
    return [candidate_id]


def expected_gain_from(actual: dict[str, Any], execution: dict[str, Any], entry: dict[str, Any]) -> float | None:
    return as_float(
        first_nonempty(
            actual.get("expected_point_gain_sum_for_submit"),
            actual.get("expected_point_gain"),
            execution.get("expected_point_gain"),
            entry.get("expected_lb_point_gain"),
            entry.get("expected_point_gain"),
            as_dict(entry.get("approval_packet")).get("expected_lb_point_gain"),
            as_dict(entry.get("approval_packet")).get("expected_point_gain"),
        )
    )


def build_promotion_args(args: argparse.Namespace, queue_doc: dict[str, Any]) -> tuple[argparse.Namespace | None, dict[str, Any]]:
    repo_root = args.repo_root.resolve()
    orchestrator_current_best = effective_orchestrator_current_best(args)
    main_local_dir = effective_main_local_dir(args)
    known = as_dict(queue_doc.get("current_best_manifest"))
    known_score = as_float(known.get("score"))
    known_candidate_id = str(known.get("candidate_id") or "")
    known_sha = str(known.get("package_sha256") or "").upper()
    synced = current_manifest(repo_root)
    synced_score = current_score(repo_root)
    synced_sha = str(synced.get("package_sha256") or "").upper()
    canonical_sha = canonical_package_sha(repo_root)

    state = {
        "known_public_best": known,
        "synced_score": synced_score,
        "synced_candidate_id": synced.get("candidate_id"),
        "synced_package_sha256": synced_sha or None,
        "canonical_package_sha256": canonical_sha,
    }

    if known_score is None or not known_candidate_id or not known_sha:
        state["reason"] = "submit_queue current_best_manifest is missing score/candidate/package_sha256"
        return None, state
    if synced_score is not None and known_score < synced_score:
        state["reason"] = "submit_queue current best is older than canonical current best"
        return None, state

    package_mirrors, manifest_mirrors, stale_mirrors = mirror_state(
        repo_root,
        expected_candidate_id=known_candidate_id,
        expected_score=known_score,
        expected_sha=known_sha,
        orchestrator_current_best=orchestrator_current_best,
        main_local_dir=main_local_dir,
    )
    state["package_mirrors"] = package_mirrors
    state["manifest_mirrors"] = manifest_mirrors

    needs_sync = (
        synced_score is None
        or known_score > synced_score
        or synced_sha != known_sha
        or canonical_sha != known_sha
        or known.get("sync_status") != "synced"
        or stale_mirrors
    )
    if not needs_sync:
        state["reason"] = "canonical current best already matches submit_queue current best"
        return None, state

    entry = best_matching_entry(queue_entries(queue_doc), known_candidate_id, known_sha)
    # Widening (2026-05-19): when queue[] doesn't match (e.g. submit-captain
    # bypass / synthetic-queue-entry not yet written by update_submit_queue),
    # fall back to submit_history[]. This makes the validator accept any
    # promoted candidate that has SOME trail of evidence (queue or history),
    # rather than requiring a queue entry that historically only
    # submit-captain LLM created.
    if entry is None:
        entry = best_matching_entry(
            submit_history_entries(queue_doc), known_candidate_id, known_sha,
        )
    mirror_only_repair = (
        entry is None
        and stale_mirrors
        and synced_score == known_score
        and synced_sha == known_sha
        and canonical_sha == known_sha
    )
    if entry is None and not mirror_only_repair:
        raise SystemExit(f"no submit_queue entry matches current_best candidate={known_candidate_id} sha={known_sha}")

    entry_doc = entry if entry is not None else {}
    actual = as_dict(entry_doc.get("actual_result"))
    execution = as_dict(entry_doc.get("submit_execution"))
    package_path = resolve_path(
        repo_root,
        first_nonempty(
            execution.get("package_path"),
            actual.get("package_path"),
            entry_doc.get("package_path"),
            repo_root / "submissions" / "submission.zip" if mirror_only_repair else None,
        ),
    )
    if package_path is None:
        raise SystemExit(f"matched queue entry for {known_candidate_id} has no staged package_path")

    previous_score = as_float(
        first_nonempty(
            known.get("previous_global_score"),
            known.get("lineage_base_score"),
            actual.get("previous_best_score"),
            actual.get("lineage_base_score"),
            execution.get("lineage_base_score"),
            entry_doc.get("lineage_base_score"),
        )
    )
    delta = as_float(
        first_nonempty(
            known.get("delta_vs_previous_global"),
            actual.get("delta_vs_previous_best"),
            actual.get("delta_vs_lineage_base"),
            execution.get("delta_vs_lineage_base"),
        )
    )
    if previous_score is None and delta is not None:
        previous_score = known_score - delta
    if delta is None and previous_score is not None:
        delta = known_score - previous_score
    if previous_score is None or delta is None:
        raise SystemExit(f"matched queue entry for {known_candidate_id} is missing previous score/delta")

    changed_entries = changed_entries_from(known, actual, execution, entry_doc)
    if not changed_entries:
        raise SystemExit(f"matched queue entry for {known_candidate_id} is missing changed_entries")

    sync_followup = as_dict(entry_doc.get("sync_followup"))
    expected_gain = expected_gain_from(actual, execution, entry_doc)
    task_id = "stack" if known_candidate_id.startswith("stack_") else str(entry_doc.get("task_id") or known.get("task_id") or "stack")
    promo_args = argparse.Namespace(
        repo_root=repo_root,
        package_path=package_path,
        package_sha256=known_sha,
        candidate_id=known_candidate_id,
        score=known_score,
        previous_score=previous_score,
        delta=delta,
        expected_gain=expected_gain,
        changed_entries=",".join(changed_entries),
        submitted_at=str(first_nonempty(execution.get("submitted_at"), actual.get("submitted_at")) or ""),
        completed_at=str(first_nonempty(execution.get("observed_complete_at"), actual.get("observed_complete_at")) or ""),
        attempt_kind=str(first_nonempty(execution.get("attempt_kind"), actual.get("attempt_kind")) or "stack"),
        task_id=task_id,
        stack_composition=",".join(stack_composition_from(actual, entry_doc, known_candidate_id)),
        risk_summary=str(
            first_nonempty(entry_doc.get("risk_summary"), as_dict(entry_doc.get("approval_packet")).get("risk_summary"))
            or ""
        ),
        sync_note=(
            "Deterministic auto-sync from .hermes/board/submit_queue.json current_best_manifest; "
            "package path and SHA were read from the matched submit_execution."
        ),
        sync_task_id=str(first_nonempty(sync_followup.get("sync_task_id"), "auto-python-sync")),
        ledger_id=f"submit-captain-{known_candidate_id}",
        ledger_description=(
            str(actual.get("description"))
            if actual.get("description")
            else f"submit-captain {known_candidate_id} changed={'+'.join(changed_entries)} sha={known_sha[:8]}"
        ),
        ledger_verdict=(
            f"COMPLETE +{delta:g} vs {previous_score:g}"
            + (f" (expected +{expected_gain:g})" if expected_gain is not None else "")
        ),
        ledger_source="deterministic auto-sync from submit_queue current_best_manifest",
        synced_at=args.synced_at,
        orchestrator_current_best=orchestrator_current_best,
        main_local_dir=main_local_dir,
        sync_legacy_mirrors=bool(getattr(args, "sync_legacy_mirrors", False)),
        skip_task_state_sync=bool(getattr(args, "skip_task_state_sync", False)),
        skip_task_state_enrich=bool(getattr(args, "skip_task_state_enrich", False)),
        allow_score_regression=False,
        dry_run=args.dry_run,
    )
    state["matched_queue_id"] = entry_doc.get("queue_id")
    state["matched_candidate_id"] = entry_doc.get("candidate_id")
    state["package_path"] = str(package_path)
    if mirror_only_repair:
        state["mirror_only_repair"] = True
    return promo_args, state


def entry_public_score(entry: dict[str, Any]) -> float | None:
    actual = as_dict(entry.get("actual_result"))
    execution = as_dict(entry.get("submit_execution"))
    return as_float(first_nonempty(actual.get("public_score"), execution.get("public_score")))


def has_stale_sync_pending(entry: dict[str, Any]) -> bool:
    status_text = " ".join(
        str(value or "")
        for value in (
            entry.get("status"),
            entry.get("submit_queue_status"),
        )
    )
    return "promote-sync-pending" in status_text or "sync-pending" in status_text


def cleanup_superseded_sync_pending(
    args: argparse.Namespace,
    queue_doc: dict[str, Any],
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    known = as_dict(state.get("known_public_best"))
    current_candidate_id = str(first_nonempty(known.get("candidate_id"), state.get("synced_candidate_id")) or "")
    current_sha = str(first_nonempty(known.get("package_sha256"), state.get("synced_package_sha256")) or "").upper()
    current_score_value = as_float(first_nonempty(known.get("score"), state.get("synced_score")))
    if not current_candidate_id or current_score_value is None:
        return []

    changed: list[dict[str, Any]] = []
    now = args.synced_at or utc_now()
    for entry in queue_entries(queue_doc):
        if not has_stale_sync_pending(entry):
            continue
        if current_sha and matches_known_best(entry, current_candidate_id, current_sha):
            continue
        public_score = entry_public_score(entry)
        if public_score is not None and public_score > current_score_value:
            continue

        old_status = entry.get("status")
        old_queue_status = entry.get("submit_queue_status")
        entry["status"] = "submitted-complete-promote-superseded"
        entry["submit_queue_status"] = "submitted-complete-promote-superseded"
        entry["submit_ready"] = False
        entry["updated_at"] = now
        sync = as_dict(entry.get("sync_followup"))
        sync.update(
            {
                "sync_status": "superseded-by-current-best",
                "superseded_at": now,
                "superseded_by_candidate_id": current_candidate_id,
                "superseded_by_score": current_score_value,
                "superseded_by_package_sha256": current_sha,
            }
        )
        entry["sync_followup"] = sync
        changed.append(
            {
                "candidate_id": entry.get("candidate_id"),
                "old_status": old_status,
                "old_submit_queue_status": old_queue_status,
                "new_status": entry["status"],
                "public_score": public_score,
            }
        )

    if changed:
        queue_doc["updated_at"] = now
        if not args.dry_run:
            path = args.repo_root.resolve() / ".hermes" / "board" / "submit_queue.json"
            path.write_text(json.dumps(queue_doc, indent=2) + "\n", encoding="utf-8")
    return changed


def run_auto_sync(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    queue_doc = read_json(repo_root / ".hermes" / "board" / "submit_queue.json", {})
    if not isinstance(queue_doc, dict):
        raise SystemExit("submit_queue.json is missing or not a JSON object")

    promo_args, state = build_promotion_args(args, queue_doc)
    if promo_args is None:
        cleaned = cleanup_superseded_sync_pending(args, queue_doc, state)
        if cleaned:
            return {
                "ok": True,
                "action": "dry_run_cleanup" if args.dry_run else "cleaned_queue_sync_state",
                **state,
                "cleaned": cleaned,
            }
        return {
            "ok": True,
            "action": "no_action",
            **state,
        }
    result = promote(promo_args)
    refreshed_queue_doc = read_json(repo_root / ".hermes" / "board" / "submit_queue.json", {})
    cleaned = cleanup_superseded_sync_pending(
        args,
        refreshed_queue_doc if isinstance(refreshed_queue_doc, dict) else {},
        {
            **state,
            "known_public_best": {
                **as_dict(state.get("known_public_best")),
                "candidate_id": promo_args.candidate_id,
                "score": promo_args.score,
                "package_sha256": promo_args.package_sha256,
            },
            "synced_score": promo_args.score,
            "synced_candidate_id": promo_args.candidate_id,
            "synced_package_sha256": promo_args.package_sha256,
        },
    )
    return {
        "ok": True,
        "action": "dry_run" if args.dry_run else "promoted",
        **state,
        "promotion": result,
        "cleaned": cleaned,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument(
        "--orchestrator-current-best",
        type=Path,
        default=None,
        help="Optional legacy mirror path for an orchestrator current-best zip; ignored unless --sync-legacy-mirrors is set.",
    )
    parser.add_argument("--main-local-dir", type=Path, default=None)
    parser.add_argument(
        "--sync-legacy-mirrors",
        action="store_true",
        help="Opt in to updating legacy .notongit current-best mirrors. Default is repo-local canonical artifacts only.",
    )
    parser.add_argument("--synced-at", default="")
    parser.add_argument(
        "--skip-task-state-sync",
        action="store_true",
        help="Do not refresh task_state.csv / NeuroGolf_Task_Status.xlsx after auto-promotion.",
    )
    parser.add_argument(
        "--skip-task-state-enrich",
        action="store_true",
        help="Patch task costs/status only; skip the derived workbook enrichment pass.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--always", action="store_true", help="Print no-action JSON too; default is silent no-op.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run_auto_sync(args)
    if result.get("action") != "no_action" or args.always:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
