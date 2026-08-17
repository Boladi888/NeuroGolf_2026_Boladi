#!/usr/bin/env python3
"""Submit-policy helpers for NeuroGolf orchestration.

This module keeps the queue and registry metadata coherent without executing
Kaggle submits. It can:

- normalize queue entries and stack plans under the current auto-rate-limited
  policy,
- compute candidate-registry preflight history from CURRENT_STATE.md, submit
  history, task outcome memory, and preserved failed manifests,
- compute/store a duplicate-guard key for identical package/lineage/change-set
  requeues,
- upsert registry entries, normalize legacy v1 entries, and bootstrap a submit-
  credit state file for the control plane.

The default behavior is side-effect free unless --write/--write-registry/
--enqueue is requested.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import itertools
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from tools.neurogolf_board_policy import (
        build_chain_keys,
        build_live_card_guard,
        read_board_state_doc,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from neurogolf_board_policy import (
        build_chain_keys,
        build_live_card_guard,
        read_board_state_doc,
    )

DEFAULT_APPROVAL_MODE = "auto-rate-limited"
ALLOWED_APPROVAL_MODES = {DEFAULT_APPROVAL_MODE, "manual-required"}
RATE_LIMIT_MINUTES = 10
DAILY_SUBMIT_CAP = 100
SUBMIT_AUTHORITY = "submit-captain only"
LEGACY_ALLOWED_HISTORY_STATUSES = {"pass", "reviewer-exception", "needs-new-evidence", "blocked-known-flat", "blocked-duplicate"}

REPO_ROOT = Path(__file__).resolve().parent.parent
BOARD_DIR = REPO_ROOT / ".hermes" / "board"
STATE_JSON = BOARD_DIR / "state.json"
CANDIDATE_REGISTRY_JSON = BOARD_DIR / "candidate_registry.json"
SUBMIT_QUEUE_JSON = BOARD_DIR / "submit_queue.json"
TASK_OUTCOME_REGISTRY_JSON = BOARD_DIR / "task_outcome_registry.json"
CURRENT_STATE_MD = REPO_ROOT / "CURRENT_STATE.md"
SUBMISSIONS_LEDGER_CSV = REPO_ROOT / "Logs" / "submissions_ledger.csv"
SUBMIT_CREDIT_STATE_JSON = BOARD_DIR / "submit_credit_state.json"
ORCHESTRATOR_FAILED_ROOT = (
    Path(os.environ.get("NEUROGOLF_NOGIT_ROOT", REPO_ROOT / ".notongit" / "NeuroGolf_nogit"))
    / "orchestrator"
    / "preserved"
    / "failed"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def to_str_list(values: Any) -> list[str]:
    if not values:
        return []
    if isinstance(values, str):
        values = [values]
    out = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            out.append(text)
    return sorted(dict.fromkeys(out))


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_state_refs() -> dict[str, str]:
    state = read_json(STATE_JSON)
    if isinstance(state, dict):
        refs = state.get("workflow_refs")
        if isinstance(refs, dict):
            return {str(k): str(v) for k, v in refs.items() if v}
    return {}


def current_paths() -> dict[str, Path]:
    refs = load_state_refs()
    return {
        "candidate_registry": Path(refs.get("candidate_registry", CANDIDATE_REGISTRY_JSON)),
        "submit_queue": Path(refs.get("submit_queue", SUBMIT_QUEUE_JSON)),
        "current_best_manifest": Path(refs.get("current_best_manifest", REPO_ROOT / "submissions" / "current_best_manifest.json")),
        "submit_credit_state": Path(refs.get("submit_credit_state", SUBMIT_CREDIT_STATE_JSON)),
    }


def read_task_outcome_registry() -> dict[str, Any]:
    doc = read_json(TASK_OUTCOME_REGISTRY_JSON)
    return doc if isinstance(doc, dict) else {}


def task_outcome_record(task_id: str | None) -> dict[str, Any] | None:
    if not task_id:
        return None
    doc = read_task_outcome_registry()
    tasks = doc.get("tasks")
    if isinstance(tasks, dict):
        rec = tasks.get(task_id)
        if isinstance(rec, dict):
            return rec
    return None


def current_state_hits(task_id: str | None) -> list[str]:
    if not task_id or not CURRENT_STATE_MD.exists():
        return []
    try:
        text = CURRENT_STATE_MD.read_text(encoding="utf-8")
    except Exception:
        return []
    hits = []
    for line in text.splitlines():
        if task_id in line:
            clean = line.strip()
            hits.append(clean[:220])
            if len(hits) >= 5:
                break
    return hits


def submit_ledger_hits(candidate: dict[str, Any]) -> list[dict[str, str]]:
    if not SUBMISSIONS_LEDGER_CSV.exists():
        return []
    needles = {
        str(candidate.get("candidate_id") or "").strip().lower(),
        str(candidate.get("task_id") or "").strip().lower(),
        str(candidate.get("package_sha256") or "").strip().lower(),
    }
    needles.discard("")
    if not needles:
        return []
    hits: list[dict[str, str]] = []
    try:
        with SUBMISSIONS_LEDGER_CSV.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row_text = " ".join(str(v) for v in row.values()).lower()
                if any(needle in row_text for needle in needles):
                    hits.append({k: str(v) for k, v in row.items() if v not in (None, "")})
                    if len(hits) >= 8:
                        break
    except Exception:
        return []
    return hits


def failed_manifest_records() -> list[dict[str, Any]]:
    if not ORCHESTRATOR_FAILED_ROOT.exists():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(ORCHESTRATOR_FAILED_ROOT.rglob("*manifest*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(doc, dict):
            doc = copy.deepcopy(doc)
            doc["_path"] = str(path)
            records.append(doc)
    return records


def current_history_sources() -> list[str]:
    sources = [
        "CURRENT_STATE.md",
        ".hermes/board/submit_queue.json submit_history",
        "preserved/failed manifests",
        ".hermes/board/task_outcome_registry.json",
    ]
    task_card_root = REPO_ROOT / "The_400_Tasks" / "Task_by_Task"
    if task_card_root.exists():
        sources.insert(1, "The_400_Tasks/Task_by_Task/<task>/")
    if (REPO_ROOT / "The_400_Tasks" / "NeuroGolf_Task_Status.xlsx").exists():
        sources.append("The_400_Tasks/NeuroGolf_Task_Status.xlsx")
    elif (REPO_ROOT / "The_400_Tasks" / "task_state.csv").exists():
        sources.append("The_400_Tasks/task_state.csv")
    if SUBMISSIONS_LEDGER_CSV.exists():
        sources.append("Logs/submissions_ledger.csv")
    return sources


def normalize_template_from(doc: dict[str, Any], key: str) -> dict[str, Any]:
    template = doc.get(key)
    return copy.deepcopy(template) if isinstance(template, dict) else {}


def _candidate_gain(candidate: dict[str, Any]) -> float:
    value = candidate.get("expected_point_gain", candidate.get("predicted_delta", 0))
    try:
        return float(value or 0)
    except Exception:
        return 0.0


def _candidate_risk(candidate: dict[str, Any]) -> float:
    if "risk_score" in candidate:
        try:
            return float(candidate.get("risk_score") or 0)
        except Exception:
            return 0.0
    flags = candidate.get("risk_flags") or []
    return float(len(flags))


def _lineage(candidate: dict[str, Any]) -> str | None:
    lineage = candidate.get("lineage_base_candidate_id")
    if lineage:
        return str(lineage)
    lineage_obj = candidate.get("lineage_base")
    if isinstance(lineage_obj, dict):
        return lineage_obj.get("candidate_id") or lineage_obj.get("id")
    return None


def _changed_tasks(candidate: dict[str, Any]) -> set[str]:
    values = candidate.get("changed_task_ids") or candidate.get("review_changed_task_ids") or candidate.get("changed_tasks") or []
    return set(to_str_list(values))


def _preflight_allowed(candidate: dict[str, Any]) -> bool:
    status = (candidate.get("preflight_history") or {}).get("status") or candidate.get("preflight_status")
    return status in {"pass", "reviewer-exception"}


def _eligible(candidate: dict[str, Any]) -> bool:
    return (
        candidate.get("reviewer_disposition") == "approved-for-submit-captain"
        and _preflight_allowed(candidate)
        and (candidate.get("approval_mode") or DEFAULT_APPROVAL_MODE) in ALLOWED_APPROVAL_MODES
    )


def plan_stacks(candidates: list[dict[str, Any]], *, max_stack_size: int = 3) -> list[dict[str, Any]]:
    """Plan lineage-compatible, independent/additive stacks up to size 3."""
    max_stack_size = min(max(1, int(max_stack_size)), 3)
    groups: dict[str | None, list[dict[str, Any]]] = {}
    for candidate in candidates:
        if _eligible(candidate):
            groups.setdefault(_lineage(candidate), []).append(candidate)

    plans: list[dict[str, Any]] = []
    for lineage, group in groups.items():
        ordered = sorted(group, key=lambda c: (-_candidate_gain(c), _candidate_risk(c), c.get("candidate_id", "")))
        chosen: list[dict[str, Any]] = []
        used_tasks: set[str] = set()
        for candidate in ordered:
            tasks = _changed_tasks(candidate)
            if tasks and used_tasks.intersection(tasks):
                continue
            chosen.append(candidate)
            used_tasks.update(tasks)
            if len(chosen) == max_stack_size:
                break
        if not chosen:
            continue
        plans.append(
            {
                "stack_id": "stack-" + "-".join(str(c.get("candidate_id")) for c in chosen),
                "candidate_ids": [c.get("candidate_id") for c in chosen],
                "stack_size": len(chosen),
                "lineage_base_candidate_id": lineage,
                "changed_task_ids": sorted(used_tasks),
                "combined_expected_point_gain": round(sum(_candidate_gain(c) for c in chosen), 6),
                "risk_notes": "lineage-compatible independent/additive set; conflicts skipped by changed_task_ids overlap",
                "approval_mode": DEFAULT_APPROVAL_MODE,
                "worthiness_gate": "5.5 verifier or approved 5.5 submit-captain policy",
            }
        )
    return sorted(plans, key=lambda p: (-float(p["combined_expected_point_gain"]), p["stack_size"]))


def _best_pair(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pair_candidates = list(itertools.combinations(candidates, 2))
    if not pair_candidates:
        return []
    best = max(
        pair_candidates,
        key=lambda pair: (
            sum(_candidate_gain(c) for c in pair),
            -sum(_candidate_risk(c) for c in pair),
        ),
    )
    return sorted(best, key=lambda c: (-_candidate_gain(c), _candidate_risk(c), c.get("candidate_id", "")))


def plan_isolation_after_weak_three_stack(
    stack_candidates: list[dict[str, Any]], two_stack_result: str | None
) -> dict[str, Any]:
    """Encode the weak 3-stack isolation fallback policy."""
    if len(stack_candidates) != 3:
        raise ValueError("weak 3-stack isolation requires exactly 3 candidates")

    pair = _best_pair(stack_candidates)
    pair_ids = {c.get("candidate_id") for c in pair}
    third = [c for c in stack_candidates if c.get("candidate_id") not in pair_ids]

    if two_stack_result is None:
        return {
            "next_attempt_kind": "stack-2",
            "candidate_ids": [c.get("candidate_id") for c in pair],
            "selection_basis": "highest expected-value / lowest-risk pair from same lineage-compatible trio",
            "approval_mode": DEFAULT_APPROVAL_MODE,
        }
    if two_stack_result == "pass":
        return {
            "next_attempt_kind": "isolate-third",
            "candidate_ids": [c.get("candidate_id") for c in third],
            "selection_basis": "2-stack passed acceptably; isolate remaining candidate next",
            "approval_mode": DEFAULT_APPROVAL_MODE,
        }
    if two_stack_result == "fail":
        return {
            "action": "return-all-to-queue-or-remediation",
            "candidate_ids": [c.get("candidate_id") for c in stack_candidates],
            "selection_basis": "2-stack also failed; stop isolation burn and route all three back",
            "approval_mode": DEFAULT_APPROVAL_MODE,
        }
    raise ValueError("two_stack_result must be None, 'pass', or 'fail'")


def choose_55_lane_rationale(item: dict[str, Any]) -> dict[str, Any]:
    """Record whether an item deserves the 5.5 lane and why."""
    reasons: list[str] = []
    if str(item.get("difficulty", "")).lower() in {"hard", "highest", "high"}:
        reasons.append("hard")
    if str(item.get("uncertainty", "")).lower() in {"high", "highest"}:
        reasons.append("high-uncertainty")
    if _candidate_gain(item) >= 1.0 or str(item.get("value", "")).lower() in {"high", "highest"}:
        reasons.append("high-value")
    return {
        "route_to_55_lane": bool(reasons),
        "rationale": ", ".join(reasons) if reasons else "lower-cost lane: mechanical/low-uncertainty/low-value",
    }


def build_duplicate_guard_material(candidate: dict[str, Any]) -> dict[str, Any]:
    lineage = candidate.get("lineage_base_candidate_id") or _lineage(candidate) or ""
    stack = candidate.get("stack_plan") if isinstance(candidate.get("stack_plan"), dict) else {}
    material = {
        "task_id": str(candidate.get("task_id") or ""),
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "package_sha256": str(candidate.get("package_sha256") or "").upper(),
        "lineage_base_candidate_id": str(lineage),
        "changed_task_ids": to_str_list(candidate.get("changed_task_ids") or candidate.get("review_changed_task_ids") or []),
        "changed_entries": to_str_list(candidate.get("changed_entries") or []),
        "stack_candidate_ids": to_str_list(stack.get("stack_candidate_ids") or candidate.get("stack_candidate_ids") or []),
        "approval_mode": str(candidate.get("approval_mode") or DEFAULT_APPROVAL_MODE),
    }
    family_bits = [
        candidate.get("approach"),
        candidate.get("risk_summary"),
        stack.get("independence_notes"),
        stack.get("stack_id"),
        candidate.get("submit_execution", {}).get("attempt_kind") if isinstance(candidate.get("submit_execution"), dict) else None,
    ]
    material["family_notes"] = [str(bit) for bit in family_bits if bit]
    return material


def build_duplicate_guard_key(candidate: dict[str, Any]) -> str:
    material = build_duplicate_guard_material(candidate)
    return hashlib.sha256(canonical_json(material).encode("utf-8")).hexdigest()


def extract_known_outcome_summary(task_id: str | None) -> tuple[str, bool]:
    rec = task_outcome_record(task_id)
    if not rec:
        return "", False
    bits = [str(rec.get(key, "")).strip() for key in ("status", "best_known_status", "summary", "likely_interpretation")]
    summary = " | ".join(bit for bit in bits if bit)
    lowered = summary.lower()
    known_flat = any(
        token in lowered
        for token in (
            "known_public_flat",
            "public-flat",
            "public_flat",
            "zero_public_weight",
            "public-zero",
            "cost-irrelevant",
            "no-reward",
            "no public weight",
            "hidden-bad",
        )
    )
    return summary, known_flat


def manifest_matches(candidate: dict[str, Any], manifests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidate_id = str(candidate.get("candidate_id") or "")
    task_id = str(candidate.get("task_id") or "")
    package_sha = str(candidate.get("package_sha256") or "").upper()
    guard_key = build_duplicate_guard_key(candidate)
    matches: list[dict[str, Any]] = []
    for manifest in manifests:
        manifest_candidate_id = str(manifest.get("candidate_id") or manifest.get("candidate") or "")
        manifest_task_id = str(manifest.get("task_id") or "")
        manifest_sha = str(manifest.get("package_sha256") or manifest.get("submitted_package_sha256") or "").upper()
        manifest_guard = None
        if manifest_sha or manifest_candidate_id or manifest_task_id:
            manifest_guard = hashlib.sha256(
                canonical_json(
                    {
                        "candidate_id": manifest_candidate_id,
                        "task_id": manifest_task_id,
                        "package_sha256": manifest_sha,
                        "lineage_base_candidate_id": str(manifest.get("lineage_base_candidate_id") or ""),
                        "changed_task_ids": to_str_list(manifest.get("changed_task_ids") or []),
                        "changed_entries": to_str_list(manifest.get("changed_entries") or []),
                    }
                ).encode("utf-8")
            ).hexdigest()
        if (
            (candidate_id and candidate_id == manifest_candidate_id)
            or (task_id and task_id == manifest_task_id and package_sha and package_sha == manifest_sha)
            or (manifest_guard and manifest_guard == guard_key)
        ):
            matches.append(manifest)
    return matches


def compose_preflight_history(
    candidate: dict[str, Any],
    *,
    queue_doc: dict[str, Any] | None = None,
    registry_doc: dict[str, Any] | None = None,
    manifests: list[dict[str, Any]] | None = None,
    self_candidate_id: str = "",
) -> dict[str, Any]:
    """Compute preflight history for `candidate` against registry + submit_history.

    `self_candidate_id`, when non-empty, excludes any registry / submit_history
    entry whose candidate_id matches it from the duplicate-match set. This
    enables idempotent upsert flows: when `--write-registry --enqueue` runs in
    a single invocation, the registry write inserts the candidate before the
    enqueue path recomputes preflight; without self-exclusion the enqueue path
    sees the freshly-inserted entry and erroneously flags `blocked-duplicate`.
    """
    queue_doc = queue_doc or {}
    registry_doc = registry_doc or {}
    manifests = manifests or failed_manifest_records()

    task_id = str(candidate.get("task_id") or "")
    candidate_id = str(candidate.get("candidate_id") or "")
    guard_key = build_duplicate_guard_key(candidate)
    self_id = str(self_candidate_id or "")

    checked_against = current_history_sources()
    if (REPO_ROOT / "The_400_Tasks" / "Task_by_Task" / f"{task_id}.md").exists():
        checked_against.append("The_400_Tasks/Task_by_Task/<task>/task card")

    submit_history = queue_doc.get("submit_history") if isinstance(queue_doc.get("submit_history"), list) else []
    submit_history_matches: list[dict[str, Any]] = []
    for row in submit_history:
        if not isinstance(row, dict):
            continue
        row_candidate_id = str(row.get("candidate_id") or row.get("candidate") or "")
        if self_id and row_candidate_id == self_id:
            continue  # do not flag self as a prior submit-history duplicate
        row_task_id = str(row.get("task_id") or "")
        row_sha = str(row.get("package_sha256") or row.get("submitted_package_sha256") or "").upper()
        row_guard = None
        if row_candidate_id or row_task_id or row_sha:
            row_guard = hashlib.sha256(
                canonical_json(
                    {
                        "candidate_id": row_candidate_id,
                        "task_id": row_task_id,
                        "package_sha256": row_sha,
                        "lineage_base_candidate_id": str(row.get("lineage_base_candidate_id") or row.get("lineage_base", {}).get("candidate_id") if isinstance(row.get("lineage_base"), dict) else ""),
                        "changed_task_ids": to_str_list(row.get("changed_task_ids") or []),
                        "changed_entries": to_str_list(row.get("changed_entries") or []),
                    }
                ).encode("utf-8")
            ).hexdigest()
        if (
            (candidate_id and candidate_id == row_candidate_id)
            or (task_id and task_id == row_task_id and str(candidate.get("package_sha256") or "").upper() == row_sha)
            or (row_guard and row_guard == guard_key)
        ):
            submit_history_matches.append(row)

    registry_entries = registry_doc.get("entries") if isinstance(registry_doc.get("entries"), list) else []
    registry_matches: list[dict[str, Any]] = []
    for entry in registry_entries:
        if not isinstance(entry, dict):
            continue
        entry_candidate_id = str(entry.get("candidate_id") or "")
        if self_id and entry_candidate_id == self_id:
            continue  # do not flag self as a prior registry duplicate
        if (
            (candidate_id and entry_candidate_id == candidate_id)
            or (guard_key == build_duplicate_guard_key(entry))
            or (
                str(entry.get("package_sha256") or "").upper() == str(candidate.get("package_sha256") or "").upper()
                and str(entry.get("task_id") or "") == task_id
            )
        ):
            registry_matches.append(entry)

    ledger_hits = submit_ledger_hits(candidate)
    known_summary, known_flat = extract_known_outcome_summary(task_id)
    current_hits = current_state_hits(task_id)
    failed_matches = manifest_matches(candidate, manifests)

    matched_task_ids = sorted(
        dict.fromkeys(
            [str(task_id)]
            + [str(row.get("task_id") or "") for row in submit_history_matches]
            + [str(entry.get("task_id") or "") for entry in registry_matches]
            + [str(m.get("task_id") or "") for m in failed_matches]
            + [str(row.get("number") or "") for row in ledger_hits if row.get("number")]
        )
    )
    matched_prior_candidates = sorted(
        dict.fromkeys(
            [str(candidate_id)]
            + [str(row.get("candidate_id") or row.get("candidate") or "") for row in submit_history_matches]
            + [str(entry.get("candidate_id") or "") for entry in registry_matches]
            + [str(m.get("candidate_id") or "") for m in failed_matches]
        )
    )
    matched_prior_candidates = [item for item in matched_prior_candidates if item]

    new_evidence_summary = str((candidate.get("preflight_history") or {}).get("new_evidence_summary") or candidate.get("new_evidence_summary") or "").strip()
    reviewer_exception_requested = bool((candidate.get("preflight_history") or {}).get("reviewer_exception_required") or candidate.get("reviewer_exception_required"))

    status = "pass"
    if submit_history_matches or registry_matches or failed_matches:
        status = "blocked-duplicate"
    elif known_flat:
        if reviewer_exception_requested and new_evidence_summary:
            status = "reviewer-exception"
        elif new_evidence_summary:
            status = "needs-new-evidence"
        else:
            status = "blocked-known-flat"
    elif new_evidence_summary and reviewer_exception_requested:
        status = "reviewer-exception"

    if status not in LEGACY_ALLOWED_HISTORY_STATUSES:
        status = "pass"

    known_outcome_summary = known_summary
    if current_hits:
        known_outcome_summary = (known_outcome_summary + " | " if known_outcome_summary else "") + " ; ".join(current_hits[:3])
    if ledger_hits:
        known_outcome_summary = (known_outcome_summary + " | " if known_outcome_summary else "") + f"ledger_hits={len(ledger_hits)}"
    if failed_matches:
        failed_note = ", ".join(str(m.get("failure_type") or m.get("next_experiment") or m.get("likely_interpretation") or m.get("_path") or "failed manifest") for m in failed_matches[:3])
        known_outcome_summary = (known_outcome_summary + " | " if known_outcome_summary else "") + failed_note

    return {
        "status": status,
        "checked_against": checked_against,
        "matched_prior_candidates": matched_prior_candidates,
        "matched_task_ids": matched_task_ids,
        "duplicate_guard_key": guard_key,
        "known_outcome_summary": known_outcome_summary,
        "new_evidence_summary": new_evidence_summary,
        "reviewer_exception_required": reviewer_exception_requested or known_flat,
    }


def normalize_candidate_entry(
    candidate: dict[str, Any],
    *,
    registry_doc: dict[str, Any] | None = None,
    queue_doc: dict[str, Any] | None = None,
    manifests: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    registry_doc = registry_doc or read_candidate_registry_doc()
    template = normalize_template_from(registry_doc, "entry_template")
    normalized = deep_merge(template, candidate)

    default_mode = (
        (registry_doc.get("review_gate") or {}).get("default_approval_mode")
        if isinstance(registry_doc.get("review_gate"), dict)
        else DEFAULT_APPROVAL_MODE
    )
    approval_mode = normalized.get("approval_mode") or default_mode or DEFAULT_APPROVAL_MODE
    if approval_mode not in ALLOWED_APPROVAL_MODES:
        raise ValueError(f"unsupported approval_mode: {approval_mode!r}")
    normalized["approval_mode"] = approval_mode
    normalized["expected_point_gain"] = normalized.get("expected_point_gain") or normalized.get("predicted_delta")
    normalized["changed_task_ids"] = to_str_list(normalized.get("changed_task_ids") or normalized.get("review_changed_task_ids") or [])
    normalized["changed_entries"] = to_str_list(normalized.get("changed_entries") or [])
    normalized["risk_flags"] = to_str_list(normalized.get("risk_flags") or [])
    normalized["chain_keys"] = build_chain_keys(normalized)
    normalized["chain_key"] = normalized["chain_keys"]["review"]
    normalized["chain_key_stage"] = "review"
    normalized["review_chain_key"] = normalized["chain_keys"]["review"]
    normalized["worthiness_chain_key"] = normalized["chain_keys"]["worthiness"]
    normalized["submit_chain_key"] = normalized["chain_keys"]["submit"]
    normalized["worker_chain_key"] = normalized["chain_keys"]["worker"]
    normalized.setdefault("worthiness_gate", template.get("worthiness_gate") or "5.5 verifier or approved 5.5 submit-captain policy")

    preflight = compose_preflight_history(
        normalized,
        queue_doc=queue_doc,
        registry_doc=registry_doc,
        manifests=manifests,
        self_candidate_id=str(normalized.get("candidate_id") or ""),
    )
    normalized["preflight_history"] = preflight
    normalized["duplicate_guard_key"] = preflight["duplicate_guard_key"]
    normalized["submit_ready"] = preflight["status"] in {"pass", "reviewer-exception"}
    normalized["submit_queue_status"] = "ready" if normalized["submit_ready"] else f"blocked:{preflight['status']}"
    return normalized


def normalize_queue_entry(entry: dict[str, Any], policy: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a queue entry with explicit approval-mode and submit budget metadata."""
    normalized = copy.deepcopy(entry)
    policy = policy or {}
    default_mode = policy.get("approval_mode_default", DEFAULT_APPROVAL_MODE)
    mode = normalized.get("approval_mode") or default_mode
    if mode not in ALLOWED_APPROVAL_MODES:
        raise ValueError(f"unsupported approval_mode: {mode!r}")
    normalized["approval_mode"] = mode

    packet = normalized.setdefault("approval_packet", {})
    if isinstance(packet, dict):
        packet.setdefault("approval_mode", mode)
        packet.setdefault("approval_basis", f"{mode}-policy")

    normalized.setdefault("submit_execution", {})
    if isinstance(normalized["submit_execution"], dict):
        normalized["submit_execution"].setdefault("approval_basis", f"{mode}-policy")
        normalized["submit_execution"].setdefault("result_status", "not-submitted")

    normalized = add_submit_budget_metadata(normalized)
    if "preflight_history" in normalized and isinstance(normalized["preflight_history"], dict):
        normalized["duplicate_guard_key"] = normalized["preflight_history"].get("duplicate_guard_key", normalized.get("duplicate_guard_key"))
        normalized["preflight_status"] = normalized["preflight_history"].get("status", normalized.get("preflight_status", "pass"))
    return normalized


def add_submit_budget_metadata(
    entry: dict[str, Any], *, credits_available: int | None = None, submits_used_today: int | None = None
) -> dict[str, Any]:
    """Attach rate-limit and daily-cap awareness without submitting."""
    updated = copy.deepcopy(entry)
    submits_used = 0 if submits_used_today is None else max(0, int(submits_used_today))
    remaining = max(0, DAILY_SUBMIT_CAP - submits_used)
    updated["submit_budget"] = {
        "mode": "token_bucket",
        "rate_limit_minutes": RATE_LIMIT_MINUTES,
        "earn_rate": "1 submit credit per 10 minutes",
        "burst_allowed": True,
        "credits_available": credits_available,
        "daily_cap": DAILY_SUBMIT_CAP,
        "submits_used_today": submits_used,
        "submits_remaining_today": remaining,
        "daily_cap_awareness": "never plan beyond 100 Kaggle submits/day; prefer stacks/highest-value singles if budget tight",
        "submit_authority": SUBMIT_AUTHORITY,
    }
    return updated


def read_candidate_registry_doc(path: Path | None = None) -> dict[str, Any]:
    doc = read_json(path or current_paths()["candidate_registry"])
    return doc if isinstance(doc, dict) else {}


def read_submit_queue_doc(path: Path | None = None) -> dict[str, Any]:
    doc = read_json(path or current_paths()["submit_queue"])
    return doc if isinstance(doc, dict) else {}


def bootstrap_submit_credit_state(queue_doc: dict[str, Any] | None = None) -> dict[str, Any]:
    queue_doc = queue_doc or read_submit_queue_doc()
    policy = queue_doc.get("policy") if isinstance(queue_doc.get("policy"), dict) else {}
    state = read_json(current_paths()["submit_credit_state"])
    if not isinstance(state, dict):
        state = {
            "schema": "neurogolf.submit_credit_state.v1",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "policy": {},
            "credits": {},
            "notes": [],
        }
    state.setdefault("schema", "neurogolf.submit_credit_state.v1")
    state["updated_at"] = utc_now()
    state["policy"] = {
        "approval_mode_default": policy.get("approval_mode_default", DEFAULT_APPROVAL_MODE),
        "allowed_approval_modes": list(policy.get("allowed_approval_modes", sorted(ALLOWED_APPROVAL_MODES))),
        "submit_rate_limit_minutes": policy.get("submit_rate_limit_minutes", RATE_LIMIT_MINUTES),
        "daily_submit_cap": policy.get("daily_submit_cap", DAILY_SUBMIT_CAP),
        "submit_authority": policy.get("submit_authority", SUBMIT_AUTHORITY),
    }
    credits = state.get("credits") if isinstance(state.get("credits"), dict) else {}
    credits.setdefault("available", 0)
    credits.setdefault("reserved", 0)
    credits.setdefault("earned_total", 0)
    credits.setdefault("spent_total", 0)
    credits.setdefault("daily_used", 0)
    credits.setdefault("last_refill_at", None)
    credits.setdefault("last_submit_at", None)
    state["credits"] = credits
    state.setdefault(
        "notes",
        [
            "Bootstrap submit-credit representation for auto-rate-limited policy.",
            "Submit captain updates this file when credits are consumed or replenished.",
        ],
    )
    state["queue_depth"] = len(queue_doc.get("queue", []) if isinstance(queue_doc.get("queue"), list) else [])
    state["eligible_queue_depth"] = sum(1 for entry in (queue_doc.get("queue", []) if isinstance(queue_doc.get("queue"), list) else []) if isinstance(entry, dict) and _eligible(entry))
    return state


def normalize_queue_doc(doc: dict[str, Any]) -> dict[str, Any]:
    updated = copy.deepcopy(doc)
    policy = updated.setdefault("policy", {})
    policy["approval_mode_default"] = DEFAULT_APPROVAL_MODE
    policy["submit_captain_required"] = True
    policy["submit_rate_limit_minutes"] = RATE_LIMIT_MINUTES
    policy["daily_submit_cap"] = DAILY_SUBMIT_CAP
    policy.setdefault("max_stack_size", 3)
    queue_entries = updated.get("queue", [])
    updated["queue"] = [normalize_queue_entry(entry, policy) for entry in queue_entries if isinstance(entry, dict)]
    updated["planned_stacks"] = plan_stacks(updated["queue"])
    return updated


def load_candidate_input(path: Path) -> dict[str, Any]:
    doc = read_json(path)
    if not isinstance(doc, dict):
        raise ValueError(f"candidate input must be a JSON object: {path}")
    return doc


def upsert_registry_entry(registry_doc: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    updated = copy.deepcopy(registry_doc)
    entries = updated.setdefault("entries", [])
    if not isinstance(entries, list):
        entries = []
    normalized = normalize_candidate_entry(
        candidate,
        registry_doc=updated,
        queue_doc=read_submit_queue_doc(),
        manifests=failed_manifest_records(),
    )
    candidate_id = str(normalized.get("candidate_id") or "")
    package_sha = str(normalized.get("package_sha256") or "").upper()
    replaced = False
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        if candidate_id and str(entry.get("candidate_id") or "") == candidate_id:
            entries[idx] = normalized
            replaced = True
            break
        if package_sha and str(entry.get("package_sha256") or "").upper() == package_sha:
            entries[idx] = normalized
            replaced = True
            break
        if build_duplicate_guard_key(entry) == normalized["duplicate_guard_key"]:
            entries[idx] = normalized
            replaced = True
            break
    if not replaced:
        entries.append(normalized)
    updated["entries"] = entries
    updated["updated_at"] = utc_now()
    return updated


def make_queue_entry_from_candidate(candidate: dict[str, Any], queue_doc: dict[str, Any], registry_doc: dict[str, Any]) -> dict[str, Any]:
    template = normalize_template_from(queue_doc, "queue_entry_template")
    normalized_candidate = normalize_candidate_entry(candidate, registry_doc=registry_doc, queue_doc=queue_doc, manifests=failed_manifest_records())
    entry = deep_merge(template, normalized_candidate)
    entry = normalize_queue_entry(entry, queue_doc.get("policy") if isinstance(queue_doc.get("policy"), dict) else {})
    entry["preflight_history"] = normalized_candidate["preflight_history"]
    entry["duplicate_guard_key"] = normalized_candidate["duplicate_guard_key"]
    entry["preflight_status"] = normalized_candidate["preflight_history"]["status"]
    entry["expected_point_gain"] = normalized_candidate.get("expected_point_gain")
    entry["changed_task_ids"] = normalized_candidate.get("changed_task_ids", [])
    entry["risk_flags"] = normalized_candidate.get("risk_flags", [])
    entry["approval_mode"] = normalized_candidate.get("approval_mode", DEFAULT_APPROVAL_MODE)
    entry["submit_ready"] = normalized_candidate.get("submit_ready", False)
    entry["chain_keys"] = normalized_candidate.get("chain_keys", {})
    entry["chain_key"] = normalized_candidate.get("submit_chain_key") or normalized_candidate.get("chain_keys", {}).get("submit", "")
    entry["chain_key_stage"] = "submit"
    entry["review_chain_key"] = normalized_candidate.get("review_chain_key") or normalized_candidate.get("chain_keys", {}).get("review", "")
    entry["worthiness_chain_key"] = normalized_candidate.get("worthiness_chain_key") or normalized_candidate.get("chain_keys", {}).get("worthiness", "")
    entry["submit_chain_key"] = normalized_candidate.get("submit_chain_key") or normalized_candidate.get("chain_keys", {}).get("submit", "")
    entry["worker_chain_key"] = normalized_candidate.get("worker_chain_key") or normalized_candidate.get("chain_keys", {}).get("worker", "")
    return entry


def upsert_queue_entry(queue_doc: dict[str, Any], candidate: dict[str, Any], registry_doc: dict[str, Any]) -> dict[str, Any]:
    updated = copy.deepcopy(queue_doc)
    queue = updated.setdefault("queue", [])
    if not isinstance(queue, list):
        queue = []
    queue_entry = make_queue_entry_from_candidate(candidate, updated, registry_doc)
    candidate_id = str(queue_entry.get("candidate_id") or candidate.get("candidate_id") or "")
    queue_id = str(queue_entry.get("queue_id") or "")
    duplicate_guard_key = str(queue_entry.get("duplicate_guard_key") or "")
    if not candidate_id:
        raise ValueError("candidate_id is required to enqueue a candidate")
    if not queue_entry.get("submit_ready"):
        raise ValueError(f"candidate {candidate_id} is not queue-eligible: {queue_entry.get('preflight_status')}")
    replaced = False
    for idx, entry in enumerate(queue):
        if not isinstance(entry, dict):
            continue
        if candidate_id and str(entry.get("candidate_id") or "") == candidate_id:
            queue[idx] = queue_entry
            replaced = True
            break
        if queue_id and str(entry.get("queue_id") or "") == queue_id:
            queue[idx] = queue_entry
            replaced = True
            break
        if duplicate_guard_key and str(entry.get("duplicate_guard_key") or "") == duplicate_guard_key:
            queue[idx] = queue_entry
            replaced = True
            break
    if not replaced:
        queue.append(queue_entry)
    updated["queue"] = queue
    updated["planned_stacks"] = plan_stacks(queue)
    updated["updated_at"] = utc_now()
    return updated


def migrate_registry_doc(registry_doc: dict[str, Any], queue_doc: dict[str, Any] | None = None) -> dict[str, Any]:
    updated = copy.deepcopy(registry_doc)
    queue_doc = queue_doc or read_submit_queue_doc()
    entries = updated.get("entries", [])
    if not isinstance(entries, list):
        entries = []
    migrated_entries = []
    for raw in entries:
        if isinstance(raw, dict):
            migrated_entries.append(normalize_candidate_entry(raw, registry_doc=updated, queue_doc=queue_doc, manifests=failed_manifest_records()))
    updated["entries"] = migrated_entries
    updated["updated_at"] = utc_now()
    review_gate = updated.setdefault("review_gate", {})
    if isinstance(review_gate, dict):
        review_gate["default_approval_mode"] = DEFAULT_APPROVAL_MODE
        review_gate.setdefault("allowed_approval_modes", sorted(ALLOWED_APPROVAL_MODES))
    preflight_gate = updated.setdefault("preflight_gate", {})
    if isinstance(preflight_gate, dict):
        preflight_gate.setdefault("task_history_preflight_required", True)
        preflight_gate.setdefault("sources", current_history_sources())
    return updated


def migrate_queue_doc(queue_doc: dict[str, Any], registry_doc: dict[str, Any]) -> dict[str, Any]:
    return normalize_queue_doc(queue_doc)


def print_json(obj: Any) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate/normalize NeuroGolf submit queue and candidate-registry policy metadata.")
    parser.add_argument("queue_json", nargs="?", type=Path, help="Path to .hermes/board/submit_queue.json")
    parser.add_argument("--write", action="store_true", help="Write normalized queue JSON back to disk")
    parser.add_argument("--candidate-json", type=Path, help="Candidate-registry or approval-packet JSON to normalize")
    parser.add_argument("--registry-json", type=Path, default=current_paths()["candidate_registry"], help="Candidate registry JSON path")
    parser.add_argument("--credit-state-json", type=Path, default=current_paths()["submit_credit_state"], help="Submit-credit state JSON path")
    parser.add_argument("--board-state-json", type=Path, default=STATE_JSON, help="Board state JSON path for live target-claim checks")
    parser.add_argument("--stage", choices=["auto", "worker", "review", "worthiness", "submit"], default="auto", help="Live-card stage to guard before creating/upserting")
    parser.add_argument("--write-registry", action="store_true", help="Upsert the candidate into candidate_registry.json")
    parser.add_argument("--enqueue", action="store_true", help="Upsert the candidate into submit_queue.json when preflight-eligible")
    parser.add_argument("--migrate-registry", action="store_true", help="Rewrite all registry entries to the current schema")
    parser.add_argument("--migrate-credit-state", action="store_true", help="Bootstrap/write the submit-credit state file")
    args = parser.parse_args(argv)

    registry_doc = read_candidate_registry_doc(args.registry_json)
    queue_doc = read_submit_queue_doc(args.queue_json or current_paths()["submit_queue"])

    if args.migrate_registry:
        migrated_registry = migrate_registry_doc(registry_doc, queue_doc)
        if args.write_registry or args.migrate_registry:
            write_json(args.registry_json, migrated_registry)
        print_json({"status": "migrated-registry", "entries": len(migrated_registry.get("entries", []) if isinstance(migrated_registry.get("entries"), list) else [])})
        return 0

    if args.migrate_credit_state:
        credit_state = bootstrap_submit_credit_state(queue_doc)
        write_json(args.credit_state_json, credit_state)
        print_json({"status": "bootstrapped-credit-state", "credits": credit_state.get("credits", {}), "queue_depth": credit_state.get("queue_depth", 0)})
        return 0

    if args.candidate_json:
        candidate = load_candidate_input(args.candidate_json)
        normalized = normalize_candidate_entry(candidate, registry_doc=registry_doc, queue_doc=queue_doc, manifests=failed_manifest_records())
        summary = {
            "candidate_id": normalized.get("candidate_id"),
            "task_id": normalized.get("task_id"),
            "approval_mode": normalized.get("approval_mode"),
            "expected_point_gain": normalized.get("expected_point_gain"),
            "preflight_status": normalized.get("preflight_history", {}).get("status"),
            "duplicate_guard_key": normalized.get("duplicate_guard_key"),
            "submit_ready": normalized.get("submit_ready"),
        }
        stage = args.stage
        if stage == "auto":
            stage = "submit" if args.enqueue else "review"
        guard_stages = [stage]
        if args.write_registry and stage == "submit":
            guard_stages.insert(0, "review")
        state_doc = read_board_state_doc(args.board_state_json)
        guards = []
        for guard_stage in dict.fromkeys(guard_stages):
            guards.append(
                build_live_card_guard(
                    guard_stage,
                    normalized,
                    registry_doc=registry_doc,
                    queue_doc=queue_doc,
                    state_doc=state_doc,
                    self_candidate_id=str(normalized.get("candidate_id") or ""),
                    self_queue_id=str(normalized.get("queue_id") or ""),
                )
            )
        blocking = [guard for guard in guards if not guard.get("ok")]
        if blocking:
            print_json({
                "status": "duplicate-live-card" if any(item.get("status") == "duplicate-live-chain-key" for item in blocking) else "duplicate-live-target-claim",
                "normalized_candidate": summary,
                "guards": guards,
            })
            return 0
        if args.write_registry:
            registry_doc = upsert_registry_entry(registry_doc, candidate)
            write_json(args.registry_json, registry_doc)
        if args.enqueue:
            queue_doc = upsert_queue_entry(queue_doc, candidate, registry_doc)
            if args.queue_json:
                write_json(args.queue_json, queue_doc)
            else:
                write_json(current_paths()["submit_queue"], queue_doc)
        if args.write_registry or args.enqueue:
            credit_state = bootstrap_submit_credit_state(queue_doc)
            write_json(args.credit_state_json, credit_state)
        print_json({"normalized_candidate": summary, "preflight_history": normalized.get("preflight_history", {})})
        return 0

    if args.queue_json is not None:
        normalized = normalize_queue_doc(queue_doc)
        if args.write:
            write_json(args.queue_json, normalized)
            write_json(args.credit_state_json, bootstrap_submit_credit_state(normalized))
        print_json(
            {
                "queue_entries": len(normalized.get("queue", [])),
                "planned_stacks": len(normalized.get("planned_stacks", [])),
                "approval_mode_default": normalized.get("policy", {}).get("approval_mode_default"),
                "submit_credit_state": str(args.credit_state_json),
            }
        )
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
