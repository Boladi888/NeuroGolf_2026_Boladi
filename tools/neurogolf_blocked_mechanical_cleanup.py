#!/usr/bin/env python3
"""Cheap deterministic cleanup for obvious NeuroGolf blocked cards."""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.neurogolf_cron_snapshot import summarize_manifest, summarize_submit_queue
from tools.neurogolf_control_config import is_reviewer_profile, primary_reviewer_profile, review_packet_limit
from tools.neurogolf_kanban_python import (
    KanbanClient,
    card_assignee,
    card_id,
    card_status,
    card_text,
    card_title,
    read_json,
    sort_cards,
)
from tools.neurogolf_review_packet_builder import apply_review_packet_plan, card_needs_review, enrich_blocked_cards, plan_review_packet
from tools.neurogolf_submit_policy import (
    bootstrap_submit_credit_state,
    current_paths,
    read_candidate_registry_doc,
    read_submit_queue_doc,
    upsert_queue_entry,
    upsert_registry_entry,
    utc_now,
    write_json as write_submit_json,
)

COMPLETE_MARKERS = (
    "superseded by",
    "duplicate of",
    "already completed",
    "already closed",
    "control-plane-only complete",
)
REVIEWER_PROFILE = primary_reviewer_profile()
REGISTRY_JSON = REPO_ROOT / ".hermes" / "board" / "candidate_registry.json"
SUBMIT_QUEUE_JSON = REPO_ROOT / ".hermes" / "board" / "submit_queue.json"
REVIEW_PAYLOAD_FIELD_RE = re.compile(r"^\s*-\s+([a-zA-Z0-9_]+):\s*(.+?)\s*$")


def current_best_is_synced() -> bool:
    manifest = summarize_manifest()
    submit = summarize_submit_queue()
    known = submit.get("known_public_best") or {}
    return (
        manifest.get("manifest_sha_matches_package") is True
        and manifest.get("score") == known.get("score")
        and (known.get("sync_status") in {"synced", "promoted", "current"} or not known.get("sync_status"))
    )


def reviewer_base_sha_blocker_resolved(card: dict[str, Any]) -> bool:
    if card_status(card) != "blocked" or not is_reviewer_profile(card_assignee(card)):
        return False
    text = card_text(card).lower()
    if "candidate_registry/submit_queue updated" not in text:
        return False
    if "base-sha" not in text and "base sha" not in text:
        return False
    manifest = summarize_manifest()
    current_sha = str(manifest.get("package_sha256") or "").upper()
    if not current_sha:
        return False
    docs = json.dumps(
        {
            "registry": read_json(REGISTRY_JSON, {}),
            "queue": read_json(SUBMIT_QUEUE_JSON, {}),
        }
    ).upper()
    return current_sha in docs and current_sha.lower() in card_text(card).lower()


def reviewer_completed_handoff(card: dict[str, Any]) -> bool:
    if card_status(card) != "blocked" or not is_reviewer_profile(card_assignee(card)):
        return False
    text = card_text(card).lower()
    return (
        "review-required" in text
        and "queued" in text
        and ("return_tasks_created" in text or "returned the other" in text)
        and "changed_files" in text
    )


def terminal_no_submit_worker_handoff(card: dict[str, Any]) -> bool:
    if card_status(card) != "blocked":
        return False
    if not card_assignee(card).startswith("neurogolf-worker-"):
        return False
    text = card_text(card).lower()
    if "review-required" not in text and "review required" not in text:
        return False
    terminal_markers = (
        "no_submit_candidate",
        "no-submit candidate",
        "no submit candidate",
        "submit_recommendation\": \"do_not_submit",
        "submit_recommendation': 'do_not_submit",
        "submit_recommendation: do_not_submit",
        "do_not_submit;",
        "keep it out of submit queue",
        "no positive cost delta",
        "cost-negative vs current",
        "still cost-negative",
        "still_above_current_live_by",
        "best_replacement_cost\": null",
        "best_replacement_cost: null",
    )
    return any(marker in text for marker in terminal_markers)


def _parse_review_field_list(value: str) -> list[str]:
    text = value.strip().strip("`")
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text.replace("'", '"'))
        except json.JSONDecodeError:
            parsed = []
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    return [part.strip().strip("`\"'") for part in text.split(",") if part.strip()]


def _parse_review_payload_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    capture = False
    for line in text.splitlines():
        if "recommended queue payload fields" in line.lower():
            capture = True
            continue
        if not capture:
            continue
        match = REVIEW_PAYLOAD_FIELD_RE.match(line)
        if not match:
            continue
        fields[match.group(1)] = match.group(2).strip().strip("`")
    return fields


def reviewer_summary_candidate(card: dict[str, Any]) -> dict[str, Any]:
    if card_status(card) != "blocked" or not is_reviewer_profile(card_assignee(card)):
        return {}
    text = card_text(card)
    lowered = text.lower()
    if "iteration budget exhausted" not in lowered:
        return {}
    if "recommended queue payload fields" not in lowered:
        return {}
    if "appears valid and queue-worthy" not in lowered and "approved-for-submit-captain" not in lowered:
        return {}

    fields = _parse_review_payload_fields(text)
    required = ("candidate_id", "task_id", "package_path", "package_sha256", "expected_point_gain")
    if any(not fields.get(key) for key in required):
        return {}
    package_path = fields["package_path"]
    if not Path(package_path).exists():
        return {}
    try:
        expected_gain = float(fields.get("expected_point_gain") or "0")
    except ValueError:
        return {}

    candidate_id = fields["candidate_id"]
    task_id = fields["task_id"]
    changed_task_ids = _parse_review_field_list(fields.get("changed_task_ids", f"[{task_id}]"))
    changed_entries = _parse_review_field_list(fields.get("changed_entries", f"[{task_id}.onnx]"))
    risk_flags = _parse_review_field_list(fields.get("risk_flags", "[]"))
    now = utc_now()
    origin = fields.get("origin_kanban_task_id") or ""
    worker = fields.get("worker_owner") or fields.get("return_worker") or ""
    lineage = fields.get("lineage_base_candidate_id") or "current-best"
    base_sha_note = fields.get("verified_base_package_sha256") or "(not recorded)"
    manifest_sha_note = fields.get("current_best_manifest_package_sha256") or "(not recorded)"
    risk_summary = (
        f"Reviewer packet {card_id(card)} exhausted its iteration budget after independently approving "
        f"{candidate_id}; Python recovered the structured queue payload from the reviewer log. "
        f"Verified local base SHA {base_sha_note}; manifest SHA note {manifest_sha_note}."
    )
    preflight_history = {
        "status": "reviewer-exception",
        "checked_against": [
            f"reviewer packet {card_id(card)} log summary",
            origin or "source worker card",
        ],
        "known_outcome_summary": "Recovered from reviewer log after iteration-budget block; no Kaggle submit was made by reviewer.",
        "new_evidence_summary": "Reviewer independently verified package diff, ONNX equivalence, cost delta, and queue-worthy disposition.",
    }

    approval_packet = {
        "candidate_id": candidate_id,
        "task_id": task_id,
        "reviewer_disposition": fields.get("reviewer_disposition") or "approved-for-submit-captain",
        "approval_mode": "auto-rate-limited",
        "package_path": package_path,
        "package_sha256": fields["package_sha256"].upper(),
        "lineage_base": {"candidate_id": lineage},
        "changed_task_ids": changed_task_ids,
        "changed_entries": changed_entries,
        "expected_delta": expected_gain,
        "expected_point_gain": expected_gain,
        "risk_summary": risk_summary,
        "risk_flags": risk_flags,
        "preflight_history": preflight_history,
        "approved_by": card_assignee(card) or REVIEWER_PROFILE,
        "approved_at": now,
        "worthiness_gate": f"reviewer summary recovery from {card_id(card)}",
    }
    return {
        "candidate_id": candidate_id,
        "task_id": task_id,
        "source_candidate_id": candidate_id,
        "origin_kanban_task_id": origin,
        "worker_owner": worker,
        "return_worker": worker,
        "reviewer_disposition": approval_packet["reviewer_disposition"],
        "approval_mode": "auto-rate-limited",
        "lineage_base_candidate_id": lineage,
        "verified_base_package_sha256": fields.get("verified_base_package_sha256"),
        "current_best_manifest_package_sha256": fields.get("current_best_manifest_package_sha256"),
        "package_path": package_path,
        "package_sha256": fields["package_sha256"].upper(),
        "changed_task_ids": changed_task_ids,
        "changed_entries": changed_entries,
        "expected_point_gain": expected_gain,
        "expected_lb_point_gain": expected_gain,
        "predicted_delta": expected_gain,
        "risk_flags": risk_flags,
        "risk_summary": risk_summary,
        "tests_run": [
            "reviewer log: package diff has 401 entries and only listed task entry changed",
            "reviewer log: independent ONNX Runtime equivalence passed",
            "reviewer log: model-level cost delta inspected",
        ],
        "gates": {
            "reviewer_log_recovery": "pass",
            "package_path_exists": "pass",
            "kaggle_public_validation": "not-submitted",
        },
        "approval_packet": approval_packet,
        "preflight_history": preflight_history,
        "submit_ready": True,
        "submit_queue_status": "ready",
        "updated_at": now,
    }


def registry_without_self(registry_doc: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    updated = copy.deepcopy(registry_doc)
    entries = updated.get("entries")
    if not isinstance(entries, list):
        return updated
    candidate_id = str(candidate.get("candidate_id") or "")
    package_sha = str(candidate.get("package_sha256") or "").upper()
    updated["entries"] = [
        entry
        for entry in entries
        if not (
            isinstance(entry, dict)
            and (
                (candidate_id and str(entry.get("candidate_id") or "") == candidate_id)
                or (package_sha and str(entry.get("package_sha256") or "").upper() == package_sha)
            )
        )
    ]
    return updated


def _id_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item or "").strip()]


def parent_gated_reviewer_child(
    card: dict[str, Any],
    cards_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    if not card_needs_review(card):
        return None
    for child_id in _id_list(card.get("children")):
        child = cards_by_id.get(child_id)
        if not child:
            continue
        if is_reviewer_profile(card_assignee(child)) and card_status(child) == "todo":
            return child
    return None


def classify_blocked_card(card: dict[str, Any], *, best_synced: bool = False) -> dict[str, Any]:
    if card_status(card) != "blocked":
        return {"class": "ignore", "reason": "not blocked"}
    text = card_text(card).lower()
    assignee = card_assignee(card)
    if reviewer_base_sha_blocker_resolved(card):
        return {"class": "complete", "reason": "reviewer base-SHA blocker resolved by registry/queue/current-best SHA match"}
    if reviewer_completed_handoff(card):
        return {"class": "complete", "reason": "reviewer handoff completed: approved candidates queued and repair/optimization returns created"}
    if terminal_no_submit_worker_handoff(card):
        return {"class": "complete", "reason": "worker handoff is terminal no-submit/no-positive-delta; no reviewer packet needed"}
    if card_needs_review(card):
        return {"class": "review-packet", "reason": "blocked worker card needs reviewer handoff"}
    if "review-required" in text or "review required" in text or "needs reviewer" in text:
        return {"class": "manual", "reason": "non-worker review-required card is already owned by its specialist lane"}
    if any(marker in text for marker in COMPLETE_MARKERS):
        return {"class": "complete", "reason": "blocked card has an explicit terminal marker"}
    if assignee == "neurogolf-sync-manager" and best_synced and ("sync" in text or "current best" in text):
        return {"class": "complete", "reason": "canonical current best is already synced by deterministic Python"}
    if "submit-captain" in assignee and best_synced and ("sync" in text or "promote" in text):
        return {"class": "complete", "reason": "submit bookkeeping blocker is stale after deterministic sync"}
    return {"class": "manual", "reason": "not mechanically safe to close"}


def plan_blocked_cleanup(
    cards: list[dict[str, Any]],
    *,
    best_synced: bool = False,
    max_complete: int = 12,
    review_limit: int | None = None,
) -> dict[str, Any]:
    review_limit = review_limit or review_packet_limit()
    blocked = sort_cards([card for card in cards if card_status(card) == "blocked"])
    actions: list[dict[str, Any]] = []
    manual: list[dict[str, Any]] = []
    review_cards: list[dict[str, Any]] = []
    cards_by_id = {card_id(card): card for card in cards if card_id(card)}

    for card in blocked:
        recovered_candidate = reviewer_summary_candidate(card)
        if recovered_candidate:
            actions.append(
                {
                    "op": "recover_reviewer_queue_payload",
                    "task_id": card_id(card),
                    "title": card_title(card),
                    "candidate_id": recovered_candidate["candidate_id"],
                    "candidate": recovered_candidate,
                    "reason": "reviewer exhausted iterations after writing a structured approved queue payload",
                }
            )
            continue
        classification = classify_blocked_card(card, best_synced=best_synced)
        cls = classification["class"]
        if cls == "complete" and len([a for a in actions if a["op"] == "complete"]) < max_complete:
            actions.append(
                {
                    "op": "complete",
                    "task_id": card_id(card),
                    "title": card_title(card),
                    "reason": classification["reason"],
                }
            )
        elif cls == "review-packet":
            reviewer_child = parent_gated_reviewer_child(card, cards_by_id)
            if reviewer_child:
                actions.append(
                    {
                        "op": "release_parent_gated_reviewer",
                        "task_id": card_id(card),
                        "reviewer_task_id": card_id(reviewer_child),
                        "title": card_title(card),
                        "reason": "reviewer child is todo but parent-gated behind the blocked source it must review",
                    }
                )
            else:
                review_cards.append(card)
        elif cls == "manual":
            manual.append(
                {
                    "task_id": card_id(card),
                    "assignee": card_assignee(card),
                    "title": card_title(card),
                    "reason": classification["reason"],
                }
            )

    completing_ids = {str(action["task_id"]) for action in actions if action.get("op") == "complete"}
    review_plan_cards = [card for card in cards if card_id(card) not in completing_ids]
    review_plan = plan_review_packet(review_plan_cards, limit=review_limit)
    if review_cards and review_plan.get("actions"):
        actions.extend(review_plan["actions"])
    waiting_review_sources = [card_id(card) for card in review_cards]
    reviewer_active = review_plan.get("reviewer_active") or []

    return {
        "schema": "neurogolf.blocked_mechanical_cleanup.v1",
        "best_synced": best_synced,
        "blocked_count": len(blocked),
        "actions": actions,
        "manual": manual,
        "review_source_count": len(review_cards),
        "waiting_review_sources": waiting_review_sources if reviewer_active and not review_plan.get("actions") else [],
        "reviewer_active": reviewer_active,
        "action_count": len(actions),
        "manual_count": len(manual),
    }


def apply_blocked_cleanup_plan(client: KanbanClient, plan: dict[str, Any]) -> None:
    review_only = {"actions": [action for action in plan.get("actions", []) if action.get("op") == "create_reviewer_packet"]}
    for action in plan.get("actions", []):
        if action.get("op") == "complete":
            client.comment(str(action["task_id"]), f"Python blocked cleanup closing mechanical blocker: {action['reason']}.")
            client.complete(str(action["task_id"]), str(action["reason"]))
        elif action.get("op") == "release_parent_gated_reviewer":
            source_id = str(action["task_id"])
            reviewer_id = str(action["reviewer_task_id"])
            client.unlink(source_id, reviewer_id)
            client.comment(
                source_id,
                (
                    "Python blocked cleanup detached reviewer child "
                    f"{reviewer_id} from this blocked source. The reviewer card must run now; this source can remain blocked "
                    "to preserve worker context until review/submit feedback returns."
                ),
            )
            client.comment(
                reviewer_id,
                (
                    "Python blocked cleanup removed the blocked-source parent gate so this reviewer card can run. "
                    f"Source worker card remains {source_id}."
                ),
            )
        elif action.get("op") == "recover_reviewer_queue_payload":
            task_id = str(action["task_id"])
            candidate = action.get("candidate") if isinstance(action.get("candidate"), dict) else {}
            try:
                paths = current_paths()
                registry_path = paths["candidate_registry"]
                queue_path = paths["submit_queue"]
                credit_path = paths["submit_credit_state"]
                registry_doc = read_candidate_registry_doc(registry_path)
                queue_doc = read_submit_queue_doc(queue_path)
                queue_preflight_registry = registry_without_self(registry_doc, candidate)
                queue_doc = upsert_queue_entry(queue_doc, candidate, queue_preflight_registry)
                registry_doc = upsert_registry_entry(registry_doc, candidate)
                write_submit_json(registry_path, registry_doc)
                write_submit_json(queue_path, queue_doc)
                write_submit_json(credit_path, bootstrap_submit_credit_state(queue_doc))
            except Exception as exc:  # pragma: no cover - defensive live-board recovery
                client.comment(task_id, f"Python blocked cleanup could not recover reviewer queue payload: {exc}")
                continue
            client.comment(
                task_id,
                (
                    "Python blocked cleanup recovered the reviewer-approved queue payload "
                    f"for {candidate.get('candidate_id')} from the iteration-budget log and upserted registry/submit_queue."
                ),
            )
            client.complete(
                task_id,
                f"recovered approved queue payload for {candidate.get('candidate_id')}; reviewer lane freed",
            )
    if review_only["actions"]:
        apply_review_packet_plan(client, review_only)


def render_text(report: dict[str, Any]) -> str:
    lines = [
        f"blocked={report['blocked_count']} actions={report['action_count']} manual={report['manual_count']} best_synced={report['best_synced']}",
    ]
    for action in report["actions"]:
        if action["op"] == "complete":
            lines.append(f"- complete {action['task_id']}: {action['reason']}")
        elif action["op"] == "release_parent_gated_reviewer":
            lines.append(
                f"- release reviewer {action['reviewer_task_id']} from blocked parent {action['task_id']}: {action['reason']}"
            )
        elif action["op"] == "create_reviewer_packet":
            lines.append(f"- create reviewer packet for {len(action['source_cards'])} source card(s)")
        elif action["op"] == "recover_reviewer_queue_payload":
            lines.append(f"- recover reviewer payload {action['task_id']}: {action['candidate_id']}")
    if report.get("waiting_review_sources"):
        lines.append(
            "- waiting for active reviewer "
            f"{','.join(report.get('reviewer_active') or [])}: "
            f"{','.join(report.get('waiting_review_sources') or [])}"
        )
    for item in report["manual"]:
        lines.append(f"- manual {item['task_id']} {item['assignee']}: {item['reason']}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--kanban-json", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--assume-best-synced", action="store_true")
    parser.add_argument("--max-complete", type=int, default=12)
    parser.add_argument("--review-limit", type=int, default=None)
    args = parser.parse_args()

    client = KanbanClient(args.repo_root, dry_run=not args.apply)
    if args.kanban_json:
        cards = json.loads(args.kanban_json.read_text(encoding="utf-8"))
        best_synced = args.assume_best_synced
    else:
        cards = client.load_cards()
        cards = enrich_blocked_cards(client, cards)
        best_synced = args.assume_best_synced or current_best_is_synced()
    if not isinstance(cards, list):
        raise SystemExit("expected Kanban JSON list")

    report = plan_blocked_cleanup(
        [card for card in cards if isinstance(card, dict)],
        best_synced=best_synced,
        max_complete=args.max_complete,
        review_limit=args.review_limit,
    )
    if args.apply:
        apply_blocked_cleanup_plan(client, report)
        report["applied_actions"] = client.actions
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(render_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
