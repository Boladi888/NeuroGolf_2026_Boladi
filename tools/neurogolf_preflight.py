#!/usr/bin/env python3
"""Read-only live preflight for NeuroGolf Kanban worker assignments.

The goal is to stop repeated work before a new card is created or assigned.
It combines live Kanban state, deterministic chain keys, target locks, and the
task outcome registry into a compact ALLOW/WARN/DENY verdict.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from tools.neurogolf_board_policy import (
        build_live_card_guard,
        is_active_card,
        normalize_task_id,
        read_json,
    )
    from tools.neurogolf_paths import board_dir, resolve_canonical_root
except ModuleNotFoundError:  # pragma: no cover
    from neurogolf_board_policy import (  # type: ignore
        build_live_card_guard,
        is_active_card,
        normalize_task_id,
        read_json,
    )
    from neurogolf_paths import board_dir, resolve_canonical_root  # type: ignore


DEFAULT_REPO_ROOT = resolve_canonical_root(Path(__file__).resolve())
ACTIVE_STATUSES = {"running", "ready", "todo", "blocked"}
ASSIGNEE_CAPACITY_DENY_STATUSES = ACTIVE_STATUSES
ASSIGNEE_PARKED_STATUSES: set[str] = set()
KNOWN_FLAT_MARKERS = (
    "known_public_flat",
    "public_flat",
    "public-flat",
    "public_non_improving",
    "public non-improving",
    "zero public weight",
)
LOCAL_COST_ONLY_MARKERS = (
    "local-cost-only",
    "local cost only",
    "cost-only",
    "cost shrink",
    "cost rewrite",
    "current-live shrink",
    "current live shrink",
)
NOVELTY_MARKERS = (
    "hidden",
    "private",
    "behavior",
    "behaviour",
    "semantic",
    "rule",
    "new evidence",
    "materially different",
    "novel",
)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _norm_text(value: Any) -> str:
    text = _as_text(value).lower()
    text = text.replace("_", "-")
    return re.sub(r"\s+", " ", text)


def _compact_task_doc_path(repo_root: Path, task_id: str) -> Path:
    return repo_root / "The_400_Tasks" / "Task_by_Task" / f"{task_id}.md"


def _card_status(card: dict[str, Any]) -> str:
    return _norm_text(card.get("status"))


def _card_assignee(card: dict[str, Any]) -> str:
    return _as_text(card.get("assignee"))


def _card_text(card: dict[str, Any]) -> str:
    parts = [
        card.get("id"),
        card.get("title"),
        card.get("body"),
        card.get("result"),
        card.get("assignee"),
    ]
    return "\n".join(_as_text(part) for part in parts if part is not None)


def _summarize_card(card: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": _as_text(card.get("id")),
        "status": _card_status(card),
        "assignee": _card_assignee(card),
        "title": _as_text(card.get("title")),
    }


def _card_mentions_task(card: dict[str, Any], task_id: str) -> bool:
    text = _card_text(card)
    return task_id in text or task_id.replace("task", "task0") in text


def _active_cards_for_task(
    cards: list[dict[str, Any]],
    task_id: str,
    *,
    self_kanban_task_id: str = "",
) -> list[dict[str, Any]]:
    out = []
    for card in cards:
        if self_kanban_task_id and _as_text(card.get("id")) == self_kanban_task_id:
            continue
        if _card_status(card) in ACTIVE_STATUSES and _card_mentions_task(card, task_id):
            out.append(card)
    return out


def _cards_for_assignee_with_statuses(
    cards: list[dict[str, Any]],
    assignee: str,
    statuses: set[str],
    *,
    self_kanban_task_id: str = "",
) -> list[dict[str, Any]]:
    if not assignee:
        return []
    out = []
    for card in cards:
        if self_kanban_task_id and _as_text(card.get("id")) == self_kanban_task_id:
            continue
        if _card_status(card) in statuses and _card_assignee(card) == assignee:
            out.append(card)
    return out


def _task_outcome_record(task_id: str, task_outcome_doc: dict[str, Any] | None) -> dict[str, Any]:
    doc = task_outcome_doc or {}
    tasks = doc.get("tasks") if isinstance(doc.get("tasks"), dict) else {}
    record = tasks.get(task_id)
    return record if isinstance(record, dict) else {}


def _method_has_local_cost_only_shape(method_family: str) -> bool:
    method = _norm_text(method_family)
    return any(marker in method for marker in LOCAL_COST_ONLY_MARKERS) or (
        "cost" in method and "shrink" in method
    )


def _method_has_novelty_shape(method_family: str, new_evidence: str) -> bool:
    text = _norm_text(method_family) + "\n" + _norm_text(new_evidence)
    return any(marker in text for marker in NOVELTY_MARKERS)


def _blocked_next_work_hit(record: dict[str, Any], method_family: str) -> str:
    blocked_items = record.get("blocked_next_work")
    if not isinstance(blocked_items, list):
        return ""
    method = _norm_text(method_family)
    if not method:
        return ""
    blocked_text = "\n".join(_norm_text(item) for item in blocked_items)
    if _method_has_local_cost_only_shape(method) and (
        "local-cost-only" in blocked_text
        or "local cost" in blocked_text
        or "cost reduction" in blocked_text
        or "cost shrink" in blocked_text
    ):
        return "local-cost-only work is blocked by task outcome registry"
    method_tokens = {tok for tok in re.split(r"[^a-z0-9]+", method) if len(tok) >= 5}
    for item in blocked_items:
        item_text = _norm_text(item)
        item_tokens = {tok for tok in re.split(r"[^a-z0-9]+", item_text) if len(tok) >= 5}
        if method_tokens and len(method_tokens & item_tokens) >= 2:
            return _as_text(item)
    return ""


def _outcome_preflight(
    task_id: str,
    method_family: str,
    new_evidence: str,
    task_outcome_doc: dict[str, Any] | None,
    *,
    reviewer_exception: bool = False,
) -> dict[str, Any]:
    record = _task_outcome_record(task_id, task_outcome_doc)
    if not record:
        return {"status": "clear", "severity": "allow", "record": None, "messages": []}

    record_text = _norm_text(
        "\n".join(
            _as_text(record.get(field))
            for field in ("status", "best_known_status", "summary")
            if record.get(field)
        )
    )
    blocked_hit = _blocked_next_work_hit(record, method_family)
    new_evidence_is_material = bool(_as_text(new_evidence)) or _method_has_novelty_shape(
        method_family, new_evidence
    )

    messages: list[str] = []
    severity = "allow"
    status = "clear"

    if blocked_hit:
        status = "blocked-by-task-outcome-registry"
        messages.append(blocked_hit)
        severity = "deny"

    if any(marker in record_text for marker in KNOWN_FLAT_MARKERS):
        if not new_evidence_is_material:
            status = "known-flat-needs-new-evidence"
            messages.append(
                "task has known public-flat/non-improving history; new behavioral or hidden-score evidence is required"
            )
            severity = "deny"
        elif severity != "deny":
            status = "known-flat-new-evidence-required"
            messages.append(
                "task has known public-flat/non-improving history; keep the new-evidence rationale on the card"
            )
            severity = "warn"

    if reviewer_exception and severity == "deny":
        messages.append("reviewer_exception=true downgraded deny to warn; card must cite the exception")
        severity = "warn"
        status = f"{status}-reviewer-exception"

    return {
        "status": status,
        "severity": severity,
        "record": {
            "task_id": task_id,
            "status": _as_text(record.get("status")),
            "best_known_status": _as_text(record.get("best_known_status")),
            "summary": _as_text(record.get("summary")),
            "allowed_next_work": record.get("allowed_next_work") or [],
            "blocked_next_work": record.get("blocked_next_work") or [],
        },
        "messages": messages,
    }


def _task_doc_findings(repo_root: Path, task_id: str, method_family: str) -> dict[str, Any]:
    path = _compact_task_doc_path(repo_root, task_id)
    if not path.exists():
        return {"path": str(path), "exists": False, "warnings": ["task doc missing"]}
    text = path.read_text(encoding="utf-8", errors="replace")
    lower = _norm_text(text)
    method = _norm_text(method_family)
    warnings: list[str] = []
    if "don't-do log" in lower and _method_has_local_cost_only_shape(method) and "public-flat" in lower:
        warnings.append("task doc contains don't-do/public-flat notes; cite why this attempt differs")
    return {"path": str(path), "exists": True, "warnings": warnings}


def build_worker_preflight(
    *,
    task_id: str,
    assignee: str = "",
    method_family: str = "",
    lineage_base_candidate_id: str = "current-best",
    new_evidence: str = "",
    cards: list[dict[str, Any]] | None = None,
    registry_doc: dict[str, Any] | None = None,
    queue_doc: dict[str, Any] | None = None,
    state_doc: dict[str, Any] | None = None,  # accepted for backward compat; unused
    task_outcome_doc: dict[str, Any] | None = None,
    repo_root: Path = DEFAULT_REPO_ROOT,
    allow_independent_verification: bool = False,
    reviewer_exception: bool = False,
    self_kanban_task_id: str = "",
) -> dict[str, Any]:
    del state_doc  # state.json lock sections were retired 2026-05-17
    task_key = normalize_task_id(task_id)
    cards = cards or []

    chain_guard = build_live_card_guard(
        "worker",
        {
            "task_id": task_key,
            "worker_owner": assignee,
            "lineage_base_candidate_id": lineage_base_candidate_id,
        },
        registry_doc=registry_doc,
        queue_doc=queue_doc,
        cards=cards,
        self_kanban_task_id=self_kanban_task_id,
    )
    same_task_cards = _active_cards_for_task(cards, task_key, self_kanban_task_id=self_kanban_task_id)
    assignee_cards = _cards_for_assignee_with_statuses(
        cards,
        assignee,
        ASSIGNEE_CAPACITY_DENY_STATUSES,
        self_kanban_task_id=self_kanban_task_id,
    )
    assignee_parked_cards = _cards_for_assignee_with_statuses(
        cards,
        assignee,
        ASSIGNEE_PARKED_STATUSES,
        self_kanban_task_id=self_kanban_task_id,
    )
    outcome = _outcome_preflight(
        task_key,
        method_family,
        new_evidence,
        task_outcome_doc,
        reviewer_exception=reviewer_exception,
    )
    task_doc = _task_doc_findings(repo_root, task_key, method_family)

    deny_reasons: list[str] = []
    warnings: list[str] = []

    if not task_key:
        deny_reasons.append("missing task_id")
    if not _as_text(method_family):
        warnings.append("method_family missing; duplicate prevention is weaker")
    if chain_guard["status"] != "ok":
        deny_reasons.extend(chain_guard["warnings"])
    if same_task_cards and not allow_independent_verification:
        ids = ", ".join(_as_text(card.get("id")) for card in same_task_cards)
        deny_reasons.append(f"live Kanban card(s) already mention {task_key}: {ids}")
    if assignee_cards and not allow_independent_verification:
        ids = ", ".join(_as_text(card.get("id")) for card in assignee_cards)
        deny_reasons.append(f"{assignee} already has active card(s): {ids}")
    if same_task_cards and allow_independent_verification:
        warnings.append("same task is live; independent-verification rationale must be explicit")
    if assignee_cards and allow_independent_verification:
        warnings.append("assignee has active work; intentional parallelism rationale must be explicit")
    if assignee_parked_cards and not assignee_cards:
        ids = ", ".join(_as_text(card.get("id")) for card in assignee_parked_cards)
        warnings.append(f"{assignee} already has parked todo queue card(s): {ids}")
    if outcome["severity"] == "deny":
        deny_reasons.extend(outcome["messages"])
    elif outcome["severity"] == "warn":
        warnings.extend(outcome["messages"])
    warnings.extend(task_doc["warnings"])

    verdict = "DENY" if deny_reasons else "WARN" if warnings else "ALLOW"
    sources = [
        "hermes kanban list --json",
        ".hermes/board/candidate_registry.json",
        ".hermes/board/submit_queue.json",
        ".hermes/board/task_outcome_registry.json",
        str(_compact_task_doc_path(repo_root, task_key)),
    ]
    footer = [
        f"preflight_verdict: {verdict}",
        f"preflight_task_id: {task_key}",
        f"preflight_assignee: {assignee or '(unassigned)'}",
        f"preflight_method_family: {_as_text(method_family) or '(missing)'}",
        f"preflight_lineage_base: {_as_text(lineage_base_candidate_id) or 'current-best'}",
        f"preflight_chain_key: {chain_guard['chain_key']}",
        f"preflight_outcome_status: {outcome['status']}",
        f"preflight_sources: {', '.join(sources)}",
    ]
    if new_evidence:
        footer.append(f"preflight_new_evidence: {new_evidence}")
    if reviewer_exception:
        footer.append("preflight_reviewer_exception: true")

    return {
        "schema": "neurogolf.worker_preflight.v1",
        "verdict": verdict,
        "ok_to_create_worker_card": verdict != "DENY",
        "task_id": task_key,
        "assignee": assignee,
        "method_family": method_family,
        "lineage_base_candidate_id": lineage_base_candidate_id,
        "new_evidence": new_evidence,
        "deny_reasons": deny_reasons,
        "warnings": warnings,
        "same_task_active_cards": [_summarize_card(card) for card in same_task_cards],
        "assignee_active_cards": [_summarize_card(card) for card in assignee_cards],
        "assignee_parked_cards": [_summarize_card(card) for card in assignee_parked_cards],
        "chain_guard": chain_guard,
        "task_outcome": outcome,
        "task_doc": task_doc,
        "required_card_footer": footer,
    }


def load_cards(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.kanban_json:
        data = json.loads(args.kanban_json.read_text(encoding="utf-8"))
    else:
        proc = subprocess.run(
            ["hermes", "kanban", "list", "--json"],
            cwd=str(args.repo_root),
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            raise SystemExit(proc.stderr or proc.stdout)
        data = json.loads(proc.stdout)
    if not isinstance(data, list):
        raise SystemExit("expected list from Kanban JSON")
    return [item for item in data if isinstance(item, dict)]


def load_doc(path: Path) -> dict[str, Any]:
    data = read_json(path)
    return data if isinstance(data, dict) else {}


def render_text(report: dict[str, Any]) -> str:
    lines = [
        f"verdict: {report['verdict']}",
        f"ok_to_create_worker_card: {report['ok_to_create_worker_card']}",
        f"task_id: {report['task_id']}",
        f"assignee: {report['assignee'] or '(unassigned)'}",
        f"method_family: {report['method_family'] or '(missing)'}",
        "",
        "deny_reasons:",
    ]
    lines.extend(f"  - {item}" for item in report["deny_reasons"] or ["(none)"])
    lines.append("")
    lines.append("warnings:")
    lines.extend(f"  - {item}" for item in report["warnings"] or ["(none)"])
    lines.append("")
    lines.append("required_card_footer:")
    lines.extend(f"  {item}" for item in report["required_card_footer"])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--kanban-json", type=Path, help="Saved output from hermes kanban list --json")
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--submit-queue", type=Path, default=None)
    parser.add_argument("--state-json", type=Path, default=None, help="DEPRECATED — locks now derive from Hermes cards")
    parser.add_argument(
        "--task-outcome-registry",
        type=Path,
        default=None,
    )
    parser.add_argument("--task", required=True, help="taskNNN or numeric task id")
    parser.add_argument("--assignee", default="")
    parser.add_argument("--method-family", default="", help="Short normalized method family or hypothesis")
    parser.add_argument("--lineage-base", default="current-best")
    parser.add_argument("--new-evidence", default="")
    parser.add_argument("--self-kanban-task-id", default="", help="Ignore this existing card id when rechecking a live card")
    parser.add_argument("--allow-independent-verification", action="store_true")
    parser.add_argument("--reviewer-exception", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on DENY")
    args = parser.parse_args(argv)
    args.repo_root = resolve_canonical_root(args.repo_root)
    canonical_board_dir = board_dir(args.repo_root)
    args.registry = args.registry or canonical_board_dir / "candidate_registry.json"
    args.submit_queue = args.submit_queue or canonical_board_dir / "submit_queue.json"
    args.task_outcome_registry = args.task_outcome_registry or canonical_board_dir / "task_outcome_registry.json"

    report = build_worker_preflight(
        task_id=args.task,
        assignee=args.assignee,
        method_family=args.method_family,
        lineage_base_candidate_id=args.lineage_base,
        new_evidence=args.new_evidence,
        cards=load_cards(args),
        registry_doc=load_doc(args.registry),
        queue_doc=load_doc(args.submit_queue),
        task_outcome_doc=load_doc(args.task_outcome_registry),
        repo_root=args.repo_root,
        allow_independent_verification=args.allow_independent_verification,
        reviewer_exception=args.reviewer_exception,
        self_kanban_task_id=args.self_kanban_task_id,
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(render_text(report))
    if args.strict and report["verdict"] == "DENY":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
