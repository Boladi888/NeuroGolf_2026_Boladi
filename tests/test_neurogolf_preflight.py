from pathlib import Path

from tools.neurogolf_preflight import build_worker_preflight


def outcome_registry():
    return {
        "tasks": {
            "task366": {
                "status": "known_public_flat_cost_rewrite_but_not_abandoned",
                "best_known_status": "tested_public_non_improving_cost_rewrite",
                "summary": (
                    "Prior task366 local-cost rewrites did not improve public score; "
                    "identity diagnostic indicated zero public weight."
                ),
                "allowed_next_work": [
                    "true hidden/private-score hypothesis beyond public cost reduction",
                    "behavioral/rule improvement that changes outputs on credible hidden-like cases",
                ],
                "blocked_next_work": [
                    "local-cost-only shrink with no behavior novelty",
                    "repeating compressed lookup/grouped lookup/identity-style public-flat probes without new evidence",
                ],
            }
        }
    }


def test_worker_preflight_denies_known_public_flat_local_cost_repeat(tmp_path: Path):
    report = build_worker_preflight(
        task_id="366",
        assignee="neurogolf-worker-3",
        method_family="local-cost-only current-live shrink",
        cards=[],
        registry_doc={},
        queue_doc={},
        state_doc={},
        task_outcome_doc=outcome_registry(),
        repo_root=tmp_path,
    )

    assert report["verdict"] == "DENY"
    assert report["ok_to_create_worker_card"] is False
    assert any("local-cost-only" in reason for reason in report["deny_reasons"])
    assert "preflight_verdict: DENY" in report["required_card_footer"]


def test_worker_preflight_warns_when_known_flat_has_reviewer_exception_and_new_evidence(tmp_path: Path):
    report = build_worker_preflight(
        task_id="task366",
        assignee="neurogolf-worker-3",
        method_family="behavior-novel hidden-like rule repair",
        new_evidence="Changes hidden-like outputs; not a local-cost-only rewrite.",
        cards=[],
        registry_doc={},
        queue_doc={},
        state_doc={},
        task_outcome_doc=outcome_registry(),
        repo_root=tmp_path,
        reviewer_exception=True,
    )

    assert report["verdict"] == "WARN"
    assert report["ok_to_create_worker_card"] is True
    assert report["deny_reasons"] == []
    assert any("known public-flat" in warning or "known public" in warning for warning in report["warnings"])


def test_worker_preflight_denies_live_same_task_and_assignee_capacity(tmp_path: Path):
    cards = [
        {
            "id": "t_existing",
            "title": "[worker task080 neurogolf-worker-4 current-live] exact shrink",
            "body": "task080 current-live shrink",
            "status": "running",
            "assignee": "neurogolf-worker-4",
        }
    ]

    report = build_worker_preflight(
        task_id="task080",
        assignee="neurogolf-worker-4",
        method_family="current-live shrink",
        cards=cards,
        registry_doc={},
        queue_doc={},
        state_doc={},
        task_outcome_doc={},
        repo_root=tmp_path,
    )

    assert report["verdict"] == "DENY"
    assert any("already mention task080" in reason for reason in report["deny_reasons"])
    assert any("already has active card" in reason for reason in report["deny_reasons"])
    assert report["same_task_active_cards"][0]["id"] == "t_existing"
    assert report["assignee_active_cards"][0]["id"] == "t_existing"


def test_worker_preflight_denies_assigned_todo_queue_without_dependency_gate(tmp_path: Path):
    cards = [
        {
            "id": "t_parked",
            "title": "[worker task081 neurogolf-worker-4 queued-next] exact shrink",
            "body": "Assigned todo queue; unsafe unless it is dependency-gated outside active dispatch.",
            "status": "todo",
            "assignee": "neurogolf-worker-4",
        }
    ]

    report = build_worker_preflight(
        task_id="task080",
        assignee="neurogolf-worker-4",
        method_family="current-live shrink",
        cards=cards,
        registry_doc={},
        queue_doc={},
        state_doc={},
        task_outcome_doc={},
        repo_root=tmp_path,
    )

    assert report["verdict"] == "DENY"
    assert report["ok_to_create_worker_card"] is False
    assert report["assignee_active_cards"][0]["id"] == "t_parked"
    assert report["assignee_parked_cards"] == []
    assert any("already has active card" in reason for reason in report["deny_reasons"])


def test_worker_preflight_denies_live_target_claim_from_card(tmp_path: Path):
    cards = [
        {
            "id": "t_claim",
            "title": "[worker task080 neurogolf-worker-4 current-live] active shrink",
            "body": "active lane",
            "assignee": "neurogolf-worker-4",
            "status": "running",
            "started_at": "2026-05-16T18:00:00Z",
        }
    ]

    report = build_worker_preflight(
        task_id="task080",
        assignee="neurogolf-worker-4",
        method_family="current-live shrink",
        lineage_base_candidate_id="baseA",
        cards=cards,
        registry_doc={},
        queue_doc={},
        task_outcome_doc={},
        repo_root=tmp_path,
    )

    assert report["verdict"] == "DENY"
    assert any("live target claim" in reason for reason in report["deny_reasons"])
    assert report["chain_guard"]["live_target_claim"]["kanban_task_id"] == "t_claim"


def test_worker_preflight_allows_clean_assignment_and_emits_footer(tmp_path: Path):
    (tmp_path / "The_400_Tasks" / "Task_by_Task").mkdir(parents=True)
    (tmp_path / "The_400_Tasks" / "Task_by_Task" / "task021.md").write_text(
        "# task021\n\n## Attempts\n\nNone yet.\n",
        encoding="utf-8",
    )

    report = build_worker_preflight(
        task_id="21",
        assignee="neurogolf-worker-6",
        method_family="current-live shrink with task-specific symmetry simplification",
        cards=[],
        registry_doc={},
        queue_doc={},
        state_doc={},
        task_outcome_doc={},
        repo_root=tmp_path,
    )

    assert report["verdict"] == "ALLOW"
    assert report["ok_to_create_worker_card"] is True
    assert report["deny_reasons"] == []
    assert "preflight_task_id: task021" in report["required_card_footer"]
    assert report["task_doc"]["exists"] is True
