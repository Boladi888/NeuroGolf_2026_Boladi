from tools.neurogolf_board_policy import (
    audit_worker_queue,
    build_chain_keys,
    build_live_card_guard,
    build_target_claim_record,
)


def test_chain_keys_follow_expected_stage_convention():
    keys = build_chain_keys(
        {
            "candidate_id": "task080_v9_bool_slices_pruned_on_5727_75_20260516",
            "task_id": "task080",
            "worker_owner": "neurogolf-worker-4",
            "lineage_base_candidate_id": "stack_task109_task128_task198_on_5725_87_20260516T043320Z",
            "stack_candidate_ids": ["task080_v9_bool_slices_pruned_on_5727_75_20260516"],
        }
    )

    assert keys == {
        "worker": "target:task080:neurogolf-worker-4:stack_task109_task128_task198_on_5725_87_20260516T043320Z",
        "review": "review:task080_v9_bool_slices_pruned_on_5727_75_20260516",
        "worthiness": "worthiness:task080_v9_bool_slices_pruned_on_5727_75_20260516",
        "submit": "submit-stack:task080_v9_bool_slices_pruned_on_5727_75_20260516",
    }


def test_live_card_guard_detects_duplicate_live_chain_key_from_registry():
    registry_doc = {
        "entries": [
            {
                "candidate_id": "task080_v9_bool_slices_pruned_on_5727_75_20260516",
                "task_id": "task080",
                "worker_owner": "neurogolf-worker-4",
                "lineage_base_candidate_id": "stack_task109_task128_task198_on_5725_87_20260516T043320Z",
                "submit_queue_status": "ready",
            }
        ]
    }

    guard = build_live_card_guard(
        "review",
        {
            "candidate_id": "task080_v9_bool_slices_pruned_on_5727_75_20260516",
            "task_id": "task080",
            "worker_owner": "neurogolf-worker-4",
            "lineage_base_candidate_id": "stack_task109_task128_task198_on_5725_87_20260516T043320Z",
        },
        registry_doc=registry_doc,
    )

    assert guard["ok"] is False
    assert guard["status"] == "duplicate-live-chain-key"
    assert guard["live_chain_matches"]
    assert guard["chain_key"] == "review:task080_v9_bool_slices_pruned_on_5727_75_20260516"


def test_live_card_guard_detects_duplicate_live_target_claim():
    cards = [
        {
            "id": "t_f83226b8",
            "title": "[worker task080 neurogolf-worker-4 current-live] active lane",
            "body": "current-live shrink lane A",
            "assignee": "neurogolf-worker-4",
            "status": "running",
            "started_at": "2026-05-16T18:00:00Z",
        }
    ]

    guard = build_live_card_guard(
        "worker",
        {
            "candidate_id": "task080_v9_bool_slices_pruned_on_5727_75_20260516",
            "task_id": "task080",
            "worker_owner": "neurogolf-worker-4",
            "lineage_base_candidate_id": "stack_task109_task128_task198_on_5725_87_20260516T043320Z",
        },
        cards=cards,
    )

    assert guard["ok"] is False
    assert guard["status"] == "duplicate-live-target-claim"
    assert guard["live_target_claim"]["task_id"] == "task080"
    assert guard["live_target_claim"]["owner_profile"] == "neurogolf-worker-4"
    assert guard["live_target_claim"]["kanban_task_id"] == "t_f83226b8"
    assert guard["chain_key"] == "target:task080:neurogolf-worker-4:stack_task109_task128_task198_on_5725_87_20260516T043320Z"

def test_live_card_guard_detects_duplicate_live_chain_key_from_registry_using_direct_stack_ids():
    registry_doc = {
        "entries": [
            {
                "candidate_id": "candA",
                "task_id": "task080",
                "worker_owner": "neurogolf-worker-4",
                "lineage_base_candidate_id": "baseA",
                "stack_candidate_ids": ["candA"],
                "submit_queue_status": "ready",
            }
        ]
    }

    guard = build_live_card_guard(
        "submit",
        {
            "candidate_id": "candA",
            "task_id": "task080",
            "worker_owner": "neurogolf-worker-4",
            "lineage_base_candidate_id": "baseA",
            "stack_candidate_ids": ["candA"],
        },
        registry_doc=registry_doc,
    )

    assert guard["ok"] is False
    assert guard["status"] == "duplicate-live-chain-key"
    assert guard["live_chain_matches"]
    assert guard["chain_key"] == "submit-stack:candA"


def test_live_card_guard_detects_duplicate_live_chain_key_from_registry_using_direct_lineage_base_candidate_id():
    registry_doc = {
        "entries": [
            {
                "candidate_id": "candA",
                "task_id": "task080",
                "worker_owner": "neurogolf-worker-4",
                "lineage_base_candidate_id": "baseA",
                "submit_queue_status": "ready",
            }
        ]
    }

    guard = build_live_card_guard(
        "worker",
        {
            "candidate_id": "candA",
            "task_id": "task080",
            "worker_owner": "neurogolf-worker-4",
            "lineage_base_candidate_id": "baseA",
        },
        registry_doc=registry_doc,
        cards=[],
    )

    assert guard["ok"] is False
    assert guard["status"] == "duplicate-live-chain-key"
    assert guard["live_chain_matches"]
    assert guard["chain_key"] == "target:task080:neurogolf-worker-4:baseA"


def test_audit_worker_queue_flags_pending_backlog_and_dispatch_storms():
    cards = [
        {
            "id": "t_live_a",
            "title": "[worker task040 neurogolf-worker-4 current-live] exact shrink",
            "body": "active lane",
            "assignee": "neurogolf-worker-4",
            "status": "running",
            "priority": 80,
            "created_at": 10,
        },
        {
            "id": "t_backlog_1",
            "title": "[BACKLOG single task] task021 fresh current-live shrink continuation",
            "body": "UNASSIGNED BACKLOG CARD — do not dispatch until an operator assigns exactly one idle worker.",
            "assignee": "neurogolf-worker-4",
            "status": "ready",
            "priority": -20,
            "created_at": 20,
            "workspace_path": "C:/wt/worker4-task021",
        },
        {
            "id": "t_unassigned_bank",
            "title": "[BACKLOG single task] task102 fresh current-live shrink continuation",
            "body": "UNASSIGNED BACKLOG CARD — do not dispatch until an operator assigns exactly one idle worker.",
            "assignee": "",
            "status": "todo",
            "priority": -20,
            "created_at": 30,
            "workspace_path": "C:/wt/worker2-task102",
        },
    ]

    report = audit_worker_queue(cards)

    assert report["ok"] is False
    assert report["pending_assignment_backlog_count"] == 1
    assert report["unassigned_backlog_bank_count"] == 1
    assert report["active_violations"][0]["card_ids"] == ["t_live_a", "t_backlog_1"]
    assert report["active_violations"][0]["reason"] == "more than one active assigned card for one worker; use a dependency chain or unassigned backlog bank"
    assert report["dispatch_risk_violations"][0]["card_ids"] == ["t_live_a", "t_backlog_1"]
    assert report["parked_by_assignee"] == {}
    assert report["assignment_plan"] == []
    assert report["low_context_pending"] == []


def test_audit_worker_queue_counts_title_only_unassigned_backlog_as_bank():
    cards = [
        {
            "id": "t_legacy_backlog",
            "title": "[BACKLOG single task] task102 old backlog card",
            "body": "legacy card without the newer body marker",
            "assignee": "",
            "status": "todo",
            "priority": 30,
            "created_at": 10,
        }
    ]

    report = audit_worker_queue(cards)

    assert report["ok"] is True
    assert report["unassigned_backlog_bank_count"] == 1
    assert report["unassigned_backlog_bank"][0]["task_id"] == "t_legacy_backlog"


def test_audit_worker_queue_does_not_treat_running_backlog_marker_as_assignment_wait():
    cards = [
        {
            "id": "t_running_backlog",
            "title": "[BACKLOG single task] task021 fresh current-live shrink continuation",
            "body": "UNASSIGNED BACKLOG CARD — stale marker left after assignment.",
            "assignee": "neurogolf-worker-4",
            "status": "running",
            "priority": -20,
            "created_at": 20,
        }
    ]

    report = audit_worker_queue(cards)

    assert report["ok"] is True
    assert report["pending_assignment_backlog_count"] == 0
    assert report["active_violations"] == []


def test_audit_worker_queue_counts_assigned_todo_as_active_until_dependency_gated():
    cards = [
        {
            "id": "t_waiting_submit",
            "title": "[worker task201 neurogolf-worker-2 waiting submit]",
            "body": "Keep context for failed-submit follow-up.",
            "assignee": "neurogolf-worker-2",
            "status": "blocked",
            "priority": 80,
            "created_at": 10,
        },
        {
            "id": "t_next_queue",
            "title": "[worker task202 neurogolf-worker-2 queued-next]",
            "body": "Queued item; without an unsatisfied dependency Hermes may promote it.",
            "assignee": "neurogolf-worker-2",
            "status": "todo",
            "priority": -20,
            "created_at": 20,
        },
    ]

    report = audit_worker_queue(cards)

    assert report["ok"] is False
    assert report["active_violations"] == [
        {
            "assignee": "neurogolf-worker-2",
            "active_count": 2,
            "card_ids": ["t_waiting_submit", "t_next_queue"],
            "statuses": ["blocked", "todo"],
            "reason": "more than one active assigned card for one worker; use a dependency chain or unassigned backlog bank",
        }
    ]
    assert report["lane_context_by_assignee"]["neurogolf-worker-2"] == ["t_waiting_submit", "t_next_queue"]
    assert report["parked_by_assignee"] == {}
    assert "neurogolf-worker-2" not in report["idle_queue_workers"]
    assert "neurogolf-worker-2" not in report["lane_free_workers"]


def test_audit_skips_lane_violations_for_pool_b_workers():
    """Pool B workers (5, 7, 8) are manual Claude Code /loop sessions with
    closed-port profiles. The Hermes dispatcher cannot fan them out, so
    having a chain of parked opt-return/solver-hunt follow-ups is the
    dependency serializer's intentional behavior — not a violation. The
    audit must NOT flag them as lane-context or dispatch-risk violations,
    otherwise the submit trigger's queue_ok gate stays False forever and
    Kaggle uploads are perpetually blocked by unrelated Pool B work.
    """
    cards = [
        # Pool B worker with the exact lane shape that was blocking the
        # submit trigger: 1 blocked review-required + 1 todo follow-up.
        {
            "id": "t_blocked",
            "title": "[optimization-return] task013 packet had no submit-ready artifact",
            "assignee": "neurogolf-worker-8",
            "status": "blocked",
            "priority": 60,
            "created_at": 10,
        },
        {
            "id": "t_followup",
            "title": "[optimization-return] task004 v9 shrink fails",
            "assignee": "neurogolf-worker-8",
            "status": "todo",
            "priority": 55,
            "created_at": 20,
        },
    ]

    report = audit_worker_queue(cards)
    # Pool B worker NOT in lane_context_violations.
    assert report["lane_context_violations"] == []
    assert report["dispatch_risk_violations"] == []
    assert report["active_violations"] == []
    # Queue safety is now OK — submit trigger can fire.
    assert report["ok"] is True


def test_audit_still_flags_lane_violations_for_pool_a_workers():
    """Pool A workers (1, 2, 3) and draining workers (4, 6) are still
    dispatcher-driven; having multiple active cards IS a real fanout risk
    that the submit trigger should gate on."""
    cards = [
        {
            "id": "t_blocked",
            "title": "[worker task201 neurogolf-worker-2 waiting]",
            "assignee": "neurogolf-worker-2",
            "status": "blocked",
            "priority": 80,
            "created_at": 10,
        },
        {
            "id": "t_todo",
            "title": "[worker task202 neurogolf-worker-2 queued]",
            "assignee": "neurogolf-worker-2",
            "status": "todo",
            "priority": 0,
            "created_at": 20,
        },
    ]

    report = audit_worker_queue(cards)
    assert report["ok"] is False
    assert len(report["lane_context_violations"]) == 1
    assert report["lane_context_violations"][0]["assignee"] == "neurogolf-worker-2"


def test_worker3_is_part_of_default_alpha_queue_pool():
    report = audit_worker_queue([])

    assert "neurogolf-worker-3" in report["queue_workers"]
    assert "neurogolf-worker-3" in report["idle_queue_workers"]


def test_default_alpha_queue_pool_is_workers_1_through_8():
    report = audit_worker_queue([])

    assert report["queue_workers"] == [
        "neurogolf-worker-1",
        "neurogolf-worker-2",
        "neurogolf-worker-3",
        "neurogolf-worker-4",
        "neurogolf-worker-5",
        "neurogolf-worker-6",
        "neurogolf-worker-7",
        "neurogolf-worker-8",
    ]
    assert report["low_context_pending"] == []
