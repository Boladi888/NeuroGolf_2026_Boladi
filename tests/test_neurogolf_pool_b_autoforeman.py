"""Smoke tests for the Pool B autoforeman.

Pool B feeds new-solver hunt cards to workers 5/7/8 for tasks NOT in the
curated ``inspect_tasks.SOLVED_TASKS`` list. Many of these tasks DO have
working solvers in ``submissions/submission.zip``; Pool B's job is to find
a BETTER one (different family, smaller, more correct on hidden test).
The card body is generated from ``current_solver_info()`` so the worker
knows what they're actually competing against.

These tests cover the selection policy, idempotency-key shape, and the
zip-aware body messaging without touching the real Hermes kanban DB.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from tools.neurogolf_board_policy import POOL_B_SOLVER_HUNT_WORKERS
from tools.neurogolf_pool_b_autoforeman import (
    ZERO_PAD_MAX_BYTES_DEFAULT,
    actively_claimed_task_ids,
    all_unsolved_task_ids,
    card_body_for,
    current_solver_info,
    plan_pool_b,
)


def _empty_docs() -> dict:
    return {"registry_doc": {}, "queue_doc": {}, "state_doc": {}, "task_outcome_doc": {}}


@pytest.fixture
def fake_submission_zip(tmp_path: Path) -> Path:
    """A tiny submission.zip with mixed task sizes:
    * task000.onnx: 100 bytes  -> zero-pad-tier
    * task001.onnx: 1000 bytes -> real working solver
    * task002.onnx: 200 bytes  -> zero-pad-tier
    * task008.onnx: 31000 bytes -> real working solver (the original bug case)
    """
    path = tmp_path / "submission.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("task000.onnx", b"\x00" * 100)
        z.writestr("task001.onnx", b"\x00" * 1000)
        z.writestr("task002.onnx", b"\x00" * 200)
        z.writestr("task008.onnx", b"\x00" * 31000)
    return path


def test_task_ids_exhausted_by_public_history_detects_public_negative_no_fresh():
    """task000 was submitted twice and both came back COMPLETE tie /
    public-no-improve. The picker should now stop suggesting it.
    """
    from tools.neurogolf_pool_b_autoforeman import task_ids_exhausted_by_public_history
    queue = {"queue": [
        {"task_id": "task000", "candidate_id": "task000_v2_conv2x2", "status": "submitted-complete-no-improve"},
        {"task_id": "task000", "candidate_id": "task000_v3_minparams", "status": "submitted-complete-no-promote-tie"},
        # task001 had one no-improve, but ALSO has a fresh ready candidate
        # waiting in queue — not exhausted yet.
        {"task_id": "task001", "candidate_id": "task001_v1", "status": "submitted-complete-no-improve"},
        {"task_id": "task001", "candidate_id": "task001_v2_fresh", "status": "ready"},
        # task002 has only fresh candidates — definitely not exhausted.
        {"task_id": "task002", "candidate_id": "task002_v1_fresh", "status": "ready"},
        # task003 has been done-promoted (success); also not "exhausted".
        {"task_id": "task003", "candidate_id": "task003_v1", "status": "submitted-complete-promote-synced"},
    ]}
    out = task_ids_exhausted_by_public_history(queue)
    assert "task000" in out
    assert "task001" not in out  # has fresh ready candidate
    assert "task002" not in out  # only ready, never failed
    assert "task003" not in out  # promote-synced is success, not exhaustion


def test_task_ids_exhausted_excludes_promote_superseded_success():
    """`submitted-complete-promote-superseded` means the candidate WAS
    promoted to LB and later replaced by a different candidate — two
    successes, not exhaustion. Earlier exact-match logic counted this
    as a public-negative and falsely marked 8 healthy tasks as exhausted."""
    from tools.neurogolf_pool_b_autoforeman import task_ids_exhausted_by_public_history
    queue = {"queue": [
        {"task_id": "task020", "candidate_id": "task020_v1", "status": "submitted-complete-promote-superseded"},
    ]}
    out = task_ids_exhausted_by_public_history(queue)
    assert "task020" not in out


def test_task_ids_exhausted_catches_suffixed_public_negative_variants():
    """Hermes appends contextual suffixes to status strings:
       submitted-failed-public-negative-stack-isolation
       submitted-failed-public-negative-on-5755.62
       submitted-complete-no-requeue-public-flat
       public-negative-isolated-replaced-by-task300-v4
    The earlier exact-match set missed ALL of these — they should now
    count as exhaustion signals."""
    from tools.neurogolf_pool_b_autoforeman import task_ids_exhausted_by_public_history
    queue = {"queue": [
        {"task_id": "task011", "candidate_id": "c", "status": "submitted-failed-public-negative-stack-isolation"},
        {"task_id": "task097", "candidate_id": "c", "status": "submitted-failed-public-negative-on-5727.97"},
        {"task_id": "task133", "candidate_id": "c", "status": "submitted-complete-no-requeue-public-flat"},
        {"task_id": "task388", "candidate_id": "c", "status": "submitted-failed-public-negative-no-submit"},
    ]}
    out = task_ids_exhausted_by_public_history(queue)
    assert "task011" in out
    assert "task097" in out
    assert "task133" in out
    assert "task388" in out


def test_task_ids_exhausted_treats_returned_to_worker_as_fresh():
    """A `returned-to-worker-after-review` candidate means the worker is
    actively reworking — the task isn't exhausted, there's pending work.
    Same for `returned-remediation-*`."""
    from tools.neurogolf_pool_b_autoforeman import task_ids_exhausted_by_public_history
    queue = {"queue": [
        {"task_id": "task255", "candidate_id": "a", "status": "submitted-complete-no-promote"},
        {"task_id": "task255", "candidate_id": "b", "status": "returned-to-worker-after-review"},
        {"task_id": "task284", "candidate_id": "c", "status": "returned-remediation-after-public-negative-stack"},
    ]}
    out = task_ids_exhausted_by_public_history(queue)
    assert "task255" not in out  # active rework override
    assert "task284" not in out  # active remediation override


def test_task_ids_exhausted_returns_empty_on_missing_queue():
    from tools.neurogolf_pool_b_autoforeman import task_ids_exhausted_by_public_history
    assert task_ids_exhausted_by_public_history(None) == set()
    assert task_ids_exhausted_by_public_history({}) == set()
    assert task_ids_exhausted_by_public_history({"queue": []}) == set()
    assert task_ids_exhausted_by_public_history({"queue": "not-a-list"}) == set()


def test_plan_pool_b_deprioritizes_exhausted_tasks_when_fresh_available():
    """End-to-end: Pool B picker pushes task000 to the END of the rotation
    once its queue history shows public-tie — so with plenty of fresh
    tasks available, task000 never gets picked. (Compare with the
    `_picks_exhausted_when_forced` test below.)"""
    queue_doc = {"queue": [
        {"task_id": "task000", "candidate_id": "task000_v2", "status": "submitted-complete-no-improve"},
    ]}
    docs = {"queue_doc": queue_doc, "registry_doc": {}, "state_doc": {}, "task_outcome_doc": {}}
    plan = plan_pool_b([], docs=docs, create_limit=5, max_per_worker=1)
    chosen = {a["task_id"] for a in plan["actions"]}
    # task000 is deprioritized to the end of available — with hundreds of
    # fresh unsolved tasks it never reaches the front of the queue.
    assert "task000" not in chosen


def test_plan_pool_b_picks_exhausted_task_when_only_option_and_attaches_framing():
    """When an exhausted task is the only one available, Pool B DOES pick
    it (because there's still LB upside from a fundamentally-different
    algorithm family) — but the resulting card body MUST tell the worker
    that prior attempts failed and a different family is needed."""
    queue_doc = {"queue": [
        {"task_id": "task000", "candidate_id": "task000_v2_conv2x2", "status": "submitted-complete-no-improve"},
        {"task_id": "task000", "candidate_id": "task000_v3_minparams", "status": "submitted-complete-no-promote-tie"},
    ]}
    docs = {"queue_doc": queue_doc, "registry_doc": {}, "state_doc": {}, "task_outcome_doc": {}}
    plan = plan_pool_b(
        [],
        docs=docs,
        create_limit=5,
        max_per_worker=1,
        unsolved_override=["task000"],
    )
    chosen = [a for a in plan["actions"] if a["task_id"] == "task000"]
    assert chosen, "task000 should be picked when it's the only available task"
    body = chosen[0]["body"]
    assert "EXHAUSTED TASK NOTICE" in body
    assert "FUNDAMENTALLY DIFFERENT" in body
    assert "task000_v2_conv2x2" in body  # prior candidate listed
    assert "task000_v3_minparams" in body


def test_unsolved_list_uses_curated_solved_set():
    """Pool B targets the complement of inspect_tasks.SOLVED_TASKS — not
    a submission.zip-derived set. Many Pool B targets DO have working
    solvers; we want workers to compete against them, not just zero-pads."""
    unsolved = all_unsolved_task_ids()
    assert "task081" not in unsolved  # in SOLVED_TASKS (current LB candidate)
    assert "task001" not in unsolved  # in SOLVED_TASKS
    assert "task000" in unsolved      # NOT in SOLVED_TASKS
    assert "task008" in unsolved      # NOT in SOLVED_TASKS (despite having a real solver)
    assert all(t.startswith("task") and len(t) == 7 for t in unsolved)


def test_current_solver_info_detects_real_solver(fake_submission_zip: Path):
    info = current_solver_info("task008", zip_path=fake_submission_zip)
    assert info["exists"] is True
    assert info["size"] == 31000
    assert info["is_zero_pad_tier"] is False


def test_current_solver_info_detects_zero_pad_tier(fake_submission_zip: Path):
    info = current_solver_info("task000", zip_path=fake_submission_zip)
    assert info["exists"] is True
    assert info["size"] == 100
    assert info["is_zero_pad_tier"] is True


def test_current_solver_info_handles_missing_entry(fake_submission_zip: Path):
    info = current_solver_info("task399", zip_path=fake_submission_zip)
    assert info["exists"] is False
    assert info["size"] == 0
    assert info["is_zero_pad_tier"] is True  # missing == nothing == zero-pad-tier


def test_current_solver_info_handles_missing_zip(tmp_path: Path):
    info = current_solver_info("task000", zip_path=tmp_path / "missing.zip")
    assert info["exists"] is False


def test_card_body_for_real_solver_tells_worker_to_beat_it():
    body = card_body_for(
        task_id="task008",
        assignee="neurogolf-worker-7",
        preflight={},
        current_solver={"exists": True, "size": 31000, "is_zero_pad_tier": False},
    )
    # Worker must know there's a real solver and what to do about it.
    assert "31000 bytes" in body
    assert "current-best in submission.zip" in body
    assert "improve on the current solver" in body
    # Honesty check: don't tell them "zero-padding baseline" when there's a real solver.
    assert "zero-padding baseline" not in body


def test_card_body_for_zero_pad_tier_tells_worker_to_build_real_solver():
    body = card_body_for(
        task_id="task016",
        assignee="neurogolf-worker-7",
        preflight={},
        current_solver={"exists": True, "size": 158, "is_zero_pad_tier": True},
    )
    assert "158 bytes" in body
    assert "constant-output" in body or "zero-pad" in body
    assert "build a real working ONNX" in body


def test_card_body_for_missing_entry_uses_zero_padding_baseline():
    body = card_body_for(
        task_id="task399",
        assignee="neurogolf-worker-7",
        preflight={},
        current_solver={"exists": False, "size": 0, "is_zero_pad_tier": True},
    )
    assert "no entry for this task" in body
    assert "FIRST working ONNX" in body


def test_card_body_for_exhausted_task_includes_prior_attempts_and_family_guidance():
    body = card_body_for(
        task_id="task000",
        assignee="neurogolf-worker-7",
        preflight={},
        current_solver={"exists": True, "size": 100, "is_zero_pad_tier": True},
        is_exhausted=True,
        submitted_history=[
            {"candidate_id": "task000_v3_minparams", "status": "submitted-complete-no-promote-tie"},
            {"candidate_id": "task000_v2_conv2x2", "status": "submitted-complete-no-improve"},
        ],
    )
    # Must mention the per-task exhausted notice and the algorithm-family
    # framing — that's the whole point of this branch.
    assert "EXHAUSTED TASK NOTICE" in body
    assert "FUNDAMENTALLY DIFFERENT" in body
    assert "task000_v3_minparams" in body
    assert "task000_v2_conv2x2" in body
    assert "submitted-complete-no-promote-tie" in body
    # And it must come BEFORE the worker instructions block so the worker
    # reads the framing before the goal lines.
    notice_idx = body.index("EXHAUSTED TASK NOTICE")
    worker_idx = body.index("Worker instructions (Pool B):")
    assert notice_idx < worker_idx


def test_card_body_for_exhausted_with_empty_history_renders_defensive_note():
    """If is_exhausted is True but history lookup returned nothing (shouldn't
    happen in normal flow but is possible after partial queue rewrites),
    the body must still render the exhaustion notice and call out the
    empty lookup instead of just being silent."""
    body = card_body_for(
        task_id="task000",
        assignee="neurogolf-worker-7",
        preflight={},
        current_solver={"exists": True, "size": 100, "is_zero_pad_tier": True},
        is_exhausted=True,
        submitted_history=[],
    )
    assert "EXHAUSTED TASK NOTICE" in body
    assert "queue history lookup returned no entries" in body
    assert "FUNDAMENTALLY DIFFERENT" in body


def test_card_body_for_non_exhausted_omits_exhaustion_notice():
    body = card_body_for(
        task_id="task008",
        assignee="neurogolf-worker-7",
        preflight={},
        current_solver={"exists": True, "size": 31000, "is_zero_pad_tier": False},
        is_exhausted=False,
    )
    assert "EXHAUSTED TASK NOTICE" not in body
    assert "FUNDAMENTALLY DIFFERENT" not in body


def test_actively_claimed_uses_active_status_and_known_worker():
    cards = [
        {"id": "a", "title": "[worker task050 neurogolf-worker-1 ...]", "assignee": "neurogolf-worker-1", "status": "running"},
        {"id": "b", "title": "[worker task051 neurogolf-worker-2 ...]", "assignee": "neurogolf-worker-2", "status": "done"},      # done → ignored
        {"id": "c", "title": "[worker task052 ext ...]",                "assignee": "external-bot",        "status": "running"},  # not a queue worker → ignored
        {"id": "d", "title": "[worker task053 neurogolf-worker-7 ...]", "assignee": "neurogolf-worker-7", "status": "ready"},
    ]
    claimed = actively_claimed_task_ids(cards)
    assert "task050" in claimed
    assert "task053" in claimed
    assert "task051" not in claimed
    assert "task052" not in claimed


def test_plan_pool_b_skips_claimed_tasks_and_picks_idle_workers():
    cards = [
        # Pool A worker is busy on task000 — Pool B foreman must skip task000.
        {"id": "busy", "title": "[worker task000 neurogolf-worker-1 ...]", "assignee": "neurogolf-worker-1", "status": "running"},
        # No cards for any Pool B worker → all idle.
    ]
    plan = plan_pool_b(cards, docs=_empty_docs(), create_limit=10, max_per_worker=1)
    assigned_tasks = {a["task_id"] for a in plan["actions"]}
    assert "task000" not in assigned_tasks
    # Every Pool B worker got a card.
    assignees = {a["assignee"] for a in plan["actions"]}
    assert assignees == set(POOL_B_SOLVER_HUNT_WORKERS)
    # Each worker gets at most one card.
    assert len(plan["actions"]) == len(POOL_B_SOLVER_HUNT_WORKERS)


def test_plan_pool_b_respects_max_per_worker():
    plan = plan_pool_b([], docs=_empty_docs(), create_limit=10, max_per_worker=1)
    by_worker: dict[str, int] = {}
    for a in plan["actions"]:
        by_worker[a["assignee"]] = by_worker.get(a["assignee"], 0) + 1
    for worker, count in by_worker.items():
        assert count <= 1, f"{worker} got {count} cards, expected <=1"


def test_plan_pool_b_respects_create_limit():
    plan = plan_pool_b([], docs=_empty_docs(), create_limit=1, max_per_worker=5)
    assert len(plan["actions"]) <= 1


def test_plan_pool_b_skips_busy_workers():
    # worker-7 is already running a card → the OTHER Pool B workers should
    # get assigned, but not worker-7.
    cards = [
        {"id": "x", "title": "[worker task100 neurogolf-worker-7 current-live]", "assignee": "neurogolf-worker-7", "status": "running"},
    ]
    plan = plan_pool_b(cards, docs=_empty_docs(), create_limit=10)
    assignees = {a["assignee"] for a in plan["actions"]}
    expected_idle = set(POOL_B_SOLVER_HUNT_WORKERS) - {"neurogolf-worker-7"}
    assert assignees == expected_idle


def test_idempotency_key_includes_worker_task_and_date():
    plan = plan_pool_b([], docs=_empty_docs(), today="20260518")
    for a in plan["actions"]:
        key = a["idempotency_key"]
        # Shape: ng-pool-b-{worker}-{task_id}-{YYYYMMDD}
        assert key.startswith("ng-pool-b-")
        assert a["assignee"] in key
        assert a["task_id"] in key
        assert "20260518" in key


def test_card_titles_use_recognized_schema():
    plan = plan_pool_b([], docs=_empty_docs())
    for a in plan["actions"]:
        title = a["title"]
        # Must match the `[worker taskNNN <worker> solver-hunt]` schema so
        # the dependency serializer and route-repair tools classify it
        # correctly.
        assert title.startswith("[worker ")
        assert "solver-hunt" in title
        assert a["task_id"] in title
        assert a["assignee"] in title


def test_plan_pool_b_reports_unsolved_totals():
    plan = plan_pool_b([], docs=_empty_docs())
    # Should match the unsolved-task count from inspect_tasks.SOLVED_TASKS.
    assert plan["unsolved_total"] == len(all_unsolved_task_ids())
    assert plan["unsolved_available_total"] <= plan["unsolved_total"]
    assert plan["schema"] == "neurogolf.pool_b_autoforeman.v1"
