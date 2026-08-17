"""Tests for the submit-picker cache helper.

The cache pre-computes the deterministic parts of submit-captain's
decision (which queue entries are ready, EV-ranked, hidden-bad-tagged,
already-failed-SHA-tagged). It's read-only advice — the agent stays in
charge — so the failure mode is "agent reads stale/wrong cache." Tests
focus on:

  1. Status filtering: ready / awaiting-credit included, everything else
     dropped.
  2. EV ranking: priority_score wins; expected_point_gain is fallback.
  3. Hidden-bad task hit: a candidate that changes a task in
     state.hidden_bad_tasks gets flagged.
  4. Failed-SHA hit: a candidate whose package_sha appears in
     failed_validations or a failed queue entry gets flagged.
  5. safe_to_submit: True iff both hit fields are empty.
  6. Schema stability: top-level keys + values match the spec.
"""
from __future__ import annotations

from tools.neurogolf_submit_picker_cache import (
    SCHEMA_VERSION,
    build_cache,
    build_ready_index,
    collect_failed_public_negative_shas,
    collect_hidden_bad_task_ids,
    is_ready_for_submit,
)


def _ready_entry(**overrides):
    base = {
        "queue_id": "submit-2026-q-1",
        "candidate_id": "task011_v_w8a_t_aaaaaaaa",
        "task_id": "task011",
        "status": "ready",
        "priority_score": 1.04,
        "package_sha256": "AAAAAAAA" + "0" * 56,
        "package_path": "/tmp/staged/task011/submission.zip",
        "changed_task_ids": ["task011"],
        "lineage_base_candidate_id": "current-best",
        "reviewer_disposition": "approved-for-submit-captain",
        "worker_owner": "neurogolf-worker-8",
    }
    base.update(overrides)
    return base


def _state(hidden_bad_tasks=None, failed=None):
    return {
        "hidden_bad_tasks": hidden_bad_tasks or [],
        "failed_validations": failed or [],
    }


def _manifest():
    return {
        "score": 5755.05,
        "candidate_id": "task013_v_w8a_live_fp16",
        "task_id": "task013",
        "package_path": "submissions/submission.zip",
        "package_sha256": "99F2EA25" + "0" * 56,
    }


# --- status filtering ---------------------------------------------------------


def test_is_ready_for_submit_accepts_ready_and_awaiting_credit():
    assert is_ready_for_submit({"status": "ready"}) is True
    assert is_ready_for_submit({"status": "awaiting-submit-credit-or-stack"}) is True


def test_is_ready_for_submit_rejects_completed_and_returned():
    for status in (
        "submitted-complete-promote-synced",
        "submitted-complete-no-improve",
        "returned-to-worker-after-review",
        "submitted-failed-public-negative-on-5727.97",
        "archived",
        "",
        None,
    ):
        assert is_ready_for_submit({"status": status}) is False


def test_is_ready_for_submit_handles_alt_status_field():
    # Some queue entries use submit_queue_status instead of status.
    assert is_ready_for_submit({"submit_queue_status": "ready"}) is True
    assert is_ready_for_submit({"submit_queue_status": "submitted-complete-promote-synced"}) is False


# --- hidden-bad task collection ----------------------------------------------


def test_collect_hidden_bad_task_ids_normalizes_int_inputs():
    state = {"hidden_bad_tasks": [4, 22, 25, 89, "task300", "300"]}
    out = collect_hidden_bad_task_ids(state)
    assert "task004" in out
    assert "task022" in out
    assert "task025" in out
    assert "task089" in out
    assert "task300" in out
    # No duplicates from "task300" / "300" both being present.
    assert out.count("task300") == 1


def test_collect_hidden_bad_task_ids_skips_garbage():
    state = {"hidden_bad_tasks": [None, "", -1, "not-a-task", []]}
    # Negative numbers normalize to task-001 which is harmless but odd;
    # accept whatever survives — the contract is "best-effort." The
    # important thing is it doesn't crash.
    out = collect_hidden_bad_task_ids(state)
    assert isinstance(out, list)
    assert "" not in out
    assert None not in out


# --- failed SHA collection ----------------------------------------------------


def test_collect_failed_shas_from_state_failed_validations():
    state = {
        "failed_validations": [
            {"package_sha256": "DEADBEEF" + "1" * 56},
            {"package_sha256": "C0FFEE00" + "2" * 56},
            {"package_sha256": ""},  # blanks skipped
            {},
        ]
    }
    out = collect_failed_public_negative_shas(state, queue=[])
    assert "deadbeef" + "1" * 56 in out
    assert "c0ffee00" + "2" * 56 in out
    assert "" not in out


def test_collect_failed_shas_from_queue_failed_statuses():
    queue = [
        {
            "package_sha256": "DEADAAAA" + "0" * 56,
            "status": "submitted-failed-public-negative-on-5727.97",
        },
        {
            "package_sha256": "C0FFEE11" + "0" * 56,
            "status": "public-negative-isolated-replaced-by-task300-v4",
        },
        {
            "package_sha256": "5AFE0000" + "0" * 56,  # not a failed status, ignored
            "status": "submitted-complete-promote-synced",
        },
    ]
    out = collect_failed_public_negative_shas(state={}, queue=queue)
    assert "deadaaaa" + "0" * 56 in out
    assert "c0ffee11" + "0" * 56 in out
    assert "5afe0000" + "0" * 56 not in out


def test_collect_failed_shas_merges_state_and_queue_sources():
    state = {"failed_validations": [{"package_sha256": "AA" + "0" * 62}]}
    queue = [{"package_sha256": "BB" + "0" * 62, "status": "submitted-failed-public-negative"}]
    out = set(collect_failed_public_negative_shas(state, queue))
    assert "aa" + "0" * 62 in out
    assert "bb" + "0" * 62 in out


# --- EV ranking + ready index ------------------------------------------------


def test_build_ready_index_sorts_by_ev_descending():
    queue = [
        _ready_entry(candidate_id="cand_low", priority_score=0.1, changed_task_ids=["task001"]),
        _ready_entry(candidate_id="cand_high", priority_score=1.65, changed_task_ids=["task002"]),
        _ready_entry(candidate_id="cand_mid", priority_score=0.5, changed_task_ids=["task003"]),
    ]
    rows = build_ready_index(queue, hidden_bad_tasks=set(), failed_shas=set())
    assert [r["candidate_id"] for r in rows] == ["cand_high", "cand_mid", "cand_low"]


def test_build_ready_index_falls_back_to_expected_point_gain_when_priority_zero():
    queue = [
        _ready_entry(candidate_id="cand", priority_score=0, expected_point_gain=0.42, changed_task_ids=["task001"]),
    ]
    rows = build_ready_index(queue, hidden_bad_tasks=set(), failed_shas=set())
    assert rows[0]["ev"] == 0.42


def test_build_ready_index_flags_hidden_bad_task_hit():
    queue = [
        _ready_entry(candidate_id="bad", task_id="task004", changed_task_ids=["task004"]),
        _ready_entry(candidate_id="good", task_id="task011", changed_task_ids=["task011"]),
    ]
    rows = build_ready_index(queue, hidden_bad_tasks={"task004"}, failed_shas=set())
    by_id = {r["candidate_id"]: r for r in rows}
    assert by_id["bad"]["hidden_bad_task_hit"] == ["task004"]
    assert by_id["bad"]["safe_to_submit"] is False
    assert by_id["good"]["hidden_bad_task_hit"] == []
    assert by_id["good"]["safe_to_submit"] is True


def test_build_ready_index_flags_failed_sha_hit():
    queue = [
        _ready_entry(candidate_id="failed", package_sha256="BADFACE0" + "0" * 56),
        _ready_entry(candidate_id="fresh", package_sha256="FFFFFFFF" + "0" * 56),
    ]
    rows = build_ready_index(
        queue,
        hidden_bad_tasks=set(),
        failed_shas={"badface0" + "0" * 56},
    )
    by_id = {r["candidate_id"]: r for r in rows}
    assert by_id["failed"]["failed_sha_hit"] is True
    assert by_id["failed"]["safe_to_submit"] is False
    assert by_id["fresh"]["failed_sha_hit"] is False
    assert by_id["fresh"]["safe_to_submit"] is True


def test_build_ready_index_excludes_non_ready_statuses():
    queue = [
        _ready_entry(candidate_id="ready_one", status="ready"),
        _ready_entry(candidate_id="done_one", status="submitted-complete-promote-synced"),
        _ready_entry(candidate_id="returned_one", status="returned-to-worker-after-review"),
    ]
    rows = build_ready_index(queue, hidden_bad_tasks=set(), failed_shas=set())
    assert [r["candidate_id"] for r in rows] == ["ready_one"]


def test_build_ready_index_falls_back_to_approval_packet_for_path_and_sha():
    """Real submit_queue entries store package_path + package_sha256 under
    `approval_packet`, not at the top level. The cache builder must look
    inside that nested dict so the picker can validate the staged zip."""
    queue = [
        {
            "queue_id": "q",
            "candidate_id": "approved_cand",
            "status": "ready",
            "priority_score": 1.0,
            "package_path": None,
            "package_sha256": None,
            "approval_packet": {
                "package_path": "/staged/x/submission.zip",
                "package_sha256": "AB12" + "0" * 60,
            },
            "changed_task_ids": ["task050"],
        },
    ]
    rows = build_ready_index(queue, hidden_bad_tasks=set(), failed_shas=set())
    assert rows[0]["package_path"] == "/staged/x/submission.zip"
    assert rows[0]["package_sha256"] == "ab12" + "0" * 60


def test_build_ready_index_handles_missing_changed_task_ids():
    """Older queue entries may not have ``changed_task_ids``; fall back
    to the single ``task_id`` so stack-compatibility still works."""
    queue = [_ready_entry(candidate_id="legacy", task_id="task050", changed_task_ids=[])]
    rows = build_ready_index(queue, hidden_bad_tasks=set(), failed_shas=set())
    assert rows[0]["changed_task_ids"] == ["task050"]


# --- end-to-end build_cache --------------------------------------------------


def test_build_cache_top_level_schema():
    cache = build_cache(
        queue_doc={"queue": [
            _ready_entry(candidate_id="a", priority_score=1.0, changed_task_ids=["task011"]),
            _ready_entry(candidate_id="b", priority_score=0.5, changed_task_ids=["task004"]),
        ]},
        state_doc=_state(hidden_bad_tasks=[4]),
        manifest_doc=_manifest(),
        task_outcome_doc={"updated_at": "2026-05-17T22:37:29Z"},
    )
    # Schema fields are stable.
    assert cache["schema"] == SCHEMA_VERSION
    assert "computed_at" in cache
    assert cache["current_best"]["score"] == 5755.05
    assert "task004" in cache["hidden_bad_task_ids"]
    assert cache["ready_count"] == 2
    # task004 entry got tagged as hidden-bad, so safe_ready_count = 1.
    assert cache["safe_ready_count"] == 1
    assert cache["blocked_ready_count"] == 1
    # Top EV among SAFE candidates is 1.0 (the task011 one).
    assert cache["top_safe_ev"] == 1.0
    assert cache["total_safe_ev"] == 1.0
    assert cache["task_outcome_registry_updated_at"] == "2026-05-17T22:37:29Z"


def test_build_cache_robust_to_empty_inputs():
    cache = build_cache(queue_doc=None, state_doc=None, manifest_doc=None, task_outcome_doc=None)
    assert cache["schema"] == SCHEMA_VERSION
    assert cache["ready_candidates_by_ev"] == []
    assert cache["hidden_bad_task_ids"] == []
    assert cache["failed_public_negative_shas"] == []
    assert cache["current_best"] == {}
    assert cache["ready_count"] == 0
    assert cache["safe_ready_count"] == 0
    assert cache["top_safe_ev"] == 0.0


def test_build_cache_does_not_count_failed_sha_in_top_ev():
    """If the highest-EV ready entry has a failed SHA, top_safe_ev should
    skip it and use the next-highest safe entry."""
    cache = build_cache(
        queue_doc={"queue": [
            _ready_entry(candidate_id="poisoned", priority_score=5.0, package_sha256="DEAD" + "0" * 60),
            _ready_entry(candidate_id="clean", priority_score=1.0, package_sha256="FFFF" + "0" * 60),
        ]},
        state_doc=_state(failed=[{"package_sha256": "DEAD" + "0" * 60}]),
        manifest_doc=_manifest(),
        task_outcome_doc={},
    )
    assert cache["top_safe_ev"] == 1.0
    assert cache["safe_ready_count"] == 1
    assert cache["blocked_ready_count"] == 1
