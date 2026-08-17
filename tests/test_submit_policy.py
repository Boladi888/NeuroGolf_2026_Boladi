from tools.neurogolf_submit_policy import (
    DEFAULT_APPROVAL_MODE,
    add_submit_budget_metadata,
    choose_55_lane_rationale,
    compose_preflight_history,
    normalize_candidate_entry,
    normalize_queue_entry,
    plan_isolation_after_weak_three_stack,
    plan_stacks,
    upsert_queue_entry,
    upsert_registry_entry,
)


def candidate(candidate_id, *, tasks, lineage="baseA", gain=1.0, risk=1.0, approved=True):
    return {
        "candidate_id": candidate_id,
        "changed_task_ids": tasks,
        "lineage_base_candidate_id": lineage,
        "expected_point_gain": gain,
        "risk_score": risk,
        "reviewer_disposition": "approved-for-submit-captain" if approved else "needs-repair",
        "preflight_history": {"status": "pass"},
    }


def test_queue_entry_defaults_to_auto_rate_limited_and_preserves_manual_override():
    normalized = normalize_queue_entry({"candidate_id": "c1", "approval_packet": {}})

    assert DEFAULT_APPROVAL_MODE == "auto-rate-limited"
    assert normalized["approval_mode"] == "auto-rate-limited"
    assert normalized["approval_packet"]["approval_mode"] == "auto-rate-limited"

    manual = normalize_queue_entry({"candidate_id": "c2", "approval_mode": "manual-required"})

    assert manual["approval_mode"] == "manual-required"


def test_stack_planning_groups_up_to_three_independent_lineage_compatible_candidates():
    candidates = [
        candidate("c1", tasks=["task001"], gain=0.4, risk=0.2),
        candidate("c2", tasks=["task002"], gain=0.8, risk=0.5),
        candidate("c3", tasks=["task003"], gain=0.5, risk=0.1),
        candidate("c4", tasks=["task004"], lineage="baseB", gain=2.0, risk=0.1),
        candidate("c5", tasks=["task002"], gain=2.5, risk=0.1),
    ]

    plans = plan_stacks(candidates)

    assert plans[0]["stack_size"] == 3
    assert plans[0]["candidate_ids"] == ["c5", "c3", "c1"]
    assert plans[0]["lineage_base_candidate_id"] == "baseA"
    assert plans[0]["changed_task_ids"] == ["task001", "task002", "task003"]
    assert plans[0]["combined_expected_point_gain"] == 3.4


def test_weak_three_stack_fallback_selects_best_value_lowest_risk_pair_then_isolates_third_on_pass():
    stack = [
        candidate("low", tasks=["task001"], gain=0.3, risk=0.1),
        candidate("mid", tasks=["task002"], gain=0.5, risk=0.3),
        candidate("high", tasks=["task003"], gain=0.8, risk=0.2),
    ]

    retry = plan_isolation_after_weak_three_stack(stack, two_stack_result=None)
    passed = plan_isolation_after_weak_three_stack(stack, two_stack_result="pass")
    failed = plan_isolation_after_weak_three_stack(stack, two_stack_result="fail")

    assert retry["next_attempt_kind"] == "stack-2"
    assert retry["candidate_ids"] == ["high", "mid"]
    assert passed["next_attempt_kind"] == "isolate-third"
    assert passed["candidate_ids"] == ["low"]
    assert failed["action"] == "return-all-to-queue-or-remediation"
    assert failed["candidate_ids"] == ["low", "mid", "high"]


def test_submit_budget_metadata_records_token_bucket_and_daily_cap_awareness():
    entry = add_submit_budget_metadata({"candidate_id": "c1"}, credits_available=2, submits_used_today=99)

    assert entry["submit_budget"]["rate_limit_minutes"] == 10
    assert entry["submit_budget"]["daily_cap"] == 100
    assert entry["submit_budget"]["credits_available"] == 2
    assert entry["submit_budget"]["submits_remaining_today"] == 1
    assert entry["submit_budget"]["submit_authority"] == "submit-captain only"


def test_55_lane_rationale_flags_hard_uncertain_high_value_items():
    rationale = choose_55_lane_rationale(
        {"task_id": "task366", "expected_point_gain": 1.2, "uncertainty": "high", "difficulty": "hard"}
    )

    assert rationale["route_to_55_lane"] is True
    assert "hard" in rationale["rationale"]
    assert "high-uncertainty" in rationale["rationale"]
    assert "high-value" in rationale["rationale"]


# ----- t_ee4a8497: --write-registry --enqueue self-duplicate regression ----


def _fresh_candidate(candidate_id="task216_winner_test", task_id="task216"):
    return {
        "candidate_id": candidate_id,
        "task_id": task_id,
        "package_sha256": "ABCD1234" * 8,  # 64 hex chars
        "changed_task_ids": [task_id],
        "changed_entries": [f"{task_id}.onnx"],
        "expected_point_gain": 0.5,
        "lineage_base_candidate_id": "current-best",
        "reviewer_disposition": "approved-for-submit-captain",
    }


def test_compose_preflight_excludes_self_from_registry_matches():
    """A candidate already present in the registry must not flag itself as a prior duplicate."""
    candidate = _fresh_candidate()
    registry_doc = {"entries": [candidate]}  # candidate is already in the registry
    queue_doc = {}

    # Without self-exclusion, the candidate sees itself as a registry-match
    # and gets blocked-duplicate. With self-exclusion via self_candidate_id,
    # the match is correctly skipped.
    without_self = compose_preflight_history(candidate, queue_doc=queue_doc, registry_doc=registry_doc)
    assert without_self["status"] == "blocked-duplicate"
    assert candidate["candidate_id"] in without_self["matched_prior_candidates"]

    with_self = compose_preflight_history(
        candidate, queue_doc=queue_doc, registry_doc=registry_doc,
        self_candidate_id=candidate["candidate_id"],
    )
    assert with_self["status"] == "pass", f"expected pass, got {with_self['status']}"


def test_compose_preflight_self_exclusion_does_not_hide_true_duplicates():
    """Self-exclusion must NOT shield against a different-candidate-id true duplicate."""
    candidate = _fresh_candidate(candidate_id="task216_new")
    other = _fresh_candidate(candidate_id="task216_old")  # same sha + task_id, different id
    registry_doc = {"entries": [other]}

    preflight = compose_preflight_history(
        candidate, queue_doc={}, registry_doc=registry_doc,
        self_candidate_id=candidate["candidate_id"],
    )
    # The other entry has the SAME package_sha + task_id, so it's a true
    # duplicate (different candidate_id). Self-exclusion shouldn't hide it.
    assert preflight["status"] == "blocked-duplicate"
    assert "task216_old" in preflight["matched_prior_candidates"]


def test_compose_preflight_self_exclusion_skips_submit_history_self_match():
    """Submit history self-match must also be excluded when self_candidate_id is set."""
    candidate = _fresh_candidate()
    queue_doc = {
        "submit_history": [
            {
                "candidate_id": candidate["candidate_id"],
                "task_id": candidate["task_id"],
                "package_sha256": candidate["package_sha256"],
            }
        ],
    }
    preflight = compose_preflight_history(
        candidate, queue_doc=queue_doc, registry_doc={},
        self_candidate_id=candidate["candidate_id"],
    )
    assert preflight["status"] == "pass"


def test_normalize_candidate_entry_propagates_self_exclusion():
    """normalize_candidate_entry must pass the candidate's own ID as self_candidate_id."""
    candidate = _fresh_candidate()
    # Pre-populate the registry with this candidate (post-write state).
    registry_doc = {
        "entries": [candidate],
        "entry_template": {},
        "review_gate": {"default_approval_mode": "auto-rate-limited"},
    }
    queue_doc = {"queue": [], "submit_history": [], "policy": {}}

    normalized = normalize_candidate_entry(
        candidate, registry_doc=registry_doc, queue_doc=queue_doc, manifests=[],
    )
    # With the fix, normalize must produce submit_ready=True for an already-
    # registered candidate that is being re-normalized as part of the
    # combined --write-registry --enqueue flow.
    assert normalized["submit_ready"] is True, (
        f"expected submit_ready=True after self-exclusion; "
        f"preflight={normalized.get('preflight_history')}"
    )
    assert normalized["submit_queue_status"] == "ready"


def test_combined_write_registry_then_enqueue_does_not_self_duplicate():
    """End-to-end regression for the t_ee4a8497 bug: upserting into the
    registry must not cause the subsequent enqueue to see itself as a
    prior duplicate.
    """
    candidate = _fresh_candidate()
    registry_doc = {
        "entries": [],
        "entry_template": {},
        "review_gate": {"default_approval_mode": "auto-rate-limited"},
    }
    queue_doc = {
        "queue": [],
        "submit_history": [],
        "queue_entry_template": {},
        "policy": {},
    }

    # Step 1: write-registry inserts the candidate.
    updated_registry = upsert_registry_entry(registry_doc, candidate)
    inserted_ids = [e.get("candidate_id") for e in updated_registry["entries"]]
    assert candidate["candidate_id"] in inserted_ids

    # Step 2: enqueue, with the post-write registry. This is the path that
    # used to raise ValueError("not queue-eligible: blocked-duplicate").
    # With the self-exclusion fix it must succeed.
    updated_queue = upsert_queue_entry(queue_doc, candidate, updated_registry)
    enqueued_ids = [e.get("candidate_id") for e in updated_queue["queue"]]
    assert candidate["candidate_id"] in enqueued_ids


def test_combined_write_registry_then_enqueue_still_blocks_real_duplicate():
    """The combined flow must still reject a candidate that's a true duplicate
    of a DIFFERENT pre-existing entry (different candidate_id, same content).
    """
    pre_existing = _fresh_candidate(candidate_id="task216_old")
    candidate = _fresh_candidate(candidate_id="task216_new")  # same sha + task_id
    registry_doc = {
        "entries": [pre_existing],
        "entry_template": {},
        "review_gate": {"default_approval_mode": "auto-rate-limited"},
    }
    queue_doc = {
        "queue": [],
        "submit_history": [],
        "queue_entry_template": {},
        "policy": {},
    }
    # upsert_registry_entry will REPLACE the pre_existing by package_sha match
    # (per upsert_registry_entry's match-by-sha branch). After that, the
    # registry has one entry with the new candidate_id and the new candidate's
    # data. The enqueue should then succeed for the (now sole) entry —
    # mirroring the legitimate-replacement semantic upsert_registry_entry
    # already implements.
    updated_registry = upsert_registry_entry(registry_doc, candidate)
    assert len(updated_registry["entries"]) == 1
    assert updated_registry["entries"][0]["candidate_id"] == "task216_new"

    # Enqueue should now succeed because the only entry IS the candidate
    # (matched by sha → replaced → same candidate_id after replacement).
    updated_queue = upsert_queue_entry(queue_doc, candidate, updated_registry)
    assert any(e.get("candidate_id") == "task216_new" for e in updated_queue["queue"])
