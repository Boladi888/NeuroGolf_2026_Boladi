"""Unit tests for tools/auto_triage_blocked deterministic logic.

Only the pure-function logic is tested: parsing handoff JSON variants, the
sub-floor decision, the no-fix marker matching, and the registry-entry builder.
The hermes-CLI integration (apply_*, hermes_show, etc.) is covered by the live
dry-run already exercised in the cleanup session.
"""
from __future__ import annotations

from tools.auto_triage_blocked import (
    NO_FIX_MARKERS,
    Handoff,
    Verdict,
    _card_has_review_required,
    build_banked_registry_entry,
    build_review_card_spec,
    classify,
    cost_to_points,
    count_attempts_this_pass,
    extract_block_reason,
    extract_json_object,
    extract_point_delta,
    get_attempted_families_this_pass,
    has_marker,
    is_dispatcher_give_up,
    is_return_or_repair_card,
    is_session_broken,
    reconstruct_handoff_from_summary,
    should_broaden_scope,
)


def _handoff_from_parsed(parsed: dict) -> Handoff:
    """Minimal Handoff factory used in tests; mirrors parse_handoff post-extraction."""
    scores = parsed.get("scores") or {}
    score_delta = parsed.get("score_delta") or {}
    point_delta = scores.get("point_delta")
    if point_delta is None:
        point_delta = score_delta.get("point_delta") or score_delta.get("delta_points")
    if point_delta is None:
        baseline = scores.get("current_cost") or parsed.get("baseline_cost") or parsed.get("base_cost")
        candidate = scores.get("candidate_cost") or parsed.get("best_cost") or parsed.get("candidate_cost")
        if baseline and candidate:
            bp, cp = cost_to_points(baseline), cost_to_points(candidate)
            if bp is not None and cp is not None:
                point_delta = cp - bp
    expected = parsed.get("expected_point_gain")
    if isinstance(expected, dict):
        for v in expected.values():
            if isinstance(v, (int, float)):
                expected = v
                break
        else:
            expected = None
    return Handoff(
        raw_text="{}",
        parsed=parsed,
        candidate_id=parsed.get("candidate_id", ""),
        task_id=parsed.get("task_id", ""),
        point_delta=point_delta,
        expected_point_gain=expected if isinstance(expected, (int, float)) else point_delta,
        package_sha=parsed.get("package_sha256", ""),
        lineage_base=parsed.get("lineage_base", ""),
        changed_entries=list(parsed.get("changed_package_entries") or []),
        tests_run=list(parsed.get("tests_run") or []),
        risk_notes=list(parsed.get("risk_notes") or []),
        algorithm_family=str(parsed.get("algorithm_family") or ""),
        retry_attempt_index=(
            int(parsed["retry_attempt_index"])
            if isinstance(parsed.get("retry_attempt_index"), (int, float))
            else None
        ),
    )


# ----- pure helpers --------------------------------------------------------


def test_cost_to_points_formula_matches_competition_spec():
    # points = max(1.0, 25 - log(memory + params))
    assert cost_to_points(73000) == max(1.0, 25.0 - __import__("math").log(73000))
    assert cost_to_points(0) is None
    assert cost_to_points(None) is None
    assert cost_to_points(1) == 25.0  # log(1) = 0


def test_extract_json_object_handles_nested_braces_and_strings():
    src = 'preamble text\n{\n  "task_id": "task001",\n  "nested": {"a": "b{c}d"},\n  "x": 1\n}\nfooter'
    out = extract_json_object(src)
    assert out.startswith("{") and out.endswith("}")
    import json
    parsed = json.loads(out)
    assert parsed["task_id"] == "task001"
    assert parsed["nested"]["a"] == "b{c}d"


def test_extract_json_object_returns_empty_on_no_match():
    assert extract_json_object("no braces here") == ""


def test_has_marker_is_case_insensitive_and_substring():
    assert has_marker("Still NOT enough", NO_FIX_MARKERS)
    assert not has_marker("everything is fine", NO_FIX_MARKERS)


def test_extract_block_reason_pulls_from_event_log():
    text = "Events (4):\n  [..] [run 7] blocked {'reason': 'review-required: task033 here.'}\n"
    assert "task033" in extract_block_reason(text)


# ----- classify decisions --------------------------------------------------


def _card(card_id="t_test", assignee="neurogolf-worker-1", title="[worker task033 ...]"):
    return {"id": card_id, "assignee": assignee, "title": title}


def test_classify_reviewer_card_is_manual():
    handoff = _handoff_from_parsed({})
    verdict = classify(_card(assignee="neurogolf-reviewer"), handoff, "review-required: task371 thing")
    assert verdict.action == "manual"
    assert "reviewer-owned" in verdict.reason


def test_classify_no_fix_marker_in_block_reason_overrides_everything():
    # Even if the handoff looks great, an explicit no-fix block reason wins.
    handoff = _handoff_from_parsed({
        "task_id": "task092",
        "candidate_id": "task092_v9",
        "scores": {"point_delta": 1.5},  # would otherwise escalate
    })
    block_reason = "review-required: task092 follow-up; no singleton/all-zero repair preserves >=0.05 gain."
    verdict = classify(_card(), handoff, block_reason)
    assert verdict.action == "no-fix"


def test_classify_no_fix_marker_in_title_does_NOT_trigger_no_fix():
    # opt-return titles can carry the PREVIOUS reviewer's framing. We must not
    # close a fresh candidate based on a stale title.
    handoff = _handoff_from_parsed({
        "task_id": "task071",
        "candidate_id": "task071_successor_v2",
        "scores": {"point_delta": 0.17},  # above floor, fresh candidate
    })
    title = "[optimization-return] task071 successor is real but still below standalone submit floor"
    block_reason = "review-required: task071 v2 candidate +0.17 EV"  # NO no-fix marker
    verdict = classify(_card(title=title), handoff, block_reason)
    assert verdict.action == "escalate-to-review"


def test_classify_sub_floor_candidate_banks():
    handoff = _handoff_from_parsed({
        "task_id": "task375",
        "candidate_id": "task375_v1",
        "scores": {"point_delta": 0.03},
    })
    verdict = classify(_card(), handoff, "review-required: task375 small win")
    assert verdict.action == "bank"
    assert verdict.new_registry_entry is not None
    assert verdict.new_registry_entry["submit_queue_status"] == "banked-for-integration"


def test_classify_above_floor_with_high_risk_marker_goes_manual():
    handoff = _handoff_from_parsed({
        "task_id": "task397",
        "candidate_id": "task397_risky",
        "scores": {"point_delta": 1.5},
        "risk_notes": ["candidate may have off-contract behavior on hidden inputs"],
    })
    verdict = classify(_card(), handoff, "review-required: task397 +1.5 LB but risky")
    assert verdict.action == "manual"
    assert "high-risk markers" in verdict.reason


def test_classify_clean_above_floor_escalates_to_review():
    handoff = _handoff_from_parsed({
        "task_id": "task033",
        "candidate_id": "task033_bg_tl_delta",
        "scores": {"point_delta": 1.18},
        "risk_notes": [],
    })
    verdict = classify(_card(), handoff, "review-required: task033 nice shrink")
    assert verdict.action == "escalate-to-review"
    assert verdict.new_review_card is not None
    assert verdict.new_review_card["chain_key"].startswith("review:")
    assert verdict.new_review_card["assignee"] == "neurogolf-reviewer"


def test_classify_unparseable_handoff_is_manual():
    handoff = Handoff(raw_text="", parse_error="no JSON object found")
    verdict = classify(_card(), handoff, "review-required: something")
    assert verdict.action == "manual"
    assert "could not parse handoff" in verdict.reason


def test_classify_handles_cost_only_handoff_via_points_formula():
    # task379-style handoff: only baseline_cost + best_cost given.
    handoff = _handoff_from_parsed({
        "task_id": "task379",
        "candidate_id": "task379_v3",
        "baseline_cost": 108414,
        "best_cost": 73000,
    })
    # Computed: 25-log(73000) - (25-log(108414)) ≈ +0.395
    verdict = classify(_card(), handoff, "review-required: task379 shrink")
    assert verdict.action == "escalate-to-review"


def test_classify_handles_nested_score_delta_objects():
    # task034-style handoff: score_delta = {current: {cost: N}, candidate: {cost: M}, delta_points: P}
    handoff = _handoff_from_parsed({
        "task_id": "task034",
        "candidate_id": "task034_v4",
        "score_delta": {
            "current": {"cost": 124798, "points": 13.27},
            "candidate": {"cost": 29452, "points": 14.71},
            "delta_cost": -95346,
            "delta_points": 1.44,
        },
    })
    verdict = classify(_card(), handoff, "review-required: task034 huge win")
    assert verdict.action == "escalate-to-review"


# ----- output builders -----------------------------------------------------


def test_build_banked_registry_entry_carries_expected_fields():
    handoff = _handoff_from_parsed({
        "task_id": "task375",
        "candidate_id": "task375_noand",
        "package_sha256": "DEADBEEF",
        "lineage_base": "current-best task019_x",
        "changed_package_entries": ["task375.onnx"],
        "scores": {"point_delta": 0.03},
    })
    card = _card(card_id="t_4a47", assignee="neurogolf-worker-7")
    entry = build_banked_registry_entry(card, handoff)
    assert entry["candidate_id"] == "task375_noand"
    assert entry["task_id"] == "task375"
    assert entry["worker_owner"] == "neurogolf-worker-7"
    assert entry["submit_queue_status"] == "banked-for-integration"
    assert entry["banked_by"].endswith("auto_triage_blocked.py")
    assert entry["package_sha256"] == "DEADBEEF"
    assert entry["chain_key"] == "review:task375_noand"


def test_build_review_card_spec_includes_chain_key_and_parent():
    handoff = _handoff_from_parsed({
        "task_id": "task033",
        "candidate_id": "task033_v1",
        "scores": {"point_delta": 1.0},
    })
    card = _card(card_id="t_worker1")
    spec = build_review_card_spec(card, handoff)
    assert spec["title"].startswith("[review task033 task033_v1]")
    assert spec["chain_key"] == "review:task033_v1"
    assert spec["parent_id"] == "t_worker1"
    assert spec["assignee"] == "neurogolf-reviewer"
    assert "chain_key: review:task033_v1" in spec["body"]


def test_build_review_card_spec_surfaces_algorithm_family_and_attempt_index():
    """submit-gate redesign: review-card body must echo back the worker's
    self-reported algorithm_family and retry_attempt_index so the user can
    see at a glance what was tried."""
    handoff = _handoff_from_parsed({
        "task_id": "task033",
        "candidate_id": "task033_v2",
        "scores": {"point_delta": 0.42},
        "algorithm_family": "where-select",
        "retry_attempt_index": 2,
    })
    card = _card(card_id="t_worker1")
    spec = build_review_card_spec(card, handoff)
    assert "algorithm_family: where-select" in spec["body"]
    assert "retry_attempt_index: 2" in spec["body"]


def test_build_review_card_spec_handles_legacy_handoff_without_family():
    """Pre-submit-gate workers don't report algorithm_family. Body must still
    render gracefully and call out the missing field rather than crashing or
    printing 'None'."""
    handoff = _handoff_from_parsed({
        "task_id": "task033",
        "candidate_id": "task033_v1",
        "scores": {"point_delta": 1.0},
    })
    card = _card(card_id="t_worker1")
    spec = build_review_card_spec(card, handoff)
    assert "algorithm_family: (worker did not report" in spec["body"]
    assert "retry_attempt_index: (worker did not report)" in spec["body"]


# ----- dispatcher-give-up detection ----------------------------------------


_GAVE_UP_SHOW = """\
Task t_test: [worker task042 neurogolf-worker-1 current-live] cursor backlog task
  status:    blocked
  !! [error] Agent crashed 2x: pid 100112 not alive

Events (8):
  [..] created {'assignee': 'neurogolf-worker-1', 'status': 'ready', 'parents': []}
  [..] [run 825] claimed {'lock': 'Trick:121156', 'expires': 0, 'run_id': 825}
  [..] [run 825] spawned {'pid': 62664}
  [..] [run 825] crashed {'pid': 62664, 'claimer': 'Trick:121156'}
  [..] [run 831] claimed {'lock': 'Trick:121156'}
  [..] [run 831] spawned {'pid': 100112}
  [..] [run 831] crashed {'pid': 100112, 'claimer': 'Trick:121156'}
  [..] gave_up {'failures': 2, 'effective_limit': 2, 'limit_source': 'dispatcher', 'error': 'pid 100112 not alive'}
"""

_REVIEW_REQUIRED_SHOW = """\
Task t_other: [worker task033 neurogolf-worker-1 current-live] cursor backlog task
  status:    blocked

Latest summary:
review-required: task033 candidate built; needs reviewer.

Comments (1):
  [..] neurogolf-worker-1: review-required handoff:
{"task_id": "task033", "candidate_id": "task033_v1", "scores": {"point_delta": 1.0}}

Events (5):
  [..] blocked {'reason': 'review-required: task033 candidate ready'}
"""

_ITERATION_BUDGET_SHOW = """\
Task t_review: [review task371] v3 const-ones strict-clean local shrink candidate
  status:    blocked

Events (3):
  [..] blocked {'reason': 'Iteration budget exhausted (60/60) — task could not complete within the allowed iterations'}
"""


def test_is_dispatcher_give_up_detects_classic_pattern():
    assert is_dispatcher_give_up(_GAVE_UP_SHOW) is True


def test_is_dispatcher_give_up_skips_review_required_handoff():
    # Card has a real worker handoff — never a give-up.
    assert is_dispatcher_give_up(_REVIEW_REQUIRED_SHOW) is False


def test_is_dispatcher_give_up_skips_iteration_budget_block():
    # LLM session WAS running; it just ran out of iterations.
    assert is_dispatcher_give_up(_ITERATION_BUDGET_SHOW) is False


def test_classify_gave_up_default_action_is_awaiting_session():
    handoff = Handoff(raw_text="", parse_error="no JSON object found")
    verdict = classify(
        _card(assignee="neurogolf-worker-1"),
        handoff,
        block_reason="",
        card_show_text=_GAVE_UP_SHOW,
    )
    assert verdict.action == "awaiting-session"
    assert verdict.gave_up is True
    assert "no LLM session bound" in verdict.reason


def test_classify_gave_up_with_unblock_action_emits_retry_verdict():
    handoff = Handoff(raw_text="", parse_error="no JSON object found")
    verdict = classify(
        _card(assignee="neurogolf-worker-1"),
        handoff,
        block_reason="",
        card_show_text=_GAVE_UP_SHOW,
        gave_up_action="unblock",
    )
    assert verdict.action == "retry-give-up"
    assert verdict.gave_up is True


def test_classify_gave_up_with_abandon_action_emits_abandon_verdict():
    handoff = Handoff(raw_text="", parse_error="no JSON object found")
    verdict = classify(
        _card(assignee="neurogolf-worker-1"),
        handoff,
        block_reason="",
        card_show_text=_GAVE_UP_SHOW,
        gave_up_action="abandon",
    )
    assert verdict.action == "abandon-give-up"
    assert verdict.gave_up is True


# ----- schema-agnostic walker -----------------------------------------------


def test_extract_point_delta_finds_direct_expected_point_gain():
    parsed = {"task_id": "task397", "expected_point_gain": 1.8656}
    pd, src = extract_point_delta(parsed)
    assert pd == 1.8656
    assert "direct" in src


def test_extract_point_delta_finds_score_current_live_and_candidate():
    """Worker-6 task066-style schema."""
    parsed = {
        "task_id": "task066",
        "score_current_live": {"macs": 0, "memory": 16962, "params": 4446, "cost": 21408},
        "score_candidate": {"macs": 0, "memory": 9906, "params": 3636, "cost": 13542},
    }
    pd, src = extract_point_delta(parsed)
    assert pd is not None and pd > 0.4  # ~+0.458
    assert "computed" in src


def test_extract_point_delta_finds_expected_gain_point_gain_nested():
    """Worker-6 task066 also has expected_gain.point_gain — must beat the cost calc."""
    parsed = {
        "task_id": "task066",
        "expected_gain": {"cost_reduction": 7866, "point_gain": 0.4579687168823643},
        "score_current_live": {"cost": 21408},
        "score_candidate": {"cost": 13542},
    }
    pd, src = extract_point_delta(parsed)
    assert pd == 0.4579687168823643
    assert "expected_gain" in src or "direct" in src


def test_extract_point_delta_finds_points_delta_vs_per_task_suffix():
    """Worker-1 task080-style: candidate.points_delta_vs_current_best_task080."""
    parsed = {
        "candidate_id": "task080_v14",
        "base": {"cost": 197573, "memory": 196352, "params": 1221},
        "candidate": {
            "cost": 188453,
            "memory": 187234,
            "params": 1219,
            "delta_cost": 9120,
            "points_delta_vs_current_best_task080": 0.04725949737316881,
        },
    }
    pd, src = extract_point_delta(parsed)
    assert pd == 0.04725949737316881


def test_extract_point_delta_finds_expected_point_gain_vs_suffix():
    """Worker-3 task127-style: selected_repair.expected_point_gain_vs_task127_live.

    Without the (_vs.*)? tail on the expected_point_gain regex, this gain is
    silently dropped and auto_triage falls back to the candidate_id-only path
    with no gain estimate on the resulting review card.
    """
    parsed = {
        "task_id": "task127",
        "selected_repair": {
            "variant": "v2_const_sep_zero_ch0_pad30",
            "cost": 3110,
            "expected_point_gain_vs_task127_live": 0.2631060125544131,
        },
    }
    pd, src = extract_point_delta(parsed)
    assert pd == 0.2631060125544131
    assert "direct" in src


def test_extract_point_delta_handles_top_level_baseline_and_candidate_cost():
    """Worker-5 task373-style: top-level current_cost + candidate_cost."""
    parsed = {
        "task_id": "task373",
        "current_cost": 1067,
        "candidate_cost": 642,
    }
    pd, src = extract_point_delta(parsed)
    assert pd is not None and pd > 0.5  # ~+0.51


def test_extract_point_delta_returns_none_for_truly_empty_handoff():
    parsed = {"task_id": "task???", "notes": "wip"}
    pd, src = extract_point_delta(parsed)
    assert pd is None
    assert src == "no-signal"


def test_classify_review_required_without_delta_escalates_with_candidate_id():
    """Multi-candidate handoffs that don't pick a winner should escalate, not stall."""
    handoff = Handoff(
        raw_text="{}",
        parsed={"task_id": "task388", "candidate": "active8_constant_pad_fp32"},
        candidate_id="active8_constant_pad_fp32",
        task_id="task388",
        point_delta=None,
        expected_point_gain=None,
    )
    verdict = classify(_card(), handoff, "review-required: task388 multi-candidate explored")
    assert verdict.action == "escalate-to-review"
    assert "let reviewer decide" in verdict.reason


def test_classify_truly_empty_handoff_falls_to_manual():
    """Only verdict 'manual' when we have NEITHER candidate_id NOR task_id."""
    handoff = Handoff(
        raw_text="{}",
        parsed={"notes": "nothing"},
        candidate_id="",
        task_id="",
        point_delta=None,
        expected_point_gain=None,
    )
    verdict = classify(_card(), handoff, "review-required: nothing definite")
    assert verdict.action == "manual"


# ----- edge cases from the audit pass --------------------------------------


def test_extract_point_delta_rejects_cost_delta_nested_key():
    """Operator-precedence regression: candidate.cost_delta is NOT a point gain.

    Pre-fix, the boolean was parsed as:
        ("point" in k) OR ("delta" in k AND "cost" not in k)
    Now correctly: ("point" in k OR "delta" in k) AND ("cost" not in k).
    """
    parsed = {
        "task_id": "task001",
        "delta": {"cost_delta": -899, "cost": -899},  # cost field; not a point delta
    }
    pd, src = extract_point_delta(parsed)
    assert pd is None
    assert src == "no-signal"


def test_extract_point_delta_accepts_numeric_string():
    """Legacy entries store expected_point_gain as a string like "1.5"."""
    parsed = {"task_id": "task100", "expected_point_gain": "1.5"}
    pd, src = extract_point_delta(parsed)
    assert pd == 1.5


def test_extract_point_delta_accepts_signed_numeric_string():
    parsed = {"task_id": "task100", "expected_point_gain": "+0.05"}
    pd, src = extract_point_delta(parsed)
    assert pd == 0.05


def test_extract_point_delta_rejects_garbage_string():
    parsed = {"task_id": "task100", "expected_point_gain": "n/a"}
    pd, src = extract_point_delta(parsed)
    # Falls through to cost-based fallback (also missing) → no-signal
    assert pd is None


def test_is_dispatcher_give_up_returns_false_after_successful_unblock():
    """Card had gave_up earlier, then was unblocked and respawned cleanly.

    Auto-triage must NOT re-flag this as awaiting-session because the latest
    event is `spawned`, not `gave_up`. Pre-fix, any historical gave_up in the
    log would match.
    """
    show_text = """\
Task t_test: [worker task042 neurogolf-worker-1 current-live] cursor backlog task
  status:    running

Events (6):
  [2026-05-17 14:54] created {'assignee': 'neurogolf-worker-1', 'status': 'ready'}
  [2026-05-17 14:54] [run 825] spawned {'pid': 62664}
  [2026-05-17 14:55] [run 825] crashed {'pid': 62664}
  [2026-05-17 14:55] gave_up {'failures': 2, 'error': 'pid 62664 not alive'}
  [2026-05-17 16:00] unblocked
  [2026-05-17 16:00] [run 840] spawned {'pid': 5496}

Runs (2):
"""
    assert is_dispatcher_give_up(show_text) is False


def test_is_dispatcher_give_up_still_detects_when_gave_up_is_latest():
    """When the LAST event in the log is gave_up, the verdict is still True."""
    show_text = """\
Task t_test: [worker task050 neurogolf-worker-5 current-live] cursor backlog task
  status:    blocked

Events (4):
  [2026-05-17 14:54] created {'assignee': 'neurogolf-worker-5', 'status': 'ready'}
  [2026-05-17 14:54] [run 829] spawned {'pid': 104608}
  [2026-05-17 14:55] [run 829] crashed {'pid': 104608}
  [2026-05-17 14:55] gave_up {'failures': 2, 'error': 'pid 104608 not alive'}

Runs (1):
"""
    assert is_dispatcher_give_up(show_text) is True


# ----- summary-only fallback (worker forgot the JSON envelope) -------------


def test_reconstruct_handoff_recovers_task238_summary_with_arrow_cost():
    """Real-world summary worker-5 posted on t_895565db (no JSON envelope)."""
    summary = (
        "task238 v1 COMPLETE: chain+identity opt. memory 89802->71606 (-18196), "
        "params 1515->1459 (-56), cost 91317->73065 (-18252), pts +0.2230. "
        "266/266 verify_against_live, 70/70 adversarial. "
        "Staged: .hermes/board/staged_submits/task238_v1_t895565db_cost73065.onnx. "
        "Build: .hermes/task238_v1_t895565db_opt.py"
    )
    card_show = "  status:    blocked\n  assignee:  -\n... assigned to neurogolf-worker-5 who is not in POOL_A_SHRINK_WORKERS..."
    card = {"id": "t_895565db", "assignee": "", "title": "[BACKLOG single task] task238"}
    handoff = reconstruct_handoff_from_summary(card, summary, card_show)
    assert handoff is not None
    assert handoff.task_id == "task238"
    assert handoff.expected_point_gain is not None
    assert abs(handoff.expected_point_gain - 0.2230) < 1e-6
    assert handoff.parsed["scores"]["current_cost"] == 91317
    assert handoff.parsed["scores"]["candidate_cost"] == 73065
    assert handoff.parsed["worker_owner"] == "neurogolf-worker-5"  # recovered from card_show
    assert handoff.parsed["_reconstructed_from"] == "block_reason_summary"
    # risk_notes flags the reconstruction so reviewer knows to verify independently.
    assert any("RECONSTRUCTED" in note for note in handoff.risk_notes)


def test_reconstruct_handoff_recovers_unicode_arrow_cost_pair():
    """The actual t_895565db summary used Unicode → (not ASCII ->)."""
    summary = "task238 v1 cost 91317→73065, pts +0.2230. Staged: foo.onnx"
    handoff = reconstruct_handoff_from_summary({"id": "t_x", "assignee": "neurogolf-worker-5"}, summary, "")
    assert handoff is not None
    assert handoff.parsed["scores"]["current_cost"] == 91317
    assert handoff.parsed["scores"]["candidate_cost"] == 73065
    assert abs(handoff.expected_point_gain - 0.2230) < 1e-6


def test_reconstruct_handoff_classifies_as_escalate_not_manual():
    """End-to-end: a reconstructed handoff above floor must reach escalate-to-review,
    not manual — that's the whole point of the fallback."""
    summary = "task238 v1 cost 91317->73065, pts +0.2230. Staged: foo.onnx"
    card = {"id": "t_895565db", "assignee": "neurogolf-worker-5", "title": "[BACKLOG] task238"}
    handoff = reconstruct_handoff_from_summary(card, summary, "")
    assert handoff is not None
    assert handoff.parse_error == ""  # must not look unparseable to classify()
    verdict = classify(card, handoff, summary)
    assert verdict.action == "escalate-to-review", f"got {verdict.action}: {verdict.reason}"
    assert verdict.new_review_card is not None


def test_classify_no_candidate_built_summary_is_no_fix():
    """Analysis-only block reasons ('no candidate built') must close the card,
    not stall as 'manual'. The worker should be freed."""
    handoff = Handoff(raw_text="", parse_error="no JSON object found")
    block_reason = (
        "task310 analysis: cost 81924 dominated by shifted_rows; no identity "
        "Muls, no large intermediate eliminations possible. No candidate built."
    )
    verdict = classify(_card(assignee=""), handoff, block_reason)
    assert verdict.action == "no-fix"


def test_classify_no_improvement_possible_summary_is_no_fix():
    handoff = Handoff(raw_text="", parse_error="no JSON object found")
    block_reason = "task399 analysis: tried Conv fold and Equal swap; no improvement possible at the current floor."
    verdict = classify(_card(assignee=""), handoff, block_reason)
    assert verdict.action == "no-fix"


def test_reconstruct_handoff_path_traversal_outside_repo_is_blocked():
    """A summary pointing to a staged ONNX outside REPO_ROOT must not compute its SHA."""
    summary = "task001 cost 100->50 pts +0.1 Staged: ../../../etc/passwd.onnx"
    handoff = reconstruct_handoff_from_summary({"id": "t_x"}, summary, "")
    assert handoff is not None
    assert handoff.package_sha == ""  # blocked by repo-containment check


def test_reconstruct_handoff_recovers_vs_style_cost_pair():
    """Summary using `cost (cand vs base)` format."""
    summary = (
        "task133 v5 Conv+ScatterND+Equal-fold saves 55636 cost (105435 vs 161071), "
        "+0.4238 pts, 267/267 verified. Staged: foo.onnx"
    )
    card = {"id": "t_x", "assignee": "neurogolf-worker-7", "title": "[BACKLOG single task] task133"}
    handoff = reconstruct_handoff_from_summary(card, summary, "")
    assert handoff is not None
    assert handoff.task_id == "task133"
    assert abs(handoff.expected_point_gain - 0.4238) < 1e-6
    # vs-form: first number is candidate, second is current
    assert handoff.parsed["scores"]["candidate_cost"] == 105435
    assert handoff.parsed["scores"]["current_cost"] == 161071
    assert handoff.parsed["worker_owner"] == "neurogolf-worker-7"  # from card.assignee


def test_reconstruct_handoff_returns_none_for_signal_free_summary():
    """If the summary has no task_id we can't recover anything safely."""
    assert reconstruct_handoff_from_summary({"id": "t_x"}, "", "") is None
    assert reconstruct_handoff_from_summary({"id": "t_x"}, "nothing useful here", "") is None


def test_reconstruct_handoff_requires_cost_pair_or_staged_path():
    """A task_id alone isn't enough — we need at least costs or a staged ONNX."""
    summary = "task005 something happened pts +0.1"
    # task_id + pts but no costs and no staged path → reject
    assert reconstruct_handoff_from_summary({"id": "t_x"}, summary, "") is None


def test_reconstructed_handoff_review_card_carries_reconstructed_marker():
    """Review cards built from reconstructed handoffs must signal the source."""
    summary = (
        "task238 v1 COMPLETE: cost 91317->73065 (-18252), pts +0.2230. "
        "Staged: .hermes/board/staged_submits/task238_v1.onnx"
    )
    card = {"id": "t_895565db", "assignee": "neurogolf-worker-5", "title": "[BACKLOG] task238"}
    handoff = reconstruct_handoff_from_summary(card, summary, "")
    assert handoff is not None
    spec = build_review_card_spec(card, handoff)
    assert spec["title"].startswith("[review-reconstructed task238")
    assert "RECONSTRUCTED HANDOFF" in spec["body"]
    assert spec["chain_key"].startswith("review:")


# ----- Phase 3 narrow-scope filter (submit_gate owns review-required) -----


class TestCardHasReviewRequired:
    def test_detects_via_title(self):
        card = {"id": "t_a", "title": "blocked: review-required: candidate X"}
        assert _card_has_review_required(card) is True

    def test_detects_via_latest_summary(self):
        card = {"id": "t_a", "title": "x", "latest_summary": "review-required: do the thing"}
        assert _card_has_review_required(card) is True

    def test_detects_via_body(self):
        card = {"id": "t_a", "title": "x", "body": "Worker said review-required for this"}
        assert _card_has_review_required(card) is True

    def test_does_not_detect_when_absent(self):
        card = {"id": "t_a", "title": "ordinary blocked card", "body": "no marker"}
        assert _card_has_review_required(card) is False

    def test_case_insensitive(self):
        card = {"id": "t_a", "title": "REVIEW-REQUIRED: shouted"}
        assert _card_has_review_required(card) is True

    def test_handles_missing_fields(self):
        """A card with NO title / body / latest_summary still must not crash."""
        assert _card_has_review_required({"id": "t_a"}) is False

    def test_handles_non_string_fields(self):
        """Defensive: a kanban card with a non-string field should not crash."""
        assert _card_has_review_required({"title": None, "body": 42}) is False


# ----- Piece 1: stale-unassign branch (session-broken give-up) ------------


def _build_card_show(events_iso: list[tuple[str, str]]) -> str:
    """Build a fake `hermes kanban show` text with the given (ts_iso, kind)
    event list. Mirrors the real format that ``is_session_broken`` parses."""
    lines = ["Events (N):"]
    for ts_iso, kind in events_iso:
        # Format: [YYYY-MM-DD HH:MM] [run N] kind {...}
        ts_short = ts_iso.replace("T", " ")[:16]
        lines.append(f"  [{ts_short}] [run 1] {kind} {{'note': 'x'}}")
    lines.append("\nRuns (1):")
    return "\n".join(lines)


class TestIsSessionBroken:
    def test_no_events_section_returns_false(self):
        assert is_session_broken("no Events here") is False

    def test_recent_give_up_returns_false(self):
        """If the latest event was 5 minutes ago, NOT session-broken."""
        import datetime as dt
        now = dt.datetime(2026, 5, 19, 12, 0, 0, tzinfo=dt.timezone.utc).timestamp()
        five_min_ago = "2026-05-19T11:55"
        card_show = _build_card_show([(five_min_ago, "gave_up")])
        assert is_session_broken(card_show, now_ts=now) is False

    def test_stale_over_1hr_returns_true(self):
        """Last event > 1hr ago → session-broken regardless of pattern."""
        import datetime as dt
        now = dt.datetime(2026, 5, 19, 12, 0, 0, tzinfo=dt.timezone.utc).timestamp()
        two_hrs_ago = "2026-05-19T10:00"
        card_show = _build_card_show([(two_hrs_ago, "gave_up")])
        assert is_session_broken(card_show, now_ts=now) is True

    def test_three_immediate_crashes_returns_true(self):
        """All recent spawns crashed <30s after spawn → session broken
        (even if the latest event was minutes ago, not an hour)."""
        import datetime as dt
        now = dt.datetime(2026, 5, 19, 12, 0, 0, tzinfo=dt.timezone.utc).timestamp()
        events = [
            ("2026-05-19T11:55", "spawned"),
            ("2026-05-19T11:55", "crashed"),
            ("2026-05-19T11:57", "spawned"),
            ("2026-05-19T11:57", "crashed"),
            ("2026-05-19T11:59", "spawned"),
            ("2026-05-19T11:59", "crashed"),
        ]
        card_show = _build_card_show(events)
        assert is_session_broken(card_show, now_ts=now) is True

    def test_long_lived_worker_then_crash_returns_false(self):
        """A spawn that ran for 5+ minutes before crashing isn't session-broken
        — that's a normal worker that hit an issue mid-run."""
        import datetime as dt
        now = dt.datetime(2026, 5, 19, 12, 0, 0, tzinfo=dt.timezone.utc).timestamp()
        events = [
            ("2026-05-19T11:30", "spawned"),
            ("2026-05-19T11:55", "crashed"),  # 25 min runtime
        ]
        card_show = _build_card_show(events)
        assert is_session_broken(card_show, now_ts=now) is False

    def test_custom_stale_threshold(self):
        import datetime as dt
        now = dt.datetime(2026, 5, 19, 12, 0, 0, tzinfo=dt.timezone.utc).timestamp()
        thirty_min_ago = "2026-05-19T11:30"
        card_show = _build_card_show([(thirty_min_ago, "gave_up")])
        # Default 1hr threshold: not stale
        assert is_session_broken(card_show, now_ts=now) is False
        # Custom 15-minute threshold: stale
        assert is_session_broken(card_show, now_ts=now, stale_seconds=900) is True


def _build_give_up_card_show(*, hours_ago: float = 0.1) -> str:
    """Build a card_show that triggers is_dispatcher_give_up. The dispatcher
    pattern requires the LAST event to be 'gave_up' with a 'pid not alive' style
    block reason in the show text."""
    import datetime as dt
    now = dt.datetime.now(dt.timezone.utc)
    ts = now - dt.timedelta(hours=hours_ago)
    ts_short = ts.strftime("%Y-%m-%d %H:%M")
    return (
        "Task t_test: [task100 return] something\n"
        "  status:    blocked\n"
        "  !! [error] Agent crashed 11x: pid 30308 not alive\n"
        "\n"
        f"Events (3):\n"
        f"  [{ts_short}] [run 1] claimed {{'lock': 'X'}}\n"
        f"  [{ts_short}] [run 1] spawned {{'pid': 100}}\n"
        f"  [{ts_short}] [run 1] crashed {{'pid': 100, 'error': 'pid 100 not alive'}}\n"
        f"  [{ts_short}] gave_up {{'failures': 11, 'effective_limit': 2, 'pid': 100}}\n"
        "\n"
        "Runs (1):\n"
    )


def test_classify_stale_unassign_when_session_broken(monkeypatch):
    """When dispatcher gave up AND session looks broken (no recent activity),
    emit stale-unassign instead of awaiting-session.

    Uses neurogolf-worker-1 (Pool A) since the 2026-05-19 Pool B exemption
    means worker-7/8 cards are NOT stale-unassigned by design. The cache is
    monkey-patched to an empty set to ensure no live `hermes profile list`
    call shapes the outcome here."""
    import datetime as dt
    import tools.auto_triage_blocked as m
    monkeypatch.setattr(m, "_POOL_B_PROFILES_CACHE", set())  # no Pool B exemptions
    two_hrs_ago = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=2)).strftime("%Y-%m-%d %H:%M")
    card_show = (
        "Task t_test: [task100 return] x\n  status: blocked\n  !! pid not alive\n\n"
        f"Events (2):\n"
        f"  [{two_hrs_ago}] [run 1] spawned {{'pid': 100}}\n"
        f"  [{two_hrs_ago}] [run 1] crashed {{'pid': 100, 'error': 'pid 100 not alive'}}\n"
        f"  [{two_hrs_ago}] gave_up {{'failures': 11, 'pid': 100}}\n"
        "\nRuns (1):\n"
    )
    card = {"id": "t_test", "assignee": "neurogolf-worker-1", "title": "[task100 return] x"}
    verdict = classify(
        card,
        Handoff(raw_text="", parse_error="empty"),
        block_reason="pid 100 not alive",
        card_show_text=card_show,
        gave_up_action="leave",
    )
    assert verdict.action == "stale-unassign"
    assert "session looks broken" in verdict.reason


class TestIsReturnOrRepairCard:
    def test_repair_bracket_title(self):
        assert is_return_or_repair_card({"title": "[repair task100] x"}) is True

    def test_repair_no_bracket(self):
        assert is_return_or_repair_card({"title": "repair task100 something"}) is True

    def test_opt_return_title(self):
        assert is_return_or_repair_card({"title": "[optimization-return] task100"}) is True

    def test_task_return_title(self):
        """[taskNNN return] format that submit-captain creates."""
        assert is_return_or_repair_card({"title": "[task117 return] crop15 fails parity"}) is True

    def test_worker_card_title_excluded(self):
        assert is_return_or_repair_card({"title": "[worker task100 neurogolf-worker-1] x"}) is False

    def test_backlog_card_title_excluded(self):
        assert is_return_or_repair_card({"title": "[BACKLOG single task] task100 cursor"}) is False

    def test_review_card_title_excluded(self):
        assert is_return_or_repair_card({"title": "[review task100 candidate_v1] x"}) is False

    def test_plain_string(self):
        assert is_return_or_repair_card("[repair task100] x") is True

    def test_missing_title(self):
        assert is_return_or_repair_card({}) is False


class TestCountAttemptsThisPass:
    def test_no_doc(self):
        assert count_attempts_this_pass(None, "task100", current_pass=1) == 0

    def test_missing_task(self):
        assert count_attempts_this_pass({"tasks": {}}, "task100", current_pass=1) == 0

    def test_matching_pass(self):
        doc = {"tasks": {"task100": {"attempts_this_pass": 3, "last_attempt_pass": 5}}}
        assert count_attempts_this_pass(doc, "task100", current_pass=5) == 3

    def test_stale_pass_returns_zero(self):
        doc = {"tasks": {"task100": {"attempts_this_pass": 3, "last_attempt_pass": 4}}}
        assert count_attempts_this_pass(doc, "task100", current_pass=5) == 0


class TestApplyStaleUnassignIntegration:
    """Integration: apply_stale_unassign with mocked run_hermes verifies
    the hermes kanban sequence (comment + assign none + maybe broaden + unblock)."""

    @staticmethod
    def _stub_proc(returncode=0, stdout=b"", stderr=b""):
        from types import SimpleNamespace
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)

    def test_plain_stale_unassign_no_broaden_calls_comment_assign_unblock(self, monkeypatch):
        from tools.auto_triage_blocked import Verdict, apply_stale_unassign

        calls: list[list[str]] = []
        monkeypatch.setattr(
            "tools.auto_triage_blocked.run_hermes",
            lambda args: (calls.append(list(args)) or self._stub_proc()),
        )
        verdict = Verdict(
            card_id="t_aaa",
            assignee="neurogolf-worker-7",
            title="[task100 return] x",
            block_reason="pid not alive",
            handoff=Handoff(raw_text=""),
            action="stale-unassign",
            reason="session broken",
            gave_up=True,
        )
        result = apply_stale_unassign(verdict, task_outcome_doc=None, current_pass=0)

        # Three hermes calls expected: comment, assign, unblock.
        ops = [c[0] for c in calls]
        assert "comment" in ops
        assert "assign" in ops
        assert "unblock" in ops
        # No broaden-scope addendum
        assert result["broaden_scope_applied"] is False
        # The assign call cleared the assignee
        assign_call = next(c for c in calls if c[0] == "assign")
        assert assign_call == ["assign", "t_aaa", "none"]

    def test_stale_unassign_with_broaden_scope_adds_directive_comment(self, monkeypatch):
        """When the card is a return + 3+ attempts this pass, an additional
        comment with the broaden-scope directive should be posted before
        unblock."""
        from tools.auto_triage_blocked import Verdict, apply_stale_unassign

        calls: list[list[str]] = []
        monkeypatch.setattr(
            "tools.auto_triage_blocked.run_hermes",
            lambda args: (calls.append(list(args)) or self._stub_proc()),
        )
        verdict = Verdict(
            card_id="t_aaa",
            assignee="neurogolf-worker-7",
            title="[task117 return] crop15 fails parity",
            block_reason="pid not alive",
            handoff=Handoff(raw_text=""),
            action="stale-unassign",
            reason="session broken",
            gave_up=True,
        )
        doc = {"tasks": {"task117": {
            "attempts_this_pass": 3,
            "attempted_families": ["where-select", "reduce-broadcast", "conv-stamp"],
            "last_attempt_pass": 5,
        }}}
        result = apply_stale_unassign(verdict, task_outcome_doc=doc, current_pass=5)

        assert result["broaden_scope_applied"] is True
        # Two comments expected (audit note + broaden directive), one assign, one unblock
        comment_calls = [c for c in calls if c[0] == "comment"]
        assert len(comment_calls) == 2
        # Broaden directive references the task by ID and lists tried families
        broaden_call = comment_calls[1]
        body = broaden_call[3] if len(broaden_call) > 3 else ""
        assert "task117" in body
        assert "where-select" in body or "auto_triage broaden-scope" in body

    def test_stale_unassign_no_broaden_when_not_return_card(self, monkeypatch):
        """Worker cards don't get broaden-scope even with 3+ attempts."""
        from tools.auto_triage_blocked import Verdict, apply_stale_unassign

        calls: list[list[str]] = []
        monkeypatch.setattr(
            "tools.auto_triage_blocked.run_hermes",
            lambda args: (calls.append(list(args)) or self._stub_proc()),
        )
        verdict = Verdict(
            card_id="t_aaa",
            assignee="neurogolf-worker-1",
            title="[worker task100 neurogolf-worker-1] x",
            block_reason="pid not alive",
            handoff=Handoff(raw_text=""),
            action="stale-unassign",
            reason="session broken",
            gave_up=True,
        )
        doc = {"tasks": {"task100": {"attempts_this_pass": 3, "last_attempt_pass": 5}}}
        result = apply_stale_unassign(verdict, task_outcome_doc=doc, current_pass=5)

        # No broaden directive on worker cards
        assert result["broaden_scope_applied"] is False
        assert len([c for c in calls if c[0] == "comment"]) == 1


class TestShouldBroadenScope:
    def test_yes_when_return_card_with_3_attempts(self):
        card = {"title": "[task117 return] x", "id": "t_a"}
        doc = {"tasks": {"task117": {"attempts_this_pass": 3, "last_attempt_pass": 5}}}
        assert should_broaden_scope(card, doc, current_pass=5) is True

    def test_no_when_worker_card_with_3_attempts(self):
        """Broaden-scope only applies to return/repair cards. Worker cards
        get scope broadening through submit_gate's retry directive."""
        card = {"title": "[worker task100 neurogolf-worker-1] x", "id": "t_a"}
        doc = {"tasks": {"task100": {"attempts_this_pass": 3, "last_attempt_pass": 5}}}
        assert should_broaden_scope(card, doc, current_pass=5) is False

    def test_no_when_return_card_with_2_attempts(self):
        card = {"title": "[task117 return] x", "id": "t_a"}
        doc = {"tasks": {"task117": {"attempts_this_pass": 2, "last_attempt_pass": 5}}}
        assert should_broaden_scope(card, doc, current_pass=5) is False

    def test_no_when_pass_advanced(self):
        """Counter resets across passes — 3 attempts on pass 4 shouldn't
        trigger broaden-scope on pass 5."""
        card = {"title": "[task117 return] x", "id": "t_a"}
        doc = {"tasks": {"task117": {"attempts_this_pass": 3, "last_attempt_pass": 4}}}
        assert should_broaden_scope(card, doc, current_pass=5) is False

    def test_custom_threshold(self):
        card = {"title": "[task117 return] x", "id": "t_a"}
        doc = {"tasks": {"task117": {"attempts_this_pass": 3, "last_attempt_pass": 5}}}
        assert should_broaden_scope(card, doc, current_pass=5, max_attempts_before_broaden=5) is False

    def test_no_when_no_task_id_in_card(self):
        """A return card with no taskNNN in title/body can't be looked up."""
        card = {"title": "[repair something-else] x", "id": "t_a"}
        doc = {"tasks": {}}
        assert should_broaden_scope(card, doc, current_pass=5) is False


def test_classify_pool_b_assignee_does_not_get_stale_unassigned(monkeypatch):
    """Profile-aware exemption (2026-05-19): Pool B (none-claude-code-manual)
    profiles like neurogolf-worker-7 do NOT have a Hermes gateway daemon.
    Dispatcher-give-up + spawn-crash patterns on these cards are EXPECTED
    behavior — the operator drives them via Claude Code /loop and is meant
    to find the blocked card via the playbook's step 2 pickup path. We
    must NOT stale-unassign because that would orphan the manual lane."""
    import datetime as dt
    import tools.auto_triage_blocked as m

    # Force the cache to mark worker-7 as Pool B (don't rely on hermes CLI)
    monkeypatch.setattr(m, "_POOL_B_PROFILES_CACHE", {"neurogolf-worker-7"})

    # Build a card_show that triggers BOTH is_dispatcher_give_up AND
    # is_session_broken (events >1hr stale).
    two_hrs_ago = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=2)).strftime("%Y-%m-%d %H:%M")
    card_show = (
        "Task t_test: BIG_CONV_b follow-up\n  status: blocked\n  !! pid not alive\n\n"
        f"Events (3):\n"
        f"  [{two_hrs_ago}] [run 1] spawned {{'pid': 100}}\n"
        f"  [{two_hrs_ago}] [run 1] crashed {{'pid': 100, 'error': 'pid 100 not alive'}}\n"
        f"  [{two_hrs_ago}] gave_up {{'failures': 2, 'pid': 100}}\n"
        "\nRuns (1):\n"
    )
    card = {"id": "t_test", "assignee": "neurogolf-worker-7", "title": "BIG_CONV_b follow-up"}

    verdict = classify(
        card,
        Handoff(raw_text="", parse_error="empty"),
        block_reason="pid 100 not alive",
        card_show_text=card_show,
        gave_up_action="leave",
    )

    # MUST NOT be stale-unassign — Pool B exemption applies.
    assert verdict.action != "stale-unassign"
    # Falls through to the existing leave/awaiting-session verdict instead.
    assert verdict.action == "awaiting-session"


def test_classify_non_pool_b_assignee_still_stale_unassigns(monkeypatch):
    """Mirror of the Pool B exemption: a regular worker profile (not in
    the manual list) still gets stale-unassigned per the original Piece 1
    behavior. This guards against the fix accidentally exempting everyone."""
    import datetime as dt
    import tools.auto_triage_blocked as m

    # Cache says ONLY worker-7 is Pool B; worker-1 is NOT
    monkeypatch.setattr(m, "_POOL_B_PROFILES_CACHE", {"neurogolf-worker-7"})

    two_hrs_ago = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=2)).strftime("%Y-%m-%d %H:%M")
    card_show = (
        "Task t_test: [task100 return] x\n  status: blocked\n  !! pid not alive\n\n"
        f"Events (3):\n"
        f"  [{two_hrs_ago}] [run 1] spawned {{'pid': 100}}\n"
        f"  [{two_hrs_ago}] [run 1] crashed {{'pid': 100, 'error': 'pid 100 not alive'}}\n"
        f"  [{two_hrs_ago}] gave_up {{'failures': 2, 'pid': 100}}\n"
        "\nRuns (1):\n"
    )
    card = {"id": "t_test", "assignee": "neurogolf-worker-1", "title": "[task100 return] x"}

    verdict = classify(
        card,
        Handoff(raw_text="", parse_error="empty"),
        block_reason="pid 100 not alive",
        card_show_text=card_show,
        gave_up_action="leave",
    )

    # Pool A worker → stale-unassign STILL fires
    assert verdict.action == "stale-unassign"


def test_classify_leave_when_give_up_but_recent_activity():
    """A give-up with recent (within last hour) spawn/crash activity should
    still get the old awaiting-session verdict — don't over-eagerly unassign."""
    import datetime as dt
    five_min_ago = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M")
    card_show = (
        "Task t_test: [task100 return] x\n  status: blocked\n  !! pid not alive\n\n"
        f"Events (2):\n"
        f"  [{five_min_ago}] [run 1] claimed {{}}\n"
        f"  [{five_min_ago}] [run 1] spawned {{'pid': 100}}\n"
        f"  [{five_min_ago}] [run 1] crashed {{'pid': 100, 'error': 'pid 100 not alive'}}\n"
        f"  [{five_min_ago}] gave_up {{'failures': 2, 'pid': 100}}\n"
        "\nRuns (1):\n"
    )
    card = {"id": "t_test", "assignee": "neurogolf-worker-7", "title": "[task100 return] x"}
    verdict = classify(
        card,
        Handoff(raw_text="", parse_error="empty"),
        block_reason="pid 100 not alive",
        card_show_text=card_show,
        gave_up_action="leave",
    )
    assert verdict.action == "awaiting-session"
