"""Tests for tools/submit_gate_mechanical.py.

Synthesize minimal ONNX models + zip files in-memory so each check runs
against deterministic artifacts. The slow score_v2 / verify_against_live
checks are tested with stub callables that simulate pass/fail outcomes.
"""
from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path

import onnx
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.auto_triage_blocked import Handoff
from tools.submit_gate_mechanical import (
    MechanicalCheckInput,
    check_package_structure,
    check_raw_shape_parity,
    check_score_reproduces,
    check_verify_against_live,
    check_zip_exists,
    check_zip_sha,
    run_mechanical_checks,
)


# ---------------------------------------------------------------------------
# Helpers to build synthetic ONNX / zip artifacts
# ---------------------------------------------------------------------------


def _make_minimal_onnx(output_shape: tuple[int, ...]) -> bytes:
    """Return a serialized ONNX model whose first output has the given shape.

    The model just maps input → output through an Identity node — enough to
    declare the shape for the parity check without invoking any heavy op.
    """
    input_dims = [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 10, 30, 30])]
    output_dims = [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, list(output_shape))]
    node = onnx.helper.make_node("Identity", inputs=["input"], outputs=["output"])
    graph = onnx.helper.make_graph([node], "g", input_dims, output_dims)
    model = onnx.helper.make_model(graph)
    return model.SerializeToString()


_make_zip_counter = 0


def _make_zip(tmp_path: Path, entries: dict[str, bytes]) -> Path:
    """Build a zip with the given {name: payload} entries.

    Uses a process-wide counter for uniqueness — ``id(entries)`` collides when
    the previous dict has been garbage-collected and its memory address is
    reused for the next one, which silently makes two distinct ``_make_zip``
    calls write to the same path (one overwrites the other).
    """
    global _make_zip_counter
    _make_zip_counter += 1
    zip_path = tmp_path / f"test_{_make_zip_counter}.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for name, payload in entries.items():
            zf.writestr(name, payload)
    return zip_path


def _handoff(
    *,
    task_id: str = "task100",
    candidate_id: str = "task100_v1",
    expected_point_gain: float = 0.5,
    algorithm_family: str = "where-select",
    package_sha: str = "",
    changed_entries: list[str] | None = None,
    parsed_scores: dict | None = None,
) -> Handoff:
    parsed = {}
    if parsed_scores is not None:
        parsed["scores"] = parsed_scores
    return Handoff(
        raw_text="{}",
        parsed=parsed,
        candidate_id=candidate_id,
        task_id=task_id,
        point_delta=expected_point_gain,
        expected_point_gain=expected_point_gain,
        package_sha=package_sha,
        lineage_base="current-best stub",
        changed_entries=changed_entries or [f"{task_id}.onnx"],
        algorithm_family=algorithm_family,
    )


# ---------------------------------------------------------------------------
# check_zip_exists
# ---------------------------------------------------------------------------


def test_check_zip_exists_passes_for_real_file(tmp_path):
    zip_path = _make_zip(tmp_path, {"task100.onnx": b"x"})
    assert check_zip_exists(zip_path) == []


def test_check_zip_exists_fails_for_missing_file(tmp_path):
    result = check_zip_exists(tmp_path / "nonexistent.zip")
    assert len(result) == 1
    assert "not found" in result[0]


def test_check_zip_exists_fails_for_directory(tmp_path):
    """Defensive: someone points us at a dir instead of a file."""
    (tmp_path / "subdir").mkdir()
    result = check_zip_exists(tmp_path / "subdir")
    assert len(result) == 1
    assert "not a file" in result[0]


# ---------------------------------------------------------------------------
# check_zip_sha
# ---------------------------------------------------------------------------


def test_check_zip_sha_passes_when_match(tmp_path):
    payload = b"some-deterministic-bytes"
    zip_path = tmp_path / "test.zip"
    zip_path.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    assert check_zip_sha(zip_path, expected) == []


def test_check_zip_sha_case_insensitive(tmp_path):
    """Workers might report uppercase SHA; our compute is lowercase. Match anyway."""
    payload = b"data"
    zip_path = tmp_path / "test.zip"
    zip_path.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest().upper()
    assert check_zip_sha(zip_path, expected) == []


def test_check_zip_sha_fails_when_mismatch(tmp_path):
    zip_path = tmp_path / "test.zip"
    zip_path.write_bytes(b"actual")
    result = check_zip_sha(zip_path, "f" * 64)
    assert len(result) == 1
    assert "package_sha256 mismatch" in result[0]


def test_check_zip_sha_fails_when_handoff_missing_sha(tmp_path):
    zip_path = tmp_path / "test.zip"
    zip_path.write_bytes(b"x")
    result = check_zip_sha(zip_path, "")
    assert len(result) == 1
    assert "missing package_sha256" in result[0]


def test_check_zip_sha_does_not_double_report_when_zip_missing(tmp_path):
    """check_zip_exists already reports missing-zip; check_zip_sha should
    quietly return [] in that case rather than adding 'mismatch' noise."""
    result = check_zip_sha(tmp_path / "nonexistent.zip", "f" * 64)
    assert result == []


# ---------------------------------------------------------------------------
# check_package_structure
# ---------------------------------------------------------------------------


def test_package_structure_passes_when_only_declared_entries_differ(tmp_path):
    base = _make_zip(tmp_path, {
        "task100.onnx": b"OLD",
        "task101.onnx": b"unchanged_101",
        "task102.onnx": b"unchanged_102",
    })
    candidate = _make_zip(tmp_path, {
        "task100.onnx": b"NEW",  # the only change
        "task101.onnx": b"unchanged_101",
        "task102.onnx": b"unchanged_102",
    })
    assert check_package_structure(candidate, base, ["task100.onnx"]) == []


def test_package_structure_fails_when_undeclared_entry_also_differs(tmp_path):
    """The task275 trap: zip differs on tasks not declared in changed_package_entries."""
    base = _make_zip(tmp_path, {
        "task100.onnx": b"OLD",
        "task101.onnx": b"OLD_101",
    })
    candidate = _make_zip(tmp_path, {
        "task100.onnx": b"NEW",
        "task101.onnx": b"DIFFERENT_101",  # also changed but not declared!
    })
    result = check_package_structure(candidate, base, ["task100.onnx"])
    assert len(result) == 1
    assert "task101.onnx" in result[0]
    assert "not declared" in result[0]


def test_package_structure_fails_when_declared_entry_missing_from_zip(tmp_path):
    candidate = _make_zip(tmp_path, {"task100.onnx": b"x"})
    base = _make_zip(tmp_path, {"task100.onnx": b"x", "task101.onnx": b"y"})
    result = check_package_structure(candidate, base, ["task100.onnx", "task101.onnx"])
    assert len(result) == 1
    assert "task101.onnx" in result[0]
    assert "missing from zip" in result[0]


def test_package_structure_skipped_when_no_current_best(tmp_path):
    """First-ever submission has no current-best to diff against — pass."""
    candidate = _make_zip(tmp_path, {"task100.onnx": b"x"})
    assert check_package_structure(candidate, None, ["task100.onnx"]) == []


def test_package_structure_fails_when_changed_entries_empty(tmp_path):
    """A submit with no declared changes is suspect — refuse to verify."""
    candidate = _make_zip(tmp_path, {"task100.onnx": b"x"})
    base = _make_zip(tmp_path, {"task100.onnx": b"x"})
    result = check_package_structure(candidate, base, [])
    assert len(result) == 1
    assert "missing changed_package_entries" in result[0]


def test_package_structure_fails_when_zip_corrupt(tmp_path):
    bad_zip = tmp_path / "corrupt.zip"
    bad_zip.write_bytes(b"NOT A ZIP")
    base = _make_zip(tmp_path, {"task100.onnx": b"x"})
    result = check_package_structure(bad_zip, base, ["task100.onnx"])
    assert len(result) == 1
    assert "unreadable" in result[0]


# ---------------------------------------------------------------------------
# check_raw_shape_parity (the task127 lesson)
# ---------------------------------------------------------------------------


def test_raw_shape_parity_passes_when_shapes_match(tmp_path):
    onnx_bytes = _make_minimal_onnx((1, 10, 30, 30))
    candidate = _make_zip(tmp_path, {"task100.onnx": onnx_bytes})
    base = _make_zip(tmp_path, {"task100.onnx": onnx_bytes})
    assert check_raw_shape_parity(candidate, base, "task100") == []


def test_raw_shape_parity_fails_on_task127_style_divergence(tmp_path):
    """The exact pattern that cost 16.69 LB pts: candidate output shape
    [1,10,7,11] vs live [1,10,30,30]."""
    candidate_onnx = _make_minimal_onnx((1, 10, 7, 11))
    live_onnx = _make_minimal_onnx((1, 10, 30, 30))
    candidate = _make_zip(tmp_path, {"task127.onnx": candidate_onnx})
    base = _make_zip(tmp_path, {"task127.onnx": live_onnx})
    result = check_raw_shape_parity(candidate, base, "task127")
    assert len(result) == 1
    assert "task127" in result[0]
    assert "divergence" in result[0]
    assert "task127 lesson" in result[0]


def test_raw_shape_parity_skipped_when_no_current_best(tmp_path):
    onnx_bytes = _make_minimal_onnx((1, 10, 30, 30))
    candidate = _make_zip(tmp_path, {"task100.onnx": onnx_bytes})
    assert check_raw_shape_parity(candidate, None, "task100") == []


def test_raw_shape_parity_skipped_when_live_missing_entry(tmp_path):
    """If the live zip doesn't have the task entry (new task addition?),
    warn but don't block."""
    candidate_onnx = _make_minimal_onnx((1, 10, 30, 30))
    candidate = _make_zip(tmp_path, {"task999.onnx": candidate_onnx})
    base = _make_zip(tmp_path, {"task001.onnx": _make_minimal_onnx((1, 10, 30, 30))})
    result = check_raw_shape_parity(candidate, base, "task999")
    assert len(result) == 1
    assert "proceeding without parity check" in result[0]


def test_raw_shape_parity_handles_no_pad_12x12(tmp_path):
    """The task019 case from our session: bool 12x12 candidate vs live 30x30."""
    candidate_onnx = _make_minimal_onnx((1, 10, 12, 12))
    live_onnx = _make_minimal_onnx((1, 10, 30, 30))
    candidate = _make_zip(tmp_path, {"task019.onnx": candidate_onnx})
    base = _make_zip(tmp_path, {"task019.onnx": live_onnx})
    result = check_raw_shape_parity(candidate, base, "task019")
    assert len(result) == 1
    assert "[1, 10, 12, 12]" in result[0] or "1, 10, 12, 12" in result[0]


# ---------------------------------------------------------------------------
# check_score_reproduces
# ---------------------------------------------------------------------------


def test_score_reproduces_passes_when_costs_match(tmp_path):
    candidate = _make_zip(tmp_path, {"task100.onnx": _make_minimal_onnx((1, 10, 30, 30))})
    # Stub score function returning a fixed result: macs=0, memory=12000, params=345 → cost 12345
    stub_score = lambda path: (0, 12000, 345)
    result = check_score_reproduces(candidate, "task100", expected_cost=12345, score_fn=stub_score)
    assert result == []


def test_score_reproduces_fails_when_costs_disagree(tmp_path):
    candidate = _make_zip(tmp_path, {"task100.onnx": _make_minimal_onnx((1, 10, 30, 30))})
    stub_score = lambda path: (0, 12000, 345)  # reports 12345
    result = check_score_reproduces(candidate, "task100", expected_cost=99999, score_fn=stub_score)
    assert len(result) == 1
    assert "score_v2 reproduce disagrees" in result[0]
    assert "99999" in result[0]
    assert "12345" in result[0]


def test_score_reproduces_within_tolerance(tmp_path):
    candidate = _make_zip(tmp_path, {"task100.onnx": _make_minimal_onnx((1, 10, 30, 30))})
    stub_score = lambda path: (0, 12001, 345)  # reports 12346, expected 12345 → diff 1
    result = check_score_reproduces(candidate, "task100", expected_cost=12345, score_fn=stub_score, tolerance=2)
    assert result == []


def test_score_reproduces_outside_tolerance(tmp_path):
    candidate = _make_zip(tmp_path, {"task100.onnx": _make_minimal_onnx((1, 10, 30, 30))})
    stub_score = lambda path: (0, 12010, 345)  # reports 12355, expected 12345 → diff 10
    result = check_score_reproduces(candidate, "task100", expected_cost=12345, score_fn=stub_score, tolerance=2)
    assert len(result) == 1


def test_score_reproduces_handles_score_fn_returning_none(tmp_path):
    candidate = _make_zip(tmp_path, {"task100.onnx": _make_minimal_onnx((1, 10, 30, 30))})
    stub_score = lambda path: None  # ONNX invalid per score_network_v2
    result = check_score_reproduces(candidate, "task100", expected_cost=12345, score_fn=stub_score)
    assert len(result) == 1
    assert "score_v2 returned None" in result[0]


def test_score_reproduces_skipped_when_expected_cost_missing(tmp_path):
    """If the handoff didn't include candidate_cost, we have nothing to verify against."""
    candidate = _make_zip(tmp_path, {"task100.onnx": _make_minimal_onnx((1, 10, 30, 30))})
    stub_score = lambda path: pytest.fail("score_fn should not be called")  # would raise if called
    result = check_score_reproduces(candidate, "task100", expected_cost=None, score_fn=stub_score)
    assert result == []


# ---------------------------------------------------------------------------
# check_verify_against_live
# ---------------------------------------------------------------------------


def test_verify_against_live_passes_when_full_match(tmp_path):
    candidate = _make_zip(tmp_path, {"task100.onnx": _make_minimal_onnx((1, 10, 30, 30))})
    stub_verify = lambda task_id, path: (267, 267, 0.0)
    result = check_verify_against_live(candidate, None, "task100", verify_fn=stub_verify)
    assert result == []


def test_verify_against_live_fails_on_partial_match(tmp_path):
    candidate = _make_zip(tmp_path, {"task100.onnx": _make_minimal_onnx((1, 10, 30, 30))})
    stub_verify = lambda task_id, path: (260, 267, 1e-5)
    result = check_verify_against_live(candidate, None, "task100", verify_fn=stub_verify)
    assert len(result) == 1
    assert "260/267" in result[0]


def test_verify_against_live_accepts_partial_when_not_required(tmp_path):
    candidate = _make_zip(tmp_path, {"task100.onnx": _make_minimal_onnx((1, 10, 30, 30))})
    stub_verify = lambda task_id, path: (265, 267, 1e-7)
    result = check_verify_against_live(
        candidate, None, "task100", verify_fn=stub_verify, require_full_pass=False,
    )
    assert result == []


def test_verify_against_live_fails_when_zero_pairs(tmp_path):
    candidate = _make_zip(tmp_path, {"task100.onnx": _make_minimal_onnx((1, 10, 30, 30))})
    stub_verify = lambda task_id, path: (0, 0, 0.0)
    result = check_verify_against_live(candidate, None, "task100", verify_fn=stub_verify)
    assert len(result) == 1
    assert "0 pairs" in result[0]


def test_verify_against_live_skipped_when_no_verify_fn(tmp_path):
    candidate = _make_zip(tmp_path, {"task100.onnx": _make_minimal_onnx((1, 10, 30, 30))})
    result = check_verify_against_live(candidate, None, "task100", verify_fn=None)
    assert result == []


# ---------------------------------------------------------------------------
# run_mechanical_checks (orchestrator)
# ---------------------------------------------------------------------------


def test_run_mechanical_checks_passes_when_everything_clean(tmp_path):
    onnx_bytes = _make_minimal_onnx((1, 10, 30, 30))
    # Base must hold a structurally-valid ONNX whose shape matches the
    # candidate's (otherwise raw-shape parity correctly fails). Different
    # bytes (so package_structure correctly detects a change) but same
    # declared output shape.
    base_onnx_bytes = _make_minimal_onnx((1, 10, 30, 30))
    candidate = _make_zip(tmp_path, {"task100.onnx": onnx_bytes})
    base = _make_zip(tmp_path, {"task100.onnx": base_onnx_bytes})
    sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
    handoff = _handoff(
        task_id="task100",
        package_sha=sha,
        changed_entries=["task100.onnx"],
        parsed_scores={"candidate_cost": 12345},
    )
    inp = MechanicalCheckInput(
        handoff=handoff,
        candidate_zip_path=candidate,
        current_best_zip_path=base,
        task_id="task100",
    )
    stub_score = lambda path: (0, 12000, 345)  # → 12345
    stub_verify = lambda task_id, path: (267, 267, 0.0)
    result = run_mechanical_checks(inp, score_fn=stub_score, verify_fn=stub_verify)
    assert result.ok is True
    assert result.failures == []


def test_run_mechanical_checks_aggregates_all_failures(tmp_path):
    """Multiple problems in one handoff should ALL surface — don't fail-fast."""
    candidate_onnx = _make_minimal_onnx((1, 10, 7, 11))  # wrong shape
    live_onnx = _make_minimal_onnx((1, 10, 30, 30))
    candidate = _make_zip(tmp_path, {
        "task100.onnx": candidate_onnx,
        "task101.onnx": b"undeclared_change",  # extra mutation
    })
    base = _make_zip(tmp_path, {
        "task100.onnx": live_onnx,
        "task101.onnx": b"original_101",
    })
    handoff = _handoff(
        task_id="task100",
        package_sha="f" * 64,  # wrong SHA
        changed_entries=["task100.onnx"],
        parsed_scores={"candidate_cost": 99999},
    )
    inp = MechanicalCheckInput(
        handoff=handoff,
        candidate_zip_path=candidate,
        current_best_zip_path=base,
        task_id="task100",
    )
    stub_score = lambda path: (0, 12000, 345)
    result = run_mechanical_checks(inp, score_fn=stub_score)
    assert result.ok is False
    # At least 4 distinct failures: SHA, package structure, raw-shape, score
    assert len(result.failures) >= 4
    all_failures = " ".join(result.failures)
    assert "package_sha256 mismatch" in all_failures
    assert "task101.onnx" in all_failures  # undeclared change
    assert "divergence" in all_failures
    assert "score_v2 reproduce disagrees" in all_failures


def test_run_mechanical_checks_no_score_fn_skips_score_check(tmp_path):
    onnx_bytes = _make_minimal_onnx((1, 10, 30, 30))
    candidate = _make_zip(tmp_path, {"task100.onnx": onnx_bytes})
    sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
    handoff = _handoff(task_id="task100", package_sha=sha)
    inp = MechanicalCheckInput(
        handoff=handoff,
        candidate_zip_path=candidate,
        current_best_zip_path=None,
        task_id="task100",
    )
    result = run_mechanical_checks(inp, score_fn=None, verify_fn=None)
    assert result.ok is True


def test_run_mechanical_checks_score_check_uses_handoff_scores_block(tmp_path):
    """Score expected_cost comes from handoff.parsed['scores']['candidate_cost']."""
    onnx_bytes = _make_minimal_onnx((1, 10, 30, 30))
    candidate = _make_zip(tmp_path, {"task100.onnx": onnx_bytes})
    sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
    handoff = _handoff(
        task_id="task100",
        package_sha=sha,
        parsed_scores={"candidate_cost": 50000},
    )
    inp = MechanicalCheckInput(
        handoff=handoff,
        candidate_zip_path=candidate,
        current_best_zip_path=None,
        task_id="task100",
    )
    stub_score = lambda path: (0, 12000, 345)  # → 12345, far from 50000
    result = run_mechanical_checks(inp, score_fn=stub_score)
    assert result.ok is False
    assert any("score_v2 reproduce disagrees" in f for f in result.failures)


def test_run_mechanical_checks_score_check_handles_candidate_block_shape(tmp_path):
    """Alternative handoff shape: candidate.cost instead of scores.candidate_cost.
    (Worker-3's task275 handoff used this shape.)"""
    onnx_bytes = _make_minimal_onnx((1, 10, 30, 30))
    candidate = _make_zip(tmp_path, {"task100.onnx": onnx_bytes})
    sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
    handoff = _handoff(task_id="task100", package_sha=sha)
    handoff.parsed = {"candidate": {"cost": 12345}}
    inp = MechanicalCheckInput(
        handoff=handoff,
        candidate_zip_path=candidate,
        current_best_zip_path=None,
        task_id="task100",
    )
    stub_score = lambda path: (0, 12000, 345)  # → 12345 matches
    result = run_mechanical_checks(inp, score_fn=stub_score)
    assert result.ok is True


def test_run_mechanical_checks_zip_missing_fails_cleanly(tmp_path):
    """If the worker referenced a non-existent zip, we should fail cleanly
    rather than crash deep in score_fn."""
    handoff = _handoff(task_id="task100", package_sha="f" * 64)
    inp = MechanicalCheckInput(
        handoff=handoff,
        candidate_zip_path=tmp_path / "missing.zip",
        current_best_zip_path=None,
        task_id="task100",
    )
    stub_score = lambda path: pytest.fail("score_fn should not be reached")
    result = run_mechanical_checks(inp, score_fn=stub_score)
    assert result.ok is False
    assert any("not found" in f for f in result.failures)
