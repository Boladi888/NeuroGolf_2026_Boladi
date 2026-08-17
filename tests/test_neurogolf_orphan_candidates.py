"""Tests for the orphan-candidate detector + promotion helper.

The detector finds candidates that are reviewer-approved in
``candidate_registry.json`` (``submit_queue_status: ready``) but have no
matching ``candidate_id`` in ``submit_queue.json::queue``. The promoter
reconstructs the missing queue entries from the registry data, with
safety guards on package existence + SHA + risk flags.
"""
from __future__ import annotations

import json
import zipfile
from hashlib import sha256
from pathlib import Path

import pytest

from tools.neurogolf_orphan_candidates import build_report, find_orphans
from tools.neurogolf_promote_orphans import (
    apply_promotions,
    plan_promotions,
    reconstruct_queue_entry,
    validate_orphan,
)


def _reg_entry(cid, *, status="ready", task_id="task011", pkg_path="", pkg_sha="", ev=1.0):
    return {
        "candidate_id": cid,
        "task_id": task_id,
        "submit_queue_status": status,
        "worker_owner": "neurogolf-worker-4",
        "expected_point_gain": ev,
        "priority_score": ev,
        "package_path": pkg_path,
        "package_sha256": pkg_sha,
        "lineage_base_candidate_id": "current-best",
        "lineage_base_score": 5750.0,
        "risk_flags": [],
        "reviewer_disposition": "approved-for-submit-captain",
        "approval_mode": "auto-rate-limited",
        "approval_packet": {
            "candidate_id": cid,
            "task_id": task_id,
            "package_path": pkg_path,
            "package_sha256": pkg_sha,
        },
    }


def _queue_entry(cid, *, status="ready"):
    return {"candidate_id": cid, "status": status}


@pytest.fixture
def good_zip(tmp_path: Path) -> tuple[Path, str]:
    """submission.zip with 401 ONNX entries; returns (path, sha)."""
    path = tmp_path / "submission.zip"
    with zipfile.ZipFile(path, "w") as z:
        for i in range(401):
            z.writestr(f"task{i:03d}.onnx", b"\x00" * 100)
    h = sha256(path.read_bytes()).hexdigest()
    return path, h


# --- find_orphans ----------------------------------------------------------


def test_find_orphans_detects_registry_only_ready():
    reg = {"entries": [
        _reg_entry("cand_orphan", task_id="task388", ev=3.27),
        _reg_entry("cand_in_queue", task_id="task011"),
    ]}
    queue = {"queue": [_queue_entry("cand_in_queue")]}
    orphans = find_orphans(reg, queue)
    assert len(orphans) == 1
    assert orphans[0]["candidate_id"] == "cand_orphan"
    assert orphans[0]["task_id"] == "task388"
    assert orphans[0]["expected_point_gain"] == 3.27


def test_find_orphans_skips_non_ready_registry_entries():
    reg = {"entries": [
        _reg_entry("done_cand", status="submitted-complete-promote-synced"),
        _reg_entry("returned_cand", status="returned-to-worker-after-review"),
    ]}
    queue = {"queue": []}
    assert find_orphans(reg, queue) == []


def test_find_orphans_sorts_by_ev_descending():
    reg = {"entries": [
        _reg_entry("a", ev=0.5),
        _reg_entry("b", ev=3.0),
        _reg_entry("c", ev=1.0),
    ]}
    queue = {"queue": []}
    out = find_orphans(reg, queue)
    assert [o["candidate_id"] for o in out] == ["b", "c", "a"]


def test_find_orphans_flags_worktree_packages():
    reg = {"entries": [
        _reg_entry(
            "cand",
            pkg_path="<REPO_ROOT>/.worktrees/t_x/.hermes/submission.zip",
        ),
    ]}
    queue = {"queue": []}
    out = find_orphans(reg, queue)
    assert out[0]["package_in_worktree"] is True
    assert out[0]["package_in_staged_candidates"] is False


def test_find_orphans_handles_empty_inputs():
    assert find_orphans(None, None) == []
    assert find_orphans({}, {}) == []
    assert find_orphans({"entries": []}, {"queue": []}) == []


def test_build_report_top_level_shape():
    reg = {"entries": [_reg_entry("a", ev=3.27), _reg_entry("b", ev=0.5)]}
    queue = {"queue": []}
    report = build_report(reg, queue)
    assert report["schema"] == "neurogolf.orphan_candidates.v1"
    assert report["orphan_count"] == 2
    assert report["total_orphan_ev"] == 3.77
    assert report["top_orphan_ev"] == 3.27
    assert len(report["orphans"]) == 2


# --- validate_orphan -------------------------------------------------------


def test_validate_orphan_passes_for_good_full_package(good_zip):
    path, sha = good_zip
    orphan = {"package_path": str(path), "package_sha256": sha, "task_id": "task004"}
    result = validate_orphan(orphan)
    assert result["ok"] is True
    assert result["shape"] == "full"


def test_validate_orphan_passes_for_delta_package(tmp_path):
    """Some staged packages are deltas (single-onnx) that submit-captain
    merges onto the canonical baseline. Observed for task388, task379,
    task062. The single entry must match the task_id."""
    path = tmp_path / "delta.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("task388.onnx", b"\x00" * 200)
    sha = sha256(path.read_bytes()).hexdigest()
    orphan = {"package_path": str(path), "package_sha256": sha, "task_id": "task388"}
    result = validate_orphan(orphan)
    assert result["ok"] is True
    assert result["shape"] == "delta"


def test_validate_orphan_rejects_delta_with_wrong_task_id(tmp_path):
    path = tmp_path / "bad_delta.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("task010.onnx", b"\x00")  # task_id mismatch
    sha = sha256(path.read_bytes()).hexdigest()
    orphan = {"package_path": str(path), "package_sha256": sha, "task_id": "task388"}
    result = validate_orphan(orphan)
    assert result["ok"] is False
    assert any("does not match task_id" in m for m in result["messages"])


def test_validate_orphan_rejects_missing_file(tmp_path):
    orphan = {"package_path": str(tmp_path / "missing.zip"), "package_sha256": ""}
    result = validate_orphan(orphan)
    assert result["ok"] is False
    assert "not on disk" in result["messages"][0]


def test_validate_orphan_rejects_sha_mismatch(good_zip):
    path, _ = good_zip
    orphan = {"package_path": str(path), "package_sha256": "deadbeef" * 8}
    result = validate_orphan(orphan)
    assert result["ok"] is False
    assert any("SHA mismatch" in m for m in result["messages"])


def test_validate_orphan_rejects_bad_zip(tmp_path):
    bad = tmp_path / "not_a_zip.zip"
    bad.write_bytes(b"garbage")
    orphan = {"package_path": str(bad)}
    result = validate_orphan(orphan)
    assert result["ok"] is False


def test_validate_orphan_rejects_wrong_entry_count(tmp_path):
    small = tmp_path / "small.zip"
    with zipfile.ZipFile(small, "w") as z:
        for i in range(50):
            z.writestr(f"task{i:03d}.onnx", b"\x00")
    sha = sha256(small.read_bytes()).hexdigest()
    orphan = {"package_path": str(small), "package_sha256": sha}
    result = validate_orphan(orphan)
    assert result["ok"] is False
    assert any("entries" in m for m in result["messages"])


# --- reconstruct_queue_entry ----------------------------------------------


def test_reconstruct_queue_entry_carries_core_fields():
    orphan = {
        "candidate_id": "task388_cand_t_xx",
        "task_id": "task388",
        "expected_point_gain": 3.27,
        "package_path": "/tmp/staged/submission.zip",
        "package_sha256": "abc" * 21,
        "lineage_base_candidate_id": "current-best",
        "worker_owner": "neurogolf-worker-4",
        "risk_flags": ["public-test-only-non-structural"],
        "reviewer_disposition": "approved-for-submit-captain",
        "approval_mode": "auto-rate-limited",
    }
    entry = reconstruct_queue_entry(orphan)
    assert entry["candidate_id"] == "task388_cand_t_xx"
    assert entry["task_id"] == "task388"
    assert entry["status"] == "ready"
    assert entry["priority_score"] == 3.27
    assert entry["package_path"] == "/tmp/staged/submission.zip"
    assert entry["reviewer_disposition"] == "approved-for-submit-captain"
    assert "promoted-from-orphan" in " ".join(entry["risk_flags"])
    # The promoted-from-orphan provenance lets future detectors see this
    # was reconstructed rather than reviewer-approved freshly.
    assert "promoted_from_orphan" in entry
    assert entry["promoted_from_orphan"]["promoted_by"].endswith("neurogolf_promote_orphans.py")


def test_reconstruct_queue_entry_generates_unique_queue_ids():
    orphan = {"candidate_id": "a", "expected_point_gain": 0.5}
    e1 = reconstruct_queue_entry(orphan, now_iso="2026-05-18T06:00:00Z")
    e2 = reconstruct_queue_entry({"candidate_id": "b", "expected_point_gain": 0.5})
    assert e1["queue_id"] != e2["queue_id"]
    assert e1["queue_id"].startswith("submit-")


# --- plan_promotions -------------------------------------------------------


def test_plan_promotions_skips_already_in_queue(good_zip):
    path, sha = good_zip
    orphan_report = {"orphans": [
        {"candidate_id": "dup", "task_id": "task001", "expected_point_gain": 1.0,
         "package_path": str(path), "package_sha256": sha,
         "package_in_staged_candidates": True, "package_in_worktree": False,
         "risk_flags": []},
    ]}
    queue = {"queue": [_queue_entry("dup")]}
    plan = plan_promotions(orphan_report=orphan_report, registry_doc={}, queue_doc=queue)
    assert plan["promote_count"] == 0
    assert plan["skip_count"] == 1
    assert "already in submit_queue" in plan["decisions"][0]["reasons"][0]


def test_plan_promotions_skips_excluded_task(good_zip):
    path, sha = good_zip
    orphan_report = {"orphans": [
        {"candidate_id": "task379_v3", "task_id": "task379", "expected_point_gain": 0.4,
         "package_path": str(path), "package_sha256": sha,
         "package_in_staged_candidates": True, "package_in_worktree": False,
         "risk_flags": []},
    ]}
    plan = plan_promotions(
        orphan_report=orphan_report, registry_doc={}, queue_doc={"queue": []},
        exclude_tasks={"task379"},
    )
    assert plan["promote_count"] == 0
    assert "exclude-task" in plan["decisions"][0]["reasons"][0]


def test_plan_promotions_skips_worktree_packages_by_default(good_zip):
    path, sha = good_zip
    orphan_report = {"orphans": [
        {"candidate_id": "task098_v1", "task_id": "task098", "expected_point_gain": 0.59,
         "package_path": str(path), "package_sha256": sha,
         "package_in_staged_candidates": False, "package_in_worktree": True,
         "risk_flags": []},
    ]}
    plan = plan_promotions(orphan_report=orphan_report, registry_doc={}, queue_doc={"queue": []})
    assert plan["promote_count"] == 0
    assert "not in staged_candidates" in plan["decisions"][0]["reasons"][0]


def test_plan_promotions_accepts_worktree_when_override_set(good_zip):
    path, sha = good_zip
    orphan_report = {"orphans": [
        {"candidate_id": "task098_v1", "task_id": "task098", "expected_point_gain": 0.59,
         "package_path": str(path), "package_sha256": sha,
         "package_in_staged_candidates": False, "package_in_worktree": True,
         "risk_flags": []},
    ]}
    plan = plan_promotions(
        orphan_report=orphan_report, registry_doc={}, queue_doc={"queue": []},
        require_staged_candidates=False,
    )
    assert plan["promote_count"] == 1


def test_plan_promotions_refuses_stale_lineage_by_default(good_zip):
    """Bug fix 2026-05-18:

    task388 was reviewer-approved at expected +3.27 against
    lineage_base_score=5743.97, but by promote time current canonical was
    5755.05 (+11.08 LB ahead). The reviewer's gain estimate is for a
    baseline that no longer exists. Promoter must refuse unless
    --allow-stale-lineage is set.
    """
    path, sha = good_zip
    orphan_report = {"orphans": [
        {"candidate_id": "task388_stale", "task_id": "task388", "expected_point_gain": 3.27,
         "package_path": str(path), "package_sha256": sha,
         "package_in_staged_candidates": True, "package_in_worktree": False,
         "risk_flags": []},
    ]}
    # Registry has the stale base_score
    registry = {"entries": [{
        "candidate_id": "task388_stale",
        "lineage_base_score": 5743.97,
    }]}
    plan = plan_promotions(
        orphan_report=orphan_report, registry_doc=registry, queue_doc={"queue": []},
        manifest_doc={"score": 5755.05},
    )
    assert plan["promote_count"] == 0
    assert any("stale lineage" in r for r in plan["decisions"][0]["reasons"])


def test_plan_promotions_allows_stale_lineage_with_override(good_zip):
    """Operator can override with --allow-stale-lineage when they've
    manually verified the candidate still wins against the new baseline."""
    path, sha = good_zip
    orphan_report = {"orphans": [
        {"candidate_id": "task388_stale", "task_id": "task388", "expected_point_gain": 3.27,
         "package_path": str(path), "package_sha256": sha,
         "package_in_staged_candidates": True, "package_in_worktree": False,
         "risk_flags": []},
    ]}
    registry = {"entries": [{
        "candidate_id": "task388_stale",
        "lineage_base_score": 5743.97,
    }]}
    plan = plan_promotions(
        orphan_report=orphan_report, registry_doc=registry, queue_doc={"queue": []},
        manifest_doc={"score": 5755.05},
        allow_stale_lineage=True,
    )
    assert plan["promote_count"] == 1


def test_plan_promotions_fresh_lineage_passes(good_zip):
    """When lineage_base_score is close to current canonical (within
    threshold), the candidate should pass."""
    path, sha = good_zip
    orphan_report = {"orphans": [
        {"candidate_id": "task011_fresh", "task_id": "task011", "expected_point_gain": 1.0,
         "package_path": str(path), "package_sha256": sha,
         "package_in_staged_candidates": True, "package_in_worktree": False,
         "risk_flags": []},
    ]}
    registry = {"entries": [{
        "candidate_id": "task011_fresh",
        "lineage_base_score": 5753.0,  # only 2 LB below current
    }]}
    plan = plan_promotions(
        orphan_report=orphan_report, registry_doc=registry, queue_doc={"queue": []},
        manifest_doc={"score": 5755.05},
    )
    assert plan["promote_count"] == 1


def test_plan_promotions_handles_missing_lineage_base_score(good_zip):
    """If lineage_base_score is missing (legacy entry), don't block — let
    other gates decide. The stale check is opportunistic, not a hard
    requirement."""
    path, sha = good_zip
    orphan_report = {"orphans": [
        {"candidate_id": "task011_no_base", "task_id": "task011", "expected_point_gain": 1.0,
         "package_path": str(path), "package_sha256": sha,
         "package_in_staged_candidates": True, "package_in_worktree": False,
         "risk_flags": []},
    ]}
    registry = {"entries": [{"candidate_id": "task011_no_base"}]}  # no lineage_base_score
    plan = plan_promotions(
        orphan_report=orphan_report, registry_doc=registry, queue_doc={"queue": []},
        manifest_doc={"score": 5755.05},
    )
    assert plan["promote_count"] == 1


def test_plan_promotions_skips_hidden_bad_risk_flag(good_zip):
    path, sha = good_zip
    orphan_report = {"orphans": [
        {"candidate_id": "task004", "task_id": "task004", "expected_point_gain": 1.0,
         "package_path": str(path), "package_sha256": sha,
         "package_in_staged_candidates": True, "package_in_worktree": False,
         "risk_flags": ["hidden-bad-family"]},
    ]}
    plan = plan_promotions(orphan_report=orphan_report, registry_doc={}, queue_doc={"queue": []})
    assert plan["promote_count"] == 0
    assert "hidden-bad" in plan["decisions"][0]["reasons"][0]


def test_plan_promotions_promotes_clean_orphan(good_zip):
    path, sha = good_zip
    orphan_report = {"orphans": [
        {"candidate_id": "task388_clean", "task_id": "task388", "expected_point_gain": 3.27,
         "package_path": str(path), "package_sha256": sha,
         "package_in_staged_candidates": True, "package_in_worktree": False,
         "risk_flags": ["public-test-only-non-structural"]},
    ]}
    plan = plan_promotions(orphan_report=orphan_report, registry_doc={}, queue_doc={"queue": []})
    assert plan["promote_count"] == 1
    assert plan["skip_count"] == 0
    assert plan["promote_total_ev"] == 3.27
    assert plan["decisions"][0]["decision"] == "promote"


# --- apply_promotions ------------------------------------------------------


def test_apply_promotions_appends_and_backs_up(tmp_path, good_zip):
    path, sha = good_zip
    queue_path = tmp_path / "submit_queue.json"
    queue_path.write_text(json.dumps({"queue": [_queue_entry("existing")]}, indent=2))
    plan = {
        "promote_count": 1,
        "promote_total_ev": 3.27,
        "promote_entries": [
            reconstruct_queue_entry({"candidate_id": "task388_new", "task_id": "task388",
                                     "expected_point_gain": 3.27,
                                     "package_path": str(path), "package_sha256": sha,
                                     "risk_flags": []}),
        ],
    }
    result = apply_promotions(plan, queue_path)
    assert result["applied_count"] == 1
    # The original queue + new entry should be on disk.
    new_doc = json.loads(queue_path.read_text())
    cids = [e.get("candidate_id") for e in new_doc["queue"]]
    assert "existing" in cids
    assert "task388_new" in cids
    # Backup file should exist.
    assert result["backup_path"]
    assert Path(result["backup_path"]).exists()


def test_apply_promotions_noop_when_promote_count_zero(tmp_path):
    queue_path = tmp_path / "submit_queue.json"
    queue_path.write_text(json.dumps({"queue": []}, indent=2))
    plan = {"promote_count": 0, "promote_total_ev": 0, "promote_entries": []}
    result = apply_promotions(plan, queue_path)
    assert result["applied_count"] == 0
    assert result["backup_path"] is None


def test_apply_promotions_refuses_malformed_queue(tmp_path):
    queue_path = tmp_path / "submit_queue.json"
    queue_path.write_text(json.dumps({"queue": "not-a-list"}))
    plan = {"promote_count": 1, "promote_total_ev": 1.0, "promote_entries": [{"candidate_id": "x"}]}
    with pytest.raises(SystemExit):
        apply_promotions(plan, queue_path)
