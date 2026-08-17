from pathlib import Path

import pytest

import json

from tools.neurogolf_sync_current_best import (
    DEFAULT_REPO_ROOT,
    _append_synthetic_queue_entry,
    _ensure_submit_history_entry,
    effective_main_local_dir,
    effective_orchestrator_current_best,
    main,
    parse_task_id,
    sha256_file,
    update_submit_queue,
)


def _sample_entry() -> dict:
    """Mirrors what manifest_entry() returns for a typical promote."""
    return {
        "candidate_id": "alternates31_on_5800_80",
        "score": 5805.19,
        "previous_global_score": 5800.8,
        "delta_vs_previous_global": 4.39,
        "task_id": "task180",
        "package_sha256": "8526E8DAB6AF3573316ECD04C80F01587F12ED5EDE15F0B8B364D6421DF1A1A2",
        "changed_entries": ["31-task-multi-stack"],
        "source_package_path": "C:/fake/staged_submits/alternates31/submission.zip",
        "sync_status": "synced",
        "synced_at": "2026-05-19T17:37:50Z",
        "last_submit_result": {
            "submitted_at": "2026-05-19T13:45:00Z",
            "public_score": 5805.19,
        },
        "predicted_delta": 5.31,
        "expected_point_gain": 5.31,
        "sync_task_id": "auto-python-sync",
    }


def test_parse_task_id_accepts_numeric_and_task_prefixed_values():
    assert parse_task_id("213") == 213
    assert parse_task_id("task213") == 213
    assert parse_task_id("stack") is None


def test_ensure_submit_history_appends_missing_entry():
    doc: dict = {"submit_history": []}
    _ensure_submit_history_entry(doc, _sample_entry())
    assert len(doc["submit_history"]) == 1
    row = doc["submit_history"][0]
    assert row["candidate_id"] == "alternates31_on_5800_80"
    assert row["package_sha256"].startswith("8526E8DA")
    assert row["status"] == "submitted-complete-promote-synced"
    assert row["kaggle_score"] == 5805.19
    assert row["actual_point_gain"] == 4.39
    assert row["expected_point_gain"] == 5.31


def test_ensure_submit_history_is_idempotent_on_candidate_id_match():
    """Re-running the sync for the same candidate must not duplicate the
    history entry. Idempotency is required because the cron retries
    aggressively after transient failures."""
    doc: dict = {"submit_history": []}
    _ensure_submit_history_entry(doc, _sample_entry())
    _ensure_submit_history_entry(doc, _sample_entry())
    assert len(doc["submit_history"]) == 1


def test_ensure_submit_history_case_insensitive_sha_match():
    """SHA comparison must be case-insensitive (some entries store
    uppercase, some lowercase)."""
    pre_existing = {
        "candidate_id": "alternates31_on_5800_80",
        "package_sha256": "8526e8dab6af3573316ecd04c80f01587f12ed5ede15f0b8b364d6421df1a1a2",  # lowercase
    }
    doc: dict = {"submit_history": [pre_existing]}
    _ensure_submit_history_entry(doc, _sample_entry())  # uppercase
    assert len(doc["submit_history"]) == 1  # not duplicated


def test_ensure_submit_history_creates_field_when_missing():
    """If submit_history doesn't exist on the doc yet, create it."""
    doc: dict = {}
    _ensure_submit_history_entry(doc, _sample_entry())
    assert "submit_history" in doc
    assert len(doc["submit_history"]) == 1


def test_append_synthetic_queue_entry_when_queue_has_no_match():
    """A promoted candidate that bypassed the queue gets a synthetic entry."""
    doc: dict = {"queue": []}
    _append_synthetic_queue_entry(doc, _sample_entry())
    assert len(doc["queue"]) == 1
    item = doc["queue"][0]
    assert item["candidate_id"] == "alternates31_on_5800_80"
    assert item["status"] == "submitted-complete-promote-synced"
    assert item["submit_queue_status"] == "submitted-complete-promote-synced"
    assert item["submit_ready"] is False
    assert item["actual_result"]["public_score"] == 5805.19


def test_append_synthetic_queue_entry_skips_when_candidate_id_already_present():
    """Don't double-insert if the queue already has a row for this candidate."""
    pre_existing = {"candidate_id": "alternates31_on_5800_80", "status": "submitted-pending"}
    doc: dict = {"queue": [pre_existing]}
    _append_synthetic_queue_entry(doc, _sample_entry())
    assert len(doc["queue"]) == 1
    assert doc["queue"][0] is pre_existing  # original untouched


def test_append_synthetic_queue_entry_skips_when_sha_already_present():
    """Match by SHA even when candidate_id differs (lineage-rename edge case)."""
    pre_existing = {
        "candidate_id": "different_name",
        "package_sha256": "8526E8DAB6AF3573316ECD04C80F01587F12ED5EDE15F0B8B364D6421DF1A1A2",
    }
    doc: dict = {"queue": [pre_existing]}
    _append_synthetic_queue_entry(doc, _sample_entry())
    assert len(doc["queue"]) == 1


def test_update_submit_queue_invariant_repair_for_orphan_promote(tmp_path: Path):
    """End-to-end: a candidate that's NOT in queue but IS being promoted
    must come out with BOTH a synthetic queue entry AND a history entry,
    so the next validator run accepts it.

    This is the alternates31_on_5800_80 production failure mode.
    """
    queue_path = tmp_path / "submit_queue.json"
    queue_path.write_text(json.dumps({
        "schema": "neurogolf.submit_queue.v1",
        "queue": [],
        "submit_history": [],
    }), encoding="utf-8")

    actions: list[str] = []
    update_submit_queue(queue_path, _sample_entry(), dry_run=False, actions=actions)

    after = json.loads(queue_path.read_text(encoding="utf-8"))
    # Current best updated as before
    assert after["current_best_manifest"]["candidate_id"] == "alternates31_on_5800_80"
    # NEW: synthetic queue entry created
    assert len(after["queue"]) == 1
    assert after["queue"][0]["candidate_id"] == "alternates31_on_5800_80"
    # NEW: submit_history populated
    assert len(after["submit_history"]) == 1
    assert after["submit_history"][0]["candidate_id"] == "alternates31_on_5800_80"


def test_update_submit_queue_does_not_double_insert_on_re_run(tmp_path: Path):
    """The cron retries every minute. Re-running the promote for the same
    candidate must not balloon queue or history."""
    queue_path = tmp_path / "submit_queue.json"
    queue_path.write_text(json.dumps({
        "schema": "neurogolf.submit_queue.v1",
        "queue": [],
        "submit_history": [],
    }), encoding="utf-8")

    actions: list[str] = []
    for _ in range(3):
        update_submit_queue(queue_path, _sample_entry(), dry_run=False, actions=actions)

    after = json.loads(queue_path.read_text(encoding="utf-8"))
    assert len(after["queue"]) == 1
    assert len(after["submit_history"]) == 1


def test_update_submit_queue_initializes_when_file_missing(tmp_path: Path):
    """2026-05-20 fix: when submit_queue.json doesn't exist (fresh repo /
    corrupted file / first promote on a new machine), update_submit_queue
    must initialize a valid skeleton with empty queue+history and STILL
    write the new current_best_manifest. Original code returned silently
    on a non-dict file, which made the auto_sync validator fail forever
    on the next tick because the promote was never recorded."""
    queue_path = tmp_path / "submit_queue.json"  # NOTE: not creating the file
    actions: list[str] = []
    update_submit_queue(queue_path, _sample_entry(), dry_run=False, actions=actions)
    assert queue_path.exists()
    doc = json.loads(queue_path.read_text(encoding="utf-8"))
    assert doc["current_best_manifest"]["candidate_id"] == "alternates31_on_5800_80"
    assert len(doc["queue"]) == 1  # synthetic entry backfilled
    assert len(doc["submit_history"]) == 1  # history backfilled
    assert doc["schema"] == "neurogolf.submit_queue.v1"


def test_update_submit_queue_initializes_when_file_is_a_list(tmp_path: Path):
    """Defensive: if the file is somehow a list (legacy / corruption),
    still recover by initializing the skeleton."""
    queue_path = tmp_path / "submit_queue.json"
    queue_path.write_text(json.dumps(["unexpected", "shape"]), encoding="utf-8")
    actions: list[str] = []
    update_submit_queue(queue_path, _sample_entry(), dry_run=False, actions=actions)
    doc = json.loads(queue_path.read_text(encoding="utf-8"))
    assert doc.get("current_best_manifest", {}).get("candidate_id") == "alternates31_on_5800_80"


def test_update_v2_manifest_keeps_top_level_in_sync_with_nested(tmp_path: Path):
    """2026-05-20 fix: after the megastack153 incident, v2 promotes MUST
    update both nested current_best AND top-level legacy fields. Original
    code only wrote nested, leaving top-level score/candidate_id stale —
    causing submit_gate's lineage-stale gate to read the wrong number."""
    from tools.neurogolf_sync_current_best import update_v2_manifest

    doc = {
        "schema": "neurogolf.candidate_registry.v2-draft",
        "score": 5835.3,  # STALE — should be overwritten
        "candidate_id": "megastack93_on_5833_74",  # STALE
        "entries": [],
    }
    entry = {
        "candidate_id": "megastack153_on_5835_30",
        "score": 5861.43,
        "task_id": "megastack153",
        "previous_global_score": 5835.3,
        "delta_vs_previous_global": 26.13,
        "package_sha256": "CB7779655379A7293F0E63FE22346499FADC1E8CA279B5937A232E76AAA1A7E0",
    }
    updated = update_v2_manifest(doc, entry, synced_at="2026-05-20T16:00:00Z", source_manifest=tmp_path / "src.json")

    # Nested is updated (existing behavior)
    assert updated["current_best"]["score"] == 5861.43
    assert updated["current_best"]["candidate_id"] == "megastack153_on_5835_30"

    # NEW: top-level legacy fields are ALSO synced
    assert updated["score"] == 5861.43
    assert updated["candidate_id"] == "megastack153_on_5835_30"
    assert updated["task_id"] == "megastack153"
    assert updated["previous_global_score"] == 5835.3
    assert updated["delta_vs_previous_global"] == 26.13


def test_write_json_atomic_survives_simulated_crash(tmp_path: Path):
    """2026-05-20 fix: write_json must be atomic. Simulate a crash by
    pointing the .tmp temp file at an unwritable location partway through
    — the destination must NOT exist (no partial write) OR contain the
    PREVIOUS contents (atomic replace either fully succeeds or didn't run).
    Original code used path.write_text() directly, which would leave a
    truncated file on crash."""
    from tools.neurogolf_sync_current_best import write_json

    target = tmp_path / "manifest.json"
    target.write_text(json.dumps({"old": "value"}) + "\n", encoding="utf-8")

    # Normal write should succeed atomically
    actions: list[str] = []
    write_json(target, {"new": "data"}, dry_run=False, actions=actions)
    assert json.loads(target.read_text(encoding="utf-8")) == {"new": "data"}

    # The .tmp file should not linger
    assert not (target.with_suffix(target.suffix + ".tmp")).exists()


def test_update_submit_queue_matched_queue_entry_still_works(tmp_path: Path):
    """If the queue ALREADY has the candidate (the historical happy-path),
    the existing entry is updated as before and no synthetic entry is added."""
    pre_existing = {
        "candidate_id": "alternates31_on_5800_80",
        "package_sha256": "8526E8DAB6AF3573316ECD04C80F01587F12ED5EDE15F0B8B364D6421DF1A1A2",
        "status": "submitted-pending",
    }
    queue_path = tmp_path / "submit_queue.json"
    queue_path.write_text(json.dumps({
        "schema": "neurogolf.submit_queue.v1",
        "queue": [pre_existing],
        "submit_history": [],
    }), encoding="utf-8")

    actions: list[str] = []
    update_submit_queue(queue_path, _sample_entry(), dry_run=False, actions=actions)

    after = json.loads(queue_path.read_text(encoding="utf-8"))
    # Still one queue entry (the existing one, status updated)
    assert len(after["queue"]) == 1
    assert after["queue"][0]["status"] == "submitted-complete-promote-synced"
    # And submit_history was populated
    assert len(after["submit_history"]) == 1


class Args:
    repo_root = DEFAULT_REPO_ROOT
    orchestrator_current_best = None
    main_local_dir = None


def test_sync_current_best_does_not_default_to_legacy_notongit_mirrors():
    args = Args()

    assert effective_orchestrator_current_best(args) is None
    assert effective_main_local_dir(args) is None


def test_sync_current_best_refuses_hash_mismatch(tmp_path: Path):
    package = tmp_path / "stage" / "submission.zip"
    package.parent.mkdir()
    package.write_bytes(b"new winner")

    with pytest.raises(SystemExit, match="package sha mismatch"):
        main(
            [
                "--repo-root",
                str(tmp_path),
                "--package-path",
                str(package),
                "--package-sha256",
                "0" * 64,
                "--candidate-id",
                "winner",
                "--score",
                "2",
                "--previous-score",
                "1",
                "--delta",
                "1",
                "--changed-entries",
                "task001.onnx",
                "--orchestrator-current-best",
                str(tmp_path / "orchestrator" / "current_best" / "submission.zip"),
            ]
        )


def test_sync_current_best_updates_package_manifest_registry_and_ledger(tmp_path: Path):
    package = tmp_path / "stage" / "submission.zip"
    package.parent.mkdir()
    package.write_bytes(b"new winner")
    package_sha = sha256_file(package)

    (tmp_path / "submissions").mkdir()
    (tmp_path / "submissions" / "submission.zip").write_bytes(b"old")
    (tmp_path / "submissions" / "current_best_manifest.json").write_text(
        '{"schema":"neurogolf.candidate_registry.v2-draft","current_best":{"candidate_id":"old","score":1},"entries":[]}\n',
        encoding="utf-8",
    )
    (tmp_path / ".hermes" / "board").mkdir(parents=True)
    (tmp_path / ".hermes" / "board" / "candidate_registry.json").write_text(
        '{"schema":"neurogolf.candidate_registry.v2-draft","entries":[{"candidate_id":"old","score":1}]}\n',
        encoding="utf-8",
    )
    (tmp_path / ".hermes" / "board" / "submit_queue.json").write_text(
        '{"queue":[{"candidate_id":"winner","submit_ready":true,"sync_followup":{}}]}\n',
        encoding="utf-8",
    )
    (tmp_path / "mainlocal" / "submissions").mkdir(parents=True)
    (tmp_path / "mainlocal" / "submission.zip").write_bytes(b"old mainlocal root")
    (tmp_path / "mainlocal" / "submissions" / "submission.zip").write_bytes(b"old mainlocal submissions")
    (tmp_path / "Logs").mkdir()
    (tmp_path / "Logs" / "submissions_ledger.csv").write_text(
        "date,number,fruit,kind,local_prediction,kaggle_score,verdict,source\n",
        encoding="utf-8",
    )

    main(
        [
            "--repo-root",
            str(tmp_path),
            "--package-path",
            str(package),
            "--package-sha256",
            package_sha,
            "--candidate-id",
            "winner",
            "--score",
            "2",
            "--previous-score",
            "1",
            "--delta",
            "1",
            "--expected-gain",
            "1",
            "--changed-entries",
            "task001.onnx",
            "--sync-task-id",
            "t_sync",
            "--orchestrator-current-best",
            str(tmp_path / "orchestrator" / "current_best" / "submission.zip"),
            "--main-local-dir",
            str(tmp_path / "mainlocal"),
            "--sync-legacy-mirrors",
        ]
    )

    assert sha256_file(tmp_path / "submissions" / "submission.zip") == package_sha
    assert sha256_file(tmp_path / "orchestrator" / "current_best" / "submission.zip") == package_sha
    assert sha256_file(tmp_path / "mainlocal" / "submission.zip") == package_sha
    assert sha256_file(tmp_path / "mainlocal" / "submissions" / "submission.zip") == package_sha
    manifest_text = (tmp_path / "submissions" / "current_best_manifest.json").read_text(encoding="utf-8")
    assert '"candidate_id": "winner"' in manifest_text
    assert '"package_sha256": "' + package_sha + '"' in manifest_text
    orchestrator_manifest = (tmp_path / "orchestrator" / "current_best" / "manifest.json").read_text(encoding="utf-8")
    assert '"candidate_id": "winner"' in orchestrator_manifest
    assert '"package_sha256": "' + package_sha + '"' in orchestrator_manifest
    mainlocal_manifest = (tmp_path / "mainlocal" / "submissions" / "current_best_manifest.json").read_text(
        encoding="utf-8"
    )
    assert '"candidate_id": "winner"' in mainlocal_manifest
    assert '"package_sha256": "' + package_sha + '"' in mainlocal_manifest

    registry_text = (tmp_path / ".hermes" / "board" / "candidate_registry.json").read_text(encoding="utf-8")
    assert registry_text.index('"candidate_id": "winner"') < registry_text.index('"candidate_id": "old"')

    queue_text = (tmp_path / ".hermes" / "board" / "submit_queue.json").read_text(encoding="utf-8")
    assert '"submit_ready": false' in queue_text
    assert '"sync_status": "synced"' in queue_text

    ledger_text = (tmp_path / "Logs" / "submissions_ledger.csv").read_text(encoding="utf-8")
    assert "submit-captain-winner" in ledger_text


def test_sync_current_best_invokes_task_state_sync_by_default(tmp_path: Path, monkeypatch):
    package = tmp_path / "stage" / "submission.zip"
    package.parent.mkdir()
    package.write_bytes(b"new winner")
    package_sha = sha256_file(package)

    (tmp_path / "submissions").mkdir()
    (tmp_path / "submissions" / "submission.zip").write_bytes(b"old")
    (tmp_path / "submissions" / "current_best_manifest.json").write_text(
        '{"schema":"neurogolf.candidate_registry.v2-draft","current_best":{"candidate_id":"old","score":1},"entries":[]}\n',
        encoding="utf-8",
    )
    (tmp_path / "Logs").mkdir()
    (tmp_path / "Logs" / "submissions_ledger.csv").write_text(
        "date,number,fruit,kind,local_prediction,kaggle_score,verdict,source\n",
        encoding="utf-8",
    )

    calls = []

    def fake_sync_task_state(**kwargs):
        calls.append(kwargs)
        return {"ok": True, "updated_task_ids": [1], "skipped": False}

    monkeypatch.setattr("tools.neurogolf_sync_task_state.sync_task_state", fake_sync_task_state)

    main(
        [
            "--repo-root",
            str(tmp_path),
            "--package-path",
            str(package),
            "--package-sha256",
            package_sha,
            "--candidate-id",
            "winner_on_1",
            "--score",
            "2",
            "--previous-score",
            "1",
            "--delta",
            "1",
            "--changed-entries",
            "task001.onnx",
        ]
    )

    assert len(calls) == 1
    assert calls[0]["changed_entries"] == ["task001.onnx"]
    assert calls[0]["candidate_id"] == "winner_on_1"
    assert calls[0]["mark_shipped"] is True
